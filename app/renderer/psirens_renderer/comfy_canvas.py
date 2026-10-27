"""ComfyUI canvas: every frame is a FLUX 3 Image Klein workflow run.

The renderer keeps the Psirens canvas contract (two lanes, keyframes,
latest-wins provisional churn) but the pixels come from ComfyUI, driven
through the Comfy SDK (Comfy API v2, served by comfy-api-proxy):

  - root   (no canvas yet / reset): workflows/psirens_root_api.json
  - edit   (canvas latent still in ComfyUI): psirens_edit_latent_api.json
           -> references the stored latent directly (no decode/encode trip)
              and freezes untouched tokens (Psirens Token Passthrough)
  - edit   (latent gone, size changed, undo past eviction):
           psirens_edit_pixels_api.json -> uploads the current frame

Short spoken instructions are expanded with BFL's FLUX 3 prompt-upsampler
system prompt by the same LLM that runs the scene editor (it knows what is
on the canvas from the scene state).
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import random
import time
import urllib.request
import uuid

from .flux2 import EMPTY_SCENE_PROMPT, Flux2Canvas
from .preview import RenderPreempted

log = logging.getLogger("renderer.comfy")
logging.getLogger("httpx").setLevel(logging.WARNING)

WF_DIR = os.environ.get("PSIRENS_WORKFLOWS", "/srv/workflows")
COMFY_URL = os.environ.get("COMFY_URL", "http://comfyui:8188").rstrip("/")
STEPS = int(os.environ.get("KLEIN_STEPS", "6"))
FAST_STEPS = int(os.environ.get("KLEIN_FAST_STEPS", "4"))
PASS_TAU = float(os.environ.get("PSIRENS_PASS_TAU", "0.35"))
PASS_COVER = float(os.environ.get("PSIRENS_PASS_COVER", "0.6"))
UPSAMPLE = os.environ.get("PSIRENS_UPSAMPLE", "1") not in ("0", "false", "no")
LLM_URL = os.environ.get("UPSAMPLE_URL", "http://vllm:8000/v1").rstrip("/")
LLM_MODEL = os.environ.get("EDITOR_MODEL", "")
UPSAMPLE_TIMEOUT_S = float(os.environ.get("UPSAMPLE_TIMEOUT_S", "6"))
ROOT_SUFFIX = os.environ.get("PSIRENS_ROOT_SUFFIX", "")

# BFL's FLUX 3 edit upsampler instructions (from the official ComfyUI
# template), run on the scene-editor LLM (it sees the current frame) instead
# of a second 9B model, so the GPU budget stays one card per role.
UPSAMPLE_SYSTEM = (
    "You are the prompt upsampler of an image-editing model. You see the "
    "current image (plus a short scene description) and the user's short "
    "request. Expand "
    "it into ONE detailed, self-contained instruction: the caption the "
    "editing model receives together with the image. Output only the "
    "instruction as one continuous English prose block, with no greetings, "
    "explanations, Markdown, JSON, coordinates, ids or angle-bracket tokens. "
    "Read a vague or conversational request as the most plausible edit of "
    "the image.\n\n"
    "When the image stays and gets bounded changes, write an imperative "
    "editing instruction (\"Remove the ...\", \"Change the ... to ...\", "
    "\"Zoom in on ...\"); a whole-image transformation opens with its scope "
    "(\"Convert the entire image into ...\", \"Shift the camera viewpoint "
    "to ...\", then where the major elements should now be).\n\n"
    "Keep every explicit edit, constraint, count, position, style cue and "
    "piece of wording, and decide what is left open as one concrete plan. "
    "Identify each referent unambiguously by its appearance, never mention "
    "content that is not there. State every end state concretely (which "
    "color, how large, which pose, which material) and give new content "
    "enough appearance to be generated. Cover the knock-on effects of the "
    "change (shadows, reflections, occlusion, perspective, scale, lighting) "
    "and say what fills a removed region. Describe the change, not the "
    "whole scene, and close with one compact clause naming what stays "
    "unchanged. Directions are frame-relative; use no evaluative or quality "
    "words. Typical length is 40 to 100 words.\n\nOutput only the "
    "instruction.")


def _jpeg_b64(img, quality: int = 85) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


class ComfyCanvas(Flux2Canvas):
    engine_name = "comfy"

    def __init__(self) -> None:
        super().__init__()
        from comfy_sdk import Comfy
        self.client = Comfy()                     # COMFY_BASE_URL from env
        self.templates = {
            name: json.load(open(os.path.join(WF_DIR, f"psirens_{name}_api.json")))
            for name in ("root", "edit_latent", "edit_pixels")}
        self.session = uuid.uuid4().hex[:12]
        self.slot: str | None = None           # committed canvas latent slot
        self.slot_size = 0
        self.scratch_slot: str | None = None
        self.rev_slots: dict[int, tuple[str | None, int]] = {}
        self._n = 0
        self._loaded = False
        self.last_stats = ""

    # ---- lifecycle ----
    def _comfy_get(self, path: str, timeout: float = 5.0):
        with urllib.request.urlopen(COMFY_URL + path, timeout=timeout) as r:
            return json.load(r)

    def load(self) -> None:
        if self._loaded:
            return
        deadline = time.monotonic() + 900
        while True:
            try:
                self._comfy_get("/system_stats")
                break
            except Exception:
                if time.monotonic() > deadline:
                    raise
                time.sleep(3)
        t0 = time.monotonic()
        try:   # warm-up: loads Klein + text encoder + VAE into VRAM
            wf = self._workflow("root", "a plain gray studio backdrop",
                                seed=1, steps=1, size=512, slot="warmup")
            self._run(wf)
            log.info("comfy warm-up done in %.1fs", time.monotonic() - t0)
        except Exception:
            log.exception("comfy warm-up failed (first render pays the load)")
        self._loaded = True

    def unload(self) -> None:
        try:
            req = urllib.request.Request(
                COMFY_URL + "/free",
                data=json.dumps({"unload_models": True,
                                 "free_memory": True}).encode(),
                headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=30).close()
        except Exception:
            log.exception("comfy /free failed")
        self._loaded = False

    def reset_session(self) -> None:
        super().reset_session()
        self.session = uuid.uuid4().hex[:12]
        self.slot = None
        self.scratch_slot = None
        self.rev_slots = {}

    # klein-in-process concepts that don't apply here
    def rebase_anchor(self) -> None:
        pass

    def set_model(self, tag: str) -> None:
        pass

    def calibrate_scene_floor(self) -> None:
        self.needs_floor_calib = False

    def ensure_embedding(self) -> None:
        pass

    def point_segment(self, nx: float, ny: float):
        return None

    def lasso_segment(self, points):
        return None

    def cut_object(self, b64: str, name: str):
        return None

    def judge_final(self, *a, **kw):
        return None

    def render_looped(self, prompt, strength, entities, instruction,
                      requirements=None, on_preview=None, camera=False,
                      draft_size=0, preserve_last=False, color_guard=False,
                      plan=None, should_abort=None, source=None, **kw):
        kw.pop("reset", None)
        return self.render(prompt, strength, entities, on_preview=on_preview,
                           source=source, instruction=instruction,
                           camera=camera, **kw)

    # ---- keyframes: remember which ComfyUI latent slot each rev used ----
    def snapshot(self, commit_rev: int, jpeg: bytes) -> None:
        super().snapshot(commit_rev, jpeg)
        self.rev_slots[commit_rev] = (self.slot, self.slot_size)

    def restore(self, commit_rev: int):
        out = super().restore(commit_rev)
        if out is not None:
            self.slot, self.slot_size = self.rev_slots.get(commit_rev,
                                                           (None, 0))
            self.scratch_slot = None
        return out

    # ---- workflow plumbing ----
    def _workflow(self, kind: str, text: str, *, seed: int, steps: int,
                  size: int, slot: str, ref_slot: str | None = None,
                  asset=None):
        from comfy_sdk import Workflow  # noqa: F401  (type only)
        g = json.loads(json.dumps(self.templates[kind]))
        g["4"]["inputs"]["text"] = text
        g["7"]["inputs"]["seed"] = seed
        g["7"]["inputs"]["steps"] = steps
        g["10"]["inputs"]["session"] = self.session
        g["10"]["inputs"]["slot"] = slot
        if kind == "root":
            g["6"]["inputs"]["width"] = size
            g["6"]["inputs"]["height"] = size
        else:
            g["24"]["inputs"]["tau"] = PASS_TAU
            g["24"]["inputs"]["max_cover"] = PASS_COVER
        if kind == "edit_latent":
            g["20"]["inputs"]["session"] = self.session
            g["20"]["inputs"]["slot"] = ref_slot
        wf = self.client.workflows.from_json(g)
        if kind == "edit_pixels":
            wf.set_input("11", "image", asset)
        return wf

    def _run(self, wf, on_preview=None, cancel_check=None):
        """Submit, stream previews, honor preemption; return PNG bytes."""
        from comfy_sdk import Preview, Progress, StatusChange
        job = self.client.submit(wf)
        step, steps = 0, 0
        for ev in job.events():
            if cancel_check is not None and cancel_check():
                try:
                    job.cancel()
                except Exception:
                    pass
                raise RenderPreempted()
            if isinstance(ev, Progress):
                if ev.steps:
                    step, steps = ev.step or 0, ev.steps
            elif isinstance(ev, Preview) and on_preview is not None:
                try:
                    img = ev.to_pil().convert("RGB")
                    on_preview(_jpeg_b64(img, 70), step, steps)
                except Exception:
                    pass
            elif isinstance(ev, StatusChange) and ev.status in (
                    "succeeded", "failed", "cancelled", "canceled"):
                break
        job = job.result()
        outs = job.get_outputs("9")
        if not outs:
            raise RuntimeError(f"comfy job {job.id} produced no frame")
        return outs[0].to_bytes()

    def _upsample(self, instruction: str, scene: str, image=None) -> str:
        if not (UPSAMPLE and LLM_MODEL and instruction):
            return instruction
        content: list = []
        if image is not None:
            small = image.copy()
            small.thumbnail((512, 512))
            content.append({"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + _jpeg_b64(small, 80)}})
        content.append({"type": "text", "text":
                        f"Scene description: {scene or 'unknown'}\n\n"
                        f"Request: {instruction}"})
        body = {
            "model": LLM_MODEL, "temperature": 0.2, "max_tokens": 220,
            "messages": [
                {"role": "system", "content": UPSAMPLE_SYSTEM},
                {"role": "user", "content": content}],
            "chat_template_kwargs": {"enable_thinking": False},
        }
        t0 = time.monotonic()
        try:
            req = urllib.request.Request(
                LLM_URL + "/chat/completions", data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=UPSAMPLE_TIMEOUT_S) as r:
                text = json.load(r)["choices"][0]["message"]["content"]
            text = " ".join(text.split()).strip()
            if len(text) < 8:
                return instruction
            log.info("upsampled in %dms: %r -> %r",
                     (time.monotonic() - t0) * 1000, instruction[:60],
                     text[:120])
            return text
        except Exception as e:
            log.info("upsample skipped (%s); using the raw instruction", e)
            return instruction

    # ---- rendering ----
    def render(self, prompt: str, strength: float, entities,
               on_preview=None, cancel_check=None, source=None,
               instruction=None, reset=False, lane=None, commit_rev=None,
               do_snapshot=False, move_ops=None, camera=False, region=None,
               promote_ctx=None, fresh_seed=False, ink=None,
               style_refs=None, style_suffix=None,
               preserve_names=None) -> str:
        from PIL import Image, ImageFilter

        if not self._loaded:
            self.load()
        provisional = (lane == "provisional"
                       or (lane is None and source in ("reflex", "overlay",
                                                       "draft")))
        editing = self.canvas is not None and not reset
        self._last_deletes = []
        self._composite_pin = False
        self._unfound_deletes = []
        steps = FAST_STEPS if provisional else STEPS

        if editing:
            call = (instruction or "").strip()
            if move_ops:
                verbs = []
                for op in move_ops:
                    n = (op.get("name") or "object").replace("_", " ")
                    where = self._pos_words(op.get("to_box") or {})
                    if op.get("delete"):
                        verbs.append(f"remove {'all the ' if op.get('all') else 'the '}{n}")
                    elif op.get("insert"):
                        verbs.append(f"add a {n} at the {where}")
                    else:
                        verbs.append(f"move the {n} to the {where}")
                call = ((call + ". ") if call else "") \
                    + ", ".join(verbs).capitalize() + "."
            if not call:
                call = f"Redraw the image to match: {prompt}"
            if not provisional:
                call = self._upsample(call, prompt, self.canvas)
            # edits share the root's seed (a chain on one noise trajectory
            # repaints its past more faithfully); camera moves and region
            # edits need a fresh roll to escape the old framing/content
            seed = (random.randrange(1 << 31)
                    if (fresh_seed or camera or region)
                    else getattr(self, "session_seed", 42))
            use_scratch = (provisional and self.scratch is not None
                           and self.scratch_rev == self.rev)
            base_img = self.scratch if use_scratch else self.canvas
            ref_slot = self.scratch_slot if use_scratch else self.slot
            self._n += 1
            slot = "scratch" if provisional else f"r{self._n}"
            png = None
            if ref_slot and self.slot_size == self.size:
                try:
                    png = self._run(self._workflow(
                        "edit_latent", call, seed=seed, steps=steps,
                        size=self.size, slot=slot, ref_slot=ref_slot),
                        on_preview, None if not provisional else cancel_check)
                except RenderPreempted:
                    raise
                except Exception as e:
                    log.info("latent edit failed (%s): falling back to pixels", e)
            if png is None:
                img = base_img
                if img.size != (self.size, self.size):
                    img = img.resize((self.size, self.size), Image.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                asset = self.client.assets.from_bytes(
                    buf.getvalue(), filename="canvas.png",
                    content_type="image/png")
                png = self._run(self._workflow(
                    "edit_pixels", call, seed=seed, steps=steps,
                    size=self.size, slot=slot, asset=asset),
                    on_preview, None if not provisional else cancel_check)
        else:
            call = (prompt or "").strip() or EMPTY_SCENE_PROMPT
            if style_suffix:
                call += ", " + style_suffix.strip(" ,.")
            elif ROOT_SUFFIX:
                call += ROOT_SUFFIX
            self.session_seed = random.randrange(1 << 31)
            seed = self.session_seed
            self._n += 1
            slot = f"r{self._n}"
            png = self._run(self._workflow(
                "root", call, seed=seed, steps=STEPS, size=self.size,
                slot=slot), on_preview, None)

        image = Image.open(io.BytesIO(png)).convert("RGB")
        if image.size != (self.size, self.size):
            image = image.resize((self.size, self.size), Image.LANCZOS)

        if editing and region and not move_ops:
            # region scope is a pixel guarantee: only the circled pixels change
            try:
                rm = self._rle_to_mask(region["mask_rle"], region["mask_size"])
                if rm.size != (self.size, self.size):
                    rm = rm.resize((self.size, self.size), Image.NEAREST)
                soft = rm.filter(ImageFilter.GaussianBlur(6))
                base = self.scratch if (provisional and self.scratch
                                        is not None) else self.canvas
                if base.size != image.size:
                    base = base.resize(image.size, Image.LANCZOS)
                image = Image.composite(image, base, soft)
                slot = None          # latent no longer matches the pixels
            except Exception:
                log.exception("region splice failed (unscoped edit kept)")

        if editing and provisional:
            self.scratch = image
            self.scratch_rev = self.rev
            self.scratch_slot = slot
        else:
            self.canvas = image
            self.slot = slot
            self.slot_size = self.size
            self.scratch = None
            self.scratch_slot = None
            self.rev += 1
            self.edits_since_root = (self.edits_since_root + 1 if editing
                                     else 0)
        log.info("[comfy] %s%s: %r", "edit" if editing else "root",
                 " (provisional)" if provisional and editing else "",
                 call[:90])
        jpeg = io.BytesIO()
        image.save(jpeg, format="JPEG", quality=88)
        jpeg = jpeg.getvalue()
        if do_snapshot and not provisional and commit_rev is not None:
            self.snapshot(commit_rev, jpeg)
        return base64.b64encode(jpeg).decode()
