"""Alternate editing engines behind the Flux2Canvas protocol.

A RemoteEngineCanvas keeps ALL of Flux2Canvas's pixel-space session state
(canvas, revisions, keyframes, SAM cutouts, the vLLM judge) but replaces the
diffusion itself with an HTTP sidecar that exposes:

    POST /load      -> {"ok": true}          load weights into VRAM
    POST /unload    -> {"ok": true}          free VRAM (process stays up)
    GET  /healthz   -> 200 when responsive (loaded or not)
    POST /generate  {"prompt", "size", "seed"}            -> {"image_b64"}
    POST /edit      {"images": [b64...], "prompt", "size", "seed"}
                                                          -> {"image_b64"}

Klein-only machinery (latent chaining, token merge/anchor, delete stencils)
does not apply: every render is a stock pixel->pixel edit. Region scope is
still guaranteed by pixel splice, and move/delete ops still travel as
templated world-state clauses.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import random
import urllib.request

from .flux2 import Flux2Canvas, EMPTY_SCENE_PROMPT, DEFAULT_PROMPT_SUFFIX

log = logging.getLogger("renderer")


class RemoteEngineCanvas(Flux2Canvas):
    def __init__(self, name: str, url: str) -> None:
        super().__init__()
        self.engine_name = name
        self.url = url.rstrip("/")
        self._remote_loaded = False

    # ---- HTTP plumbing ----
    def _post(self, path: str, payload: dict | None = None,
              timeout: float = 180.0) -> dict:
        req = urllib.request.Request(
            self.url + path, data=json.dumps(payload or {}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)

    # ---- lifecycle ----
    def load(self) -> None:
        if self._remote_loaded:
            return
        log.info("engine %s: loading remote weights", self.engine_name)
        self._post("/load", timeout=900.0)
        self._remote_loaded = True
        log.info("engine %s: loaded", self.engine_name)

    def unload(self) -> None:
        try:
            self._post("/unload", timeout=120.0)
        except Exception:
            log.exception("engine %s: unload failed (continuing)",
                          self.engine_name)
        self._remote_loaded = False
        self._sam = None
        self._sam_proc = None
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
        log.info("engine %s: unloaded", self.engine_name)

    # klein-latent concepts are no-ops here
    def rebase_anchor(self) -> None:
        pass

    def set_model(self, tag: str) -> None:
        pass

    # ---- rendering ----
    @staticmethod
    def _b64(img) -> str:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def render_looped(self, prompt: str, strength: float, entities,
                      instruction: str, requirements=None, on_preview=None,
                      camera: bool = False, draft_size: int = 0,
                      preserve_last: bool = False, color_guard: bool = False,
                      plan=None, should_abort=None, source=None,
                      **kw) -> str:
        # no draft/promote latent ladder on remote engines: single shot
        kw.pop("reset", None)
        return self.render(prompt, strength, entities,
                           on_preview=on_preview, source=source,
                           instruction=instruction, **kw)

    def render(self, prompt: str, strength: float, entities,
               on_preview=None, cancel_check=None, source=None,
               instruction=None, reset=False, lane=None, commit_rev=None,
               do_snapshot=False, move_ops=None, camera=False, region=None,
               promote_ctx=None, fresh_seed=False, ink=None,
               style_refs=None, style_suffix=None,
               preserve_names=None) -> str:
        from PIL import Image, ImageFilter

        if not self._remote_loaded:
            self.load()
        editing = self.canvas is not None and not reset
        provisional = lane == "provisional"
        seed = random.randrange(1 << 31)
        self._last_deletes = []
        self._composite_pin = False
        self._unfound_deletes = []

        if editing:
            call_prompt = (instruction or prompt or "").strip()
            if move_ops:
                # geometry intent travels as templated world-state clauses
                # (same contract as the klein path, minus the composites)
                clauses = "".join(self._move_clause(op) for op in move_ops)
                verbs = []
                for op in move_ops:
                    n = (op.get("name") or "object").replace("_", " ")
                    if op.get("delete"):
                        verbs.append(f"remove {'all the ' if op.get('all') else 'the '}{n}")
                    elif op.get("insert"):
                        verbs.append(f"add a {n} at the {self._pos_words(op.get('to_box') or {})}")
                    else:
                        verbs.append(f"move the {n} to the {self._pos_words(op.get('to_box') or {})}")
                call_prompt = ((call_prompt + ". ") if call_prompt else "") \
                    + ", ".join(verbs).capitalize() + "." + clauses
            if not call_prompt:
                call_prompt = f"Redraw to exactly match: {prompt}"
            base = self.scratch if (provisional and self.scratch is not None) \
                else self.canvas
            out = self._post("/edit", {
                "images": [self._b64(base)], "prompt": call_prompt,
                "size": self.size, "seed": seed}, timeout=600.0)
        else:
            call_prompt = ((prompt or "").strip() or EMPTY_SCENE_PROMPT)
            if style_suffix:
                call_prompt += " " + style_suffix
            else:
                call_prompt += DEFAULT_PROMPT_SUFFIX
            payload = {"prompt": call_prompt, "size": self.size,
                       "seed": seed}
            if style_refs:
                # refs ride as edit conditioning when the engine supports it
                payload["images"] = [r for r in style_refs[:2]]
                payload["reference_only"] = True
            out = self._post("/generate" if not style_refs else "/edit",
                             payload, timeout=600.0)

        image = Image.open(io.BytesIO(
            base64.b64decode(out["image_b64"]))).convert("RGB")
        if image.size != (self.size, self.size):
            image = image.resize((self.size, self.size), Image.LANCZOS)

        if editing and region and not move_ops:
            # region scope is a pixel guarantee, engine-independent: only
            # the circled pixels may change
            try:
                import numpy as np
                rm = self._rle_to_mask(region["mask_rle"],
                                       region["mask_size"])
                if rm.size != (self.size, self.size):
                    rm = rm.resize((self.size, self.size), Image.NEAREST)
                soft = rm.filter(ImageFilter.GaussianBlur(6))
                base = self.scratch if (provisional and self.scratch
                                        is not None) else self.canvas
                image = Image.composite(image, base, soft)
            except Exception:
                log.exception("region splice failed (unscoped edit kept)")

        if editing and provisional:
            self.scratch = image
        else:
            self.canvas = image
            self.canvas_lat = None
            self.scratch = None
            self.scratch_lat = None
            self.scratch_anchor = None
            self.anchor = None
            self.rev += 1
            if not editing:
                self.needs_floor_calib = True
            self.edits_since_root = (self.edits_since_root + 1 if editing
                                     else 0)
        log.info("[%s] %s: %r", self.engine_name,
                 "edit" if editing else "root", call_prompt[:70])

        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=85)
        jpeg = buf.getvalue()
        if do_snapshot and not provisional and commit_rev is not None:
            self.snapshot(commit_rev, jpeg)
        return base64.b64encode(jpeg).decode()


ENGINE_URLS = {
    "u15": "http://engine-u15:8000",
    "h3": "http://engine-h3:8000",
}


def _engine_url(engine: str) -> str:
    import os
    return os.environ.get(f"ENGINE_{engine.upper()}_URL",
                          ENGINE_URLS[engine])


def free_other_sidecars(keep: str) -> None:
    """Tell every REMOTE engine sidecar that isn't `keep` to release its
    GPU memory. Best-effort and idempotent: a sidecar that's already empty,
    still booting, or briefly unreachable just no-ops. This is the safety
    net that makes engine switching survive leftover VRAM — nothing but the
    active engine may hold weights, so the next load always has room."""
    if keep == "comfy":
        return
    for name, url in ENGINE_URLS.items():
        if name == keep:
            continue
        try:
            req = urllib.request.Request(
                _engine_url(name).rstrip("/") + "/unload",
                data=b"{}", headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120.0):
                pass
            log.info("freed sidecar %s (keeping %s)", name, keep)
        except Exception as e:
            log.info("sidecar %s free skipped (%s)", name, e)


def make_canvas(engine: str):
    """Engine factory used by the renderer's set_engine switch."""
    if engine == "comfy":
        from .comfy_canvas import ComfyCanvas
        return ComfyCanvas()
    if engine == "klein":
        return Flux2Canvas()
    if engine in ENGINE_URLS:
        return RemoteEngineCanvas(engine, _engine_url(engine))
    raise ValueError(f"unknown engine {engine!r}")
