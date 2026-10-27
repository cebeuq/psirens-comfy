"""FLUX.2 Klein canvas: edit-native rendering.

The core idea (owner's call): only the ROOT image is an original
generation — everything after is the model's EDIT feature applied to the
current canvas. The picture you saw stays the picture you have; a change
touches only what the instruction names. "Morph, not teleport" for real,
instead of simulated with a fixed seed.

  - render target has no edit_instruction, or reset=True, or no canvas yet
    -> text-to-image from the full prompt (root / re-anchor)
  - otherwise -> edit conditioned on the current canvas

Long edit chains age the picture through two compounding losses, and the
stock pipeline call hides both, so edits run through _call_klein (a
faithful replica of Flux2KleinPipeline.__call__, diffusers 0.39) with two
structural fixes:

  1. LATENT CHAINING — the stock call VAE-encodes the conditioning image
     every edit, but the previous edit's final latents are ALREADY in that
     exact space (normalized+patchified+packed). Conditioning on them
     directly skips one decode->encode roundtrip per edit; the roundtrip
     loss that used to accumulate never happens. Pixels only get
     re-encoded where pixels were genuinely produced outside the model
     (composite moves, restores of evicted latents, resolution changes).
  2. TOKEN PASSTHROUGH — the model regenerates EVERY token from noise each
     edit; untouched regions survive only via attention to the reference,
     so they get subtly repainted (texture crisping, saturation creep)
     over and over. Generation tokens and conditioning tokens live on the
     same grid, index-aligned — so after denoising, tokens whose delta to
     the conditioning sits at the repaint-noise floor are copied BIT-EXACT
     from the conditioning. What the edit didn't touch stops aging, no
     matter how long the chain runs.
"""
from __future__ import annotations

import base64
import io
import logging
import os
import random

from .preview import LatentPreviewer, RenderPreempted

log = logging.getLogger("renderer.flux2")

MODEL = os.environ.get("FLUX2_MODEL", "black-forest-labs/FLUX.2-klein-4B")
# quality mode: a second, bigger Klein loaded alongside (quantized, both
# transformers are a few GB). Empty string disables the second model.
MODEL_HQ = os.environ.get("FLUX2_MODEL_HQ", "ModelsLab/FLUX.2-klein-9B")
# nvfp4-wo (weight-only: most of the speed, bf16 activations keep the
# repaint noise floor low enough for drift protection) | nvfp4 (fastest,
# but W4A4 noise disables token passthrough) | none
QUANT = os.environ.get("FLUX2_QUANT", "nvfp4-wo")
DEFAULT_PREF = os.environ.get("FLUX2_DEFAULT", "fast")   # fast | hq
SIZE = int(os.environ.get("FLUX2_SIZE", "512"))
STEPS = int(os.environ.get("FLUX2_STEPS", "4"))
FAST_STEPS = int(os.environ.get("FLUX2_FAST_STEPS", "2"))
GUIDANCE = float(os.environ.get("FLUX2_GUIDANCE", "1.0"))
# FLUX2_SEED: set an integer for deterministic renders (debugging);
# unset/empty = a fresh random seed EVERY render, so regenerating the same
# thing gives a genuinely new roll. Continuity does not depend on the seed
# anymore — it comes from edit-native conditioning + latent chaining.
_SEED_ENV = os.environ.get("FLUX2_SEED", "").strip()
FIXED_SEED = int(_SEED_ENV) if _SEED_ENV else None
PREVIEW = os.environ.get("FLUX2_PREVIEW", "1") not in ("0", "false", "no")
PREVIEW_SIZE = int(os.environ.get("FLUX2_PREVIEW_SIZE", "288"))

DEFAULT_PROMPT_SUFFIX = ", high quality"
EMPTY_SCENE_PROMPT = "a blank canvas, soft neutral gray studio backdrop"
FAST_SOURCES = {"reflex", "overlay", "draft"}
# point-promptable SAM for the mouse layer: "the user clicked HERE — what
# object did they mean?" (SAM 3 is detector-only; SAM 1 point prompts are
# built for exactly this: heavy image encoding once per frame, then each
# click is a ~10ms mask decode)
# latent chaining + token passthrough (see module docstring). Kill switch:
# FLUX2_LATENT_CHAIN=0 restores the stock pixel-roundtrip pipeline call.
LATENT_CHAIN = os.environ.get("FLUX2_LATENT_CHAIN", "1") not in ("0", "false",
                                                                 "no")
# Per-token RMS delta (normalized latent units) separating repaint noise
# from intended change. Measured on-device (tools/probe_drift.py): pure
# repaint noise floor p95 ~= 0.14, the SUBTLEST intended global edit
# ("slightly warmer") median ~= 0.21, local-edit regions 0.3-0.8. 0.17
# sits in the gap: drift freezes, every intended edit passes. 0 disables
# passthrough while keeping latent chaining.
PASS_TAU = float(os.environ.get("FLUX2_PASS_TAU", "0.16"))
# quantized weights repaint with a HIGHER noise floor than bf16 (different
# numerics -> less correlated trajectories); each model gets its own tau.
# Defaults re-measured in production via the pass-stats log line.
PASS_TAU_HQ = float(os.environ.get("FLUX2_PASS_TAU_HQ", str(PASS_TAU)))
# ---- anchor passthrough: kills the FP4 random walk ----
# The old passthrough compared each render against the PREVIOUS render's
# latents, so any noise token that escaped tau once had its noisy value
# adopted as the next reference — drift accumulated as a random walk.
# Anchored mode compares against a per-token ANCHOR (the value from the
# last render that INTENTIONALLY changed it): sub-tau_lo excursions snap
# back to the anchor bit-exactly (erased, never integrated); adopting a
# new value requires clearing tau_hi (clearly above the measured noise
# floor) or persisting in the tau_lo..tau_hi band for K consecutive
# renders (noise doesn't persist across varied prompts; real subtle
# edits do). FLUX2_ANCHOR=0 restores the legacy passthrough bit for bit.
ANCHOR = os.environ.get("FLUX2_ANCHOR", "1") not in ("0", "false", "no")
ANCHOR_K = int(os.environ.get("FLUX2_ANCHOR_K", "3"))
# band-DENSITY gate: FP4 band noise is scattered singleton clusters while
# any real subtle edit (even a local one) is a dense contiguous region.
# Band tokens in a sparse 7x7 neighborhood snap back to the anchor, so
# noise can never persist into the conditioning and never be adopted —
# measured: 12-edit no-op chain goes from 27.6dB (legacy) / 32.3dB
# (anchor alone, declining) to 138dB FLAT (bit-exact), with global and
# local edit survival intact. 0 disables the gate.
ANCHOR_DENS = float(os.environ.get("FLUX2_ANCHOR_DENS", "0.35"))
# coherence override: FP4 repaint noise is scattered (tau_lo is calibrated
# at the no-op floor's p99, so a true no-op exceeds it on ~1% of tokens);
# a FOLLOWED global instruction moves a large coherent fraction. When more
# than this fraction of tokens moved, the render IS the edit — freezing
# would eat the model's obedience (the reason finals under-followed
# subtle instructions while the pre-merge preview looked right).
MERGE_COVER = float(os.environ.get("FLUX2_MERGE_COVER", "0.30"))
# ...and the moved tokens must agree on a DIRECTION: FP4 noise is
# directionally random (the mean delta vector across tokens cancels to
# ~1/sqrt(N) of the per-token magnitude), while a followed global
# instruction shifts the image coherently. Both gates must pass.
MERGE_COH = float(os.environ.get("FLUX2_MERGE_COH", "0"))
_TAU_HI = os.environ.get("FLUX2_PASS_TAU_HI", "").strip()
PASS_TAU_HI = float(_TAU_HI) if _TAU_HI else None       # None = derived
_TAU_HI_HQ = os.environ.get("FLUX2_PASS_TAU_HI_HQ", "").strip()
PASS_TAU_HI_HQ = float(_TAU_HI_HQ) if _TAU_HI_HQ else PASS_TAU_HI
# auto-calibrate per-model thresholds at load from THIS build's measured
# no-op repaint floor (quantization numerics change the floor; a stale
# constant silently miscalibrates the whole freeze stack). The env
# constants above stay as clamp references and hard overrides.
AUTOTAU = os.environ.get("FLUX2_AUTOTAU", "1") not in ("0", "false", "no")
# ---- FP4 floor levers (probe-gated; defaults preserve today's quant) ----
# comma-separated fqn substrings of Linears to KEEP in bf16. Candidates:
# the seq-len-1 adaLN modulation GEMVs (norm1, norm1_context, norm_out) —
# near-zero FLOPs saved by quantizing them, and their per-channel error is
# spatially COHERENT, which 3x3 pooling cannot average away.
FP4_KEEP_BF16 = [s.strip() for s in
                 os.environ.get("FLUX2_FP4_KEEP_BF16", "").split(",")
                 if s.strip()]
# "off" -> static per-tensor scales: quantization error becomes a local
# function of each token's own features instead of re-rolling globally
# whenever ANY token in the activation changes
FP4_TENSOR_SCALE = os.environ.get("FLUX2_FP4_TENSOR_SCALE", "dynamic")
# text-encoder quantization: the bf16 TE was the single biggest VRAM item
# on the card (16GB for the 9B's). NVFP4 shrinks it ~4x; prompt-embedding
# LMs tolerate weight quantization well. "none" restores bf16.
TE_QUANT = os.environ.get("FLUX2_TE_QUANT", "nvfp4")
# cuDNN: the blanket disable was a GB10/Spark hardware workaround (silent
# VAE conv corruption). On sane GPUs it just slows the VAE; the startup
# roundtrip verify is the categorical guard either way. FLUX2_CUDNN=0
# restores the disable if a future host misbehaves.
CUDNN_OK = os.environ.get("FLUX2_CUDNN", "1") not in ("0", "false", "no")
# torch.compile of the transformer: measured 1.3x on edits (1235->945ms)
# for ~40-60s one-time warmup at startup (plus a one-time recompile the
# first render after a resolution switch)
COMPILE = os.environ.get("FLUX2_COMPILE", "1") not in ("0", "false", "no")
SAM_MODEL = os.environ.get("SAM_MODEL", "facebook/sam-vit-base")
# ONE fixed repair prompt for every composite, never phrased per-edit:
# geometry is already done in code — the model's only job is paint
REPAIR_PROMPT = ("Refine this image: blend all elements seamlessly, fix any "
                 "seams, artifacts, or unfinished regions, and fill in any "
                 "gaps naturally. Erase any faint ghost silhouettes, stains, "
                 "or leftover shadows where an object used to be — those "
                 "areas are plain background. Do not move, add, or remove "
                 "any real object.")


MAX_KEYFRAMES = int(os.environ.get("FLUX2_MAX_KEYFRAMES", "64"))

# ---- draft-verify-refine loop (design: docs/DRAFT-LOOP notes) ----
# Every eligible committed edit first renders a low-res DRAFT, the editor
# LLM judges instruction compliance (before+after+requirements), failed
# drafts get corrective redraws, and the accepted draft is PROMOTED to
# full resolution by upsampling its latents and resuming the mu-shifted
# schedule mid-way with the SAME shared noise field (drawn once at full
# grid; drafts use its block-mean). DRAFT_LOOP=0 restores today's path.
DRAFT_LOOP = os.environ.get("DRAFT_LOOP", "0") not in ("0", "false", "no")
DRAFT_SIZE = int(os.environ.get("DRAFT_SIZE", "256"))
DRAFT_STEPS = int(os.environ.get("DRAFT_STEPS", "4"))
DRAFT_MAX_ITERS = int(os.environ.get("DRAFT_MAX_ITERS", "2"))
DRAFT_BUDGET_MS = int(os.environ.get("DRAFT_LOOP_BUDGET_MS", "2200"))
# schedule node (index into the full 4-step shifted schedule) where the
# promotion resumes; measured/tunable
PROMOTE_SIGMA_NODE = int(os.environ.get("PROMOTE_SIGMA_NODE", "1"))
PROMOTE_MODE = os.environ.get("PROMOTE_MODE", "hires")  # hires|recipe|direct
JUDGE_URL = os.environ.get("JUDGE_URL", "http://vllm:8000/v1")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL",
                             "nvidia/Qwen3.6-35B-A3B-NVFP4")
# fail verdicts stream up to ~450 tokens (analysis + corrective +
# revised instruction) ≈ 3s on the 35B; 1200ms silently timed out every
# long verdict and fail-opened the loop (unverified promotes)
JUDGE_TIMEOUT_MS = int(os.environ.get("JUDGE_TIMEOUT_MS", "4500"))


class Flux2Canvas:
    """Blocking renderer; call from an executor thread.

    v4 shared infrastructure (all four edit-native designs agreed):
      - TWO LANES: the committed canvas only receives committed edits;
        provisional churn (reflex/overlay/draft) edits a scratch canvas
        seeded from the committed one, so it can never age the real picture.
      - KEYFRAMES: every committed render is archived (JPEG, ~50KB) keyed by
        its commit revision. Undo = restore, pixel-exact, no diffusion.
    """

    def __init__(self) -> None:
        # two resident models: "fast" (4B) and optional "hq" (9B), both
        # NVFP4-quantized + compiled. model_pref picks which one renders;
        # latents only chain within one model (a switch re-anchors from
        # pixels once, since the VAE spaces are not assumed interchangeable)
        self.pipes: dict = {}
        self.model_pref = DEFAULT_PREF
        self.seed = FIXED_SEED if FIXED_SEED is not None else 0
        self.size = SIZE            # switchable from the UI (512/768/1024)
        self.canvas = None          # committed lane (PIL, for display/SAM)
        self.canvas_lat = None      # committed lane (packed, (h,w), model)
        self.scratch = None         # provisional lane (seeded from canvas)
        self.scratch_lat = None
        self.scratch_rev = -1       # committed rev the scratch was seeded from
        self.anchor = None          # per-token anchor state (committed lane)
        self.scratch_anchor = None  # provisional-lane clone
        self.pass_tau = {}          # per-model auto-calibrated tau_lo
        self.pass_tau_hi = {}       # per-model auto-calibrated tau_hi
        self.needs_floor_calib = False   # scene changed: re-measure floor
        self._merge_anchored = False  # last render's merge used the anchor
        self.rev = 0                # committed revision counter
        self.edits_since_root = 0
        # commit_rev -> (JPEG bytes, latents-on-cpu or None)
        self.keyframes: dict[int, tuple] = {}
        self._custom_ok = True      # custom klein loop healthy this session
        self.previewers: dict = {}  # per model: latent spaces may differ
        self._preview_ok = True
        self._sam = None            # lazy-loaded segmenter for composites
        self._sam_proc = None
        self._sam3 = False
        self._last_deletes: list = []   # (name, hole mask) of last composite

    @property
    def pipe(self):
        """The active pipeline (legacy accessor; render() captures its own
        reference so a mid-render switch can't swap pipes under it)."""
        return self.pipes.get(self.model_pref) or next(
            iter(self.pipes.values()), None)

    def set_model(self, tag: str) -> None:
        if tag not in ("fast", "hq"):
            return
        if tag not in self.pipes and self.pipes:
            log.info("model %r not loaded; keeping %r", tag, self.model_pref)
            return
        if tag != self.model_pref:
            log.info("model pref %s -> %s", self.model_pref, tag)
            self.model_pref = tag
            self.needs_floor_calib = True

    def _previewer(self, tag: str) -> LatentPreviewer:
        if tag not in self.previewers:
            self.previewers[tag] = LatentPreviewer(PREVIEW_SIZE)
        return self.previewers[tag]

    # ---- geometric compositing (positional edits) ----
    #
    # Edit models are great at paint and terrible at geometry — so code does
    # the geometry. The reflex computed an exact box delta; SAM (prompted
    # with the CURRENT box, no text needed) masks the object; we cut,
    # scale, and paste it at the target box; the hole gets a blurred-
    # background placeholder; then ONE generic repair edit blends it all.

    def _load_sam(self) -> None:
        if self._sam is not None:
            return
        import torch

        self._sam3 = "sam3" in SAM_MODEL.lower()
        log.info("loading %s for positional composites ...", SAM_MODEL)
        if self._sam3:
            from transformers import Sam3Model, Sam3Processor
            self._sam = Sam3Model.from_pretrained(
                SAM_MODEL, torch_dtype=torch.float32).to("cuda").eval()
            self._sam_proc = Sam3Processor.from_pretrained(SAM_MODEL)
        else:
            from transformers import SamModel, SamProcessor
            self._sam = SamModel.from_pretrained(
                SAM_MODEL, torch_dtype=torch.float32).to("cuda").eval()
            self._sam_proc = SamProcessor.from_pretrained(SAM_MODEL)
        log.info("sam ready (%s)", "sam3" if self._sam3 else "sam1")

    @staticmethod
    def _pos_words(box: dict) -> str:
        x, y = box.get("x", 0.5), box.get("y", 0.5)
        col = "left" if x < 0.34 else ("right" if x > 0.66 else "center")
        row = "top" if y < 0.34 else ("bottom" if y > 0.66 else "middle")
        if col == "center" and row == "middle":
            return "center"
        if row == "middle":
            return col
        if col == "center":
            return row
        return f"{row} {col}"

    def _move_clause(self, op: dict) -> str:
        """Templated world-state clause for one move op (pure geometry,
        never LLM-phrased): tells the repair pass the moved object exists
        exactly once and its old spot is empty."""
        name = (op.get("name") or op.get("id") or "object").replace("_", " ")
        for art in ("a ", "an ", "the "):
            if name.lower().startswith(art):
                name = name[len(art):]
        if op.get("insert"):
            return (f" A {name} has just been placed at the "
                    f"{self._pos_words(op.get('to_box') or {})} — it is "
                    f"real and must stay. Blend it into the scene: match "
                    f"the lighting direction, color temperature, and "
                    f"perspective, and give it a soft natural contact "
                    f"shadow. Keep its shape, colors, patterns, and "
                    f"design EXACTLY as they are.")
        if op.get("delete") and op.get("all"):
            return (f" Every {name} has been completely removed from the "
                    f"image; the areas where they were are now plain empty "
                    f"background — fill them in naturally with the "
                    f"surrounding scene. Do not add any new {name}.")
        if op.get("delete"):
            return (f" The {name} has been completely removed from the "
                    f"image; the {self._pos_words(op.get('from_box') or {})} "
                    f"area where it was is now plain empty background — "
                    f"fill it in naturally with the surrounding scene. Do "
                    f"not add a new {name}.")
        return (f" There is exactly one {name} in the image, at the "
                f"{self._pos_words(op.get('to_box') or {})}; the "
                f"{self._pos_words(op.get('from_box') or {})} area behind "
                f"it is plain empty background. Repaint the {name} so it "
                f"belongs in its new spot: match the scene's lighting "
                f"direction, color temperature, and perspective, give it "
                f"a soft natural contact shadow, and if it overlaps or "
                f"blocks another object, adjust its exact position "
                f"slightly so it sits plausibly against the nearest wall "
                f"or clear floor space. Keep its identity, colors, and "
                f"details unchanged.")

    @staticmethod
    def _iou(a, b) -> float:
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    def _mask_box(self, image, box_px, name: str | None = None):
        """Segment the object inside box_px (x0,y0,x1,y1). Returns a PIL
        'L' mask, or None on failure (caller falls back to the raw box)."""
        import numpy as np
        import torch
        from PIL import Image

        try:
            self._load_sam()
            if self._sam3:
                # concept segmentation: the entity NAME finds the object,
                # the box picks WHICH instance (IoU against our box) —
                # robust even when the tracked box has drifted off-center
                W, H = image.size
                text = (name or "").replace("_", " ").strip() or None
                kw = dict(images=image, return_tensors="pt")
                if text:
                    kw["text"] = text
                else:
                    kw["input_boxes"] = [[list(box_px)]]
                    kw["input_boxes_labels"] = [[1]]
                inputs = self._sam_proc(**kw).to("cuda")
                with torch.no_grad():
                    out = self._sam(**inputs)
                res = self._sam_proc.post_process_instance_segmentation(
                    out, threshold=0.4, mask_threshold=0.5,
                    target_sizes=[(H, W)])[0]
                if len(res["scores"]) == 0:
                    log.info("sam3 found no %r; using raw box", text)
                    return None
                ious = [self._iou(box_px, [float(v) for v in b])
                        for b in res["boxes"].tolist()]
                best = int(max(range(len(ious)), key=lambda i: ious[i]))
                mask = (res["masks"][best].cpu().numpy() > 0).astype(np.uint8) * 255
                return Image.fromarray(mask, mode="L")
            inputs = self._sam_proc(
                image, input_boxes=[[list(box_px)]],
                return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = self._sam(**inputs)
            masks = self._sam_proc.image_processor.post_process_masks(
                out.pred_masks.cpu(), inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu())[0][0]
            scores = out.iou_scores[0, 0]
            best = masks[int(scores.argmax())].numpy().astype(np.uint8) * 255
            return Image.fromarray(best, mode="L")
        except Exception:
            log.exception("sam mask failed; using raw box")
            return None

    def _mask_instances(self, image, name: str,
                        threshold: float = 0.3) -> list | None:
        """Per-instance boolean masks of a named concept (SAM3 text
        prompt), whole-scene instances excluded. None on failure/no SAM3."""
        import torch

        try:
            self._load_sam()
            if not self._sam3:
                return None
            W, H = image.size
            text = (name or "").replace("_", " ").strip()
            if not text:
                return None
            inputs = self._sam_proc(images=image, text=text,
                                    return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = self._sam(**inputs)
            # recall over precision for deletes: a missed instance means
            # the object visibly survives; extra candidates are contained
            # by the whole-scene guard and the overlap rules downstream
            res = self._sam_proc.post_process_instance_segmentation(
                out, threshold=threshold, mask_threshold=0.5,
                target_sizes=[(H, W)])[0]
            insts = []
            for m in res["masks"]:
                mk = (m.cpu().numpy() > 0)
                if 0.65 >= mk.mean() > 0:     # whole-scene guard
                    insts.append(mk)
            return insts
        except Exception:
            log.exception("sam instance query failed")
            return None

    def _mask_all(self, image, name: str, keep: list | None = None,
                  keep_boxes: list | None = None):
        """Union mask of ALL instances of a named concept — 'remove all
        the palm trees' must hit every tree, not the single best-overlap
        instance _mask_box picks. `keep` names are SAM-masked and their
        instances are EXCLUDED: 'remove the people in the background,
        keep the couple' once cut the couple too and refilled them as
        strangers. Returns a PIL 'L' mask or None."""
        import numpy as np
        from PIL import Image

        insts = self._mask_instances(image, name)
        if not insts:
            if insts is not None:
                log.info("sam3 found no %r for delete-all", name)
            return None
        keep_mask = None
        if keep_boxes:
            # the parser POINTED at the kept things — rectangle shields
            # beat any name-matching heuristic
            w, h = image.size
            keep_mask = np.zeros((h, w), dtype=bool)
            for b in keep_boxes[:3]:
                x0 = max(0, int(float(b.get("x", 0)) * w))
                y0 = max(0, int(float(b.get("y", 0)) * h))
                x1 = min(w, x0 + int(float(b.get("w", 0)) * w))
                y1 = min(h, y0 + int(float(b.get("h", 0)) * h))
                keep_mask[y0:y1, x0:x1] = True
        for kn in ([] if keep_boxes else (keep or []))[:3]:
            kms = [km.astype(bool) for km in (self._mask_instances(
                image, str(kn).replace("_", " "),
                threshold=0.3) or [])]
            if not kms:
                continue
            # foreground-scale instances only: "keep the couple" once
            # matched every person in the scene and shielded them all
            biggest = max(km.sum() for km in kms)
            for km in kms:
                if km.sum() >= 0.6 * biggest:
                    keep_mask = (km if keep_mask is None
                                 else keep_mask | km)
        union = np.zeros_like(insts[0], dtype=bool)
        dropped = 0
        for mk in insts:
            if keep_mask is not None:
                ov = (mk & keep_mask).sum() / max(1, mk.sum())
                if ov > 0.3:
                    dropped += 1
                    continue
            union |= mk
        if dropped:
            log.info("delete-all %r: %d instances protected by keep "
                     "list %s", name, dropped, keep)
        if not union.any():
            return None
        log.info("delete-all %r: %d instances, %.0f%% of frame",
                 name, len(insts), union.mean() * 100)
        return Image.fromarray(union.astype(np.uint8) * 255, mode="L")

    def _enforce_deletes(self, image, deletes: list, ref=None):
        """One strict verification pass: deleted means DELETED, but only
        a TRUE resurrection may be scrubbed. A true resurrection is an
        instance of the deleted name that (a) sits inside the deleted
        region (>50% overlap) and (b) RESEMBLES the original deleted
        pixels (an anchor/merge restoration is near-identical; clean new
        floor, fill smears, blankets, or neighboring furniture are not —
        a looser test once scrub-looped a whole room into mud).
        deletes: [(name, composite-hole PIL mask)]; ref = pre-delete
        image. Returns (image, changed)."""
        import numpy as np
        from PIL import Image, ImageFilter

        if ref is None:
            return image, False
        fixed = image
        changed = False
        self._scrub_free = np.zeros((image.size[1], image.size[0]),
                                    dtype=bool)
        res_arr = np.asarray(image.convert("L"), dtype=np.float32)
        ref_arr = np.asarray(
            ref.convert("L").resize(image.size), dtype=np.float32)
        for entry in deletes:
            name, dmask = entry[0], entry[1]
            all_op = bool(entry[2]) if len(entry) > 2 else False
            # delete-ALL is judged by NAME, not by resemblance: ANY
            # instance of the name is a failure — a freshly regenerated
            # different-looking sofa is exactly the escape the old
            # mae>40 skip allowed. Single-instance deletes keep the
            # strict tests (a looser loop once scrubbed a room to mud).
            thr = 0.3 if all_op else 0.45
            insts = self._mask_instances(fixed, name, threshold=thr)
            if not insts:
                continue
            dm = np.asarray(dmask.resize(fixed.size), dtype=np.uint8) > 64
            scrub = np.zeros_like(dm)
            n = 0
            for mk in insts:
                inside = (mk & dm).sum() / max(mk.sum(), 1)
                if inside < (0.3 if all_op else 0.5):
                    if not all_op:
                        continue
                mae = float(np.abs(res_arr[mk] - ref_arr[mk]).mean())
                if mae > 40 and not all_op:
                    continue        # does not resemble the deleted object
                scrub |= mk
                n += 1
                log.info("delete verify: %r resurrected (overlap %.0f%% "
                         "mae %.0f)", name, inside * 100, mae)
            if not n:
                continue
            hole = Image.fromarray(scrub.astype(np.uint8) * 255, "L")
            hole = hole.filter(ImageFilter.MaxFilter(9)).filter(
                ImageFilter.GaussianBlur(3))
            if not changed:
                fixed = fixed.copy()
            fixed.paste(self._bg_fill(fixed, hole), (0, 0), hole)
            self._scrub_free |= (np.asarray(hole) > 16)
            changed = True
        return fixed, changed

    @staticmethod
    def _box_px(box: dict, w: int, h: int):
        """Center-normalized {x,y,w,h} -> pixel (x0,y0,x1,y1), clamped."""
        bw, bh = box.get("w", 0.3) * w, box.get("h", 0.3) * h
        cx, cy = box.get("x", 0.5) * w, box.get("y", 0.5) * h
        x0 = max(0, int(cx - bw / 2)); y0 = max(0, int(cy - bh / 2))
        x1 = min(w, int(cx + bw / 2)); y1 = min(h, int(cy + bh / 2))
        return x0, y0, max(x1, x0 + 2), max(y1, y0 + 2)

    @staticmethod
    def _bg_fill(image, hole_mask):
        """Fill the hole with surrounding background via low-res onion-peel
        (grow known pixels inward), then upsample. Cheap (~quarter-res
        numpy) and produces background-colored fill, not object-colored."""
        import numpy as np
        from PIL import Image, ImageFilter

        W, H = image.size
        s = 4
        small = image.resize((W // s, H // s), Image.BILINEAR)
        msk = hole_mask.resize((W // s, H // s), Image.BILINEAR)
        im = np.asarray(small).astype(np.float32)
        known = np.asarray(msk) < 64
        for _ in range(max(W, H) // s):
            if known.all():
                break
            acc = np.zeros_like(im)
            cnt = np.zeros(known.shape, np.float32)
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                k = np.roll(np.roll(known, dy, 0), dx, 1)
                v = np.roll(np.roll(im, dy, 0), dx, 1)
                acc += v * k[..., None]
                cnt += k
            newly = (~known) & (cnt > 0)
            if not newly.any():
                break
            im[newly] = acc[newly] / cnt[newly][..., None]
            known |= newly
        filled = Image.fromarray(im.clip(0, 255).astype(np.uint8))
        return filled.resize((W, H), Image.BILINEAR).filter(
            ImageFilter.GaussianBlur(2))

    def _cut_upload(self, obj_img, name: str):
        """SAM-cut the named object out of an upload. Returns
        (cropped RGB, cropped L mask); mask is full-white when
        segmentation finds nothing (swatch/graphic fallback)."""
        from PIL import Image, ImageFilter

        mask = None
        try:
            insts = self._mask_instances(obj_img, name, threshold=0.3)
            if insts:
                m = insts[0]
                for extra in insts[1:2]:
                    m = m | extra
                mask = Image.fromarray(
                    (m.astype("uint8") * 255)).convert("L")
        except Exception:
            log.exception("insert: SAM on upload failed")
        if mask is None or not mask.getbbox():
            mask = Image.new("L", obj_img.size, 255)
            log.info("insert: no segmentation for %r — full-image paste",
                     name)
        # close small holes: a holey mask pastes a mangled sprite
        mask = mask.filter(ImageFilter.MaxFilter(7)).filter(
            ImageFilter.MinFilter(7))
        bb = mask.getbbox()
        return obj_img.crop(bb), mask.crop(bb)

    def cut_object(self, b64: str, name: str) -> str | None:
        """Background-removed RGBA preview of an uploaded object (the
        exact cut the insert will use), as base64 PNG."""
        import base64 as _b64
        import io as _io

        try:
            from PIL import Image
            obj_img = Image.open(_io.BytesIO(
                _b64.b64decode(b64))).convert("RGB")
            obj, om = self._cut_upload(obj_img,
                                       str(name or "object")
                                       .replace("_", " "))
            rgba = obj.convert("RGBA")
            rgba.putalpha(om)
            if max(rgba.size) > 640:
                rgba.thumbnail((640, 640))
            buf = _io.BytesIO()
            rgba.save(buf, format="PNG")
            return _b64.b64encode(buf.getvalue()).decode()
        except Exception:
            log.exception("cut_object failed")
            return None

    def _composite_insert(self, out, op: dict):
        """UPLOAD insertion: cut the named object out of the user's
        uploaded image with SAM and paste the ACTUAL pixels at the drop
        point — identity-perfect by construction; the repair render then
        blends lighting/shadow. Honors the user's interactive placement
        (position, size, rotation) when given."""
        import base64 as _b64
        import io as _io

        import numpy as np
        from PIL import Image, ImageFilter

        w, h = out.size
        obj_img = Image.open(_io.BytesIO(
            _b64.b64decode(op["b64"]))).convert("RGB")
        name = str(op.get("name") or "object").replace("_", " ")
        obj, om = self._cut_upload(obj_img, name)
        tb = op.get("to_box") or {}
        tw = max(24, int(float(tb.get("w", 0.3)) * w))
        th = max(24, int(tw * obj.size[1] / max(1, obj.size[0])))
        obj = obj.resize((tw, th), Image.LANCZOS)
        om = om.resize((tw, th), Image.LANCZOS).filter(
            ImageFilter.GaussianBlur(1.5))
        rot = float(op.get("rot") or 0.0) % 360.0
        if rot:
            obj = obj.rotate(-rot, expand=True,
                             resample=Image.BICUBIC)
            om = om.rotate(-rot, expand=True,
                           resample=Image.BICUBIC)
            tw, th = obj.size
        cx = int(float(tb.get("x", 0.5)) * w)
        cy = int(float(tb.get("y", 0.5)) * h)
        px = max(0, min(w - tw, cx - tw // 2))
        py = max(0, min(h - th, cy - th // 2))
        try:
            occ = self._occupancy_mask(out)
            fl = self._floor_mask(out)
            sol = self._solve_placement(
                occ, tw, max(4, int(th * 0.28)), px, py + th, w, h,
                floor=fl,
                skip_n=int(getattr(self, "_place_attempt", 0)))
            if sol is None:
                # deterministic last resort: the longest clear floor
                # run anywhere in the lower half — always floor-true
                import numpy as _np3
                clear_map = fl & ~occ
                best_run = None
                for yy in range(h - 8, int(h * 0.45), -12):
                    row = clear_map[max(0, yy - 6):yy, :].all(axis=0)
                    run = 0
                    start = 0
                    for xx in range(w + 1):
                        if xx < w and row[xx]:
                            run += 1
                        else:
                            if best_run is None or run > best_run[2]:
                                best_run = (xx - run, yy, run)
                            run = 0
                if best_run and best_run[2] > 40:
                    x0r, yyr, rl = best_run
                    sc = min(1.0, rl / max(1, tw))
                    sol = (x0r + max(0, (rl - int(tw * sc)) // 2),
                           yyr, max(0.4, sc))
                    log.info("insert fallback: floor run %dpx at y=%d",
                             rl, yyr)
            if sol is not None:
                px2, fy2, sc2 = sol
                if sc2 < 0.999:
                    tw2 = max(2, int(tw * sc2))
                    th2 = max(2, int(th * sc2))
                    obj = obj.resize((tw2, th2), Image.LANCZOS)
                    om = om.resize((tw2, th2), Image.LANCZOS)
                    tw, th = tw2, th2
                log.info("insert solver: x %d->%d floor %d->%d "
                         "scale %.2f", px, px2, py + th, fy2, sc2)
                px, py = px2, fy2 - th
        except Exception:
            log.exception("insert solver failed (direct paste)")
        out.paste(obj, (px, py), om)
        grown = Image.new("L", out.size, 0)
        grown.paste(om, (px, py))
        grown = grown.filter(ImageFilter.MaxFilter(9)).filter(
            ImageFilter.GaussianBlur(3))
        self._last_free |= (np.asarray(grown) > 16)
        log.info("composite insert %s: pasted %dx%d at (%d,%d)",
                 op.get("id") or name, tw, th, px, py)
        return out

    def _occupancy_mask(self, image, exclude=None):
        """Boolean map of floor-blocking objects, via SAM concept
        queries. `exclude` (bool array) removes the object being moved
        so it never blocks itself."""
        import numpy as np
        w, h = image.size
        occ = np.zeros((h, w), dtype=bool)
        try:
            # SAM3 detects SPECIFIC nouns far better than category
            # words — "furniture" found 3-5% of a fully furnished room
            for concept in ("sofa", "couch", "bed", "table",
                            "coffee table", "desk", "chair", "armchair",
                            "cabinet", "dresser", "bookcase", "shelf",
                            "lamp", "plant", "television", "tv stand",
                            "console", "nightstand", "bench", "ottoman",
                            "wardrobe"):
                for mk in (self._mask_instances(image, concept,
                                                threshold=0.35) or []):
                    occ |= mk.astype(bool)
        except Exception:
            log.exception("occupancy: SAM query failed (empty map)")
        if exclude is not None:
            inter = (occ & exclude).sum()
            occ &= ~exclude
            log.info("occupancy: %.0f%% of frame occupied "
                     "(excluded self overlap %d px)",
                     100 * occ.mean(), int(inter))
        else:
            log.info("occupancy: %.0f%% of frame occupied",
                     100 * occ.mean())
        return occ

    def _floor_mask(self, image):
        """Where the walkable floor actually is (SAM concept query) —
        non-overlap alone let inserts float in mid-air, because air
        doesn't contain furniture either."""
        import numpy as np
        w, h = image.size
        fl = np.zeros((h, w), dtype=bool)
        try:
            for concept in ("floor", "wooden floor", "carpet", "rug"):
                for mk in (self._mask_instances(image, concept,
                                                threshold=0.3) or []):
                    fl |= mk.astype(bool)
        except Exception:
            log.exception("floor mask failed (lower-third fallback)")
        if not fl.any():
            fl[int(h * 0.72):, :] = True
        log.info("floor mask: %.0f%% of frame", 100 * fl.mean())
        return fl

    @staticmethod
    def _solve_placement(occ, foot_w, foot_h, want_px, floor_y,
                         img_w, img_h, side=None, horizon_frac=0.42,
                         floor=None, skip_n=0):
        """Find a clear FOOTPRINT slot for an object: scan x outward
        from the wanted position (restricted to the commanded side if
        given), first along the original floor line, then stepping
        DEEPER into the room (floor line up, perspective scale down).
        Returns (px, floor_y, scale) — px is the footprint's left edge.
        The footprint is the bottom band of the sprite (what must touch
        clear floor); upper overlap (e.g. in front of a window) is
        allowed."""
        import numpy as np
        horizon = int(img_h * horizon_frac)

        best = [None, -1e9]
        if floor is not None:
            import numpy as _np
            # REACHABLE floor only: true floor connects downward to
            # the bottom of the frame; a coffee-table top (which the
            # concept query happily calls "floor") has a gap beneath
            # it and fails this test
            reach = _np.logical_and.accumulate(
                floor[::-1], axis=0)[::-1]
            grown = floor & False
            grown |= reach
            # allow small vertical gaps (rug edges): dilate reach up
            # a few rows through floor pixels
            for _ in range(3):
                grown[:-4] |= grown[4:] & floor[:-4]
            floor = grown

        def clear(px, fy, fw, fh, require_floor):
            x0 = max(0, int(px))
            x1 = min(img_w, int(px + fw))
            y0 = max(0, int(fy - fh))
            y1 = min(img_h, int(fy))
            if x1 - x0 < fw * 0.9:
                return False        # out of frame
            band = occ[y0:y1, x0:x1]
            if band.size == 0:
                return False
            occ_frac = float(band.mean())
            fl_frac = 1.0
            if floor is not None:
                base = floor[max(0, y1 - max(2, fh // 4)):y1, x0:x1]
                fl_frac = float(base.mean()) if base.size else 0.0
            score = -occ_frac * 3
            if score > best[1] and (floor is None or fl_frac > 0.5):
                # best-effort candidates must STILL be floor-supported:
                # a well-scored mid-air slot is worse than nothing
                best[0], best[1] = (int(px), int(fy)), score
            ok = occ_frac < 0.08
            if require_floor and floor is not None:
                ok = ok and fl_frac > 0.5
            return ok

        for depth_step in range(7):
            dscale = 1.0
            fy = floor_y
            if depth_step:
                fy = floor_y - int(img_h * 0.035 * depth_step)
                if fy <= horizon + int(img_h * 0.05):
                    break
                dscale = max(0.45, (fy - horizon)
                             / max(1, floor_y - horizon))
            step = max(6, img_w // 50)
            require_floor = (floor is not None
                             and (depth_step > 0 or floor.any()))
            # crowded rooms: a smaller object fits a gap a full-size
            # one cannot — plausible-and-smaller beats absurd-and-big
            for size_scale in (1.0, 0.8, 0.65, 0.5):
                scale = dscale * size_scale
                fw = max(8, int(foot_w * scale))
                fh = max(4, int(foot_h * scale))
                for dx in range(0, img_w, step):
                    for sgn in (+1, -1):
                        px = want_px + sgn * dx
                        if (side == "left"
                                and px + fw / 2 > img_w * 0.55):
                            continue
                        if (side == "right"
                                and px + fw / 2 < img_w * 0.45):
                            continue
                        if clear(px, fy, fw, fh, require_floor):
                            if skip_n > 0:
                                # a judged-bad slot must not be re-
                                # picked deterministically on retry
                                skip_n -= 1
                                continue
                            return int(px), int(fy), scale
        if side is not None:
            # the commanded side may hold NO clear floor at all (a bed
            # spanning the whole left wall): relax the side constraint
            # and take the nearest clear slot TOWARD that side — a
            # near-side placement beats a floating one every time
            edge_px = (int(img_w * 0.08) if side == "left"
                       else int(img_w * 0.92))
            relaxed = Flux2Canvas._solve_placement(
                occ, foot_w, foot_h, edge_px, floor_y, img_w, img_h,
                side=None, horizon_frac=horizon_frac, floor=floor,
                skip_n=skip_n)
            if relaxed is not None:
                log.info("placement solver: side relaxed (%s fully "
                         "occupied)", side)
                return relaxed
        if best[0] is not None and best[1] > -0.75:
            # least-bad FLOOR-SUPPORTED spot, but only if it is not
            # itself garbage (an 83%-occupied "best" once stacked
            # planters on a bed)
            log.info("placement solver: best-effort slot (score %.2f)",
                     best[1])
            return best[0][0], best[0][1], 1.0
        return None

    def composite_moves(self, image, move_ops: list[dict]):
        """Apply exact box-delta moves/scales in pixels. Returns new image.

        The tracked entity box is often smaller/offset vs. the object the
        image model actually painted — so the box supplies only the DELTA
        (how far, how much bigger); the segmentation mask supplies the
        object's TRUE extent. Never crop to the tracked box: that moves a
        box-shaped slice and leaves rectangular ghosts.
        """
        from PIL import Image, ImageFilter

        import numpy as np

        w, h = image.size
        out = image.copy()
        self._last_deletes = []
        self._unfound_deletes = []
        self._last_free = np.zeros((h, w), dtype=bool)
        for op in move_ops:
            if op.get("free_box"):
                # placement freedom: no pixels change here, but the
                # region is marked FREE so an instructed add can appear
                # (the merge otherwise pins the destination back to the
                # conditioning — a restaged dresser once vanished
                # because only its OLD hole was free)
                fb = op["free_box"]
                x0 = max(0, int((float(fb.get("x", 0.5))
                                 - float(fb.get("w", 0.3)) / 2) * w))
                y0 = max(0, int((float(fb.get("y", 0.5))
                                 - float(fb.get("h", 0.3)) / 2) * h))
                x1 = min(w, x0 + int(float(fb.get("w", 0.3)) * w))
                y1 = min(h, y0 + int(float(fb.get("h", 0.3)) * h))
                if x1 > x0 and y1 > y0:
                    self._last_free[y0:y1, x0:x1] = True
                    log.info("placement free box: (%d,%d)-(%d,%d)",
                             x0, y0, x1, y1)
                continue
            if op.get("insert"):
                try:
                    out = self._composite_insert(out, op)
                except Exception:
                    log.exception("composite insert failed (op skipped)")
                continue
            src = self._box_px(op["from_box"], w, h)
            fb = op["from_box"]
            tb = op.get("to_box") or fb          # deletes have no target
            dx = int((tb.get("x", 0.5) - fb.get("x", 0.5)) * w)
            dy = int((tb.get("y", 0.5) - fb.get("y", 0.5)) * h)
            scale = (tb.get("w", 0.3) / fb.get("w", 0.3)) if fb.get("w") else 1.0

            # a click-drag carries the EXACT mask the user saw highlighted;
            # re-detecting by name could grab a different instance
            mask = None
            if op.get("mask_rle") and op.get("mask_size"):
                mask = self._rle_to_mask(op["mask_rle"], op["mask_size"])
                if mask.size != image.size:
                    mask = mask.resize(image.size, Image.NEAREST)
            if mask is None and op.get("delete") and op.get("all"):
                mask = self._mask_all(image,
                                      op.get("name") or op.get("id"),
                                      keep=op.get("keep"),
                                      keep_boxes=op.get("keep_boxes"))
                if mask is None:
                    # NAME-based delete with nothing found: the object is
                    # not in the picture — a no-op, NEVER a box fallback
                    # (the placeholder box is the whole frame; falling
                    # through once hole-filled an entire image because the
                    # LLM invented a "clock" that did not exist)
                    log.info("delete-all %r: nothing found, skipping",
                             op.get("name") or op.get("id"))
                    self._unfound_deletes.append(
                        str(op.get("name") or op.get("id") or "")
                        .replace("_", " "))
                    continue
            if mask is None:
                mask = self._mask_box(image, src,
                                      name=op.get("name") or op.get("id"))
            if mask is None:
                mask = Image.new("L", image.size, 0)
                mask.paste(255, src)
            bb = mask.getbbox() or src        # the object's REAL extent
            # 1) hole: fill the old position with SURROUNDING BACKGROUND
            #    (onion-peel), never a blur of the object — an object-
            #    colored stain reads as an intentional shadow and survives
            #    the repair pass as a ghost
            hole_mask = mask.filter(ImageFilter.MaxFilter(9)).filter(
                ImageFilter.GaussianBlur(3))
            out.paste(self._bg_fill(image, hole_mask), (0, 0), hole_mask)
            self._last_free |= (np.asarray(hole_mask) > 16)
            if op.get("delete"):
                # delete: the hole fill IS the edit; the repair pass paints
                # real background over it (no paste). Record the hole so
                # the post-render enforcement can catch resurrections.
                self._last_deletes.append(
                    (str(op.get("name") or op.get("id") or ""), hole_mask,
                     bool(op.get("all"))))
                log.info("composite delete %s: real bbox %s",
                         op.get("id"), mask.getbbox())
                continue
            # 2) cutout of the real object, scaled about its center,
            #    translated by the commanded delta
            mask = mask.filter(ImageFilter.MaxFilter(7)).filter(
                ImageFilter.MinFilter(7))
            bb = mask.getbbox() or bb
            cut = image.crop(bb)
            m = mask.crop(bb)
            bw, bh = bb[2] - bb[0], bb[3] - bb[1]
            nw, nh = max(2, int(bw * scale)), max(2, int(bh * scale))
            cut = cut.resize((nw, nh), Image.LANCZOS)
            m = m.resize((nw, nh), Image.LANCZOS)
            cx, cy = (bb[0] + bb[2]) // 2 + dx, (bb[1] + bb[3]) // 2 + dy
            px, py = cx - nw // 2, cy - nh // 2
            if op.get("floor_lock"):
                # objects keep their ground plane: the parser's y is
                # jitter, not intent, on horizontal moves — IGNORE dy
                # completely and keep the exact original floor line
                # (scaled objects grow UP from it).
                py = bb[3] - nh
                log.info("floor lock: bottom edge fixed at y=%d", bb[3])
            # PLACEMENT SOLVER: never paste onto occupied space. The
            # occupancy map (SAM, minus the object itself) is scanned
            # for a clear footprint slot on the commanded side — first
            # along the original floor line, then deeper into the room
            # with perspective scaling.
            try:
                import numpy as _np2
                self_mask = _np2.zeros((h, w), dtype=bool)
                mm = _np2.asarray(mask) > 64
                self_mask[:mm.shape[0], :mm.shape[1]] |= mm
                occ = self._occupancy_mask(image, exclude=self_mask)
                fl = self._floor_mask(image)
                fl |= self_mask          # the object's old spot IS floor
                tbx = float((op.get("to_box") or {}).get("x", 0.5))
                side = ("left" if tbx < 0.42
                        else "right" if tbx > 0.58 else None)
                sol = self._solve_placement(
                    occ, nw, max(4, int(nh * 0.28)), px, py + nh,
                    w, h, side=side, floor=fl,
                    skip_n=int(getattr(self, "_place_attempt", 0)))
                if sol is not None:
                    px2, fy2, sc2 = sol
                    if sc2 < 0.999:
                        nw2 = max(2, int(nw * sc2))
                        nh2 = max(2, int(nh * sc2))
                        cut = cut.resize((nw2, nh2), Image.LANCZOS)
                        m = m.resize((nw2, nh2), Image.LANCZOS)
                        nw, nh = nw2, nh2
                    log.info("placement solver: x %d->%d, floor %d->%d,"
                             " scale %.2f", px, px2, py + nh, fy2, sc2)
                    px, py = px2, fy2 - nh
                else:
                    log.info("placement solver: no clear slot — "
                             "keeping requested spot")
            except Exception:
                log.exception("placement solver failed (direct paste)")
            px = max(0, min(w - nw, px))
            py = max(0, min(h - nh, py))
            out.paste(cut, (px, py), m)
            x0d, y0d = max(0, px), max(0, py)
            self._last_free[y0d:min(h, py + nh), x0d:min(w, px + nw)] = True
            # the SOURCE of a move is a deletion: register its hole so
            # the resurrection hunter scrubs any left-behind original
            # (a moved bar cart once stayed put AND appeared at the
            # destination)
            src_hole = Image.new("L", out.size, 0)
            src_hole.paste(mask.crop(bb), (bb[0], bb[1]))
            self._last_deletes.append(
                (str(op.get("name") or op.get("id") or ""), src_hole,
                 False))
            log.info("composite move %s: real bbox %s, delta (%+d,%+d) "
                     "scale %.2f", op.get("id"), bb, dx, dy, scale)
        return out

    # ---- click segmentation (mouse layer) ----
    #
    # The user clicks a point; SAM 1's point prompt answers "which object
    # did they mean?" — any object, no names, no tracked entities needed.
    # The expensive image encoding is cached per committed revision (done
    # once, ideally precomputed while the queue is idle); each click after
    # that is a ~10ms prompt decode.


    @staticmethod
    def _rle_encode(flat) -> list[int]:
        """Row-major binary RLE; counts alternate 0-runs/1-runs, first is
        a 0-run."""
        import numpy as np

        change = np.flatnonzero(np.diff(flat))
        counts = np.diff(np.concatenate(
            ([0], change + 1, [flat.size]))).tolist()
        return ([0] + counts) if flat[0] else counts

    @staticmethod
    def _ink_composite(image, strokes: list):
        """Marker-red freehand strokes drawn onto a COPY of the frame —
        the sketch route's conditioning. The original is never touched."""
        from PIL import ImageDraw

        img = image.copy()
        w, h = img.size
        d = ImageDraw.Draw(img)
        lw = max(3, int(min(w, h) * 0.008))
        for s in strokes:
            pts = [(p[0] * w, p[1] * h) for p in s]
            if len(pts) >= 2:
                d.line(pts, fill=(255, 32, 32), width=lw, joint="curve")
        return img

    @staticmethod
    def _rle_to_mask(counts: list[int], size: list[int]):
        import numpy as np
        from PIL import Image

        w, h = int(size[0]), int(size[1])
        a = np.zeros(w * h, dtype=np.uint8)
        pos, val = 0, 0
        for c in counts:
            if val:
                a[pos:pos + c] = 255
            pos += c
            val ^= 1
        return Image.fromarray(a.reshape(h, w), mode="L")

    def _sam3_box_instances(self, box_px: list) -> list | None:
        """SAM3 instances prompted by a pixel box (no text). Returns
        [np bool masks] filtered by the whole-scene guard, or None."""
        import torch

        try:
            self._load_sam()
            if not self._sam3 or self.canvas is None:
                return None
            W, H = self.canvas.size
            inputs = self._sam_proc(
                images=self.canvas, input_boxes=[[list(box_px)]],
                input_boxes_labels=[[1]],
                return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = self._sam(**inputs)
            res = self._sam_proc.post_process_instance_segmentation(
                out, threshold=0.3, mask_threshold=0.5,
                target_sizes=[(H, W)])[0]
            return [(m.cpu().numpy() > 0) for m in res["masks"]
                    if 0.65 >= float((m > 0).float().mean()) > 0]
        except Exception:
            log.exception("sam3 box query failed")
            return None

    def lasso_segment(self, points: list) -> dict | None:
        """A freehand TRACE around an object -> the object's snapped mask.
        The loose polygon is a prompt, not the answer: its bounding box
        drives a SAM3 box query and candidates are scored by IoU against
        the polygon fill — a sloppy trace still selects the real object.
        Falls back to the polygon fill itself when nothing matches."""
        if self.canvas is None or not points or len(points) < 3:
            return None
        import numpy as np
        from PIL import Image, ImageDraw

        w, h = self.canvas.size
        poly = [(float(x) * w, float(y) * h) for x, y in points]
        pm = Image.new("L", (w, h), 0)
        ImageDraw.Draw(pm).polygon(poly, fill=255)
        parr = np.asarray(pm) > 0
        pfrac = float(parr.mean())
        if pfrac < 0.0005:
            return None
        ys, xs = np.nonzero(parr)
        box = [float(xs.min()), float(ys.min()),
               float(xs.max()), float(ys.max())]
        best = parr
        best_iou = 0.35        # SAM must clearly match the trace to win
        for m in (self._sam3_box_instances(box) or []):
            inter = (m & parr).sum()
            union = (m | parr).sum()
            iou = inter / max(union, 1)
            if iou > best_iou:
                best, best_iou = m, iou
        m = best
        ys, xs = np.nonzero(m)
        if not len(xs):
            return None
        bb = [int(xs.min()), int(ys.min()),
              int(xs.max()) + 1, int(ys.max()) + 1]
        frac = float(m.mean())
        log.info("lasso: trace %.1f%% -> mask %.1f%% (snap iou %.2f)",
                 pfrac * 100, frac * 100, best_iou)
        return {"candidates": [{
            "rle": self._rle_encode(m.reshape(-1)), "size": [w, h],
            "tight_box_px": bb, "frac": round(frac, 4),
            "box": {"x": round((bb[0] + bb[2]) / 2 / w, 4),
                    "y": round((bb[1] + bb[3]) / 2 / h, 4),
                    "w": round((bb[2] - bb[0]) / w, 4),
                    "h": round((bb[3] - bb[1]) / h, 4)},
        }]}

    def point_segment(self, nx: float, ny: float) -> dict | None:
        """Segment the object at normalized point (nx, ny) on the
        committed canvas: a SAM3 box query around the point, keeping the
        instances that CONTAIN it, biggest first (a plain click means the
        general object; repeated clicks drill into the smaller ones).
        Returns {"candidates": [...]} or None."""
        if self.canvas is None:
            return None
        import numpy as np

        w, h = self.canvas.size
        px, py = float(nx) * w, float(ny) * h
        r = 0.2
        box = [max(0.0, px - r * w), max(0.0, py - r * h),
               min(float(w), px + r * w), min(float(h), py + r * h)]
        insts = self._sam3_box_instances(box) or []
        ix, iy = min(w - 1, max(0, int(px))), min(h - 1, max(0, int(py)))
        cands = []
        for m in insts:
            if not m[iy, ix]:
                continue          # the click must land ON the object
            frac = float(m.mean())
            if not (0.0005 < frac <= 0.65):
                continue
            ys, xs = np.nonzero(m)
            bb = [int(xs.min()), int(ys.min()),
                  int(xs.max()) + 1, int(ys.max()) + 1]
            cands.append({
                "rle": self._rle_encode(m.reshape(-1)), "size": [w, h],
                "tight_box_px": bb, "frac": round(frac, 4),
                "box": {"x": round((bb[0] + bb[2]) / 2 / w, 4),
                        "y": round((bb[1] + bb[3]) / 2 / h, 4),
                        "w": round((bb[2] - bb[0]) / w, 4),
                        "h": round((bb[3] - bb[1]) / h, 4)},
            })
        cands.sort(key=lambda c: -c["frac"])
        dedup: list[dict] = []
        for c in cands:
            if not any(abs(c["frac"] - d["frac"]) / max(d["frac"], 1e-6) < 0.15
                       for d in dedup):
                dedup.append(c)
        return {"candidates": dedup} if dedup else None

    # ---- keyframe store ----

    def snapshot(self, commit_rev: int, jpeg: bytes) -> None:
        # archive the canvas LATENTS with the JPEG: a restore then re-enters
        # the latent chain exactly, instead of restarting it from a lossy
        # JPEG decode + VAE re-encode (~1MB cpu per keyframe)
        lat = None
        if self.canvas_lat is not None:
            packed, hw, mtag = self.canvas_lat
            lat = (packed.detach().to("cpu"), hw, mtag)
        anc = None
        if self.anchor is not None:
            anc = {"lat": self.anchor["lat"].detach().to("cpu"),
                   "hw": self.anchor["hw"], "tag": self.anchor["tag"],
                   "age": self.anchor["age"].to("cpu"),
                   "pending": self.anchor["pending"].to("cpu")}
        self.keyframes[commit_rev] = (jpeg, lat, anc)
        if len(self.keyframes) > MAX_KEYFRAMES:
            root = min(self.keyframes)      # the root keyframe stays pinned
            oldest = min(k for k in self.keyframes if k != root)
            del self.keyframes[oldest]

    def restore(self, commit_rev: int) -> str | None:
        """Pixel-exact restore of an archived keyframe. No diffusion."""
        entry = self.keyframes.get(commit_rev)
        if entry is None:
            log.info("restore miss for rev %d (falling back to re-anchor)",
                     commit_rev)
            return None
        jpeg, lat = entry[0], entry[1]
        anc = entry[2] if len(entry) > 2 else None
        from PIL import Image
        self.canvas = Image.open(io.BytesIO(jpeg)).convert("RGB")
        self.canvas_lat = ((lat[0].to("cuda"), lat[1], lat[2])
                           if lat is not None else None)
        self.scratch = None
        self.scratch_lat = None
        self.scratch_anchor = None
        if ANCHOR:
            if anc is not None:
                self.anchor = {"lat": anc["lat"].to("cuda"),
                               "hw": anc["hw"], "tag": anc["tag"],
                               "age": anc["age"].to("cuda"),
                               "pending": anc["pending"].to("cuda")}
            elif self.canvas_lat is not None:
                # pre-anchor keyframe: rebase from the restored latents
                self.anchor = self._new_anchor(*self.canvas_lat)
            else:
                self.anchor = None
        log.info("restore rev %d (0ms, no render)", commit_rev)
        return base64.b64encode(jpeg).decode()

    def unload(self) -> None:
        """Release all GPU memory (the free-GPU control). load() reverses
        it; render() also lazy-loads, so a target arriving while unloaded
        just pays the load time instead of failing."""
        if not self.pipes:
            return
        import gc

        import torch

        self.reset_session()
        self.pipes = {}
        self._sam = None
        self._sam_proc = None
        self.previewers = {}                 # drops GPU lstsq state
        gc.collect()
        torch.cuda.empty_cache()
        log.info("flux2 unloaded (gpu freed)")

    @staticmethod
    def _quantize(transformer) -> None:
        """NVFP4 (Blackwell tensor cores) with the reference recipe's
        selective quantization: tiny layers violate FP4 kernel shape
        constraints and lose more to quantize overhead than they save."""
        import torch.nn as nn
        from torchao.prototype.mx_formats.inference_workflow import (
            NVFP4DynamicActivationNVFP4WeightConfig)
        from torchao.quantization import quantize_

        skipped = [0]

        def big_linear(m, fqn):
            if not (isinstance(m, nn.Linear)
                    and min(m.in_features, m.out_features) >= 1024):
                return False
            if any(k in fqn for k in FP4_KEEP_BF16):
                skipped[0] += 1
                return False
            return True

        if QUANT == "nvfp4-wo":
            from torchao.prototype.mx_formats.inference_workflow import (
                NVFP4WeightOnlyConfig)
            cfg = NVFP4WeightOnlyConfig()
        else:
            try:
                import mslk  # noqa: F401
                triton_ok = True
            except Exception:
                triton_ok = False
            cfg = NVFP4DynamicActivationNVFP4WeightConfig(
                use_dynamic_per_tensor_scale=(FP4_TENSOR_SCALE != "off"),
                use_triton_kernel=triton_ok)
        quantize_(transformer, cfg, filter_fn=big_linear)
        if FP4_KEEP_BF16:
            log.info("FP4 keep-bf16 filter %s left %d linears unquantized",
                     FP4_KEEP_BF16, skipped[0])

    def load(self) -> None:
        import gc

        import torch
        from diffusers import Flux2KleinPipeline

        if self.pipes:
            return

        # GB10/Spark: cuDNN silently mis-executed this VAE's convolutions,
        # hence the historical blanket disable. On an RTX PRO it is
        # clean and faster; _verify_vae() below is the categorical guard
        # against silent numeric corruption on ANY host.
        torch.backends.cudnn.enabled = CUDNN_OK

        # hq loads FIRST: its bf16 weights peak at ~18GB before NVFP4
        # shrinks them to ~5GB, so it needs the emptiest possible GPU
        specs = []
        if MODEL_HQ.strip():
            specs.append(("hq", MODEL_HQ.strip()))
        specs.append(("fast", MODEL))
        for tag, mid in specs:
            log.info("loading %s model %s (%dpx, %d steps, quant=%s, "
                     "compile=%s, cudnn=%s) ...", tag, mid, SIZE, STEPS,
                     QUANT, COMPILE, CUDNN_OK)
            try:
                pipe = Flux2KleinPipeline.from_pretrained(
                    mid, torch_dtype=torch.bfloat16)
            except Exception:
                if tag == "fast":
                    raise               # no fast model = no renderer
                log.exception("hq model failed to load; fast-only")
                continue
            pipe.to("cuda")
            pipe.set_progress_bar_config(disable=True)
            self._verify_vae(pipe)
            if QUANT.startswith("nvfp4"):
                try:
                    self._quantize(pipe.transformer)
                    log.info("%s transformer quantized (%s)", tag, QUANT)
                except Exception:
                    log.exception("%s quantization failed; running bf16",
                                  tag)
            if TE_QUANT.startswith("nvfp4"):
                try:
                    self._quantize(pipe.text_encoder)
                    import gc as _gc
                    _gc.collect()
                    torch.cuda.empty_cache()
                    log.info("%s text encoder quantized (%s); vram %.1fGB",
                             tag, TE_QUANT,
                             torch.cuda.memory_allocated() / 1e9)
                except Exception:
                    log.exception("%s TE quantization failed; bf16 kept",
                                  tag)
            if COMPILE:
                try:
                    pipe.transformer.compile_repeated_blocks(fullgraph=True)
                    log.info("%s regional compile enabled", tag)
                except Exception:
                    try:
                        pipe.transformer = torch.compile(pipe.transformer)
                        log.info("%s full compile enabled", tag)
                    except Exception:
                        log.exception("%s compile failed; eager", tag)
            self.pipes[tag] = pipe
            # warm both paths per model (also pays compile cost per shape)
            keep = self.model_pref
            self.model_pref = tag
            self.render("warmup scene", 0.5, [])
            self.render("warmup", 0.5, [],
                        instruction="make it slightly warmer")
            if AUTOTAU:
                try:
                    self._calibrate_tau(tag)
                except Exception:
                    log.exception("%s tau auto-calibration failed "
                                  "(static defaults stay)", tag)
            if DRAFT_LOOP and self.canvas is not None:
                # pre-pay the draft compile shape (gen+ref at DRAFT_SIZE)
                try:
                    for ds in {DRAFT_SIZE, 512}:
                        cond_d = self._encode_canvas(pipe, self.canvas,
                                                     size=ds)
                        self._call_klein(pipe, self._previewer(tag),
                                         "make it slightly warmer",
                                         DRAFT_STEPS, cond=cond_d,
                                         allow_pass=False, size=ds)
                        log.info("%s draft shape warmed (%dpx)", tag, ds)
                except Exception:
                    log.exception("%s draft warmup failed (loop will "
                                  "fall back to direct)", tag)
            self.model_pref = keep
            gc.collect()
            torch.cuda.empty_cache()
            log.info("%s resident; vram now %.1fGB", tag,
                     torch.cuda.memory_allocated() / 1e9)
        if self.model_pref not in self.pipes:
            self.model_pref = "fast"
        self.reset_session()        # warmup canvases must not leak
        log.info("flux2 ready (models=%s, pref=%s, previews=%s)",
                 "+".join(self.pipes), self.model_pref,
                 "on" if PREVIEW and self._preview_ok else "off")

    def _calibrate_tau(self, tag: str) -> None:
        """Measure THIS model's no-op repaint floor (its quantized
        numerics' fingerprint) and derive the passthrough thresholds from
        it — a threshold constant measured under one quantization recipe
        silently miscalibrates under another. Runs two identical no-op
        edits against the warmup canvas: the first supplies the pooled
        floor percentiles, the pair doubles as a determinism canary.
        Clamped to [0.5x, 2.5x] of the configured constant so an
        unrepresentative warmup scene cannot suppress all subtle edits;
        env vars remain hard overrides."""
        import torch
        import torch.nn.functional as _F

        if self.canvas_lat is None or self.canvas_lat[2] != tag:
            return
        pipe = self.pipes[tag]
        previewer = self._previewer(tag)
        cond = (self.canvas_lat[0], self.canvas_lat[1])
        # VARIED phrasings: different prompt embeds shift the repaint
        # trajectory more than one repeated phrasing, and THAT is the
        # floor the thresholds must clear (probe-verified). The first
        # phrasing runs twice as a determinism canary.
        noops = ["Keep the image exactly the same.",
                 "Do not change anything in the image.",
                 "Leave the picture as it is."]
        pooled_all = []
        first = None
        for i, noop in enumerate([noops[0]] + noops):
            _, l1 = self._call_klein(pipe, previewer, noop, STEPS,
                                     cond=cond, allow_pass=False)
            if i == 0:
                first = l1[0]
                continue
            if i == 1 and not torch.equal(first, l1[0]):
                log.warning("%s repaint is NONdeterministic (identical "
                            "inputs gave different latents) — floor "
                            "includes true noise", tag)
            hw = l1[1]
            delta = ((l1[0].float() - cond[0].float())
                     .pow(2).mean(dim=-1).sqrt()[0])
            pooled_all.append(_F.avg_pool2d(
                delta.reshape(1, 1, *hw), 3, stride=1,
                padding=1).reshape(-1))
        allp = torch.cat(pooled_all)
        p99 = float(torch.quantile(allp, 0.99))
        pmax = float(allp.max())
        base = PASS_TAU_HQ if tag == "hq" else PASS_TAU
        # probe-winning recipe: lo at the multi-phrasing p99, hi safely
        # above the floor's MAXIMUM so no noise spike ever adopts as
        # "genuine" (the density gate handles the band in between)
        lo = min(max(p99, 0.5 * base), 2.5 * base)
        hi = max(1.3 * pmax, lo + 0.06)
        if p99 > 1.5 * base or p99 < base / 1.5:
            log.warning("%s measured floor p99=%.3f differs >1.5x from "
                        "configured tau %.2f (clamped to %.3f)",
                        tag, p99, base, lo)
        self.pass_tau[tag] = lo
        self.pass_tau_hi[tag] = hi
        log.info("%s thresholds calibrated: floor p99=%.3f max=%.3f -> "
                 "tau_lo=%.3f tau_hi=%.3f", tag, p99, pmax, lo, hi)

    def _verify_vae(self, pipe) -> None:
        """Fail LOUDLY if the VAE roundtrip corrupts images.

        Silent numeric corruption (wrong kernels, driver bugs, dtype
        surprises) produces plausible-looking noise that only a human
        notices. This guard catches the whole category at startup: encode
        and decode a synthetic gradient and require it to survive.
        """
        import numpy as np
        import torch
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (SIZE, SIZE))
        d = ImageDraw.Draw(img)
        for y in range(SIZE):
            d.line([(0, y), (SIZE, y)], fill=(y % 256, 60, 255 - y % 256))
        d.ellipse([SIZE // 4, SIZE // 4, SIZE // 2, SIZE // 2],
                  fill=(255, 40, 40))
        x = torch.from_numpy(np.array(img)).float().permute(2, 0, 1)[None]
        x = x / 127.5 - 1
        with torch.no_grad():
            lat = pipe.vae.encode(
                x.to("cuda", torch.bfloat16)).latent_dist.sample()
            dec = pipe.vae.decode(lat).sample.float().cpu()
        mse = float(((dec.clamp(-1, 1) - x) ** 2).mean())
        if mse > 0.05:
            raise RuntimeError(
                f"VAE roundtrip corrupt (mse={mse:.3f}) — refusing to serve "
                f"garbage frames. Check cuDNN/driver/kernel numerics.")
        log.info("VAE roundtrip verified (mse=%.4f)", mse)

    def calibrate_scene_floor(self) -> None:
        """Measure THIS scene's no-op repaint floor and derive the merge
        thresholds from it. The load-time calibration (warmup scene) is
        not transferable: the floor is scene-dependent, and the FP4 text
        encoder adds a per-prompt systematic shift on top — on a complex
        scene the stale thresholds sat far below the real floor, which
        both ate subtle edits and let the coverage override misfire.
        One extra render (~1s), run in idle GPU time after each root."""
        import torch
        import torch.nn.functional as _F

        if not self.needs_floor_calib or self.canvas_lat is None:
            return
        tag = self.canvas_lat[2]
        pipe = self.pipes.get(tag)
        if pipe is None:
            return
        self.needs_floor_calib = False
        previewer = self._previewer(tag)
        cond = (self.canvas_lat[0], self.canvas_lat[1])
        pooled_all = []
        for noop in ("Keep everything exactly the same.",
                     "Do not change anything in the image."):
            _, l1 = self._call_klein(pipe, previewer, noop, STEPS,
                                     cond=cond, allow_pass=False)
            delta = ((l1[0].float() - cond[0].float())
                     .pow(2).mean(dim=-1).sqrt()[0])
            pooled_all.append(_F.avg_pool2d(
                delta.reshape(1, 1, *l1[1]), 3, stride=1,
                padding=1).reshape(-1))
        allp = torch.cat(pooled_all)
        p99 = float(torch.quantile(allp, 0.99))
        pmax = float(allp.max())
        self.pass_tau[tag] = max(p99, 0.05)
        self.pass_tau_hi[tag] = max(1.3 * pmax, p99 + 0.06)
        log.info("scene floor calibrated (%s): p99=%.3f max=%.3f -> "
                 "tau_lo=%.3f tau_hi=%.3f", tag, p99, pmax,
                 self.pass_tau[tag], self.pass_tau_hi[tag])

    def set_size(self, size: int) -> None:
        """Live resolution switch. The current canvas keeps its pixels;
        the next edit re-renders it at the new size (Klein accepts a
        reference image of a different resolution)."""
        if size != self.size:
            log.info("resolution %d -> %d", self.size, size)
            self.size = size
            self.needs_floor_calib = True

    def reset_session(self) -> None:
        self.seed = FIXED_SEED if FIXED_SEED is not None else 0
        self.canvas = None
        self.canvas_lat = None
        self.scratch = None
        self.scratch_lat = None
        self.scratch_rev = -1
        self.anchor = None
        self.scratch_anchor = None      # pass_tau survives: it's per-model
        self.rev = 0
        self.edits_since_root = 0
        self.keyframes = {}

    # ---- custom klein loop (latent chaining + token passthrough) ----

    def _encode_canvas(self, pipe, image, size=None):
        """Pixels -> packed normalized conditioning latents. Only needed
        where pixels were produced OUTSIDE the model (composite moves,
        restores whose latents were evicted, model switches) — and for
        DRAFT conditioning (canvas downscaled to the draft size)."""
        import torch

        if size is not None and image.size != (size, size):
            from PIL import Image as _I
            image = image.resize((size, size), _I.LANCZOS)
        w, h = image.size
        x = pipe.image_processor.preprocess(image, height=h, width=w)
        gen = torch.Generator("cuda").manual_seed(self.seed)
        with torch.no_grad():
            lat = pipe._encode_vae_image(x.to("cuda", pipe.vae.dtype), gen)
        hw = (lat.shape[2], lat.shape[3])
        return pipe._pack_latents(lat), hw

    def _tau_lo(self, tag: str) -> float:
        return self.pass_tau.get(tag,
                                 PASS_TAU_HQ if tag == "hq" else PASS_TAU)

    def _tau_hi_for(self, tag: str) -> float:
        env = PASS_TAU_HI_HQ if tag == "hq" else PASS_TAU_HI
        if env:
            return env
        lo = self._tau_lo(tag)
        return self.pass_tau_hi.get(tag, max(1.5 * lo, lo + 0.06))

    @staticmethod
    def _new_anchor(packed, hw, tag: str) -> dict:
        import torch
        n = packed.shape[1]
        return {"lat": packed.detach().clone(), "hw": hw, "tag": tag,
                "age": torch.zeros(n, dtype=torch.int32,
                                   device=packed.device),
                "pending": torch.zeros(n, dtype=torch.int32,
                                       device=packed.device)}

    @staticmethod
    def _clone_anchor(a: dict | None) -> dict | None:
        if a is None:
            return None
        return {"lat": a["lat"].clone(), "hw": a["hw"], "tag": a["tag"],
                "age": a["age"].clone(), "pending": a["pending"].clone()}

    @staticmethod
    def _merge_tokens(latents, image_latents, out_hw, tau_lo, tau_hi,
                      anchor=None, k_adopt=ANCHOR_K, free=None,
                      deletes_active=False):
        """Post-denoise token merge — the anti-drift layer.

        Legacy (anchor=None): freeze tokens whose 3x3-pooled delta vs the
        CONDITIONING is below tau_lo (bit-exact copy). This is exactly the
        pre-anchor behavior.

        Anchored: compare against the per-token ANCHOR (the value each
        token had after the last render that intentionally changed it) and
        classify into three bands:
          frozen  (pooled <= tau_lo): snap back to the anchor bit-exactly —
                  a noise excursion is ERASED, never integrated, so error
                  cannot random-walk between adoption events.
          genuine (pooled > tau_hi, 3x3-dilated for clean borders): the
                  repaint stands and the anchor adopts it immediately.
          band    (between): the repaint is DISPLAYED (subtle edits appear
                  at full strength immediately) but the anchor only adopts
                  after k_adopt CONSECUTIVE band renders — FP4 noise does
                  not persist across differently-prompted renders; a real
                  subtle edit persists in the conditioning and does.
        Even a false adoption is a one-time bounded jump (<= ~tau_hi
        pooled), not a walk. Mutates anchor in place; returns latents.
        """
        import torch
        import torch.nn.functional as _F
        n = out_hw[0] * out_hw[1]
        ref = anchor["lat"] if anchor is not None else image_latents
        delta = ((latents.float() - ref.float())
                 .pow(2).mean(dim=-1).sqrt()[0])
        pooled = _F.avg_pool2d(delta.reshape(1, 1, *out_hw), 3,
                               stride=1, padding=1)
        q = torch.quantile(pooled.reshape(-1), torch.tensor(
            [0.5, 0.95], device=delta.device)).tolist()
        cover = float((pooled > tau_lo).float().mean())
        # a DELETE render may never be classified as an intentional
        # global edit — its only legitimate change is the hole regions.
        # (The first version of this guard referenced `self` inside this
        # staticmethod: NameError on the first render of every process,
        # which tripped the stock-pipeline kill switch and silently
        # disabled ALL pinning and enforcement for the whole session.)
        if deletes_active:
            cover = 0.0
        if cover >= MERGE_COVER:
            d = (latents.float() - ref.float())[0]          # (N, C)
            coh = float(d.mean(dim=0).norm()
                        / d.norm(dim=-1).mean().clamp_min(1e-9))
            if coh >= MERGE_COH:
                # global edit: a coherent majority of the image moved in
                # a shared direction on purpose — the model output stands
                # untouched and the anchor adopts it wholesale
                if anchor is not None:
                    anchor["lat"] = latents.detach().clone().to(
                        anchor["lat"].dtype)
                    anchor["pending"].zero_()
                    anchor["age"].zero_()
                log.info("merge: global edit (cover %.0f%% coh %.3f) — "
                         "model output stands (pooled p50=%.3f p95=%.3f)",
                         cover * 100, coh, q[0], q[1])
                return latents
            log.info("merge: high cover %.0f%% but incoherent "
                     "(coh %.3f < %.2f) — treated as noise floor",
                     cover * 100, coh, MERGE_COH)
        if deletes_active:
            # DELETE containment is unconditional geometry: outside the
            # hole mask every token snaps bit-exactly to the
            # conditioning; inside, the repair's output stands. No tau,
            # no coverage, no anchor. (cover=0 alone still pinned only
            # below-tau tokens — ~4% of a real delete render.)
            if free is not None:
                keepm = (~free).reshape(1, n, 1)
            else:
                keepm = torch.ones(1, n, 1, dtype=torch.bool,
                                   device=latents.device)
            latents = torch.where(keepm,
                                  image_latents.to(latents.dtype),
                                  latents)
            log.info("merge: delete stencil (%d/%d tokens free)",
                     int(free.sum()) if free is not None else 0, n)
            return latents
        if anchor is None:
            if free is not None and getattr(
                    Flux2Canvas, "_hardpin_flag", False):
                # STENCIL hard containment: inside the mask the model's
                # output stands verbatim; outside is FORCED back to the
                # conditioning — no tau, no coverage bail. Paint cannot
                # land outside the stencil.
                keepm = (~free).reshape(1, n, 1)
                latents = torch.where(keepm,
                                      image_latents.to(latents.dtype),
                                      latents)
                log.info("merge: hard-pin stencil (%d/%d tokens free)",
                         int(free.sum()), n)
                return latents
            changed = (pooled > tau_lo)
            if free is not None:
                # composite-modified tokens: the conditioning there is a
                # synthetic FILL, not truth — freezing to it stamps a
                # blurry smear over the model's clean repaint (the reason
                # finals looked worse than the live preview). The model's
                # output always stands inside the modified region.
                changed = changed | free.reshape(1, 1, *out_hw)
            changed = (_F.max_pool2d(changed.float(), 3, stride=1,
                                     padding=1) > 0)
            keep = (~changed).reshape(1, n, 1)
            kept = int(keep.sum())
            if kept:
                latents = torch.where(keep,
                                      image_latents.to(latents.dtype),
                                      latents)
            log.info("passthrough: kept %d/%d tokens "
                     "(tau=%.2f delta p50=%.3f p95=%.3f)",
                     kept, n, tau_lo, q[0], q[1])
            return latents
        hot = (pooled > tau_hi)
        if free is not None:
            hot = hot | free.reshape(1, 1, *out_hw)
        genuine = (_F.max_pool2d(hot.float(), 3, stride=1,
                                 padding=1) > 0).reshape(-1)
        frozen = (pooled <= tau_lo).reshape(-1) & ~genuine
        band = ~frozen & ~genuine
        # band-density gate (see ANCHOR_DENS): sparse band tokens are FP4
        # noise — snap them back so noise never enters the conditioning
        # and can never be adopted. Dense band regions are real subtle
        # edits — display immediately, adopt after k_adopt persistence.
        if ANCHOR_DENS > 0:
            dens = _F.avg_pool2d(band.reshape(1, 1, *out_hw).float(), 7,
                                 stride=1, padding=3).reshape(-1)
            keep_band = band & (dens >= ANCHOR_DENS)
        else:
            keep_band = band
        snap = frozen | (band & ~keep_band)
        if int(snap.sum()):
            latents = torch.where(snap.reshape(1, n, 1),
                                  ref.to(latents.dtype), latents)
        pend = torch.where(keep_band, anchor["pending"] + 1,
                           torch.zeros_like(anchor["pending"]))
        adopt = genuine | (keep_band & (pend >= k_adopt))
        if int(adopt.sum()):
            am = adopt.reshape(1, n, 1)
            anchor["lat"] = torch.where(am, latents.to(anchor["lat"].dtype),
                                        anchor["lat"])
            pend = torch.where(adopt, torch.zeros_like(pend), pend)
        anchor["pending"] = pend
        anchor["age"] = torch.where(adopt, torch.zeros_like(anchor["age"]),
                                    anchor["age"] + 1)
        log.info("anchor merge: snap %d band %d genuine %d adopted %d "
                 "/%d (tau %.2f/%.2f pooled p50=%.3f p95=%.3f)",
                 int(snap.sum()), int(keep_band.sum()), int(genuine.sum()),
                 int(adopt.sum()), n, tau_lo, tau_hi, q[0], q[1])
        return latents

    def _call_klein(self, pipe, previewer, prompt: str, steps: int,
                    cond=None, on_preview=None, cancel_check=None,
                    allow_pass=True, tau=None, tau_hi=None, anchor=None,
                    free_px=None, size=None, noise_override=None,
                    init_latents=None, sigma_node=None, edit_keep=None,
                    exp_ref_guide=None, exp_adain=None,
                    exp_attn_bias=None):
        """Faithful replica of Flux2KleinPipeline.__call__ (diffusers 0.39)
        with the three things the stock call can't do: conditioning from
        raw latents, per-token passthrough, and step hooks for previews/
        preemption. Returns (PIL image, (packed_latents, (h, w))).
        """
        import numpy as np
        import torch
        from diffusers.pipelines.flux2.pipeline_flux2_klein import (
            compute_empirical_mu, retrieve_timesteps)

        device = "cuda"
        size = size or self.size
        generator = torch.Generator(device).manual_seed(self.seed)
        prompt_embeds, text_ids = pipe.encode_prompt(prompt=prompt,
                                                     device=device)
        num_ch = pipe.transformer.config.in_channels // 4
        latents, latent_ids = pipe.prepare_latents(
            1, num_ch, size, size, prompt_embeds.dtype, device,
            generator)
        if noise_override is not None:
            # shared noise bank: the draft and its promotion resolve the
            # SAME noise field (block-mean related), so what the judge
            # approved at low res is what the full render refines
            latents = noise_override.to(device, latents.dtype)
        grid = size // (pipe.vae_scale_factor * 2)
        out_hw = (grid, grid)

        image_latents = image_latent_ids = cond_hw = None
        if cond is not None:
            # cond may be one (packed, hw) tuple or a LIST of them —
            # _prepare_image_ids natively indexes multiple references
            conds = cond if isinstance(cond, list) else [cond]
            image_latents = torch.cat(
                [c[0] for c in conds], dim=1).to(
                device, pipe.transformer.dtype)
            cond_hw = conds[0][1]
            image_latent_ids = pipe._prepare_image_ids(
                [torch.zeros(1, num_ch * 4, *c[1])
                 for c in conds]).to(device)

        sigmas = np.linspace(1.0, 1 / steps, steps)
        mu = compute_empirical_mu(image_seq_len=latents.shape[1],
                                  num_steps=steps)
        timesteps, _ = retrieve_timesteps(pipe.scheduler, steps, device,
                                          sigmas=sigmas, mu=mu)
        if hasattr(pipe.scheduler, "set_begin_index"):
            pipe.scheduler.set_begin_index(0)

        start_i = 0
        if init_latents is not None and sigma_node is not None:
            # PROMOTION: resume the schedule mid-way from an accepted
            # draft's upsampled latents. Blend x = s*noise + (1-s)*x0 at
            # an ACTUAL shifted schedule node so the model stays on its
            # trained trajectory; only the remaining steps run.
            k = max(0, min(int(sigma_node), steps - 1))
            s0 = float(pipe.scheduler.sigmas[k])
            x0 = init_latents.to(device, latents.dtype)
            latents = s0 * latents + (1.0 - s0) * x0
            if hasattr(pipe.scheduler, "set_begin_index"):
                pipe.scheduler.set_begin_index(k)
            timesteps = timesteps[k:]
            start_i = k
            log.info("promotion resume: node %d sigma %.3f (%d steps)",
                     k, s0, len(timesteps))

        with torch.no_grad():
            for i, t in enumerate(timesteps):
                if cancel_check is not None and i > 0 and cancel_check():
                    raise RenderPreempted()
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                lmi = latents.to(pipe.transformer.dtype)
                lids = latent_ids
                jkw = None
                if image_latents is not None:
                    lmi = torch.cat([latents, image_latents], dim=1).to(
                        pipe.transformer.dtype)
                    lids = torch.cat([latent_ids, image_latent_ids], dim=1)
                    if exp_attn_bias:
                        # EXPERIMENTAL: additive attention bias that
                        # upweights the reference tokens' keys by
                        # log(lambda) — attention toward refs scales
                        # by lambda if the transformer forwards the
                        # mask to SDPA
                        import math as _math
                        bias = torch.zeros(
                            (1, 1, 1, lmi.shape[1]),
                            device=lmi.device, dtype=lmi.dtype)
                        bias[..., latents.shape[1]:] = _math.log(
                            float(exp_attn_bias))
                        jkw = {"attention_mask": bias}
                with pipe.transformer.cache_context("cond"):
                    noise_pred = pipe.transformer(
                        hidden_states=lmi, timestep=timestep / 1000,
                        guidance=None, encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids, img_ids=lids,
                        joint_attention_kwargs=jkw, return_dict=False)[0]
                noise_pred = noise_pred[:, : latents.size(1)]
                if (exp_ref_guide and image_latents is not None):
                    # EXPERIMENTAL reference guidance: extrapolate past
                    # the reference's contribution (CFG math aimed at
                    # the refs): pred = base + lam*(with_ref - base)
                    with pipe.transformer.cache_context("uncond"):
                        np_base = pipe.transformer(
                            hidden_states=latents.to(
                                pipe.transformer.dtype),
                            timestep=timestep / 1000, guidance=None,
                            encoder_hidden_states=prompt_embeds,
                            txt_ids=text_ids, img_ids=latent_ids,
                            joint_attention_kwargs=None,
                            return_dict=False)[0]
                    np_base = np_base[:, : latents.size(1)]
                    lam = float(exp_ref_guide)
                    noise_pred = np_base + lam * (noise_pred - np_base)
                latents = pipe.scheduler.step(noise_pred, t, latents,
                                              return_dict=False)[0]
                if exp_adain is not None:
                    # EXPERIMENTAL AdaIN: pull the generation's latent
                    # channel statistics toward the style reference's
                    ref_l, ad_s = exp_adain
                    rl = ref_l.to(latents.device, latents.dtype)
                    mu = latents.mean(dim=1, keepdim=True)
                    sd = latents.std(dim=1, keepdim=True) + 1e-6
                    rmu = rl.mean(dim=1, keepdim=True)
                    rsd = rl.std(dim=1, keepdim=True) + 1e-6
                    styled = (latents - mu) / sd * rsd + rmu
                    latents = latents + float(ad_s) * (styled - latents)
                if on_preview is not None and i < steps - 1:
                    try:
                        jpeg = previewer.jpeg(latents)
                        if jpeg:
                            on_preview(jpeg, i + 1, steps)
                    except Exception:
                        log.exception("preview frame failed (continuing)")

        # token merge: freeze/snap-back what the edit left at the repaint-
        # noise floor (see _merge_tokens). Requires the conditioning grid
        # to match the output grid — after a resolution switch it doesn't,
        # and that render is a full repaint anyway.
        tau = PASS_TAU if tau is None else tau
        self._merge_anchored = False
        self._promote_lost_big = False
        if (allow_pass and tau > 0 and image_latents is not None
                and cond_hw == out_hw):
            a = (anchor if ANCHOR and anchor is not None
                 and anchor.get("hw") == out_hw else None)
            self._merge_anchored = a is not None
            free = None
            if free_px is not None:
                from PIL import Image as _Img
                import numpy as _np
                m = _Img.fromarray(free_px.astype("uint8") * 255, "L")
                m = m.resize(out_hw[::-1], _Img.BILINEAR)
                free = torch.from_numpy(
                    (_np.asarray(m) > 16)).to(latents.device)
            latents = self._merge_tokens(
                latents, image_latents, out_hw, tau,
                (tau_hi if tau_hi is not None else max(1.5 * tau,
                                                       tau + 0.06)),
                anchor=a, free=free,
                deletes_active=bool(
                    getattr(self, "_composite_pin", False)
                    or (getattr(self, "_last_deletes", None)
                        and not getattr(self, "_regen_deletes",
                                        False))))
            # EDIT-PRESERVING PROMOTE: the resume re-noises the approved
            # draft to ~97% noise, so the model re-derives the image from
            # the OLD conditioning — fine-detail removals (cuffs, stitch
            # lines) get regrown from it and the approved change is lost
            # even though the preview showed it. Deterministic restore:
            # tokens the DRAFT deliberately changed (edit_keep) that came
            # back ~equal to the old conditioning are overwritten with
            # the approved draft's upsampled latents, and the anchor is
            # rebased there so later merges don't resurrect the old
            # pixels. Only provably-lost tokens are touched.
            if (edit_keep is not None and init_latents is not None):
                import torch.nn.functional as _F
                dlt = (latents - image_latents.to(latents.dtype)) \
                    .abs().mean(-1).reshape(1, 1, *out_hw)
                dlt = _F.avg_pool2d(dlt, 3, stride=1,
                                    padding=1).reshape(-1)
                ek = edit_keep.to(latents.device)
                lost = (dlt <= tau) & ek
                nl = int(lost.sum())
                ek_total = max(1, int(ek.sum()))
                # act only when the edit was SUBSTANTIALLY lost: a
                # surviving edit always leaves a few under-tau halo
                # tokens, and treating those as "lost" retried nearly
                # every promote. Under 35% of the changed region (or a
                # trivial absolute count) = the edit made it through.
                if nl < max(40, int(0.35 * ek_total)):
                    if nl:
                        log.info("edit-preserve: %d/%d changed tokens "
                                 "under tau — edit survived, no action",
                                 nl, ek_total)
                    nl = 0
                # splice ONLY tiny patches: upsampled 256-draft latents
                # are blurry, and splicing 900+ tokens of them into sharp
                # full-res line art decoded as smeared vertical streaks
                # (user-observed) — and polluted the anchor with them.
                # Large losses are handled by the caller re-running the
                # promote from a later sigma node instead (more draft
                # signal, no splicing).
                if nl and nl <= int(0.04 * latents.shape[1]):
                    lm = lost.reshape(1, -1, 1)
                    x0 = init_latents.to(latents.device, latents.dtype)
                    latents = torch.where(lm, x0, latents)
                    if a is not None:
                        a["lat"] = torch.where(
                            lm, x0.to(a["lat"].dtype), a["lat"])
                    log.info("edit-preserve: %d/%d tokens restored from "
                             "approved draft", nl, latents.shape[1])
                elif nl:
                    self._promote_lost_big = True
                    log.info("edit-preserve: %d/%d tokens lost — too "
                             "large to splice, flagging promote retry",
                             nl, latents.shape[1])

        mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(
            device, latents.dtype)
        std = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1)
                         + pipe.vae.config.batch_norm_eps).to(
            device, latents.dtype)
        packed = latents
        lat = packed.permute(0, 2, 1).reshape(1, num_ch * 4, grid, grid)
        lat = lat * std + mean
        lat = pipe._unpatchify_latents(lat)
        with torch.no_grad():
            img = pipe.vae.decode(lat.to(pipe.vae.dtype),
                                  return_dict=False)[0]
        image = pipe.image_processor.postprocess(img, output_type="pil")[0]
        return image, (packed, out_hw)

    def _shared_noise(self, pipe, seed: int, size: int):
        """Noise drawn ONCE per (seed, full-grid); the draft uses its
        block-mean (variance-renormalized), so draft and promotion
        resolve the same field. Packed layout (1, N, C), row-major."""
        import torch

        num_ch = pipe.transformer.config.in_channels // 4
        fg = self.size // (pipe.vae_scale_factor * 2)
        key = (seed, fg)
        bank = getattr(self, "_noise_bank", None)
        if bank is None:
            bank = self._noise_bank = {}
        full = bank.get(key)
        if full is None:
            gen = torch.Generator("cuda").manual_seed(seed)
            full, _ = pipe.prepare_latents(
                1, num_ch, self.size, self.size, torch.bfloat16,
                "cuda", gen)
            bank.clear()            # one live entry is enough
            bank[key] = full
        if size == self.size:
            return full
        import torch.nn.functional as F
        dg = size // (pipe.vae_scale_factor * 2)
        C = full.shape[-1]
        # adaptive pooling handles ANY grid ratio (768->512 is 1.5x and
        # crashed the integer-block reshape); variance renormalized by
        # the linear ratio — reduces exactly to blockmean*r when integer
        x = full.reshape(1, fg, fg, C).permute(0, 3, 1, 2).float()
        x = F.adaptive_avg_pool2d(x, dg) * (fg / dg)
        return (x.permute(0, 2, 3, 1).reshape(1, dg * dg, C)
                .to(full.dtype))

    @staticmethod
    def _upsample_packed(pipe, packed, dg: int, fg: int):
        """Accepted draft latents (1, dg*dg, C4) -> (1, fg*fg, C4),
        upsampled in TRUE latent space: packed tokens are PATCHIFIED
        (each token is a 2x2 spatial block), so interpolating across
        tokens scrambles sub-patch structure into confetti artifacts
        (observed). Unpack -> bilinear -> repack keeps spatial meaning."""
        import torch.nn.functional as F

        C4 = packed.shape[-1]
        C = C4 // 4
        x = packed.permute(0, 2, 1).reshape(1, C4, dg, dg).float()
        true = pipe._unpatchify_latents(x)          # (1, C, 2dg, 2dg)
        true = F.interpolate(true, size=(fg * 2, fg * 2),
                             mode="bilinear", align_corners=False)
        # exact inverse of _unpatchify_latents, then flatten-pack
        x = (true.reshape(1, C, fg, 2, fg, 2)
             .permute(0, 1, 3, 5, 2, 4)
             .reshape(1, C4, fg, fg))
        return (x.reshape(1, C4, fg * fg).permute(0, 2, 1)
                .to(packed.dtype))

    def _jpeg_b64(self, image, quality=85) -> str:
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=quality)
        return base64.b64encode(buf.getvalue()).decode()

    @staticmethod
    def _guard_keep(exclude) -> "object":
        """64x64 cell mask of pixels the guards may measure (True=keep):
        cells mostly covered by the edit's OWN modified region are its
        legitimate footprint and must not count as collateral (a 33-tree
        removal shifts global color legitimately — measured 0.139 false
        positive). None -> measure everything."""
        import numpy as np
        from PIL import Image as _I

        if exclude is None:
            return None
        m = _I.fromarray(exclude.astype("uint8") * 255, "L").resize(
            (64, 64), _I.BILINEAR)
        keep = np.asarray(m, dtype=np.float32) < 96
        if keep.mean() < 0.3:
            return None            # edit covers the frame: guards abstain
        return keep

    @staticmethod
    def _struct_shift(a, b, exclude=None) -> float:
        """Fraction of coarse grayscale cells that changed OUTSIDE the
        edit's own footprint, 0..1. A local edit touches little; a zoom/
        reframe/restyle shifts nearly every cell — and the VLM judge is
        inconsistent about flagging that, so it is detected in code."""
        import numpy as np

        keep = Flux2Canvas._guard_keep(exclude)
        if exclude is not None and keep is None:
            return 0.0

        def cells(img):
            g = np.asarray(img.resize((64, 64)).convert("L"),
                           dtype=np.float32)
            return g.reshape(16, 4, 16, 4).mean(axis=(1, 3))

        changed = np.abs(cells(a) - cells(b)) > 12.0      # (16, 16)
        if keep is not None:
            keep16 = keep.reshape(16, 4, 16, 4).mean(axis=(1, 3)) > 0.5
            if not keep16.any():
                return 0.0
            changed = changed[keep16]
        return float(changed.mean())

    @staticmethod
    def _color_shift(a, b, exclude=None) -> float:
        """Mean absolute per-channel shift OUTSIDE the edit's footprint,
        0..1. A global tint/exposure collateral is code-detectable — the
        VLM judge measurably missed a 35% orange tint at 256px."""
        import numpy as np

        keep = Flux2Canvas._guard_keep(exclude)
        if exclude is not None and keep is None:
            return 0.0
        x = np.asarray(a.resize((64, 64)), dtype=np.float32)
        y = np.asarray(b.resize((64, 64)), dtype=np.float32)
        if keep is not None:
            x = x[keep]
            y = y[keep]
            return float(np.abs(x.mean(axis=0)
                                - y.mean(axis=0)).mean() / 255.0)
        return float(np.abs(x.mean(axis=(0, 1))
                            - y.mean(axis=(0, 1))).mean() / 255.0)

    _hardpin_flag = False        # class flag read by _merge_tokens

    def _stencil_px(self, target: dict):
        """Parser target box -> pixel mask (SAM-snapped, dilated, unioned
        with the raw box). None on degenerate coverage (fail open)."""
        import numpy as np
        from PIL import Image as _I, ImageFilter

        if self.canvas is None or not target:
            return None
        W, H = self.canvas.size
        b = target.get("b") or []
        if len(b) != 4:
            return None
        x0, y0, x1, y1 = (b[0] * W // 100, b[1] * H // 100,
                          b[2] * W // 100, b[3] * H // 100)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        mask = np.zeros((H, W), dtype=bool)
        pad_x, pad_y = int((x1 - x0) * 0.12), int((y1 - y0) * 0.12)
        mask[max(0, y0 - pad_y):min(H, y1 + pad_y),
             max(0, x0 - pad_x):min(W, x1 + pad_x)] = True
        try:
            insts = self._sam3_box_instances(
                [float(x0), float(y0), float(x1), float(y1)]) or []
            for m in insts[:3]:
                grown = _I.fromarray(m.astype("uint8") * 255, "L").filter(
                    ImageFilter.MaxFilter(15))
                mask |= (np.asarray(grown) > 0)
        except Exception:
            log.exception("stencil SAM snap failed (raw box kept)")
        frac = float(mask.mean())
        if frac > 0.85 or frac < 0.005:
            log.info("stencil degenerate (%.2f) — unscoped", frac)
            return None
        log.info("stencil: %.0f%% of frame (%r)", frac * 100,
                 target.get("n"))
        return mask

    def judge_final(self, base_rev: int, requirements: list,
                    instruction: str | None,
                    target_box: list | None = None) -> dict | None:
        """Post-verify: judge the CURRENT canvas against the keyframe it
        was edited from. Used for pixel-mechanical edits (removals,
        moves, region edits) that cannot be drafted — every committed
        edit passes through the LLM one way or the other. None = judge
        unavailable or no before-frame (fail-open)."""
        from .judge import judge_edit

        entry = self.keyframes.get(base_rev)
        if entry is None or self.canvas is None or not requirements:
            return None
        from PIL import Image as _I
        before = (_I.open(io.BytesIO(entry[0])).convert("RGB")
                  .resize((512, 512)))
        after = self.canvas.resize((512, 512))
        exclude = getattr(self, "_last_free", None)
        # the edit's own intended footprint is not collateral: a rug
        # swap or couch add legitimately changes its whole target box,
        # and measuring it as drift burned retries (zebra-carpet case)
        if isinstance(target_box, dict):
            # callers pass either [x0,y0,x1,y1] percent corners or an
            # {x,y,w,h} fraction box — accept both, forever (a dict of
            # len 4 once passed the length check and KeyError'd mid-
            # verify, dropping the whole render to the mock canvas)
            try:
                bx = float(target_box["x"]) * 100
                by = float(target_box["y"]) * 100
                bw = float(target_box["w"]) * 100
                bh = float(target_box["h"]) * 100
                target_box = [bx, by, bx + bw, by + bh]
            except Exception:
                target_box = None
        if target_box and len(target_box) == 4:
            import numpy as _np
            w, h = self.canvas.size
            m = _np.zeros((h, w), dtype=bool)
            x0 = max(0, int(target_box[0] * w / 100))
            y0 = max(0, int(target_box[1] * h / 100))
            x1 = min(w, int(target_box[2] * w / 100))
            y1 = min(h, int(target_box[3] * h / 100))
            if x1 > x0 and y1 > y0:
                m[y0:y1, x0:x1] = True
                exclude = (m if exclude is None
                           else (exclude | m))
        cs = self._color_shift(before, after, exclude=exclude)
        ss = self._struct_shift(before, after, exclude=exclude)
        if cs > 0.07 or ss > 0.55:
            log.info("post-verify: collateral guard tripped "
                     "(color %.3f struct %.2f)", cs, ss)
            return {"verdict": "fail", "met": [False],
                    "what_failed": "unrequested global change (guard)",
                    "corrective_action": "reroll_seed",
                    "revised_instruction": ""}
        return judge_edit(self._jpeg_b64(before, 80),
                          self._jpeg_b64(after, 80),
                          instruction or "apply the requested change",
                          requirements, JUDGE_URL, JUDGE_MODEL,
                          JUDGE_TIMEOUT_MS / 1000.0)

    def render_looped(self, prompt: str, strength: float,
                      entities, instruction: str, requirements: list,
                      on_preview=None, should_abort=None,
                      camera: bool = False, draft_size: int = 0,
                      preserve_last: bool = False,
                      color_guard: bool = False,
                      plan: dict | None = None,
                      **commit_kwargs) -> str:
        """Draft -> judge -> (correct)* -> promote. Falls back to the
        direct path on ANY failure — never worse than today. Runs
        blocking inside the render worker's committed slot."""
        import time as _time

        from .judge import judge_edit

        t0 = _time.monotonic()
        tag = (self.model_pref if self.model_pref in self.pipes
               else next(iter(self.pipes)))
        pipe = self.pipes[tag]
        previewer = self._previewer(tag)
        if self.canvas is None:
            # ROOT: no before-image to draft against — render, then judge
            # AFTER-ONLY against the requirements; one fresh-seed retry
            from .judge import judge_edit
            out = self.render(prompt, strength, entities,
                              instruction=instruction, camera=camera,
                              on_preview=on_preview, **commit_kwargs)
            try:
                # a fresh generation has no "before": preservation-style
                # requirements are meaningless and false-fail (observed)
                rr = (requirements[:-1] if preserve_last
                      else list(requirements or []))
                if rr and self.canvas is not None:
                    after = self._jpeg_b64(
                        self.canvas.resize((512, 512)), 80)
                    v = judge_edit(None, after, instruction or prompt,
                                   rr, JUDGE_URL, JUDGE_MODEL,
                                   JUDGE_TIMEOUT_MS / 1000.0)
                    if v is not None and v.get("verdict") == "fail":
                        log.info("root verify FAIL (%s) — one fresh "
                                 "reroll", str(v.get("what_failed"))[:70])
                        out = self.render(prompt, strength, entities,
                                          instruction=instruction,
                                          camera=camera,
                                          on_preview=on_preview,
                                          **commit_kwargs)
                    elif v is not None:
                        log.info("root verify PASS")
            except Exception:
                log.exception("root verify failed (frame kept)")
            # STYLED roots additionally verify against the reference
            # image itself: the pixel guard catches near-verbatim
            # copies, but the judge catches SEMANTIC copies (same
            # scene, same composition, restyled) that correlation
            # misses — 4/5 style benchmarks once shipped copies.
            try:
                srefs = commit_kwargs.get("style_refs") or []
                if srefs and self.canvas is not None:
                    after = self._jpeg_b64(
                        self.canvas.resize((512, 512)), 80)
                    v2 = judge_edit(
                        srefs[0], after,
                        "generate a new scene in the reference's style",
                        ["The second image depicts a clearly DIFFERENT "
                         "scene/subject than the first image (the first "
                         "is only a style reference) — same visual "
                         "style is fine, same content is a failure",
                         f"The second image depicts: {prompt}"],
                        JUDGE_URL, JUDGE_MODEL,
                        JUDGE_TIMEOUT_MS / 1000.0)
                    if v2 is not None and v2.get("verdict") == "fail":
                        log.info("styled root judged a COPY (%s) — "
                                 "re-rolling suffix-only",
                                 str(v2.get("what_failed"))[:60])
                        ck = dict(commit_kwargs)
                        ck.pop("style_refs", None)
                        ck["fresh_seed"] = True
                        out = self.render(prompt, strength, entities,
                                          instruction=instruction,
                                          camera=camera,
                                          on_preview=on_preview, **ck)
            except Exception:
                log.exception("styled-root copy check failed (kept)")
            return out
        if not instruction:
            return self.render(prompt, strength, entities,
                               instruction=instruction, camera=camera,
                               on_preview=on_preview, **commit_kwargs)
        dsize = min(draft_size or DRAFT_SIZE, self.size)
        self._cg_flag = color_guard
        try:
            return self._render_looped_inner(
                pipe, previewer, prompt, strength, entities, instruction,
                requirements, on_preview, should_abort, camera, dsize,
                commit_kwargs, preserve_last=preserve_last,
                plan=plan or {})
        except Exception:
            # the loop must NEVER surface an error: any internal failure
            # (a crash here once fell through to the MOCK canvas) becomes
            # a plain direct render — exactly today's behavior
            log.exception("draft loop failed — direct render")
            return self.render(prompt, strength, entities,
                               instruction=instruction, camera=camera,
                               on_preview=on_preview, **commit_kwargs)

    def _render_looped_inner(self, pipe, previewer, prompt, strength,
                             entities, instruction, requirements,
                             on_preview, should_abort, camera, dsize,
                             commit_kwargs, preserve_last=False,
                             plan=None):
        import time as _time

        from .judge import judge_edit

        t0 = _time.monotonic()
        plan = plan or {}
        tier = plan.get("strategy") or "standard"
        budget_ms = int(plan.get("budget_ms") or DRAFT_BUDGET_MS)
        max_iters = int(plan.get("max_iters") or DRAFT_MAX_ITERS)
        n_drafts = max(1, int(plan.get("n_drafts") or 1))
        attempts_total = max(max_iters, n_drafts - 1)
        stencil = None
        if plan.get("target"):
            stencil = self._stencil_px(plan["target"])
        if tier != "standard":
            log.info("draft loop start: tier=%s budget=%dms iters=%d "
                     "drafts=%d stencil=%s contact=%s grounded=%s", tier,
                     budget_ms, max_iters, n_drafts,
                     stencil is not None, bool(plan.get("contact")),
                     plan.get("judge_profile") == "grounded")
        attempt_log: list = []
        before_small = self.canvas.resize((dsize, dsize))
        before_b64 = self._jpeg_b64(before_small, quality=80)
        # sketch route: drafts are conditioned on the MARKED frame so the
        # model paints over the user's strokes; before_b64 stays the
        # CLEAN canvas so the judge fails any surviving red ink
        pink = (plan or {}).get("ink")
        if pink and pink.get("strokes"):
            cond_draft = self._encode_canvas(
                pipe, self._ink_composite(self.canvas, pink["strokes"]),
                size=dsize)
        else:
            cond_draft = self._encode_canvas(pipe, self.canvas, size=dsize)

        # seed policy mirrors render(): camera edits need a fresh roll
        if FIXED_SEED is not None:
            seed = FIXED_SEED
        elif camera:
            seed = random.randrange(1 << 31)
        else:
            seed = getattr(self, "session_seed", 42)

        # ---- constraint ladder ----------------------------------------
        # The parser's magnitude rating is only the STARTING rung; when
        # the judge reports the constraint itself blocked the edit (the
        # object never changed / change too weak / wrong region), the
        # retry runs one rung looser instead of repeating a doomed ask:
        #   rung 0  "keep everything else exactly the same" + stencil
        #   rung 1  "adjust the rest as needed", stencil widened to the
        #           contact pair (or dropped)
        #   rung 2  full-redraw tier, no stencil, fresh seed
        ladder = plan.get("ladder") or None
        rung = 0
        rung_cap = 0
        rung_base = None
        if ladder and ladder.get("base"):
            rung_base = str(ladder["base"]).rstrip(".!?")
            rung = max(0, min(2, int(ladder.get("start_rung", 0))))
            rung_cap = min(2, rung + max(0, int(plan.get("max_climb", 1))))
            # the stencil must match the starting rung, not just climbs:
            # a relational edit starting at rung 1 needs the widened
            # fence (both contact parties) from the first attempt
            if rung >= 2:
                stencil = None
            elif rung == 1 and stencil is not None:
                stencil = (self._stencil_px(plan["widen"])
                           if plan.get("widen") else None)

        def _rung_text(base_text: str, r: int) -> str:
            if r >= 2:
                full = ladder.get("prompt_full") or prompt
                return (f"{base_text}. Redraw the whole image as needed "
                        f"to achieve this, keeping the same scene and "
                        f"subjects: {full}.")
            if r == 1:
                return f"{base_text}. Adjust the rest of the image as needed."
            return f"{base_text}. Keep everything else exactly the same."

        cur_instruction = (_rung_text(rung_base, rung) if rung_base
                           else instruction)
        last_cat = None
        best = None                       # (latents, seed, instruction)
        dg = dsize // (pipe.vae_scale_factor * 2)
        fg = self.size // (pipe.vae_scale_factor * 2)
        focus = None                 # judge-aimed retry mask
        ok = False
        # +2 headroom: overtime ladder climbs may exceed attempts_total
        for it in range(attempts_total + 3):
            self.seed = seed
            noise_d = self._shared_noise(pipe, seed, dsize)
            mask = stencil if stencil is not None else focus
            try:
                if mask is not None:
                    Flux2Canvas._hardpin_flag = stencil is not None
                    try:
                        img_d, lat_d = self._call_klein(
                            pipe, previewer, cur_instruction, DRAFT_STEPS,
                            cond=cond_draft, allow_pass=True, size=dsize,
                            tau=self._tau_lo(
                                self.canvas_lat[2] if self.canvas_lat
                                else "fast"),
                            noise_override=noise_d, free_px=mask)
                    finally:
                        Flux2Canvas._hardpin_flag = False
                else:
                    img_d, lat_d = self._call_klein(
                        pipe, previewer, cur_instruction, DRAFT_STEPS,
                        cond=cond_draft, allow_pass=False, size=dsize,
                        noise_override=noise_d)
            except Exception:
                log.exception("draft render failed — direct path")
                break
            if on_preview is not None:
                try:
                    on_preview(self._jpeg_b64(img_d, quality=75),
                               1, DRAFT_STEPS)
                except Exception:
                    pass
            best = (lat_d[0], seed, cur_instruction, img_d)
            verdict = judge_edit(before_b64, self._jpeg_b64(img_d, 80),
                                 cur_instruction, requirements,
                                 JUDGE_URL, JUDGE_MODEL,
                                 JUDGE_TIMEOUT_MS / 1000.0,
                                 attempts=attempt_log or None)
            if verdict is None:
                log.info("draft loop: judge unavailable — promoting "
                         "current draft")
                break
            ok = (verdict.get("verdict") == "pass"
                  and all(verdict.get("met") or [True]))
            if ok and getattr(self, "_cg_flag", False):
                cs = self._color_shift(before_small, img_d)
                ss = self._struct_shift(before_small, img_d)
                if cs > 0.07 or ss > 0.55:
                    ok = False
                    what = ("global color shift" if cs > 0.07
                            else "global reframe/restyle")
                    verdict = dict(verdict, verdict="fail",
                                   what_failed=f"{what} (guard)")
                    log.info("draft loop: collateral guard tripped "
                             "(color %.3f struct %.2f)", cs, ss)
            if ok and plan.get("contact"):
                # zoom re-judge for relational edits: the full-frame
                # judge passes near-misses at draft resolution (a
                # recolored-but-still-raised hot tub "passed"); a crop
                # of the contact region makes the relation visible.
                # A zoom fail flips the verdict and climbs the ladder.
                vz = self._draft_relation_check(
                    img_d, plan, rung_base or cur_instruction)
                if vz is not None and not (
                        vz.get("verdict") == "pass"
                        and all(vz.get("met") or [True])):
                    ok = False
                    zcat = vz.get("failure_category")
                    if zcat not in ("too_weak", "target_missing",
                                    "wrong_region", "wrong_attribute"):
                        zcat = "too_weak"
                    verdict = dict(vz, verdict="fail",
                                   failure_category=zcat)
                    log.info("draft loop: zoom relation check FAIL (%s)",
                             str(vz.get("what_failed", ""))[:70])
            elapsed = (_time.monotonic() - t0) * 1000
            log.info("draft loop it%d: %s (%.0fms) %s", it,
                     "PASS" if ok else "FAIL", elapsed,
                     "" if ok else verdict.get("what_failed", "")[:80])
            if ok:
                break
            # climbable? decided BEFORE the budget cap: promoting a
            # KNOWN failure to save ~1s wastes the entire render, so a
            # pending ladder climb earns one overtime attempt (hard
            # ceiling 2.5x budget still applies)
            pre_cat = verdict.get("failure_category") or "none"
            climbable = (rung_base is not None and rung < rung_cap
                         and (pre_cat in ("too_weak", "target_missing",
                                          "wrong_region")
                              or (pre_cat == last_cat and pre_cat not in
                                  ("none", "collateral",
                                   "not_judgeable"))))
            overtime = climbable and elapsed < budget_ms * 2.5
            if should_abort is not None and should_abort():
                log.info("draft loop: queue abort — promoting best-so-far")
                break
            if ((it >= attempts_total or elapsed > budget_ms)
                    and not overtime):
                log.info("draft loop: budget/queue cap — promoting "
                         "best-so-far")
                break
            met = verdict.get("met") or []
            # content requirements all satisfied but the verdict is still
            # FAIL -> the failure is collateral (the judge sometimes
            # signals this in verdict+prose while met[] stays all-true):
            # keep the seed the judge liked, freeze everything else
            if (preserve_last and len(met) >= 2 and all(met[:-1])
                    and "keep the camera framing" not in cur_instruction):
                # content is RIGHT, collateral is wrong (asked for a
                # floaty, got a zoom too): keep the seed the judge liked
                # and freeze everything else explicitly
                cur_instruction = (cur_instruction.rstrip(".")
                                   + ". Do not change anything else: "
                                   "keep the camera framing, background, "
                                   "style, and every other object "
                                   "exactly as they are.")
                log.info("draft loop: preservation-only fail — freeze "
                         "clause added, seed kept")
                continue
            act = verdict.get("corrective_action") or "reroll_seed"
            cat = verdict.get("failure_category") or "none"
            attempt_log.append(
                f"attempt {it + 1} failed: {cat}; "
                f"{str(verdict.get('what_failed', ''))[:60]}; "
                f"action {act}")
            fb = verdict.get("fail_box")
            if (plan.get("judge_profile") == "grounded" and fb
                    and cat in ("wrong_region", "too_weak",
                                "target_missing")):
                aimed = self._stencil_px({"n": "fail", "b": fb})
                if aimed is not None:
                    focus = (aimed if stencil is None
                             else (stencil | aimed))
                    log.info("draft loop: retry aimed at fail_box %s", fb)
            if act == "escalate_full_res":
                best = None
                break
            # ladder climb: the judge saying "unchanged / too weak /
            # wrong region" — or failing the same way twice — means the
            # CONSTRAINT blocked the edit, not the wording. Loosen one
            # rung before spending attempts on rewording. Collateral
            # failures (cat none/collateral) never climb: there the
            # constraint is doing its job.
            climb = (rung_base is not None and rung < rung_cap
                     and (cat in ("too_weak", "target_missing",
                                  "wrong_region")
                          or (cat == last_cat and cat not in
                              ("none", "collateral", "not_judgeable"))))
            last_cat = cat
            if climb:
                rung += 1
                if rung >= 2:
                    stencil = None
                    focus = None
                elif stencil is not None:
                    widened = (self._stencil_px(plan["widen"])
                               if plan.get("widen") else None)
                    stencil = widened      # None -> fence dropped
                cur_instruction = _rung_text(rung_base, rung)
                seed = random.randrange(1 << 31)
                log.info("ladder climb -> rung %d (cat=%s, stencil=%s)",
                         rung, cat, stencil is not None)
                continue
            if act == "rephrase_instruction":
                rev = (verdict.get("revised_instruction") or "").strip()
                if 3 <= len(rev.split()) <= 45:
                    if rung_base is not None:
                        # keep the ladder: revised wording becomes the
                        # new base, current rung's suffix reapplied
                        rung_base = rev.rstrip(".!?")
                        cur_instruction = _rung_text(rung_base, rung)
                    else:
                        cur_instruction = rev
                else:
                    seed = random.randrange(1 << 31)
            elif act == "strengthen_edit":
                if rung_base is not None:
                    rung_base = (rung_base + ". Make the change clearly "
                                 "visible and pronounced")
                    cur_instruction = _rung_text(rung_base, rung)
                else:
                    base = cur_instruction.rstrip(".")
                    cur_instruction = (base + ". Make the change clearly "
                                       "visible and pronounced.")
            else:                          # reroll_seed / region_hint
                seed = random.randrange(1 << 31)

        # RESTAGE (the ladder's last rung): a relational edit that failed
        # every attempt INCLUDING full-redraw means the model will not
        # restructure the conditioned object via prose (a raised hot tub
        # stays raised at any instruction tier — the conditioning pins
        # its build). Decompose into two PROVEN operations: composite-
        # delete the object (SAM cut + fill), then repaint it from the
        # instruction against clean background — an add, which works.
        # gate: contact edit, loop never passed, ladder available. Not
        # conditioned on reaching rung 2 — degenerate judge verdicts
        # (category "none", prose "1") can block climbing entirely, and
        # an exhausted relational edit deserves the backstop either way.
        if (not ok and plan.get("contact")
                and rung_base is not None):
            ct = plan["contact"]
            tgt = plan.get("target") or {}
            name = tgt.get("n") or ct.get("a") or "object"
            tb = tgt.get("b")
            if tb and tb[2] > tb[0] and tb[3] > tb[1]:
                fb = {"x": (tb[0] + tb[2]) / 200.0,
                      "y": (tb[1] + tb[3]) / 200.0,
                      "w": max(0.05, (tb[2] - tb[0]) / 100.0),
                      "h": max(0.05, (tb[3] - tb[1]) / 100.0)}
            else:
                fb = {"x": 0.5, "y": 0.5, "w": 1.0, "h": 1.0}
            log.info("draft loop: exhausted at rung %d — restage "
                     "(delete '%s' + repaint)", rung, name)
            try:
                ck = {k: v for k, v in (commit_kwargs or {}).items()
                      if k not in ("move_ops", "fresh_seed")}
                # the repaint must be an explicit ADD: after the cut the
                # object exists in neither pixels nor (sometimes) scene
                # text, and "make the X embedded" against nothing no-ops
                # — the tub was deleted and never repainted
                re_instr = (f"Paint a new {name} in the cleared area, "
                            f"built so that: {rung_base}. It is fully "
                            f"integrated into its surroundings. Adjust "
                            f"the rest of the image as needed.")
                out = self.render(
                    prompt, max(strength, 0.5), entities,
                    instruction=re_instr,
                    move_ops=[{"id": name.replace(" ", "_"),
                               "name": name, "delete": True,
                               "all": False, "from_box": fb}],
                    fresh_seed=True, on_preview=on_preview, **ck)
                if self.canvas is not None:
                    vr = self._draft_relation_check(self.canvas, plan,
                                                    rung_base)
                    if vr is not None and vr.get("verdict") != "pass":
                        log.warning("restage still failing (%s) — "
                                    "committed best effort",
                                    str(vr.get("what_failed", ""))[:70])
                    else:
                        log.info("restage verified: %s",
                                 "PASS" if vr else "unjudged")
                return out
            except Exception:
                log.exception("restage failed — falling through")

        if best is None or PROMOTE_MODE == "direct":
            self.seed = seed
            return self.render(prompt, strength, entities,
                               instruction=cur_instruction, camera=camera,
                               on_preview=on_preview, **commit_kwargs)
        lat_acc, seed_acc, instr_acc, img_acc = best
        promote_ctx = None
        if PROMOTE_MODE == "hires":
            # where did the APPROVED draft change the image? (draft
            # latents vs the pre-edit conditioning, per-token, dilated,
            # upsampled to the full-res token grid) — the promote uses
            # this to restore any approved change the resume loses
            edit_keep = None
            try:
                import torch
                import torch.nn.functional as _F
                tag = self.canvas_lat[2] if self.canvas_lat else "fast"
                cl = cond_draft[0].to(lat_acc.device, torch.float32)
                dd = (lat_acc.to(torch.float32) - cl).abs().mean(-1) \
                    .reshape(1, 1, dg, dg)
                dd = _F.avg_pool2d(dd, 3, stride=1, padding=1)
                # STRONG changes only (tau_hi, no dilation): the tau_lo
                # + dilate mask over-flagged halo tokens and made ~80%
                # of promotes "lose" something, doubling render time
                # with a pointless node-2 retry (measured: +1.35s on
                # nearly every edit)
                m = (dd > self._tau_hi_for(tag)).float()
                m = _F.interpolate(m, size=(fg, fg), mode="nearest")
                m = m.reshape(-1) > 0.5
                frac = float(m.float().mean())
                # near-global masks (camera moves, restyles) carry no
                # localization signal — skip, the promote re-derives fine
                if 20 <= int(m.sum()) and frac < 0.85:
                    edit_keep = m
            except Exception:
                log.exception("edit-keep mask failed (promote unguarded)")
            promote_ctx = {
                "seed": seed_acc,
                "init_latents": self._upsample_packed(pipe, lat_acc,
                                                      dg, fg),
                "noise": self._shared_noise(pipe, seed_acc, self.size),
                "sigma_node": PROMOTE_SIGMA_NODE,
                "free_px": stencil,
                "hardpin": stencil is not None,
                "edit_keep": edit_keep,
                "draft_image": img_acc,
            }
        else:                              # recipe transfer
            promote_ctx = {"seed": seed_acc}
        # NOTE: the post-promote contact verify + one-shot repair was
        # removed by user decision: it re-rendered ("re-noised") over an
        # already-approved result with no way back when the repair came
        # out worse. Relational strictness lives in the DRAFT loop's
        # zoom re-judge instead, where a rejection just costs one cheap
        # draft and never overwrites an accepted image.
        out = self.render(prompt, strength, entities,
                          instruction=instr_acc, camera=camera,
                          on_preview=on_preview,
                          promote_ctx=promote_ctx, **commit_kwargs)
        return out

    def _draft_relation_check(self, img_d, plan, claim: str):
        """Zoomed second opinion on a PASSING relational draft: crop the
        contact region (widen/target box, else the whole draft), upscale
        to 512, and re-judge the claim after-only. Returns the verdict
        dict or None (judge unavailable — fail-open)."""
        from .judge import judge_edit

        try:
            box = plan.get("widen") or plan.get("target")
            crop = img_d
            if box and box.get("b"):
                W, H = img_d.size
                b = box["b"]
                # pad by half the box's own size per side: a relation is
                # only visible WITH surroundings (a crop of just the hot
                # tub can't show whether it sits above or in the ground)
                bw, bh = (b[2] - b[0]) / 2.0, (b[3] - b[1]) / 2.0
                x0 = max(0, int((b[0] - bw) * W / 100))
                y0 = max(0, int((b[1] - bh) * H / 100))
                x1 = min(W, int((b[2] + bw) * W / 100))
                y1 = min(H, int((b[3] + bh) * H / 100))
                if x1 - x0 >= 16 and y1 - y0 >= 16:
                    crop = img_d.crop((x0, y0, x1, y1))
            crop = crop.resize((512, 512))
            ct = plan.get("contact") or {}
            reqs = [f"The requested change is clearly applied: {claim}"]
            if ct.get("a") and ct.get("b"):
                reqs.append(
                    f"The {ct['a']} and the {ct['b']} are positioned "
                    f"relative to each other exactly as the instruction "
                    f"requires (embedded / level / touching as asked) — "
                    f"not merely restyled")
            return judge_edit(None, self._jpeg_b64(crop, 85), claim,
                              reqs, JUDGE_URL, JUDGE_MODEL,
                              JUDGE_TIMEOUT_MS / 1000.0)
        except Exception:
            log.exception("draft relation check failed (fail-open)")
            return None

    def rebase_anchor(self) -> None:
        """Forget pre-edit appearance: after a VERIFIED structural
        change, the old anchor still holds the replaced content and
        every later merge can snap fragments of it back (live failure:
        window frames haunting a verified wall-fill)."""
        try:
            if not ANCHOR or self.canvas is None:
                return
            if self.canvas_lat is not None:
                l0, l1, tag = self.canvas_lat
                self.anchor = self._new_anchor(l0, l1, tag)
            else:
                self.anchor = None      # re-anchors from pixels next edit
            log.info("anchor rebased (verified structural change)")
        except Exception:
            log.exception("anchor rebase failed (kept old anchor)")

    @staticmethod
    def _style_sim(img_a, img_b) -> float:
        """RGB multi-channel normalized correlation at 64px — grayscale
        alone missed 4/5 near-copies whose structure matched but whose
        luminance profile differed."""
        import numpy as _np
        from PIL import Image as _I

        def sig(im):
            g = _np.asarray(im.convert("RGB").resize(
                (64, 64), _I.LANCZOS), dtype=_np.float32)
            g = g - g.mean(axis=(0, 1), keepdims=True)
            return g / (g.std(axis=(0, 1), keepdims=True) + 1e-6)

        return float((sig(img_a) * sig(img_b)).mean())

    def render(self, prompt: str, strength: float,
               entities: list[dict] | None,
               on_preview=None, cancel_check=None,
               source: str | None = None,
               instruction: str | None = None,
               reset: bool = False,
               lane: str | None = None,
               commit_rev: int | None = None,
               do_snapshot: bool = False,
               move_ops: list[dict] | None = None,
               camera: bool = False,
               region: dict | None = None,
               promote_ctx: dict | None = None,
               fresh_seed: bool = False,
               ink: dict | None = None,
               style_refs: list[str] | None = None,
               style_suffix: str | None = None,
               preserve_names: list[str] | None = None) -> str:
        import torch

        if not self.pipes:          # lazy reload after a free-GPU period
            self.load()
        # capture the active model ONCE: a mid-render pref switch must not
        # swap pipes under the denoise loop
        tag = (self.model_pref if self.model_pref in self.pipes
               else next(iter(self.pipes)))
        pipe = self.pipes[tag]
        previewer = self._previewer(tag)

        provisional = (lane == "provisional"
                       or (lane is None and source in FAST_SOURCES))
        steps = FAST_STEPS if provisional else STEPS
        # positional edits: geometry happens in CODE before the model runs;
        # a move with no other changes still needs a (generic) repair edit.
        # The repair carries a deterministic cardinality clause built from
        # the op's own geometry — without it, a LARGE moved object leaves a
        # silhouette-shaped fill that the repair "fixes" by painting the
        # object back, and the user sees it duplicated.
        if not move_ops:
            # stale delete flags must not leak into unrelated renders
            # (they force the pinned merge path)
            self._last_deletes = []
            self._composite_pin = False
        if move_ops and self.canvas is not None and not reset:
            # the geometry clauses ALWAYS travel with the ops: a pure
            # geometric edit gets the repair preamble, a mixed edit keeps
            # its spoken instruction and the clauses ride along (dropping
            # them lets the repaint resurrect deleted objects)
            clauses = "".join(self._move_clause(op) for op in move_ops)
            base_repair = REPAIR_PROMPT
            if self._last_deletes and all(op.get("delete")
                                          for op in move_ops):
                # absence must be stated POSITIVELY: at guidance 1.0
                # there is no negative prompt, and "fill in any gaps
                # naturally" invites the model to paint the furniture
                # back into its own hole
                base_repair = (
                    "Repaint the blurred empty areas as plain "
                    "background that seamlessly continues the "
                    "surrounding floor, walls, and surfaces — bare, "
                    "clean, open space. ")
            instruction = ((base_repair + clauses) if not instruction
                           else instruction + clauses)
        editing = (instruction and self.canvas is not None and not reset)
        # ROOTS roll fresh noise (saying the same prompt again gives a new
        # take) and that seed is INHERITED by every edit in the chain:
        # a chain sharing one noise trajectory repaints its own past far
        # more faithfully (measured: mixed seeds ~double the repaint
        # delta), which is what keeps drift slow and the pooled freeze
        # effective. New root = new seed = genuinely new roll.
        if promote_ctx is not None:
            # promotion of an accepted draft: the draft's seed IS the
            # recipe — every other seed rule stands down
            self.seed = promote_ctx["seed"]
        elif fresh_seed and FIXED_SEED is None:
            # post-verify retry: the previous roll just failed the judge
            self.seed = random.randrange(1 << 31)
        elif FIXED_SEED is not None:
            self.seed = FIXED_SEED
        elif not editing:
            self.session_seed = random.randrange(1 << 31)
            self.seed = self.session_seed
        else:
            self.seed = getattr(self, "session_seed", 42)
        base_lat = None
        if editing and provisional:
            # scratch lane: seed from the committed canvas whenever it has
            # advanced, so provisional churn re-bases instead of compounding
            if self.scratch is None or self.scratch_rev != self.rev:
                self.scratch = self.canvas
                self.scratch_lat = self.canvas_lat
                self.scratch_rev = self.rev
                # provisional churn gets its own anchor CLONE: it evolves
                # with the scratch chain but never mutates the committed one
                self.scratch_anchor = self._clone_anchor(self.anchor)
            base_image = self.scratch
            base_lat = self.scratch_lat
        elif editing:
            base_image = self.canvas
            base_lat = self.canvas_lat
        preserved_cuts = []
        if (editing and preserve_names and strength >= 0.75
                and base_image is not None and not move_ops):
            # SCENE RELOCATION: the world redraws, the subject's exact
            # pixels survive. Cut each preserved object now, paste it
            # back over the redrawn scene, feathered — a strong redraw
            # otherwise reinvents the character (identity loss), a weak
            # edit paints the new setting INTO the old room (overlay
            # artifacts). Both observed.
            try:
                from PIL import ImageFilter as _IF2
                for pname in preserve_names[:3]:
                    insts = self._mask_instances(
                        base_image, str(pname).replace("_", " "),
                        threshold=0.3)
                    if not insts:
                        continue
                    import numpy as _np4
                    mk = insts[0]
                    from PIL import Image as _I4
                    pm = _I4.fromarray(
                        (mk.astype("uint8") * 255)).convert("L")
                    pm = pm.filter(_IF2.MaxFilter(7)).filter(
                        _IF2.MinFilter(7))
                    pb = pm.getbbox()
                    if not pb:
                        continue
                    preserved_cuts.append(
                        (base_image.crop(pb),
                         pm.crop(pb).filter(_IF2.GaussianBlur(2)), pb))
                    log.info("preserve cut: %r bbox %s", pname, pb)
            except Exception:
                log.exception("preserve cut failed (plain redraw)")
        if editing and move_ops:
            self._pre_composite = base_image
            base_image = self.composite_moves(base_image, move_ops)
            self._regen_deletes = bool(move_ops) and all(
                op.get("regen") for op in move_ops if op.get("delete"))
            # ALL composite renders (inserts, moves, deletes) pin
            # outside their modified regions: the repair once repainted
            # 99% of the frame as a "global edit" and relocated a
            # perfectly solver-placed object onto the coffee table.
            # Regenerate-route renders stay free by contract.
            self._composite_pin = not self._regen_deletes
            if self._last_deletes and not self._regen_deletes:
                # REAL containment for deletes: the hard-pin stencil
                # forces every token OUTSIDE the holes bit-exactly back
                # to the conditioning. (cover=0 only ever pinned tokens
                # below tau_lo — ~4% of a delete render's frame.)
                # Regenerate-route deletes skip it: their contract is a
                # full redraw from the rewritten scene text.
                Flux2Canvas._hardpin_flag = True
            base_lat = None      # pixels changed outside the model: re-encode
            # deletes SAM could not segment are not OBJECTS — they are
            # surface features (wrinkles, sketch lines, shadows, glare).
            # The scissors path silently no-ops on them ("remove the
            # wrinkles" failed post-verify twice in a row), so reroute:
            # an imperative repaint clause + enough strength to actually
            # resurface, instead of the delete-pure 0.3-0.4 pin regime.
            uf = getattr(self, "_unfound_deletes", None)
            if uf:
                names = ", ".join(uf)
                clause = (f"Completely remove all {names} from the "
                          f"image, repainting those areas smooth, "
                          f"clean, and uniform.")
                instruction = (clause if not instruction
                               else instruction.rstrip(".") + ". "
                               + clause)
                if strength < 0.55:
                    strength = 0.55
                log.info("delete: no mask for [%s] — rerouted to "
                         "repaint (strength %.2f)", names, strength)
            # FRESH seed for composite renders: the session seed is the
            # exact noise that painted the deleted object, so reusing it
            # REGENERATES the object near pixel-identically (verified:
            # resurrections at 96-100% overlap, MAE 14). Room stability
            # on composite renders comes from the merge pinning tokens
            # to the conditioning, not from the seed.
            if FIXED_SEED is None:
                self.seed = random.randrange(1 << 31)
        if editing and camera and FIXED_SEED is None and promote_ctx is None:
            # a camera move must escape the session seed: the inherited
            # noise trajectory deterministically reproduces the previous
            # framing (measured on deletes at 96-100% overlap), so the
            # same seed fights the reframe the instruction asks for
            self.seed = random.randrange(1 << 31)
        region_px = None
        if region and editing and not move_ops:
            try:
                import numpy as _np
                rm = self._rle_to_mask(region["mask_rle"],
                                       region["mask_size"])
                if rm.size != (self.size, self.size):
                    from PIL import Image as _I
                    rm = rm.resize((self.size, self.size), _I.NEAREST)
                region_px = _np.asarray(rm) > 64
                # a scoped edit must be able to CHANGE its region: the
                # session seed deterministically regenerates old content
                if FIXED_SEED is None:
                    self.seed = random.randrange(1 << 31)
            except Exception:
                log.exception("region mask decode failed (unscoped edit)")
                region_px = None
        style_cond = None
        style_pils = []
        if editing:
            call_prompt = instruction
        else:
            call_prompt = ((prompt.strip() or EMPTY_SCENE_PROMPT)
                           + DEFAULT_PROMPT_SUFFIX)
            steps = STEPS               # roots always get the full pass
            # STYLE BANK (measured, round 4): user-uploaded style refs
            # attach as conditioning on roots, plus the VLM-distilled
            # suffix — the combination beats either alone on real refs
            # (6.8 vs 6.4 refs / 5.5 text). Never on edits: the canvas
            # already carries the style, and ref tokens would enter the
            # anchor/passthrough merge.
            if style_refs:
                try:
                    import base64 as _b64
                    import io as _io
                    from PIL import Image as _I
                    pils = []
                    for rb in style_refs[:2]:
                        im = _I.open(_io.BytesIO(_b64.b64decode(rb)))
                        pils.append(im.convert("RGB").resize(
                            (self.size, self.size), _I.LANCZOS))
                    style_cond = [self._encode_canvas(pipe, im)
                                  for im in pils]
                    style_pils = pils
                    if style_suffix:
                        call_prompt += ", " + style_suffix.strip(" ,.")
                    # "loosely inspired" measured best for content
                    # (8.8 vs 7.0 for "exact style", same fidelity);
                    # same-subject copying is caught by the copy guard
                    # below — no phrasing prevents it (0.93-0.98 sim
                    # under every wording tried)
                    call_prompt += (", loosely inspired by the visual "
                                    "style of the reference image"
                                    + ("s" if len(pils) > 1 else ""))
                    log.info("style conditioning: %d refs", len(pils))
                except Exception:
                    log.exception("style refs unusable — plain root")
                    style_cond = None

        image = None
        lat = None
        if promote_ctx is not None and promote_ctx.get("hardpin"):
            Flux2Canvas._hardpin_flag = True
        if LATENT_CHAIN and self._custom_ok:
            try:
                cond = None
                if style_cond is not None:
                    cond = style_cond
                if editing:
                    # latents chain only within one model; a switch (or a
                    # composite) re-anchors from pixels once
                    if ink and ink.get("strokes"):
                        # DRAWING MODE sketch: the model must SEE the
                        # user's strokes to paint over them — condition
                        # on the marked-up frame (this render only; the
                        # clean canvas remains truth and judge baseline)
                        marked = self._ink_composite(base_image,
                                                     ink["strokes"])
                        cond = self._encode_canvas(pipe, marked)
                        log.info("ink conditioning: %d strokes composited",
                                 len(ink["strokes"]))
                    elif (base_lat is not None and base_lat[2] == tag):
                        cond = (base_lat[0], base_lat[1])
                    else:
                        cond = self._encode_canvas(pipe, base_image)
                anc = None
                if ANCHOR and editing and not move_ops:
                    # a composite (move/delete) rebuilt the world in
                    # pixels — the old anchor still holds the REMOVED
                    # content, and merging against it snap-restores
                    # whatever the repaint makes look similar (a deleted
                    # couch respawned bit-exactly this way). Composite
                    # renders merge against their own conditioning and
                    # the anchor rebases from the result.
                    anc = (self.scratch_anchor if provisional
                           else self.anchor)
                    if anc is not None and anc["tag"] != tag:
                        anc = None      # anchors never cross models
                pk = {}
                p_free = None
                p_hard = False
                if promote_ctx is not None and "init_latents" in promote_ctx:
                    pk = {"init_latents": promote_ctx["init_latents"],
                          "noise_override": promote_ctx["noise"],
                          "sigma_node": promote_ctx["sigma_node"],
                          "edit_keep": promote_ctx.get("edit_keep")}
                    p_free = promote_ctx.get("free_px")
                    p_hard = bool(promote_ctx.get("hardpin"))
                image, lat = self._call_klein(
                    pipe, previewer, call_prompt, steps, cond=cond,
                    on_preview=(on_preview
                                if PREVIEW and self._preview_ok else None),
                    cancel_check=cancel_check,
                    # a big edit (LLM magnitude >= 0.7) intends global
                    # change — freezing "unchanged" tokens would pin parts
                    # of the old framing in place; a reframe moved every
                    # pixel, so nothing may be frozen either
                    allow_pass=((not editing or strength < 0.7)
                    or region_px is not None or p_free is not None)
                    and style_cond is None,
                    tau=self._tau_lo(tag), tau_hi=self._tau_hi_for(tag),
                    anchor=anc,
                    free_px=(p_free if p_free is not None
                             else region_px if region_px is not None
                             else (getattr(self, "_last_free", None)
                                   if move_ops else None)),
                    **pk)
                lat = (lat[0], lat[1], tag)
                # the promote lost too much of the approved draft to
                # splice (see edit-preserve): re-run ONCE resuming a node
                # later — less noise, more draft signal survives, so the
                # approved change carries into full res without splicing
                if (getattr(self, "_promote_lost_big", False)
                        and pk.get("sigma_node") is not None):
                    self._promote_lost_big = False
                    pk2 = dict(pk)
                    pk2["sigma_node"] = min(int(pk["sigma_node"]) + 1,
                                            steps - 1)
                    log.info("promote retry at node %d (draft signal "
                             "preserved)", pk2["sigma_node"])
                    image, lat = self._call_klein(
                        pipe, previewer, call_prompt, steps, cond=cond,
                        on_preview=(on_preview
                                    if PREVIEW and self._preview_ok
                                    else None),
                        cancel_check=cancel_check,
                        allow_pass=((not editing or strength < 0.7)
                        or region_px is not None or p_free is not None)
                        and style_cond is None,
                        tau=self._tau_lo(tag),
                        tau_hi=self._tau_hi_for(tag),
                        anchor=anc, free_px=p_free, **pk2)
                    lat = (lat[0], lat[1], tag)
                if (getattr(self, "_promote_lost_big", False)
                        and promote_ctx is not None
                        and promote_ctx.get("draft_image") is not None):
                    # both promote attempts lost the APPROVED change —
                    # the user already saw and approved the draft, so
                    # losing it silently is strictly worse than a
                    # softer upscale. Keep the draft itself. (Live
                    # failure: a verified wall-fill vanished on
                    # promote, resurrecting window fragments.)
                    from PIL import Image as _I
                    log.info("promote lost the approved draft twice — "
                             "keeping the upscaled draft")
                    image = promote_ctx["draft_image"].resize(
                        (self.size, self.size), _I.LANCZOS)
                    lat = None
                self._promote_lost_big = False
            except RenderPreempted:
                raise
            except Exception:
                self._custom_fails = getattr(self, "_custom_fails", 0) + 1
                if self._custom_fails >= 2:
                    log.exception(
                        "custom klein loop failed twice — stock pipeline "
                        "for the rest of this session (latent chaining + "
                        "passthrough OFF)")
                    self._custom_ok = False
                else:
                    log.exception(
                        "custom klein loop failed (1st) — stock fallback "
                        "for THIS render only, custom loop retries next")

        if image is not None and style_cond is not None and style_pils:
            # COPY GUARD: reference conditioning can reproduce a ref
            # near-verbatim when the spoken subject matches the ref's
            # subject (live repro + round 4: copies sit at 0.70-0.99
            # similarity, clean styled outputs max 0.56). A styled root
            # that lands too close to any ref is re-rolled once with
            # the distilled suffix only — styled, never a copy.
            try:
                worst = max(self._style_sim(image, sp)
                            for sp in style_pils)
                log.info("copy guard: styled root similarity %.2f", worst)
                if worst > 0.60:
                    log.info("copy guard: root/ref similarity %.2f — "
                             "re-rolling suffix-only", worst)
                    cg_prompt = ((prompt.strip() or EMPTY_SCENE_PROMPT)
                                 + DEFAULT_PROMPT_SUFFIX)
                    if style_suffix:
                        cg_prompt += ", " + style_suffix.strip(" ,.")
                    image, lat = self._call_klein(
                        pipe, previewer, cg_prompt, steps, cond=None,
                        on_preview=(on_preview
                                    if PREVIEW and self._preview_ok
                                    else None),
                        cancel_check=cancel_check, allow_pass=True,
                        tau=self._tau_lo(tag),
                        tau_hi=self._tau_hi_for(tag), anchor=None)
                    lat = (lat[0], lat[1], tag)
            except RenderPreempted:
                raise
            except Exception:
                log.exception("copy guard failed (frame kept)")
        if image is None:               # stock path: kill switch or fallback
            generator = torch.Generator("cuda").manual_seed(self.seed)
            final_latents = [None]

            def step_cb(pipe, step, timestep, cb_kwargs):
                if cancel_check is not None and step < steps - 1 and cancel_check():
                    raise RenderPreempted()
                slat = cb_kwargs.get("latents")
                if slat is not None:
                    final_latents[0] = slat
                    if on_preview is not None and step < steps - 1:
                        try:
                            jpeg = previewer.jpeg(slat)
                            if jpeg:
                                on_preview(jpeg, step + 1, steps)
                        except Exception:
                            log.exception("preview frame failed (continuing)")
                return {}

            kwargs = dict(
                prompt=call_prompt,
                image=([base_image] if editing else None),
                height=self.size, width=self.size,
                num_inference_steps=steps,
                guidance_scale=GUIDANCE,
                generator=generator,
            )
            if PREVIEW and self._preview_ok:
                try:
                    image = pipe(
                        **kwargs,
                        callback_on_step_end=step_cb,
                        callback_on_step_end_tensor_inputs=["latents"],
                    ).images[0]
                except TypeError:
                    log.warning("pipeline rejected step callbacks; previews off")
                    self._preview_ok = False
                    image = pipe(**kwargs).images[0]
            else:
                image = pipe(**kwargs).images[0]
            if final_latents[0] is not None:
                try:
                    previewer.fit(final_latents[0], image)
                except Exception:
                    log.exception("latent2rgb fit failed (continuing)")

        Flux2Canvas._hardpin_flag = False
        image = image.convert("RGB")
        # REGION SPLICE: a circled selection is a hard scope contract.
        # The tau-merge alone could not hold it — the merge classified a
        # region wall-edit as "global edit, model output stands" and let
        # the repaint rebuild half the window wall outside the circle
        # (live failure), and post-verify's restore+retry just failed the
        # same way twice. Pixels outside the (feathered) region now revert
        # to the pre-edit canvas deterministically; the model keeps full
        # context, the user keeps their circle.
        if preserved_cuts and image is not None:
            try:
                import numpy as _np5
                if not hasattr(self, "_last_free") or                         self._last_free is None:
                    self._last_free = _np5.zeros(
                        (image.size[1], image.size[0]), dtype=bool)
                for cut, pm, pb in preserved_cuts:
                    image.paste(cut, (pb[0], pb[1]), pm)
                    self._last_free[pb[1]:pb[3], pb[0]:pb[2]] = True
                lat = None      # pixels changed outside the model
                log.info("preserved %d subjects pasted back",
                         len(preserved_cuts))
            except Exception:
                log.exception("preserve paste failed (kept redraw)")
        if (editing and region_px is not None and base_image is not None):
            try:
                import numpy as _np
                from PIL import Image as _I
                from PIL import ImageFilter as _IF
                m = _I.fromarray((region_px * 255).astype("uint8"))
                if m.size != image.size:
                    m = m.resize(image.size, _I.NEAREST)
                m = m.filter(_IF.GaussianBlur(8))
                base = base_image
                if base.size != image.size:
                    base = base.resize(image.size, _I.LANCZOS)
                image = _I.composite(image.convert("RGB"),
                                     base.convert("RGB"), m)
                lat = None      # pixels changed outside the model: re-encode
                log.info("region splice: outside-circle pixels restored")
            except Exception:
                log.exception("region splice failed (frame kept)")
        # deleted means deleted: verify -> scrub -> REFINE loop. Scrubbing
        # alone leaves a background-fill smear where the resurrection was;
        # the refine pass repaints the scrubbed frame into clean scene
        # (conditioned on the scrubbed pixels, never merged against an
        # anchor), then the verify runs again. Two cycles max.
        dels = getattr(self, "_last_deletes", None)
        if editing and not provisional and dels:
            try:
                fixed, chg = self._enforce_deletes(
                    image, dels, ref=getattr(self, "_pre_composite", None))
                if chg:
                    # ONE free-masked refine: the room stays pinned, only
                    # the scrubbed regions repaint. No re-verify loop — a
                    # looser loop once scrubbed a clean room into mud.
                    # Fresh seed again: the previous roll just proved it
                    # regenerates the object.
                    if FIXED_SEED is None:
                        self.seed = random.randrange(1 << 31)
                    cond2 = self._encode_canvas(pipe, fixed)
                    image, lat2 = self._call_klein(
                        pipe, previewer, call_prompt, steps,
                        cond=cond2, allow_pass=True,
                        tau=self._tau_lo(tag),
                        tau_hi=self._tau_hi_for(tag),
                        free_px=getattr(self, "_scrub_free", None))
                    image = image.convert("RGB")
                    lat = (lat2[0], lat2[1], tag)
                    log.info("delete refine pass rendered")
            except Exception:
                log.exception("delete enforcement failed (frame kept)")
            self._last_deletes = []
            self._last_free = None
            self._composite_pin = False
            self._pre_composite = None
        if editing and provisional:
            self.scratch = image        # committed canvas untouched
            self.scratch_lat = lat      # None on the stock path: re-encode
        else:
            self.canvas = image
            self.canvas_lat = lat
            self.scratch = None
            self.scratch_lat = None
            self.scratch_anchor = None
            self.rev += 1
            if not editing:
                self.needs_floor_calib = True    # new scene, new floor
            self.edits_since_root = self.edits_since_root + 1 if editing else 0
            # unified anchor lifecycle: every committed render either ran
            # the anchored merge (anchor updated in place) or REBASES the
            # anchor wholesale from its output — covering roots, big edits
            # (allow_pass off), composites, model/resolution switches, and
            # restore re-encodes with one rule that can't go stale
            if ANCHOR:
                if lat is not None and not self._merge_anchored:
                    self.anchor = self._new_anchor(lat[0], lat[1], tag)
                elif lat is None:
                    self.anchor = None      # stock path: no latent chain
        log.info("%s%s%s: %r",
                 "scratch " if (editing and provisional) else "",
                 "edit" if editing else "root",
                 f" (+{self.edits_since_root})" if editing and not provisional else "",
                 (instruction if editing else call_prompt)[:70])

        if lat is not None:
            try:
                previewer.fit(lat[0], image)
            except Exception:
                log.exception("latent2rgb fit failed (continuing)")

        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=85)
        jpeg = buf.getvalue()
        if do_snapshot and not provisional and commit_rev is not None:
            self.snapshot(commit_rev, jpeg)
        return base64.b64encode(jpeg).decode()
