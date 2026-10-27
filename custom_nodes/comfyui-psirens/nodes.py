"""Psirens nodes: drift-free iterative editing + voice prompts.

Long edit chains on an edit model (FLUX 3 Image Klein, FLUX.2 Klein, ...)
age the picture in two ways:

  1. every edit round-trips the canvas through pixels (decode -> encode),
  2. the model regenerates EVERY token each edit, so regions the
     instruction never mentioned get subtly repainted, again and again.

Canvas Save/Load fix (1) by keeping the output latent inside ComfyUI and
referencing it directly next time. Token Passthrough fixes (2): after
sampling, each latent token whose change versus the reference sits at the
model's repaint-noise floor is copied back bit-exactly from the reference.
What the edit didn't touch stops aging, however long the session runs.
"""
from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from . import canvas_store

log = logging.getLogger("psirens")
CATEGORY = "psirens"


class PsirensCanvasSave:
    """Store a latent as a canvas slot (passes it through unchanged)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "samples": ("LATENT",),
            "session": ("STRING", {"default": "default"}),
            "slot": ("STRING", {"default": "c0",
                                "tooltip": "c<rev> for committed revisions, "
                                           "'scratch' for provisional edits"}),
        }, "optional": {
            "keyframe": ("INT", {"default": -1, "min": -1, "max": 1 << 30,
                                 "tooltip": "also keep this canvas as undo "
                                            "keyframe <slot>@<keyframe> "
                                            "(set by the Talk button; -1 = no)"}),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "save"
    CATEGORY = CATEGORY

    def save(self, samples, session, slot, keyframe=-1):
        canvas_store.put(session, slot, samples)
        if keyframe is not None and keyframe >= 0:
            canvas_store.put(session, f"{slot}@{keyframe}", samples)
        return (samples,)


class PsirensCanvasLoad:
    """Fetch a stored canvas latent (fails loudly if the slot is gone, so
    the caller can fall back to re-encoding pixels)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "session": ("STRING", {"default": "default"}),
            "slot": ("STRING", {"default": "c0"}),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "load"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, session, slot):
        return float("nan")      # the store changes between runs: never cache

    def load(self, session, slot):
        lat = canvas_store.get(session, slot)
        if lat is None:
            raise ValueError(f"psirens canvas slot {session}/{slot} not found")
        return (lat,)


class PsirensEmptyLike:
    """Zero latent with the shape of another one (the sampling target for
    an edit referenced by a stored latent)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"samples": ("LATENT",)}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "make"
    CATEGORY = CATEGORY

    def make(self, samples):
        return ({"samples": torch.zeros_like(samples["samples"])},)


class PsirensTokenPassthrough:
    """Copy untouched tokens back from the reference, bit-exact.

    Per latent position (= one transformer token for FLUX 2/3 latents) the
    RMS change across channels is measured in units of the reference's
    per-channel std, smoothed over a 3x3 neighborhood (repaint noise is
    scattered, real edits are contiguous). Positions below `tau` keep the
    reference value. If more than `max_cover` of the frame moved, the edit
    is global (relight, restyle) and is passed through untouched.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "samples": ("LATENT",),
            "reference": ("LATENT",),
            "tau": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 5.0,
                              "step": 0.01,
                              "tooltip": "noise floor; 0 disables"}),
            "max_cover": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0,
                                    "step": 0.01}),
        }}

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("samples", "stats")
    FUNCTION = "merge"
    CATEGORY = CATEGORY

    def merge(self, samples, reference, tau, max_cover):
        x = samples["samples"]
        r = reference["samples"].to(device=x.device, dtype=x.dtype)
        if tau <= 0 or x.shape != r.shape or x.ndim != 4:
            return (samples, "passthrough off (shape or tau)")
        if not torch.any(r):
            return (samples, "no reference yet (first image)")
        std = r.float().std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6)
        d = ((x.float() - r.float()) / std).pow(2).mean(dim=1, keepdim=True)
        d = F.avg_pool2d(d, 3, stride=1, padding=1,
                         count_include_pad=False).sqrt()
        moved = (d >= tau).float().mean().item()
        q = torch.quantile(d.flatten()[: 1 << 20],
                           torch.tensor([0.5, 0.9, 0.99], device=d.device))
        stats = (f"moved={moved:.3f} p50={q[0]:.3f} p90={q[1]:.3f} "
                 f"p99={q[2]:.3f} tau={tau:.2f}")
        if moved > max_cover:
            log.info("psirens passthrough: global edit, kept all (%s)", stats)
            return (samples, stats + " global")
        keep = d < tau
        out = dict(samples)
        out["samples"] = torch.where(keep, r, x)
        log.info("psirens passthrough: %s", stats)
        return (out, stats)


class PsirensVoicePrompt:
    """A prompt you can speak. The Psirens mic button fills `scene` (the
    full scene description) and `instruction` (the latest spoken edit) and
    queues the workflow. Output: the instruction once a canvas exists in
    `session`/`slot`, else the scene (so the first utterance creates the
    picture and every later one edits it). Typing works too."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "scene": ("STRING", {"multiline": True, "default": ""}),
            "instruction": ("STRING", {"multiline": True, "default": ""}),
            "session": ("STRING", {"default": "ui"}),
            "slot": ("STRING", {"default": "latest"}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "emit"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, **kw):
        return float("nan")

    def emit(self, scene, instruction, session, slot):
        has_canvas = canvas_store.get(session, slot) is not None
        text = instruction if (has_canvas and instruction.strip()) else scene
        return (text.strip() or instruction.strip(),)


class PsirensCanvasReference:
    """Point an edit model at the current canvas, if there is one.

    With a stored canvas: adds it as the reference latent to both
    conditionings and returns a matching empty latent to sample into.
    Without one (first run, new session): passes the conditionings through
    and returns an empty latent of width x height, so the same graph does
    the first generation AND every edit after it."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "session": ("STRING", {"default": "ui"}),
            "slot": ("STRING", {"default": "latest"}),
            "width": ("INT", {"default": 1024, "min": 256, "max": 4096,
                              "step": 16}),
            "height": ("INT", {"default": 1024, "min": 256, "max": 4096,
                               "step": 16}),
        }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent", "reference")
    FUNCTION = "apply"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, **kw):
        return float("nan")

    def apply(self, positive, negative, session, slot, width, height):
        import node_helpers
        ref = canvas_store.get(session, slot)
        if ref is None:
            empty = {"samples": torch.zeros(
                [1, 128, height // 16, width // 16])}
            return (positive, negative, empty, empty)
        lat = ref["samples"]
        positive = node_helpers.conditioning_set_values(
            positive, {"reference_latents": [lat]}, append=True)
        negative = node_helpers.conditioning_set_values(
            negative, {"reference_latents": [lat]}, append=True)
        return (positive, negative, {"samples": torch.zeros_like(lat)}, ref)


class PsirensCanvasReset:
    """Forget a session's canvas (start a new picture)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"session": ("STRING", {"default": "ui"})}}

    RETURN_TYPES = ()
    FUNCTION = "reset"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    def reset(self, session):
        canvas_store.drop(session)
        return {}


NODE_CLASS_MAPPINGS = {
    "PsirensCanvasSave": PsirensCanvasSave,
    "PsirensCanvasLoad": PsirensCanvasLoad,
    "PsirensEmptyLike": PsirensEmptyLike,
    "PsirensTokenPassthrough": PsirensTokenPassthrough,
    "PsirensVoicePrompt": PsirensVoicePrompt,
    "PsirensCanvasReference": PsirensCanvasReference,
    "PsirensCanvasReset": PsirensCanvasReset,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PsirensCanvasSave": "Psirens Canvas Save",
    "PsirensCanvasLoad": "Psirens Canvas Load",
    "PsirensEmptyLike": "Psirens Empty Latent Like",
    "PsirensTokenPassthrough": "Psirens Token Passthrough",
    "PsirensVoicePrompt": "Psirens Voice Prompt",
    "PsirensCanvasReference": "Psirens Canvas Reference",
    "PsirensCanvasReset": "Psirens Canvas Reset",
}
