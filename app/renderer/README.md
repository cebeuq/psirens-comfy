# Continuity Renderer

Consumes the voice service's event feed and paints the live canvas
("morph, not teleport" — SPEC.MD §2.4).

## Contract

- Connects to the voice service at `ws://orchestrator:8000/ws/feed`.
- Listens for `render_target` events:
  `{prompt: str, strength: 0..1, source: "draft"|"commit", generation: int}`
  — `strength` is how hard to move the image (drafts are gentle nudges,
  commits are decisive morphs).
- Sends back `{"type": "canvas_frame", "data": {jpeg_b64, seq, source,
  strength}}` messages; the voice service republishes them to the browser
  (they are never written to session recordings).
- During a diffusion render it also streams step previews — the same
  `canvas_frame` shape plus `{preview: true, step, steps}` — so the browser
  watches the image "develop" instead of waiting seconds for a finished
  frame (ComfyUI-style latent2rgb).

## Current implementation: Z-Image Turbo (+ mock fallback)

`RENDERER_MODE=zimage` (default): real diffusion via Tongyi-MAI/Z-Image-Turbo
(diffusers ZImagePipeline, bf16, 512px, 6 steps, guidance 1.0). Continuity
v1 = FIXED SEED per session + the evolving structured prompt. Entity boxes
can be drawn as a debug overlay (`ENTITY_OVERLAY=1`; default off — the
boxes obscure the image and diffusion can't obey them until ControlNet
layout conditioning lands, so they only show where the entity layer THINKS
objects are).
Renders are LATEST-TARGET-WINS: new render_targets replace the pending one,
so the canvas always chases the most recent intent with no backlog.

Measured on spark1 (GB10): ~5.5–6.2s per 512px frame solo, 13–15s while the
editor LLMs are simultaneously busy. Tune via ZIMAGE_SIZE / ZIMAGE_STEPS in
.env. `RENDERER_MODE=mock` restores the instant color-field mock.

Step previews (`ZIMAGE_PREVIEW=1` default, `ZIMAGE_PREVIEW_SIZE=288`):
after each denoising step the current latent is projected to RGB with a
linear map (~ms, no VAE decode) and pushed as a preview frame. The
projection matrix is NOT hardcoded per model — it is least-squares-fitted
from (final latent, decoded image) pairs, accumulated across every render
(normal equations, ridge-stabilized). This makes previews model-agnostic:
swap the checkpoint and the fit re-learns the latent space automatically,
including packed/token latent layouts (previews then appear at patch
resolution). First fit happens on the warmup render.

Next renderer steps: ControlNet Union box conditioning (real spatial
obedience), img2img morphs (denoise = strength), periodic clean re-anchor
(spec §2.4).
