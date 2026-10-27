"""Model-agnostic latent->RGB step previews (the ComfyUI latent2rgb idea).

The projection matrix is least-squares-fitted from (final latent, decoded
image) pairs accumulated across renders — no per-model coefficient tables,
so it works for any checkpoint including packed/token latent layouts.
"""
from __future__ import annotations

import base64
import io
import logging
import math

log = logging.getLogger("renderer.preview")


class RenderPreempted(Exception):
    """A newer target arrived; this render was aborted between steps."""


class LatentPreviewer:
    def __init__(self, preview_size: int = 288):
        self.preview_size = preview_size
        self._xtx = None          # (C+1, C+1) float64 normal equations
        self._xty = None          # (C+1, 3)   float64
        self._proj = None         # (C+1, 3)   float32 on GPU

    @staticmethod
    def _latent_grid(latents):
        """Normalize latents to (H, W, C) float32, whatever the layout."""
        t = latents[0].detach().float()
        if t.dim() == 3:                      # (C, H, W)
            return t.permute(1, 2, 0)
        if t.dim() == 2:                      # (L, C) token sequence
            n, c = t.shape
            side = int(math.isqrt(n))
            if side * side != n:
                side = max(1, side)
                t = t[: side * side]
            return t.reshape(side, side, c)
        raise ValueError(f"unexpected latent shape {tuple(latents.shape)}")

    def fit(self, latents, image) -> None:
        """Accumulate (latents, decoded RGB) into the running lstsq fit."""
        import torch

        grid = self._latent_grid(latents)               # (H, W, C)
        h, w, c = grid.shape
        small = image.convert("RGB").resize((w, h))
        rgb = torch.frombuffer(bytearray(small.tobytes()), dtype=torch.uint8)
        rgb = rgb.reshape(h * w, 3).to(grid.device).float()
        x = torch.cat([grid.reshape(h * w, c),
                       torch.ones(h * w, 1, device=grid.device)], dim=1)
        x64, y64 = x.double(), rgb.double()
        if self._xtx is None or self._xtx.shape[0] != c + 1:
            self._xtx = torch.zeros(c + 1, c + 1, dtype=torch.float64,
                                    device=grid.device)
            self._xty = torch.zeros(c + 1, 3, dtype=torch.float64,
                                    device=grid.device)
        self._xtx += x64.T @ x64
        self._xty += x64.T @ y64
        eye = torch.eye(c + 1, dtype=torch.float64, device=grid.device)
        self._proj = torch.linalg.solve(
            self._xtx + 1e-3 * eye, self._xty).float()

    def jpeg(self, latents) -> str | None:
        """Cheap latent->RGB projection frame (~ms). None until fitted."""
        import torch
        from PIL import Image

        if self._proj is None:
            return None
        grid = self._latent_grid(latents)
        h, w, c = grid.shape
        if self._proj.shape[0] != c + 1:
            return None
        x = torch.cat([grid.reshape(h * w, c),
                       torch.ones(h * w, 1, device=grid.device)], dim=1)
        rgb = (x @ self._proj).clamp(0, 255).to(torch.uint8).cpu()
        img = Image.frombuffer("RGB", (w, h), rgb.numpy().tobytes())
        img = img.resize((self.preview_size, self.preview_size),
                         Image.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=70)
        return base64.b64encode(buf.getvalue()).decode()
