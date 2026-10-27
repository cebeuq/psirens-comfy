"""Mock canvas: synthesizes frames so the product loop can be felt before any
diffusion model exists. A prompt maps to a stable color field; each new render
target alpha-blends from the previous frame, scaled by strength — mimicking
img2img denoise-strength morphs."""
from __future__ import annotations

import base64
import colorsys
import hashlib
import io
import textwrap

from PIL import Image, ImageDraw

SIZE = 512


COLOR_RGB = {
    "red": (190, 55, 45), "crimson": (170, 25, 55), "green": (55, 160, 75),
    "emerald": (35, 150, 85), "blue": (55, 105, 195), "indigo": (70, 60, 140),
    "purple": (135, 65, 175), "violet": (150, 80, 190),
    "yellow": (215, 185, 55), "gold": (205, 165, 55), "amber": (220, 155, 55),
    "orange": (220, 135, 45), "pink": (225, 115, 165), "magenta": (200, 60, 160),
    "teal": (45, 160, 150), "cyan": (60, 180, 200), "turquoise": (55, 175, 170),
    "brown": (135, 95, 55), "lavender": (170, 150, 210),
    "black": (22, 22, 28), "white": (225, 225, 218), "gray": (125, 125, 125),
    "grey": (125, 125, 125),
}
_DARKEN = {"dark", "darkness", "blackness", "dim", "shadowy", "moody", "night"}
_BRIGHTEN = {"bright", "blazing", "vivid", "sunlit", "brilliant"}


def _prompt_color(prompt: str) -> tuple[int, int, int]:
    """Semantic when possible: color words in the prompt drive the canvas
    color (later mentions weigh more); hash fallback otherwise."""
    words = [w.strip(".,;:!?()").lower() for w in prompt.split()]
    hits = [(i, COLOR_RGB[w]) for i, w in enumerate(words) if w in COLOR_RGB]
    if hits:
        tw = sum(i + 1 for i, _ in hits)
        rgb = [sum((i + 1) * c[k] for i, c in hits) / tw for k in range(3)]
    else:
        h = hashlib.sha256(prompt.encode()).digest()
        hue, sat, val = h[0] / 255, 0.35 + (h[1] / 255) * 0.4, 0.35 + (h[2] / 255) * 0.35
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        rgb = [r * 255, g * 255, b * 255]
    wordset = set(words)
    if wordset & _DARKEN:
        rgb = [c * 0.45 for c in rgb]
    if wordset & _BRIGHTEN:
        rgb = [min(255, c * 1.35) for c in rgb]
    return tuple(int(c) for c in rgb)


def _target_image(prompt: str) -> Image.Image:
    base = _prompt_color(prompt)
    im = Image.new("RGB", (SIZE, SIZE))
    top = tuple(min(255, int(c * 1.5)) for c in base)
    for y in range(SIZE):
        t = y / SIZE
        row = tuple(int(top[i] * (1 - t) + base[i] * t * 0.6) for i in range(3))
        im.paste(Image.new("RGB", (SIZE, 1), row), (0, y))
    draw = ImageDraw.Draw(im)
    wrapped = textwrap.fill(prompt or "empty scene", width=44)
    draw.multiline_text((18, SIZE - 20 - 14 * (wrapped.count("\n") + 1)),
                        wrapped, fill=(255, 255, 255))
    return im


def _lerp_box(a: dict, b: dict, t: float) -> dict:
    return {k: a[k] + (b[k] - a[k]) * t for k in ("x", "y", "w", "h")}


def _draw_entities(im: Image.Image, entities: list[dict],
                   alpha: float = 1.0) -> Image.Image:
    """Draw labeled boxes for entities. Boxes use center x/y + w/h, 0..1."""
    if not entities:
        return im
    overlay = im.convert("RGBA")
    draw = ImageDraw.Draw(overlay, "RGBA")
    a = int(220 * alpha)
    W, H = im.size
    for e in entities:
        b = e["box"]
        x0 = (b["x"] - b["w"] / 2) * W
        y0 = (b["y"] - b["h"] / 2) * H
        x1 = (b["x"] + b["w"] / 2) * W
        y1 = (b["y"] + b["h"] / 2) * H
        fill = _prompt_color(e.get("description") or e["id"]) + (int(60 * alpha),)
        draw.rounded_rectangle([x0, y0, x1, y1], radius=10,
                               outline=(255, 255, 255, a), width=2, fill=fill)
        draw.text((x0 + 6, y0 + 4), e.get("name") or e["id"],
                  fill=(255, 255, 255, a))
    return overlay.convert("RGB")


class MockCanvas:
    def __init__(self) -> None:
        self.current = Image.new("RGB", (SIZE, SIZE), (16, 20, 24))
        self.entities: dict[str, dict] = {}

    def frames_for_target(self, prompt: str, strength: float,
                          entities: list[dict] | None = None,
                          n_steps: int = 4) -> list[str]:
        """Morph from the current frame toward the target; entity boxes SLIDE
        between their old and new positions (matched by id; new ones fade in).
        Returns jpeg base64 strings. Background movement is scaled by strength
        — a draft (~0.2) drifts, a commit (0.7+) decisively transforms."""
        target = _target_image(prompt)
        new_entities = {e["id"]: e for e in (entities or [])}
        old_entities = self.entities

        frames = []
        start = self.current
        for i in range(1, n_steps + 1):
            t = i / n_steps
            frame = Image.blend(start, target, min(1.0, strength * t))
            drawn = []
            faded = []
            for eid, e in new_entities.items():
                if eid in old_entities:
                    box = _lerp_box(old_entities[eid]["box"], e["box"], t)
                    drawn.append({**e, "box": box})
                else:
                    faded.append(e)  # new: fade in
            frame = _draw_entities(frame, drawn, alpha=1.0)
            frame = _draw_entities(frame, faded, alpha=t)
            # removed entities: fade out
            gone = [old_entities[i2] for i2 in old_entities if i2 not in new_entities]
            frame = _draw_entities(frame, gone, alpha=1.0 - t)
            frames.append(frame)

        self.current = Image.blend(start, target, min(1.0, strength))
        self.entities = new_entities

        out = []
        for frame in frames:
            buf = io.BytesIO()
            frame.save(buf, format="JPEG", quality=80)
            out.append(base64.b64encode(buf.getvalue()).decode())
        return out
