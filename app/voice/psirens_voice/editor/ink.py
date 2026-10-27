"""Ink decider: reads a marked-up canvas and classifies the drawing.

The user draws freehand strokes over the live canvas. A stroke is either
an INSTRUCTION written as a gesture (an X over a thing = delete it, a
circle = select it) or CONTENT (a rough sketch of the thing they want
painted). The decision determines whether the image model ever sees the
ink: gestures discard it and route through proven pipelines; sketches
keep it as conditioning so the model paints over the drawing.

The VLM decides, but with deterministic evidence attached (stroke count,
closure, coverage) so it classifies from measurements, not vibes.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging

from ..llm_http import stream_chat_blocking

log = logging.getLogger("psirens.ink")

_SCHEMA = {
    "type": "object",
    "properties": {
        # what the model SEES under/around the ink, written before the
        # decision (ground first, then decide — house rule)
        # pattern style proven against xgrammar (a negated [^\n] class
        # is rejected by its EBNF lexer and 500s the request)
        "observation": {"type": "string", "maxLength": 200,
                        "pattern":
                            "^[A-Za-z][A-Za-z0-9 ,.;:'()!?/-]{7,198}$"},
        "reading": {"type": "string",
                    "enum": ["erase", "select", "sketch", "move"]},
        # the object the ink refers to (erase/select) or depicts (sketch)
        "target": {"type": "string", "maxLength": 50},
        # imperative for the downstream pipeline: erase -> a removal
        # sentence; sketch -> what to paint in place of the strokes
        "instruction": {"type": "string", "maxLength": 220},
    },
    "required": ["observation", "reading", "target", "instruction"],
    "additionalProperties": False,
}

_RUBRIC = (
    "You read a freehand drawing a user made ON TOP of an image (shown "
    "with the strokes composited in bright red). Decide what the "
    "drawing IS:\n"
    "- erase: the strokes CROSS OUT an existing object (an X, a "
    "scribble, strike-through over something). instruction = one "
    "sentence ordering that object's removal, naming it plainly.\n"
    "- select: the strokes ENCLOSE an existing object or area without "
    "depicting anything (a circle/loop around a thing). target = the "
    "most salient OBJECT(S) inside the loop — the things standing "
    "there, never the floor/wall/surface they rest on and never "
    "scenery visible through a window. instruction = one short "
    "sentence naming what was selected.\n"
    "- sketch: the strokes DEPICT something new (a shape, outline, or "
    "doodle of an object that is NOT in the image). instruction = one "
    "imperative sentence telling an image editor to replace the rough "
    "red sketch with a realistic version of the depicted object, "
    "matching the scene's style, lighting, and scale.\n"
    "- move: the strokes form an ARROW (a line with an arrowhead, "
    "possibly starting at or near an object) indicating that object "
    "should move in the arrow's direction. target = the object at the "
    "arrow's TAIL (where it starts). instruction = one sentence naming "
    "the object and the direction.\n"
    "A closed loop around empty space or around a region with several "
    "objects is select. A closed loop that clearly draws a NEW shape "
    "(a lamp silhouette, a window) is sketch — judge by whether the "
    "shape depicts something. A straight-ish line WITH an arrowhead is "
    "move — never erase, never sketch. erase requires BOTH "
    "an unmistakable cross-out (X or dense scribble) AND a clearly "
    "identifiable object underneath it — when either is uncertain, "
    "choose sketch: a wrong sketch is easily redone, a wrong erase "
    "destroys something the user wanted kept. Write observation "
    "first: what the strokes cover and what they look like. Name "
    "targets by what they are ('the flat-screen TV'), never by "
    "coordinates."
)


def _hints(strokes: list[list[list[float]]]) -> dict:
    """Deterministic stroke measurements the model gets as text."""
    n = len(strokes)
    xs = [p[0] for s in strokes for p in s]
    ys = [p[1] for s in strokes for p in s]
    if not xs:
        return {"n_strokes": 0}
    bbox = [min(xs), min(ys), max(xs), max(ys)]
    closures = []
    for s in strokes:
        if len(s) < 8:
            closures.append(False)
            continue
        dx = s[0][0] - s[-1][0]
        dy = s[0][1] - s[-1][1]
        diag = max(1e-6, ((bbox[2] - bbox[0]) ** 2
                          + (bbox[3] - bbox[1]) ** 2) ** 0.5)
        closures.append((dx * dx + dy * dy) ** 0.5 < 0.2 * diag)
    lines = []
    for s_ in strokes[:4]:
        if len(s_) < 2:
            continue
        span = ((s_[-1][0] - s_[0][0]) ** 2
                + (s_[-1][1] - s_[0][1]) ** 2) ** 0.5
        path = sum(((s_[i + 1][0] - s_[i][0]) ** 2
                    + (s_[i + 1][1] - s_[i][1]) ** 2) ** 0.5
                   for i in range(len(s_) - 1))
        lines.append({
            "start": [round(s_[0][0], 3), round(s_[0][1], 3)],
            "end": [round(s_[-1][0], 3), round(s_[-1][1], 3)],
            "straightness": round(span / max(path, 1e-6), 2),
        })
    return {
        "n_strokes": n,
        "bbox": [round(v, 3) for v in bbox],
        "closed_loops": sum(closures),
        "coverage_pct": round((bbox[2] - bbox[0])
                              * (bbox[3] - bbox[1]) * 100, 1),
        "strokes": lines,
    }


# Vision requests carry image prefill that is far heavier than text:
# four simultaneous reads (one multi-drop) crashed the vLLM engine with
# a CUDA launch failure and took the whole pipeline down. All image-
# bearing calls go through this gate — at most two in flight.
_VISION_GATE = asyncio.Semaphore(2)


def composite_ink(jpeg_b64: str, strokes: list) -> str:
    """Draw the strokes in marker red onto the frame; returns b64 JPEG."""
    from PIL import Image, ImageDraw

    img = Image.open(io.BytesIO(base64.b64decode(jpeg_b64))).convert("RGB")
    w, h = img.size
    d = ImageDraw.Draw(img)
    lw = max(3, int(min(w, h) * 0.008))
    for s in strokes:
        pts = [(p[0] * w, p[1] * h) for p in s]
        if len(pts) >= 2:
            d.line(pts, fill=(255, 32, 32), width=lw, joint="curve")
        for p in pts[:1] + pts[-1:]:
            d.ellipse([p[0] - lw / 2, p[1] - lw / 2,
                       p[0] + lw / 2, p[1] + lw / 2], fill=(255, 32, 32))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


async def read_ink(marked_b64: str, strokes: list, base_url: str,
                   model: str, timeout_s: float = 6.0) -> dict | None:
    """One guided-JSON VLM call. None on any failure (caller falls back
    to the sketch route — the only route that needs no interpretation)."""
    hints = _hints(strokes)
    body = {
        "model": model,
        "temperature": 0.1,
        "max_tokens": 220,
        "messages": [
            {"role": "system", "content": _RUBRIC},
            {"role": "user", "content": [
                {"type": "text", "text":
                    "Stroke measurements: " + json.dumps(hints)},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{marked_b64}"}},
            ]},
        ],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "ink_reading", "schema": _SCHEMA}},
    }
    try:
        loop = asyncio.get_running_loop()
        async with _VISION_GATE:
            content = await loop.run_in_executor(
                None, stream_chat_blocking, base_url, body, timeout_s)
        out = json.loads(content)
        log.info("ink read: %s target=%r (%s)", out.get("reading"),
                 out.get("target"), out.get("observation", "")[:80])
        return out
    except Exception as e:
        log.warning("ink decider failed (%s) — sketch fallback", e)
        return None


# ---------------------------------------------------------------------
# multi-gesture ("put-that-there") support: several gestures drawn
# during one utterance, identified in ONE call via numbered labels.

_GROUPS_SCHEMA = {
    "type": "object",
    "properties": {
        "groups": {
            "type": "array", "maxItems": 6,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer", "minimum": 1,
                           "maximum": 6},
                    "kind": {"type": "string",
                             "enum": ["object_marked", "empty_area",
                                      "sketch"]},
                    # object_marked -> what the strokes cover;
                    # empty_area  -> where it is ("the wall left of
                    # the window"); sketch -> what it depicts
                    "content": {"type": "string", "maxLength": 90},
                },
                "required": ["id", "kind", "content"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["groups"],
    "additionalProperties": False,
}

_GROUPS_RUBRIC = (
    "The image contains several numbered freehand drawings (red "
    "strokes, each group labeled with a white number in a red box). "
    "For EACH numbered group report: kind=object_marked when the "
    "strokes mark/cross/enclose an existing object (content = that "
    "object, named plainly); kind=empty_area when they mark a spot "
    "with nothing notable in it (content = where that spot is, "
    "relative to the room); kind=sketch when the strokes DEPICT a new "
    "object (content = what they depict). Name the FOREGROUND objects "
    "the strokes directly touch — furniture and items in the room, "
    "never the floor/wall they rest on, and never scenery visible "
    "through a window (a couch in front of a skyline is 'the couch', "
    "not 'building'). Report every group exactly once."
)


def composite_ink_labeled(jpeg_b64: str, groups: list[list]) -> str:
    """Strokes drawn per group with a numbered tag at each group's
    top-left corner, so one VLM call can identify all of them."""
    from PIL import Image, ImageDraw

    img = Image.open(io.BytesIO(base64.b64decode(jpeg_b64))).convert("RGB")
    w, h = img.size
    d = ImageDraw.Draw(img)
    lw = max(3, int(min(w, h) * 0.008))
    for gi, strokes in enumerate(groups, start=1):
        xs = [p[0] for s in strokes for p in s]
        ys = [p[1] for s in strokes for p in s]
        for s in strokes:
            pts = [(p[0] * w, p[1] * h) for p in s]
            if len(pts) >= 2:
                d.line(pts, fill=(255, 32, 32), width=lw, joint="curve")
        if xs:
            tx = max(2, min(xs) * w - 14)
            ty = max(2, min(ys) * h - 30)
            d.rectangle([tx, ty, tx + 26, ty + 26], fill=(255, 32, 32))
            d.text((tx + 8, ty + 3), str(gi), fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


async def read_ink_groups(marked_b64: str, n_groups: int, base_url: str,
                          model: str, timeout_s: float = 8.0,
                          group_boxes: list | None = None
                          ) -> list[dict] | None:
    """Identify every labeled gesture group in one call. None on any
    failure (caller degrades to unannotated parsing)."""
    body = {
        "model": model,
        "temperature": 0.1,
        "max_tokens": 300,
        "messages": [
            {"role": "system", "content": _GROUPS_RUBRIC},
            {"role": "user", "content": [
                {"type": "text",
                 "text": (f"There are exactly {n_groups} numbered "
                          f"groups. Group boxes (percent of image): "
                          + json.dumps(group_boxes or []))},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{marked_b64}"}},
            ]},
        ],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "ink_groups", "schema": _GROUPS_SCHEMA}},
    }
    try:
        loop = asyncio.get_running_loop()
        async with _VISION_GATE:
            content = await loop.run_in_executor(
                None, stream_chat_blocking, base_url, body, timeout_s)
        out = json.loads(content).get("groups") or []
        log.info("ink groups: %s",
                 [(g.get("id"), g.get("kind"),
                   str(g.get("content"))[:40]) for g in out])
        return out
    except Exception as e:
        log.warning("ink group reader failed (%s)", e)
        return None


# ---------------------------------------------------------------------
# upload decider-lite: name the thing in a dropped image so the
# insertion op can SAM-cut it and the entity can be tracked.

_UPLOAD_SCHEMA = {
    "type": "object",
    "properties": {
        "object": {"type": "string", "maxLength": 50,
                   "pattern": "^[A-Za-z][A-Za-z0-9 '-]{1,48}$"},
        "kind": {"type": "string",
                 "enum": ["isolated_object", "scene_with_object",
                          "full_scene", "swatch_or_graphic"]},
        "role": {"type": "string", "enum": ["insert", "style"]},
        "domain": {"type": "string", "maxLength": 40,
                   "pattern": "^[a-z][a-z0-9 /-]{2,38}$"},
        "style_desc": {"type": "string", "maxLength": 120,
                       "pattern": "^[A-Za-z][A-Za-z0-9 ,;:'()/-]{5,118}$"},
    },
    "required": ["object", "kind", "role", "domain", "style_desc"],
    "additionalProperties": False,
}

_UPLOAD_RUBRIC = (
    "The user dropped this image onto their canvas. Decide its role: "
    "'insert' = a photo of a thing they want placed IN the picture "
    "(furniture, an object, a product shot, a pet). 'style' = an "
    "example of a LOOK they want the picture to have (an artwork, "
    "illustration, painting, poster, stylized or art-directed photo, "
    "color/pattern swatch, portfolio piece). Also name the single most "
    "prominent foreground object as a short noun phrase ('pink "
    "flamingo pool floaty', 'leather armchair'). kind: isolated_object "
    "(one object, plain/removed background), scene_with_object (a "
    "photo where one object clearly dominates), full_scene (a whole "
    "environment with no single subject), swatch_or_graphic (a "
    "color/pattern/logo/graphic). domain: the creative field this "
    "image belongs to, lowercase ('interior design', 'fashion "
    "illustration', 'product photography', 'travel poster'). "
    "style_desc: one compact phrase naming its visual style."
)

# ---------------------------------------------------------------------
# reference director: the LLM chooses HOW to use the tray, per
# generation, from measured knowledge of the image model.

_DIRECTOR_SCHEMA = {
    "type": "object",
    "properties": {
        "use": {
            "type": "array", "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "ref": {"type": "integer", "minimum": 1,
                            "maximum": 8},
                    "as": {"type": "string",
                           "enum": ["style", "palette", "object",
                                    "ignore"]},
                    "sticky": {"type": "boolean"},
                },
                "required": ["ref", "as", "sticky"],
                "additionalProperties": False,
            },
        },
        "suffix": {"type": "string", "maxLength": 220,
                   "pattern":
                   "^[A-Za-z][A-Za-z0-9 ,;:'()/-]{0,218}$"},
        "note": {"type": "string", "maxLength": 90,
                 "pattern": "^[A-Za-z][A-Za-z0-9 ,;:'()/-]{0,88}$"},
    },
    "required": ["use", "suffix", "note"],
    "additionalProperties": False,
}

_DIRECTOR_RUBRIC = (
    "You direct how uploaded reference images feed an image generator "
    "(FLUX.2 Klein). You know its MEASURED behavior:\n"
    "1. Attaching a ref as image conditioning transfers its style well "
    "ONLY when the ref and the requested image share a domain "
    "(interior ref for an interior request). Cross-medium transfer "
    "(oil painting ref for a photo request) fails through "
    "conditioning — carry it through the text suffix instead.\n"
    "2. Conditioning on refs AND a distilled text suffix together "
    "beats either alone. Always write the suffix from the refs you "
    "chose as 'style'; describe shared style only, never objects.\n"
    "3. Two different-subject refs beat one. HARD RULE: if a ref "
    "PICTURES the same kind of thing the user is asking for (a "
    "bedroom ref for a bedroom request, a poster ref for a poster "
    "request), that ref must be 'palette', NEVER 'style' — "
    "conditioning on it reproduces it. Carry its look through the "
    "suffix instead.\n"
    "4. Lettering, titles, and signatures leak from refs; if a chosen "
    "ref contains prominent text, append 'no text or lettering "
    "copied from references' to the suffix.\n"
    "5. 'palette' refs contribute only their colors: list their "
    "DOMINANT COLORS explicitly in the suffix (e.g. 'palette of "
    "burnt orange, brown, and cream'), do not condition on them.\n"
    "6. 'object' refs are photos of things to insert; never condition "
    "on them for style.\n"
    "7. Phrases like 'that new image', 'this one', or an unqualified "
    "'that image' spoken right after an upload refer to the NEWEST "
    "reference — bind them to it.\n"
    "Pick per-reference use for THIS request. The user's spoken "
    "instructions about references always win; mark those choices "
    "sticky=true. Refs whose style or domain would fight the request "
    "get 'ignore'. Each listed ref number must appear exactly once."
)


async def direct_references(tray: list[dict], utterance: str,
                            scene_hint: str, base_url: str, model: str,
                            timeout_s: float = 8.0) -> dict | None:
    """Per-generation reference plan chosen by the LLM from the tray
    and its playbook of the generator's measured behavior."""
    content = []
    n = len(tray[:8])
    for i, entry in enumerate(tray[:8], start=1):
        newest = " — NEWEST, just uploaded" if i == n else ""
        content.append({"type": "text", "text":
                        f"reference {i} (a {entry.get('domain', '?')}: "
                        f"{entry.get('style_desc', '?')}"
                        + (f"; user assigned role "
                           f"{entry['role']}" if entry.get("role")
                           not in (None, "auto") else "")
                        + newest + "):"})
        content.append({"type": "image_url", "image_url": {
            "url": "data:image/jpeg;base64," + entry["b64"]}})
    content.append({"type": "text", "text":
                    f"The user said: {utterance!r}. "
                    f"Current scene: {scene_hint or 'empty canvas'}."})
    body = {
        "model": model,
        "temperature": 0.1,
        "max_tokens": 300,
        "messages": [
            {"role": "system", "content": _DIRECTOR_RUBRIC},
            {"role": "user", "content": content},
        ],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "ref_plan", "schema": _DIRECTOR_SCHEMA}},
    }
    try:
        loop = asyncio.get_running_loop()
        async with _VISION_GATE:
            out = await loop.run_in_executor(
                None, stream_chat_blocking, base_url, body, timeout_s)
        plan = json.loads(out)
        log.info("ref director: %s | suffix=%r | %s",
                 [(u["ref"], u["as"]) for u in plan.get("use", [])],
                 plan.get("suffix", "")[:80], plan.get("note", ""))
        return plan
    except Exception as e:
        log.warning("ref director failed (%s)", e)
        return None

_STYLE_SCHEMA = {
    "type": "object",
    "properties": {
        "suffix": {"type": "string", "maxLength": 220,
                   "pattern": "^[A-Za-z][A-Za-z0-9 ,;:'()/-]{10,215}$"},
    },
    "required": ["suffix"],
    "additionalProperties": False,
}

_STYLE_RUBRIC = (
    "These images are the user's STYLE references. Distill their "
    "shared visual style into one compact comma-separated descriptor "
    "usable as an image-generation prompt suffix: medium and "
    "rendering technique, palette, lighting, linework/texture, "
    "background treatment. Describe only the STYLE that the images "
    "share, never the objects pictured. Example: 'soft pastel "
    "watercolor, visible paper texture, loose wet washes, delicate "
    "linework, white background'."
)


async def distill_style(refs_b64: list[str], base_url: str, model: str,
                        timeout_s: float = 8.0) -> str | None:
    """One compact style suffix distilled from the uploaded refs —
    round 4 measured refs+suffix TOGETHER as the best zero-training
    style carrier (6.8 vs 6.4 refs-only / 5.5 text-only)."""
    content = [{"type": "image_url", "image_url": {
        "url": f"data:image/jpeg;base64,{b}"}} for b in refs_b64[:2]]
    body = {
        "model": model,
        "temperature": 0.1,
        "max_tokens": 120,
        "messages": [
            {"role": "system", "content": _STYLE_RUBRIC},
            {"role": "user", "content": content},
        ],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "style_distill", "schema": _STYLE_SCHEMA}},
    }
    try:
        loop = asyncio.get_running_loop()
        async with _VISION_GATE:
            out = await loop.run_in_executor(
                None, stream_chat_blocking, base_url, body, timeout_s)
        suffix = (json.loads(out).get("suffix") or "").strip()
        log.info("style distilled: %r", suffix)
        return suffix or None
    except Exception as e:
        log.warning("style distill failed (%s)", e)
        return None


async def read_upload(b64: str, base_url: str, model: str,
                      timeout_s: float = 6.0) -> dict | None:
    body = {
        "model": model,
        "temperature": 0.1,
        "max_tokens": 90,
        "messages": [
            {"role": "system", "content": _UPLOAD_RUBRIC},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}"}},
            ]},
        ],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "upload_read", "schema": _UPLOAD_SCHEMA}},
    }
    try:
        loop = asyncio.get_running_loop()
        async with _VISION_GATE:
            content = await loop.run_in_executor(
                None, stream_chat_blocking, base_url, body, timeout_s)
        out = json.loads(content)
        log.info("upload read: %r (%s)", out.get("object"),
                 out.get("kind"))
        return out
    except Exception as e:
        log.warning("upload decider failed (%s)", e)
        return None
