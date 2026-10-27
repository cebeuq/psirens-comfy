"""Deterministic patch -> edit-instruction phrasing for the edit-native
renderer (FLUX.2 Klein).

The pipeline is delta-native end to end: voice deltas become scene patches
become IMAGE EDIT instructions applied to the current canvas — the root
image is the only original generation. This module turns a patch (field
changes + entity ops) into one short imperative sentence the image model
edits by. Pure code, zero LLM calls.
"""
from __future__ import annotations

FIELD_VERBS = {
    "subject": "change the main subject to",
    "setting": "change the setting to",
    "composition": "change the composition to",
    "lighting": "change the lighting to",
    "palette": "shift the colors to",
    "mood": "make the mood",
    "medium": "restyle the image as",
}

SCENE_TEXT_FIELDS = ("subject", "setting", "composition", "lighting",
                     "palette", "mood", "medium")


def _stemset(text: str) -> set[str]:
    """Crude plural-insensitive content words for staleness matching."""
    return {w.strip(".,!?\"'").rstrip("s").lower()
            for w in (text or "").replace("_", " ").split()
            if len(w.strip(".,!?\"'")) > 2}


_PHRASE_STOP = {"with", "and", "the", "a", "an", "in", "on", "of", "at",
                "by", "under", "over", "plus", "featuring", "strung"}


def dropped_phrases(old: str, new: str, max_phrases: int = 2) -> list[str]:
    """Contiguous word runs present in OLD field text but gone from NEW —
    the language-agnostic signature of a removal ('street with palm
    trees' -> 'street' drops 'palm trees'). Used to derive pixel deletes
    without depending on the LLM emitting remove ops. Runs are trimmed of
    connective stopwords; words that merely MOVED are ignored."""
    import difflib
    ow = (old or "").split()
    nw = (new or "").split()
    new_stems = _stemset(new or "")
    out: list[str] = []
    sm = difflib.SequenceMatcher(a=[w.lower() for w in ow],
                                 b=[w.lower() for w in nw])
    for tag, i1, i2, _, _ in sm.get_opcodes():
        if tag not in ("delete", "replace"):
            continue
        run = [w.strip(".,!?\"'") for w in ow[i1:i2]]
        run = [w for w in run
               if w.rstrip("s").lower() not in new_stems]   # moved != gone
        while run and run[0].lower() in _PHRASE_STOP:
            run = run[1:]
        while run and run[-1].lower() in _PHRASE_STOP:
            run = run[:-1]
        phrase = " ".join(run).strip()
        if len(phrase) >= 3 and len(run) <= 4:
            out.append(phrase)
        if len(out) >= max_phrases:
            break
    return out


def stale_fields_for(state, labels: list[str],
                     field_overrides: dict | None = None) -> list[str]:
    """Scene TEXT fields that still assert a thing the user just removed —
    the render prompt would contradict the removal on every later render,
    and a big edit would resurrect the object from the stale text. A field
    is stale when ALL content words of a removed label appear in it
    (plural-insensitive), so "red car" never flags "red barn".
    field_overrides: pending patch values that replace the state's."""
    out = []
    for f in SCENE_TEXT_FIELDS:
        text = (field_overrides or {}).get(f, getattr(state, f))
        tstem = _stemset(text)
        if any(_stemset(lab) and _stemset(lab) <= tstem for lab in labels):
            out.append(f)
    return out


def _pos_phrase(box: dict) -> str:
    x, y = box.get("x", 0.5), box.get("y", 0.5)
    col = "left" if x < 0.34 else ("right" if x > 0.66 else "center")
    row = "top" if y < 0.34 else ("bottom" if y > 0.66 else "middle")
    if col == "center" and row == "middle":
        return "in the center"
    if row == "middle":
        return f"on the {col}"
    if col == "center":
        return f"at the {row}"
    return f"at the {row} {col}"


def _size_word(box: dict) -> str:
    w = box.get("w", 0.3)
    return "large" if w > 0.45 else ("small" if w < 0.18 else "")


def extract_move_ops(entity_ops: list[dict],
                     old_entities: list | None) -> list[dict]:
    """Exact box deltas for existing entities — executed GEOMETRICALLY by
    the renderer (cut/scale/paste + generic repair), never phrased into
    language the image model can't follow precisely."""
    old = {e.id: e for e in (old_entities or [])}
    out = []
    for op in entity_ops or []:
        if op.get("op") != "update" or not op.get("box"):
            continue
        prev = old.get(op.get("id"))
        if prev is None:
            continue
        ob, nb = prev.box, op["box"]
        moved = (abs(nb.get("x", ob.x) - ob.x) > 0.005
                 or abs(nb.get("y", ob.y) - ob.y) > 0.005
                 or abs(nb.get("w", ob.w) - ob.w) > 0.01
                 or abs(nb.get("h", ob.h) - ob.h) > 0.01)
        if moved:
            out.append({
                "id": op["id"],
                "from_box": {"x": ob.x, "y": ob.y, "w": ob.w, "h": ob.h},
                "to_box": {k: nb.get(k, getattr(ob, k))
                           for k in ("x", "y", "w", "h")},
            })
    return out


def phrase_edit(fields: dict, entity_ops: list[dict],
                old_entities: list | None = None,
                skip_move_ids: set | frozenset = frozenset()) -> str | None:
    """One short imperative instruction covering the whole patch, or None
    when there is nothing to phrase (caller falls back to a root render)."""
    old = {e.id: e for e in (old_entities or [])}

    # reconcile contradictions BEFORE phrasing: a patch that both removes
    # and updates/adds the same id would phrase as "remove the kitchen;
    # move the kitchen ..." — and the image editor obediently deletes the
    # object. When an id has a remove alongside any other op, the remove
    # is the mistake; drop it.
    ops = list(entity_ops or [])
    ids_with_keep = {op.get("id") for op in ops
                     if op.get("op") in ("add", "update")}
    ops = [op for op in ops
           if not (op.get("op") == "remove" and op.get("id") in ids_with_keep)]

    parts: list[str] = []

    for op in ops:                          # noqa: duplicated loop header
        name = (op.get("name") or op.get("id", "object")).replace("_", " ")
        kind = op.get("op")
        box = op.get("box") or {}
        if kind == "add":
            desc = op.get("description") or name
            size = _size_word(box)
            parts.append(f"add {desc}{f', {size},' if size else ''} "
                         f"{_pos_phrase(box)}")
        elif kind == "remove":
            parts.append(f"remove the {name}")
        elif kind == "update":
            prev = old.get(op.get("id"))
            if box and op.get("id") not in skip_move_ids:
                if prev is not None and abs(box.get("w", prev.box.w)
                                            - prev.box.w) > 0.02:
                    grew = box.get("w", prev.box.w) > prev.box.w
                    parts.append(f"make the {name} "
                                 f"{'bigger' if grew else 'smaller'}")
                parts.append(f"move the {name} {_pos_phrase(box)}")
            if op.get("description"):
                parts.append(f"change the {name} to {op['description']}")

    for field, value in (fields or {}).items():
        if value and field in FIELD_VERBS:
            parts.append(f"{FIELD_VERBS[field]} {value}")

    if not parts:
        return None
    # dedupe while keeping order (update ops can phrase twice)
    seen: set[str] = set()
    parts = [p for p in parts if not (p in seen or seen.add(p))]
    return "; ".join(parts) + ". Keep everything else exactly the same."
