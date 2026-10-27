"""In-process latent store: the canvas lives in ComfyUI between runs.

Iterative editing usually round-trips every edit through pixels (decode,
save, upload, re-encode). Each trip loses a little, and a voice session is
dozens of edits long. Keeping the last output LATENT inside ComfyUI lets
the next edit reference it directly ("latent chaining"), so nothing is
re-encoded unless pixels really changed outside the model.

Slots are free-form strings per session; the Psirens renderer uses
"c<rev>" for committed revisions (= undo keyframes) and "scratch" for the
provisional lane. Oldest slots are evicted past MAX_SLOTS per session.
"""
from __future__ import annotations

import threading
from collections import OrderedDict

MAX_SLOTS = 64
MAX_SESSIONS = 8

_lock = threading.Lock()
_sessions: "OrderedDict[str, OrderedDict[str, dict]]" = OrderedDict()


def put(session: str, slot: str, latent: dict) -> None:
    entry = {k: (v.detach().to("cpu") if hasattr(v, "detach") else v)
             for k, v in latent.items()}
    with _lock:
        slots = _sessions.pop(session, None) or OrderedDict()
        slots.pop(slot, None)
        slots[slot] = entry
        while len(slots) > MAX_SLOTS:
            slots.popitem(last=False)
        _sessions[session] = slots
        while len(_sessions) > MAX_SESSIONS:
            _sessions.popitem(last=False)


def get(session: str, slot: str) -> dict | None:
    with _lock:
        slots = _sessions.get(session)
        if not slots or slot not in slots:
            return None
        slots.move_to_end(slot)
        return dict(slots[slot])


def restore(session: str, slot: str, keyframe: int) -> bool:
    """Make keyframe <slot>@<keyframe> the current <slot> again (undo)."""
    with _lock:
        slots = _sessions.get(session)
        src = slots.get(f"{slot}@{keyframe}") if slots else None
        if src is None:
            return False
        slots[slot] = dict(src)
        slots.move_to_end(slot)
        return True


def drop(session: str) -> None:
    with _lock:
        _sessions.pop(session, None)


def describe() -> dict:
    with _lock:
        return {s: list(slots) for s, slots in _sessions.items()}
