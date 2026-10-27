"""Deterministic scene state -> prompt text. This string is the output of the
whole voice intent engine (what an image model would receive)."""
from __future__ import annotations

from .state import SceneState


def _names_entity(text: str, state: SceneState) -> bool:
    """True when the text mentions a tracked entity by name.

    One home per object: if the subject line names something the entity
    layer already describes, emitting both makes the image model paint the
    object TWICE (the double-Lamborghini bug). The entity phrase wins — it
    carries the position and the freshest description.
    """
    t = text.lower()
    for e in state.entities:
        name = (e.name or e.id).lower()
        if name and name in t:
            return True
    return False


def render_prompt(state: SceneState) -> str:
    parts: list[str] = []
    if state.subject and not _names_entity(state.subject, state):
        parts.append(state.subject)
    if state.setting:
        parts.append(state.setting)
    for e in state.entities:
        parts.append(f"{e.description or e.name} {e.position_phrase()}")
    if state.composition:
        parts.append(state.composition)
    if state.lighting:
        parts.append(f"{state.lighting} lighting")
    if state.palette:
        parts.append(f"{state.palette} palette")
    if state.mood:
        parts.append(f"{state.mood} mood")
    if state.medium:
        parts.append(state.medium)
    return ", ".join(parts)
