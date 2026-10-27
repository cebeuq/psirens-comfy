"""Patch contract between the delta-editor LLM and the scene state.

The model outputs an explicit list of edits rather than a full-state object
with nulls: small models follow "name only what changes" far more reliably
than "null the unchanged fields", and it roughly halves the output tokens.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from ..scene.state import ALL_FIELDS

# JSON Schema handed to vLLM for constrained decoding. Keep in sync with
# ScenePatch below (pydantic validates what constrained decoding produced).
_BOX_SCHEMA = {
    "type": "object",
    "properties": {
        "x": {"type": "number", "minimum": 0, "maximum": 1},
        "y": {"type": "number", "minimum": 0, "maximum": 1},
        "w": {"type": "number", "minimum": 0.01, "maximum": 1},
        "h": {"type": "number", "minimum": 0.01, "maximum": 1},
    },
    "required": ["x", "y", "w", "h"],
    "additionalProperties": False,
}

# Property order matters: constrained decoding emits keys in schema order,
# and the model effectively decides its patch at the FIRST key. entity_edits
# comes first so object-level ops are considered before the familiar field
# edits can absorb everything. Both arrays are required (often []).
PATCH_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "entity_edits": {
            "type": "array",
            "maxItems": 12,
            "items": {
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": ["add", "update", "remove"]},
                    "id": {"type": "string"},
                    "name": {"type": ["string", "null"]},
                    "description": {"type": ["string", "null"]},
                    "box": {"anyOf": [_BOX_SCHEMA, {"type": "null"}]},
                },
                "required": ["op", "id"],
                "additionalProperties": False,
            },
        },
        "edits": {
            "type": "array",
            "maxItems": 7,
            "items": {
                "type": "object",
                "properties": {
                    "field": {"type": "string", "enum": list(ALL_FIELDS)},
                    "value": {"type": "string"},
                },
                "required": ["field", "value"],
                "additionalProperties": False,
            },
        },
        # SURVEYOR/ATELIER grounding: one short factual sentence about
        # the exact image region the edit concerns, written BEFORE any
        # decision fields (order matters under constrained decoding —
        # ground first, then decide). Null when no image is attached.
        "grounding": {"type": ["string", "null"], "maxLength": 220},
        # STENCIL target: the region the edit should be confined to, as
        # [x0,y0,x1,y1] percentages of the image, plus a short name.
        # Null for global edits (style/lighting/setting/camera).
        "target": {
            "anyOf": [
                {"type": "object",
                 "properties": {
                     "n": {"type": "string", "maxLength": 40},
                     "b": {"type": "array", "minItems": 4, "maxItems": 4,
                           "items": {"type": "integer", "minimum": 0,
                                     "maximum": 100}}},
                 "required": ["n", "b"], "additionalProperties": False},
                {"type": "null"}]},
        # relational contact pair ("the slide's end touches the water"):
        # the two things whose contact matters. Null otherwise.
        "contact": {
            "anyOf": [
                {"type": "object",
                 "properties": {"a": {"type": "string", "maxLength": 40},
                                "b": {"type": "string", "maxLength": 40}},
                 "required": ["a", "b"], "additionalProperties": False},
                {"type": "null"}]},
        # SWITCHBOARD router signal
        "edit_difficulty": {"type": "string",
                            "enum": ["trivial", "standard", "relational",
                                     "open"]},
        "edit_magnitude": {"type": "number", "minimum": 0, "maximum": 1},
        # camera zoom, executed GEOMETRICALLY by the renderer (the image
        # model's reference conditioning pins the framing at any strength,
        # so zooms in prose do nothing — verified with saved frames)
        "reframe": {
            "anyOf": [
                {"type": "object",
                 "properties": {
                     "kind": {"type": "string",
                              "enum": ["zoom_out", "zoom_in", "pan_up",
                                       "pan_down", "pan_left",
                                       "pan_right"]},
                     "factor": {"type": "number", "minimum": 1.1,
                                "maximum": 3.0},
                 },
                 "required": ["kind", "factor"],
                 "additionalProperties": False},
                {"type": "null"},
            ],
        },
        # the model's own brushstroke: one short imperative sentence for the
        # image EDITOR (applied to the current canvas). Optional — a
        # deterministic fallback phrases the patch when this fails validation
        "edit_instruction": {"type": ["string", "null"]},
        # things in the image the edit must NOT touch (grounding)
        "preserve": {"type": "array", "items": {"type": "string"},
                     "maxItems": 5},
        # tight boxes around the preserved things AS SEEN in the
        # attached image (parallel to preserve, same order)
        "preserve_boxes": {"type": "array",
                           "items": _BOX_SCHEMA, "maxItems": 5},
        # HOW removals execute (see the removal guide in the system
        # prompt): "surgical" = SAM cut+fill per object; "regenerate" =
        # rewrite the scene text and redraw the whole image; null when
        # the patch removes nothing
        "removal_route": {"anyOf": [
            {"type": "string", "enum": ["surgical", "regenerate"]},
            {"type": "null"}]},
        "reasoning": {"type": ["string", "null"]},
    },
    # reframe and edit_instruction are REQUIRED (nullable): as optional
    # keys the model habitually omitted them even when the prompt asked
    # for them; forcing the keys makes each decision explicit per patch
    "required": ["entity_edits", "edits", "grounding", "target",
                 "contact", "edit_difficulty", "edit_magnitude", "reframe",
                 "edit_instruction", "removal_route"],
    "additionalProperties": False,
}


def patch_schema(history_revs: list[int] | None = None) -> dict:
    """The per-call schema. When a timeline exists, history_action is added
    with restore/replay targets ENUMERATED from real commit ids — a
    hallucinated frame reference is structurally impossible, not merely
    caught (the PROVENANCE mechanism)."""
    import copy

    schema = copy.deepcopy(PATCH_JSON_SCHEMA)
    if history_revs:
        schema["properties"]["history_action"] = {
            "type": ["object", "null"],
            "properties": {
                "action": {"type": "string",
                           "enum": ["restore", "restore_edit", "replay"]},
                "restore_rev": {"type": "integer", "enum": history_revs},
                "replay_revs": {"type": "array", "maxItems": 2,
                                "items": {"type": "integer",
                                          "enum": history_revs}},
            },
            "required": ["action", "restore_rev"],
            "additionalProperties": False,
        }
    return schema


class Edit(BaseModel):
    field: str
    value: str


class BoxEdit(BaseModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    w: float = Field(gt=0, le=1)
    h: float = Field(gt=0, le=1)


class EntityEdit(BaseModel):
    op: str
    id: str
    name: str | None = None
    description: str | None = None
    box: BoxEdit | None = None


class HistoryAction(BaseModel):
    action: str                        # restore | restore_edit | replay
    restore_rev: int
    replay_revs: list[int] = Field(default_factory=list, max_length=2)


class Reframe(BaseModel):
    kind: str                          # zoom_out | zoom_in
    factor: float = Field(ge=1.1, le=3.0)


class Target(BaseModel):
    n: str
    b: list[int] = Field(min_length=4, max_length=4)


class Contact(BaseModel):
    a: str
    b: str


class ScenePatch(BaseModel):
    edits: list[Edit] = Field(max_length=7)
    entity_edits: list[EntityEdit] = Field(default_factory=list,
                                           max_length=12)
    edit_magnitude: float = Field(ge=0, le=1)
    edit_instruction: str | None = None
    removal_route: str | None = None
    preserve: list[str] = Field(default_factory=list, max_length=5)
    preserve_boxes: list[BoxEdit] = Field(default_factory=list,
                                          max_length=5)
    history_action: HistoryAction | None = None
    reframe: Reframe | None = None
    grounding: str | None = None
    target: Target | None = None
    contact: Contact | None = None
    edit_difficulty: str = "standard"
    reasoning: str | None = None

    def field_patch(self) -> dict:
        # last edit wins if the model names a field twice
        return {e.field: e.value for e in self.edits if e.field in ALL_FIELDS}

    def entity_ops(self) -> list[dict]:
        return [e.model_dump() for e in self.entity_edits]
