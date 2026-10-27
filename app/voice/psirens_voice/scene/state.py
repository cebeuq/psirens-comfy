"""Scene state: the single structured source of truth (spec §2.3).

Content fields (words mostly patch these): subject, setting, composition.
Style fields (tone mostly biases these):   lighting, palette, mood, medium.
Entity layer: tracked objects with normalized position boxes — the enabler
for precise relative edits ("move the chair closer to the left" is box math,
not vision). Boxes use center x/y + w/h, all normalized 0..1.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

CONTENT_FIELDS = ("subject", "setting", "composition")
STYLE_FIELDS = ("lighting", "palette", "mood", "medium")
ALL_FIELDS = CONTENT_FIELDS + STYLE_FIELDS


class Box(BaseModel):
    x: float = Field(ge=0, le=1)   # center
    y: float = Field(ge=0, le=1)   # center
    w: float = Field(gt=0, le=1)
    h: float = Field(gt=0, le=1)


class Entity(BaseModel):
    id: str
    name: str
    description: str = ""
    box: Box
    z: int = 0

    def position_phrase(self) -> str:
        """Deterministic 3x3-grid phrase for prompt rendering."""
        col = "left" if self.box.x < 0.34 else ("right" if self.box.x > 0.66 else "center")
        row = "top" if self.box.y < 0.34 else ("bottom" if self.box.y > 0.66 else "middle")
        if col == "center" and row == "middle":
            return "in the center"
        if row == "middle":
            return f"on the {col}"
        if col == "center":
            return f"at the {row}"
        return f"at the {row} {col}"


class SceneState(BaseModel):
    subject: str = ""
    setting: str = ""
    composition: str = ""
    lighting: str = ""
    palette: str = ""
    mood: str = ""
    medium: str = ""
    entities: list[Entity] = []

    def apply_patch(self, patch: dict) -> tuple["SceneState", list[str]]:
        """Returns (new state, changed field names). None/absent = unchanged."""
        data = self.model_dump()
        changed = []
        for f in ALL_FIELDS:
            v = patch.get(f)
            if v is not None and v != data[f]:
                data[f] = v
                changed.append(f)
        return SceneState(**data), changed

    def apply_entity_edits(self, edits: list[dict]) -> tuple["SceneState", list[str]]:
        """Ops: {op: add|update|remove, id, name?, description?, box?, z?}.
        Returns (new state, changed labels like 'entity:chair')."""
        entities = {e.id: e.model_copy(deep=True) for e in self.entities}
        order = [e.id for e in self.entities]
        changed = []
        for edit in edits:
            op, eid = edit.get("op"), edit.get("id")
            if not eid:
                continue
            if op == "add":
                if "box" not in edit or edit["box"] is None:
                    continue
                entities[eid] = Entity(
                    id=eid, name=edit.get("name") or eid,
                    description=edit.get("description") or "",
                    box=Box(**edit["box"]), z=edit.get("z") or 0)
                if eid not in order:
                    order.append(eid)
                changed.append(f"entity:{eid}")
            elif op == "update" and eid in entities:
                e = entities[eid]
                if edit.get("name"):
                    e.name = edit["name"]
                if edit.get("description"):
                    e.description = edit["description"]
                if edit.get("box"):
                    e.box = Box(**edit["box"])
                if edit.get("z") is not None:
                    e.z = edit["z"]
                changed.append(f"entity:{eid}")
            elif op == "remove" and eid in entities:
                del entities[eid]
                order.remove(eid)
                changed.append(f"entity:{eid}")
        new = self.model_copy(deep=True)
        new.entities = [entities[i] for i in order if i in entities]
        return new, changed
