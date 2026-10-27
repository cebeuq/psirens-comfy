"""Reflex integrator: pure logic that turns classified micro-intents into
scene-overlay mutations. No I/O, no model — fully unit-testable.

Holds per-utterance working state:
  - working copies of entity boxes (seeded from canonical on first touch)
  - style-axis accumulation levels (greener x3 -> deeper green text)
  - a command context (selected target, last motion vector) so bare
    continuations ("a little more") repeat the previous motion
  - an undo stack of full working-state snapshots
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable

from ..scene.state import SceneState

MOVE_STEP = {"tiny": 0.03, "small": 0.06, "medium": 0.12, "big": 0.22}
SCALE_FACTOR = {"tiny": 1.08, "small": 1.15, "medium": 1.3, "big": 1.6}
DIR_VEC = {"left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1)}

# style axes materialize into scene-state field text by accumulated level 1..3
STYLE_TABLE: dict[str, tuple[str, list[str]]] = {
    "greener": ("palette", ["a hint of green", "rich green tones", "deep saturated greens"]),
    "redder": ("palette", ["a hint of red", "rich red tones", "deep saturated reds"]),
    "bluer": ("palette", ["a hint of blue", "rich blue tones", "deep saturated blues"]),
    "warmer": ("lighting", ["slightly warm light", "warm golden light", "hot amber glow"]),
    "cooler": ("lighting", ["slightly cool light", "cool blue light", "icy blue cast"]),
    "brighter": ("lighting", ["well lit", "bright vivid light", "blazing bright light"]),
    "darker": ("lighting", ["dimmed light", "dark shadowy light", "near darkness"]),
    "moodier": ("mood", ["slightly moody", "moody, atmospheric", "heavy, brooding"]),
}
# axis pairs cancel each other's levels
OPPOSITES = {"warmer": "cooler", "cooler": "warmer",
             "brighter": "darker", "darker": "brighter"}


def style_phrases(axis: str) -> tuple[str, list[str]] | None:
    """Field + level phrases for an axis; supports dynamic 'color:X' axes."""
    if axis.startswith("color:"):
        c = axis.split(":", 1)[1]
        return ("palette", [f"a hint of {c}", f"rich {c} tones",
                            f"deep saturated {c}"])
    return STYLE_TABLE.get(axis)


@dataclass
class ReflexResult:
    kind: str                 # "move" | "scale" | "style" | "undo" | "stop"
    detail: str
    fields: dict[str, str]
    entity_ops: list[dict]


@dataclass
class _Working:
    boxes: dict[str, dict] = field(default_factory=dict)   # id -> {x,y,w,h}
    style_levels: dict[str, int] = field(default_factory=dict)


class ReflexIntegrator:
    def __init__(self, get_canonical: Callable[[], SceneState]):
        self._get_canonical = get_canonical
        # selection and motion momentum persist ACROSS utterances, so a
        # follow-up utterance like "a little more" continues the last motion
        self.selected: str | None = None
        self.last_vector: tuple[float, float] | None = None
        self.last_action: dict | None = None
        self.reset_utterance()

    def reset_utterance(self) -> None:
        # style intensity persists across utterances: "darker... [pause]
        # ...darker" must continue from the reached level, not restart at 1
        # (a visible brightness regression otherwise). Boxes re-seed from
        # canonical; undo scope stays per-utterance.
        prev_styles = dict(getattr(self, "work", _Working()).style_levels)
        self.work = _Working(style_levels=prev_styles)
        self.undo_stack: list[_Working] = []
        self.actions_applied = 0
        self.residue = False   # fragments the grammar couldn't consume
        self.log: list[str] = []

    # ---- helpers ----
    def _canonical_box(self, eid: str) -> dict | None:
        for e in self._get_canonical().entities:
            if e.id == eid:
                return e.box.model_dump()
        return None

    def _resolve_target(self, target: str) -> str | None:
        if target not in ("selected", "none", None):
            self.selected = target
            return target
        return self.selected

    def _touch_box(self, eid: str) -> dict | None:
        if eid not in self.work.boxes:
            box = self._canonical_box(eid)
            if box is None:
                return None
            self.work.boxes[eid] = dict(box)
        return self.work.boxes[eid]

    def _snapshot(self) -> None:
        self.undo_stack.append(copy.deepcopy(self.work))
        if len(self.undo_stack) > 20:
            self.undo_stack.pop(0)

    # ---- main entry ----
    def apply(self, intent: dict) -> ReflexResult | None:
        """Apply one classified intent. Returns the applied result, or None
        if nothing happened (action=none marks residue)."""
        action = intent.get("action")

        if action == "none":
            self.residue = True
            return None
        if action == "stop":
            self.last_vector = None
            self.last_action = None
            return ReflexResult("stop", "hold", self.fields(), self.entity_ops())

        if action == "undo":
            if not self.undo_stack:
                # nothing to undo in THIS utterance: the target of the undo
                # is the previous commit — signal a timeline revert
                return ReflexResult("undo_previous", "revert previous commit",
                                    {}, [])
            self.work = self.undo_stack.pop()
            self.actions_applied += 1
            self.log.append("undo")
            return ReflexResult("undo", "reverted last action",
                               self.fields(), self.entity_ops())

        if action == "move":
            direction = intent.get("direction", "none")
            amount = intent.get("amount", "medium")
            eid = self._resolve_target(intent.get("target", "none"))
            if eid is None:
                self.residue = True
                return None
            # closer/away on a 2D canvas = grow/shrink
            if direction in ("closer", "away"):
                return self._apply_scale(eid, amount,
                                         "plus" if direction == "closer" else "minus")
            if direction in DIR_VEC:
                step = MOVE_STEP[amount]
                vec = (DIR_VEC[direction][0] * step, DIR_VEC[direction][1] * step)
            elif self.last_vector is not None:
                scale = MOVE_STEP[amount] / MOVE_STEP["medium"]
                vec = (self.last_vector[0] * scale, self.last_vector[1] * scale)
            else:
                # "move the chair" with no direction yet: select and wait
                self.last_action = {"action": "move", "target": eid}
                return None
            box = self._touch_box(eid)
            if box is None:
                self.residue = True
                return None
            self._snapshot()
            box = self.work.boxes[eid]
            box["x"] = min(1.0, max(0.0, box["x"] + vec[0]))
            box["y"] = min(1.0, max(0.0, box["y"] + vec[1]))
            self.last_vector = vec
            self.last_action = {"action": "move", "target": eid}
            self.actions_applied += 1
            self.log.append(f"move {eid} by ({vec[0]:+.2f},{vec[1]:+.2f})")
            return ReflexResult("move", f"{eid} -> ({box['x']:.2f},{box['y']:.2f})",
                               self.fields(), self.entity_ops())

        if action == "scale":
            eid = self._resolve_target(intent.get("target", "none"))
            if eid is None:
                self.residue = True
                return None
            return self._apply_scale(eid, intent.get("amount", "medium"),
                                     intent.get("sign", "plus"))

        if action == "style":
            axis = intent.get("style_axis", "none")
            # the classifier's legacy color axes and the lexicon's color:X
            # axes are the same thing — unify so levels accumulate together
            axis = {"redder": "color:red", "greener": "color:green",
                    "bluer": "color:blue"}.get(axis, axis)
            if axis == "none" or style_phrases(axis) is None:
                return None  # style with no axis: harmless no-op
            self._snapshot()
            # a new color replaces previous color accumulation ("red red red
            # ... shifts to green" should GO green, not blend forever)
            if axis.startswith("color:"):
                for k in [k for k in self.work.style_levels
                          if k.startswith("color:") and k != axis]:
                    del self.work.style_levels[k]
            sign = 1 if intent.get("sign", "plus") == "plus" else -1
            opp = OPPOSITES.get(axis)
            if sign > 0 and opp and self.work.style_levels.get(opp, 0) > 0:
                self.work.style_levels[opp] -= 1
            else:
                lvl = self.work.style_levels.get(axis, 0) + sign
                self.work.style_levels[axis] = min(3, max(0, lvl))
            self.last_action = {"action": "style", "axis": axis}
            self.actions_applied += 1
            self.log.append(f"style {axis} -> {self.work.style_levels.get(axis)}")
            return ReflexResult("style", f"{axis} level {self.work.style_levels.get(axis, 0)}",
                               self.fields(), self.entity_ops())

        return None

    def _apply_scale(self, eid: str, amount: str, sign: str) -> ReflexResult | None:
        box = self._touch_box(eid)
        if box is None:
            self.residue = True
            return None
        self._snapshot()
        f = SCALE_FACTOR[amount]
        if sign == "minus":
            f = 1 / f
        box = self.work.boxes[eid]
        # the same utterance's size word re-arrives as the tail grows
        # ("really big" -> "really big and flush ...") — re-interpret,
        # never compound (two stacked 1.6x once blew a chair to full
        # image width mid-sentence)
        prev = getattr(self, "_last_scale", None)
        if (prev and prev.get("eid") == eid
                and self.last_action == {"action": "scale",
                                         "target": eid}):
            box["w"] = min(1.0, max(0.02, box["w"] / prev["f"]))
            box["h"] = min(1.0, max(0.02, box["h"] / prev["f"]))
        self._last_scale = {"eid": eid, "f": f}
        box["w"] = min(1.0, max(0.02, box["w"] * f))
        box["h"] = min(1.0, max(0.02, box["h"] * f))
        self.last_action = {"action": "scale", "target": eid}
        self.actions_applied += 1
        self.log.append(f"scale {eid} x{f:.2f}")
        return ReflexResult("scale", f"{eid} x{f:.2f}",
                           self.fields(), self.entity_ops())

    # ---- overlay export ----
    def fields(self) -> dict[str, str]:
        out: dict[str, str] = {}
        by_field: dict[str, list[str]] = {}
        for axis, level in self.work.style_levels.items():
            if level <= 0:
                continue
            sp = style_phrases(axis)
            if sp is None:
                continue
            fname, phrases = sp
            by_field.setdefault(fname, []).append(phrases[min(level, 3) - 1])
        for fname, phrases in by_field.items():
            out[fname] = ", ".join(phrases)
        return out

    def entity_ops(self) -> list[dict]:
        return [{"op": "update", "id": eid, "name": None, "description": None,
                 "box": dict(box)}
                for eid, box in self.work.boxes.items()]
