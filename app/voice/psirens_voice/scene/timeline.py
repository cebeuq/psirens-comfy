"""Linear append-only timeline of scene-state snapshots.

Every applied patch appends a full snapshot. Revert-to-commit-N appends a NEW
entry whose state is a copy of N's snapshot — history is never rewritten, redo
is just another revert. Branching/merging (spec §2.3) is deliberately deferred;
full snapshots make it possible later.
"""
from __future__ import annotations

from typing import Any

from .state import SceneState


class Timeline:
    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []
        self._next_id = 1

    @property
    def current_state(self) -> SceneState:
        if not self.entries:
            return SceneState()
        return SceneState(**self.entries[-1]["state_after"])

    def append(self, *, kind: str, state_after: SceneState,
               utterance: str = "", tone: dict | None = None,
               patch: dict | None = None, edit_magnitude: float | None = None,
               revert_target: int | None = None) -> dict:
        entry = {
            "commit_id": self._next_id,
            "kind": kind,  # "edit" | "revert"
            "utterance": utterance,
            "tone": tone or {},
            "patch": patch or {},
            "edit_magnitude": edit_magnitude,
            "revert_target": revert_target,
            "state_after": state_after.model_dump(),
        }
        self._next_id += 1
        self.entries.append(entry)
        return entry

    def get(self, commit_id: int) -> dict | None:
        return next((e for e in self.entries if e["commit_id"] == commit_id), None)

    def revert_to(self, commit_id: int) -> dict | None:
        target = self.get(commit_id)
        if target is None:
            return None
        return self.append(kind="revert",
                           state_after=SceneState(**target["state_after"]),
                           revert_target=commit_id)
