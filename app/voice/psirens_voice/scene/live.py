"""LiveScene: the two-layer scene state of the v2 streaming architecture.

- canonical: the committed timeline (undo/history) — only the slow editor
  writes it, on commit.
- overlay: provisional edits sketched by the fast loop from *unfinished*
  speech. Replaced wholesale by each new draft, cleared on commit.

effective = canonical ⊕ overlay is what the canvas shows at any moment.
`generation` increments whenever canonical changes; drafts fired against an
older generation are stale and get discarded.
"""
from __future__ import annotations

from .renderer import render_prompt
from .state import SceneState
from .timeline import Timeline


class LiveScene:
    def __init__(self) -> None:
        self.timeline = Timeline()
        self.overlay: dict[str, str] = {}
        self.overlay_entity_ops: list[dict] = []
        self.generation = 0
        self.pending_commit = False  # slow editor busy: fast loop holds fire
        # reflex loop (L0) state, reset whenever canonical changes
        self.reflex_dirty = False     # reflex has acted this utterance
        self.reflex_residue = False   # utterance had content the grammar missed
        self.reflex_pending = False   # classification in flight / fragment queued
        self.reflex_revert_requested = False  # "no no go back" across utterances
        self.reflex_log: list[str] = []

    @property
    def canonical_state(self) -> SceneState:
        return self.timeline.current_state

    def effective_state(self) -> SceneState:
        state, _ = self.canonical_state.apply_patch(self.overlay)
        state, _ = state.apply_entity_edits(self.overlay_entity_ops)
        return state

    def effective_prompt(self) -> str:
        return render_prompt(self.effective_state())

    def set_overlay(self, patch: dict[str, str],
                    entity_ops: list[dict] | None = None) -> list[str]:
        """Replace the whole overlay with a new draft; returns the labels the
        overlay now changes relative to canonical (fields and entity:<id>)."""
        canonical = self.canonical_state
        self.overlay = {f: v for f, v in patch.items()
                        if v and v != getattr(canonical, f)}
        self.overlay_entity_ops = list(entity_ops or [])
        _, ent_changed = canonical.apply_entity_edits(self.overlay_entity_ops)
        return list(self.overlay) + ent_changed

    def canonical_changed(self) -> None:
        """Called by the slow editor after a commit/revert lands."""
        self.overlay = {}
        self.overlay_entity_ops = []
        self.reflex_dirty = False
        self.reflex_residue = False
        self.reflex_log = []
        self.generation += 1
