"""ReflexSession — L0 of the loop stack: word-rate canvas control.

v3: PURE RULES, ZERO LLM CALLS. Segments the finalized-word stream into
short fragments, matches them against deterministic grammars (motion, scale,
style/color, undo, stop), and applies results through the ReflexIntegrator
to the LiveScene overlay. Publishes reflex_action + provisional prompt and
render events so the canvas moves *while the user is still talking*.

The three rules every fired intent obeys (all four v3 designs agreed):
  - SPAN CONSUMPTION: the reflex consumes only the words that support its
    action. Any other content words are residue — the editor still sees the
    full utterance, so "A red Lamborghini. On Beverly Hills." both nudges
    the color AND reaches the editor for the setting.
  - COMMANDS, NOT DESCRIPTIONS: color words fire only with imperative
    support ("make it red", "redder"); attributive mentions ("a red car")
    defer to the editor.
  - UNDO IS EXPLICIT AND FIRES ONCE: only bare negation + undo vocabulary
    with nothing else after it ("no no", "go back", "undo that"), latched
    to once per utterance. Complaints and object-removal requests fall
    through to the editor.
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING

from ..prosody.engine import normalize_word
from ..scene.live import LiveScene
from .integrator import ReflexIntegrator
from .lexicon import (COLOR_WORDS, COMPARATIVES, IMPERATIVE_MARKERS,
                      INTENSIFIERS, MOTION_WORDS, SIZE_WORDS, STOP_WORDS_OK,
                      STYLE_WORDS, TONAL, detect_control_intent,
                      detect_style_intent)

if TYPE_CHECKING:
    from ..config import Config
    from ..events import EventBus

log = logging.getLogger("psirens.reflex")

# undo fires ONLY when the fragment's content is entirely within this set
UNDO_VOCAB = {"no", "nope", "undo", "back", "revert", "go", "wait",
              "actually", "that", "it", "again"}
UNDO_TRIGGERS = {"undo", "back", "revert"}

# every word the reflex grammars can possibly consume — used to compute
# residue (content the reflex did NOT handle, which must reach the editor)
REFLEX_VOCAB = (COLOR_WORDS | set(COMPARATIVES) | set(TONAL) | INTENSIFIERS
                | IMPERATIVE_MARKERS | MOTION_WORDS | SIZE_WORDS | STYLE_WORDS
                | STOP_WORDS_OK | UNDO_VOCAB
                | {"tiny", "touch", "hair", "slightly", "tad", "nudge",
                   "smaller", "tinier", "shrink", "bigger", "larger", "huge",
                   "grow", "enlarge", "expand", "wider", "taller",
                   "please", "just", "then", "now", "okay", "ok", "yeah"})
_REFLEXY = REFLEX_VOCAB

BASE_STOPWORDS = {"a", "an", "the", "and", "to", "of", "it", "that", "this",
                  "in", "on", "with", "is", "its"}


def residue_words(text: str, stopwords: set[str]) -> list[str]:
    """Content words of an utterance that NO reflex grammar could have
    consumed. Non-empty residue means the editor must still process the
    utterance even when the reflex acted (span consumption, fix B)."""
    out = []
    for raw in text.split():
        w = normalize_word(raw)
        if w and w not in stopwords and w not in REFLEX_VOCAB:
            out.append(w)
    return out


class ReflexSession:
    def __init__(self, cfg: "Config", bus: "EventBus", live: LiveScene,
                 loop: asyncio.AbstractEventLoop):
        r = cfg.reflex
        self.fragment_gap_ms = r.fragment_gap_ms
        self.fragment_max_words = r.fragment_max_words
        self.strength = r.strength
        self.fillers = {f.lower() for f in cfg.commit.fillers}
        self.stopwords = self.fillers | BASE_STOPWORDS
        # negation/correction words need what FOLLOWS them ("no no, go back"):
        # a tail whose content is only these waits for more words
        self.defer_words = {"no", "nope", "wait", "actually", "go", "going",
                            "keep"}

        self.bus = bus
        self.live = live
        self.integrator = ReflexIntegrator(lambda: live.canonical_state)

        self._utt_words: list[str] = []
        self._commit_words: list[str] | None = None  # full utterance at commit
        self._new_since_fire = 0
        self._last_sig: tuple | None = None
        self._last_sig_nwords = 0
        self._last_word_at = 0.0
        self._commit_seen = False
        self._utt_gen = 0
        self._undo_fired_gen = -1     # undo latch: once per utterance
        self._t_ms = 0

        self._queue = bus.subscribe()
        self._consume_task = loop.create_task(self._consume())
        self._pace_task = loop.create_task(self._pace())

    def _update_pending(self) -> None:
        self.live.reflex_pending = self._new_since_fire > 0

    def _finish_utterance(self) -> None:
        # residue: content words of the whole utterance that no reflex
        # grammar could have consumed. If the reflex acted AND residue
        # exists, the editor must still process the utterance.
        if self._commit_words is not None:
            residue = residue_words(" ".join(self._commit_words), self.stopwords)
            if self.live.reflex_dirty or self.live.reflex_revert_requested:
                self.integrator.residue = bool(residue)
                self.live.reflex_residue = bool(residue)
                if residue:
                    log.info("reflex residue -> editor still runs: %s", residue)
            self._commit_words = None
        self._commit_seen = False
        self._last_sig = None
        self._last_sig_nwords = 0
        self._utt_gen += 1
        self.integrator.reset_utterance()
        self._update_pending()

    async def _consume(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            ev = await self._queue.get()
            self._t_ms = max(self._t_ms, ev.t_session_ms)
            if ev.type == "word":
                self._utt_words.append(ev.data["w"])
                self._new_since_fire += 1
                self._last_word_at = loop.time()
                self._update_pending()
            elif ev.type == "commit" or (
                    ev.type == "hold"
                    and str(ev.data.get("reason", "")).startswith("discard_")):
                self._commit_seen = True
                self._commit_words = list(self._utt_words)
                tail = self._utt_words[-8:]
                nwords = len(self._utt_words)
                self._utt_words = []
                self._new_since_fire = 0
                # ALWAYS re-apply the rules at flush, not only for unseen
                # words: mid-utterance fires may have evaluated against a
                # scene the in-flight editor hadn't populated yet (no
                # entities -> no match). At flush the state is current;
                # signature dedupe + the undo latch make re-runs harmless.
                if tail:
                    self._apply_rules(tail, nwords)
                self._finish_utterance()
                self._update_pending()

    async def _pace(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            await asyncio.sleep(0.1)
            if not self._new_since_fire or self._commit_seen:
                continue
            gap_ok = (loop.time() - self._last_word_at) * 1000 >= self.fragment_gap_ms
            if gap_ok or self._new_since_fire >= self.fragment_max_words:
                nwords = len(self._utt_words)
                tail = self._utt_words[-8:]
                self._new_since_fire = 0
                self._apply_rules(tail, nwords)
                self._update_pending()

    # ---- rules-only classification (synchronous, 0ms, zero calls) ----

    def _match_intent(self, norm_content: list[str]) -> dict | None:
        # 1. undo: bare negation + undo vocabulary and NOTHING else.
        #    "no no" / "no, go back" / "undo that" fire; "no, remove the
        #    car" and "no, it removed the one" have residue words -> defer.
        content_set = set(norm_content)
        if (norm_content
                and all(w in UNDO_VOCAB for w in norm_content)
                and (content_set & UNDO_TRIGGERS
                     or norm_content.count("no") + norm_content.count("nope") >= 2)):
            if self._undo_fired_gen == self._utt_gen:
                return None       # latched: one undo per utterance
            return {"action": "undo", "target": "selected",
                    "direction": "none", "amount": "medium",
                    "sign": "plus", "style_axis": "none"}
        # 2. style/color (imperative only — see lexicon)
        if (intent := detect_style_intent(norm_content, self.stopwords)):
            return intent
        # 3. move / scale / stop
        entity_ids = [e.id for e in self.live.canonical_state.entities]
        return detect_control_intent(norm_content, entity_ids)

    def _apply_rules(self, tail_words: list[str], nwords: int) -> None:
        content = [w for w in tail_words
                   if normalize_word(w) not in self.stopwords]
        if not content:
            return
        # negation-only tails ("No,") defer mid-utterance — the meaning is
        # in what follows; the words stay buffered for the next fire
        if (not self._commit_seen
                and all(normalize_word(w) in self.defer_words for w in content)):
            return
        fragment = re.sub(r"[.!?]+$", "", " ".join(tail_words)).strip()
        norm = [normalize_word(w) for w in tail_words]
        norm_content = [w for w in norm if w and w not in self.stopwords]

        gen = self._utt_gen
        live_gen = self.live.generation
        intent = self._match_intent(norm_content)
        if intent is None:
            return
        if gen != self._utt_gen or live_gen != self.live.generation:
            return

        # dedupe: the same classification over a barely-grown tail is a
        # re-read of what we already applied, not a new command. Undo is
        # additionally latched (above) — it can never refire in an utterance.
        sig = (intent.get("action"), intent.get("target"),
               intent.get("direction"), intent.get("style_axis"),
               intent.get("sign"))
        if sig == self._last_sig and nwords - self._last_sig_nwords < 2:
            return
        result = self.integrator.apply(intent)
        if result is None:
            return
        self._last_sig = sig
        self._last_sig_nwords = nwords

        if result.kind == "undo_previous":
            self._undo_fired_gen = self._utt_gen
            self.live.reflex_revert_requested = True
            self.bus.publish("reflex_action", {
                "fragment": fragment, "intent": intent,
                "kind": "undo", "detail": result.detail,
                "state": self.live.effective_state().model_dump(),
                "provisional_fields": [],
                "latency_ms": 0,
            }, self._t_ms)
            return
        if intent["action"] == "undo":
            self._undo_fired_gen = self._utt_gen

        self.live.set_overlay(result.fields, result.entity_ops)
        self.live.reflex_dirty = True
        self.live.reflex_log = list(self.integrator.log)
        effective = self.live.effective_state()
        prompt = self.live.effective_prompt()

        self.bus.publish("reflex_action", {
            "fragment": fragment,
            "intent": intent,
            "kind": result.kind,
            "detail": result.detail,
            "state": effective.model_dump(),
            "provisional_fields": (list(result.fields)
                                   + [f"entity:{op['id']}" for op in result.entity_ops]),
            "latency_ms": 0,
        }, self._t_ms)
        self.bus.publish("prompt_updated", {
            "commit_id": None, "prompt": prompt, "provisional": True,
        }, self._t_ms)
        from ..scene.edits import extract_move_ops, phrase_edit
        move_ops = extract_move_ops(result.entity_ops,
                                    self.live.canonical_state.entities)
        self.bus.publish("render_target", {
            "prompt": prompt,
            "strength": self.strength,
            "source": "reflex",
            "generation": live_gen,
            "entities": [e.model_dump() for e in effective.entities],
            "edit_instruction": phrase_edit(
                result.fields, result.entity_ops,
                self.live.canonical_state.entities,
                skip_move_ids=frozenset(m["id"] for m in move_ops)),
            "move_ops": move_ops or None,
        }, self._t_ms)

    async def close(self) -> None:
        self._consume_task.cancel()
        self._pace_task.cancel()
        self.bus.unsubscribe(self._queue)
