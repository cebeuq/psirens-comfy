"""EditorSession: the one LLM in the v3 architecture, driven by a ModePolicy.

Consumes commit (and word / canvas_frame) events, calls the delta-editor
LLM per the active mode's policy, applies patches to the timeline, and
publishes patch/state/prompt events. Also handles revert requests from the
UI.

Mode behaviors implemented here (see ..modes):
  - verbatim overlay (vow): mid-sentence, new content words are appended
    verbatim to the render prompt — zero LLM calls.
  - speculative calls (headstart): fired at soft pauses, provisional until
    the real pause promotes them (no second call) or new words supersede.
  - clause commits (oneartist): short clause boundaries commit directly.
  - idle critic (easel): after a commit's render, a silent gap triggers one
    vision call that compares the canvas to the state and repairs drift.
  - vision attach (all modes): per-policy; frames are the current canvas.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import TYPE_CHECKING

from ..modes import DEFAULT_MODE, MODES, ModePolicy, wants_frame
from ..reflex.session import BASE_STOPWORDS, residue_words
from ..scene.edits import (dropped_phrases, extract_move_ops,
                           phrase_edit, stale_fields_for)
from ..scene.ledger import build_ledger, inverse_scan
from ..scene.live import LiveScene
from ..scene.renderer import render_prompt
from .client import REMOVAL_VERBS, DeltaEditorClient, EditorError
from .strategy import ROUTER, STRATEGIES, resolve as resolve_strategy

if TYPE_CHECKING:
    from ..config import Config
    from ..events import EventBus

log = logging.getLogger("psirens.editor")

# words that signal the sentence may INVERT its meaning ("...actually no,
# keep them") — STET freezes mid-sentence rendering for the rest of the
# utterance when one appears; the commit sorts it out
REVERSAL_WORDS = {"actually", "no", "nope", "wait", "except", "but",
                  "instead", "scratch", "forget", "keep", "nevermind",
                  "never", "dont", "don't", "not"}

def _box_iou(a: dict, b: dict) -> float:
    """IoU of two center-format boxes {x,y,w,h} in normalized coords."""
    ix = max(0.0, (min(a["x"] + a["w"] / 2, b["x"] + b["w"] / 2)
                   - max(a["x"] - a["w"] / 2, b["x"] - b["w"] / 2)))
    iy = max(0.0, (min(a["y"] + a["h"] / 2, b["y"] + b["h"] / 2)
                   - max(a["y"] - a["h"] / 2, b["y"] - b["h"] / 2)))
    inter = ix * iy
    return inter / max(a["w"] * a["h"] + b["w"] * b["h"] - inter, 1e-9)


class EditorSession:
    def __init__(self, cfg: "Config", bus: "EventBus",
                 loop: asyncio.AbstractEventLoop,
                 live: LiveScene | None = None,
                 mode: ModePolicy | None = None):
        e = cfg.editor
        model = os.environ.get(e.model_env, "LiquidAI/LFM2.5-8B-A1B")
        self.client = DeltaEditorClient(
            base_url=e.base_url, model=model, temperature=e.temperature,
            max_tokens=e.max_tokens, timeout_s=e.timeout_s)
        self.max_retries = e.max_retries
        self.minimality_max_fields = e.minimality_max_fields
        self.minimality_below_magnitude = e.minimality_below_magnitude
        self.mode = mode or MODES[DEFAULT_MODE]
        self.live = live or LiveScene()
        self.timeline = self.live.timeline
        self.bus = bus
        self.fillers = {f.lower() for f in cfg.commit.fillers}

        # conversation memory (append-only, prefix-cache friendly)
        self.history: list[dict] | None = [] if self.mode.conversation else None
        self.calls = 0
        self.last_frame_b64: str | None = None
        self._frame_count = 0
        self._verify_debt = False

        # mid-sentence machinery
        self._utt_word_count = 0
        self._pending_words: list[str] = []   # unconsumed words (clause mode)
        self._overlay_new: list[str] = []     # verbatim words not yet rendered
        self._overlay_all: list[str] = []
        self._last_word_at: float | None = None
        self._last_overlay_at = 0.0
        self._spec: tuple[int, object, int] | None = None  # (word_count, patch, ms)
        self._spec_in_flight = False
        self._reversal_seen = False     # STET: freeze after inversion words
        self._eager_fired_words = 0     # SHUTTER/PRESTIGE: eager EOU parse
        self._last_p_eot = 0.0
        self._last_eager_at = -1e9
        self._clause_busy = False
        self._tone: dict = {}
        self._t_ms = 0

        # active mouse selection: spoken edits scope to this mask
        self.active_region: dict | None = None
        # agentic edit mode: STUDIO (atelier) is the product default by
        # user decision — the mode picker was removed from the UI. Other
        # strategies remain in the registry for test tooling
        # (frame_grab's set_edit_mode) and the env escape hatch.
        self.strategy = resolve_strategy(
            os.environ.get("PSIRENS_EDIT_MODE", "atelier"))
        # constraint-ladder handoff from _pick_instruction to the plan
        self._last_ladder: dict | None = None
        # ink/speech concurrency: commits wait for in-flight ink, and
        # concurrent speech demotes ink to targeting (see _handle_ink)
        self._ink_busy = False
        self._ink_busy_ts = 0.0
        self._last_commit_ts = 0.0
        self._ink_move_ts = 0.0
        # streamed strokes for put-that-there ((t_session_ms, stroke));
        # collected at commit when speech overlaps, else routed solo
        # after 3s of pen idle
        self._stroke_buf: list = []
        self._stroke_idle_task = None

        # idle critic state
        self._critic_due = False
        self._frames_at_commit = 0
        # drift budget counter (CONSERVATOR)
        self._edits_since_refresh = 0

        self._loop = loop
        self._queue = bus.subscribe()
        self._task = loop.create_task(self._consume())
        self._pace_task = loop.create_task(self._pace())
        log.info("editor session mode=%s (vision=%s, conversation=%s)",
                 self.mode.name, self.mode.vision, self.mode.conversation)

    # ------------------------------------------------------------------
    # event intake

    async def _consume(self) -> None:
        while True:
            ev = await self._queue.get()
            try:
                if ev.type == "word":
                    self._on_word(ev)
                elif ev.type == "canvas_frame":
                    if not ev.data.get("preview"):
                        self.last_frame_b64 = ev.data.get("jpeg_b64")
                        self._frame_count += 1
                elif ev.type == "vad_state":
                    # eager EOU: the end-of-utterance model says "done"
                    # ~1s before the commit gate confirms — start the parse
                    # NOW; the commit promotes it if no more words arrived.
                    # Fire only on the low->high EDGE with a cooldown: the
                    # signal HOLDS high through the pause, and re-firing on
                    # every trickling word burned 12 calls per utterance.
                    p = ev.data.get("p_eot", 0)
                    rising = p >= 0.9 and self._last_p_eot < 0.9
                    self._last_p_eot = p
                    now = self._loop.time()
                    if (self.mode.eager_eou and rising
                            and now - self._last_eager_at > 1.0
                            and self._utt_word_count > self._eager_fired_words
                            and not self._spec_in_flight
                            and not self.live.pending_commit
                            and self._pending_words):
                        self._eager_fired_words = self._utt_word_count
                        self._last_eager_at = now
                        asyncio.ensure_future(self._speculate(eager=True))
                elif ev.type == "pre_commit":
                    self._on_pre_commit(ev)
                elif ev.type == "ink":
                    asyncio.ensure_future(
                        self._handle_ink(ev.data or {}, ev.t_session_ms))
                elif ev.type == "upload_drop":
                    asyncio.ensure_future(
                        self._handle_upload_drop(ev.data or {},
                                                 ev.t_session_ms))
                elif ev.type == "ink_stroke":
                    self._on_ink_stroke(ev)
                elif ev.type == "ink_clear":
                    self._stroke_buf = []
                    self._ink_busy = False
                    self.bus.publish("ink_routed", {
                        "req_id": "clear", "action": "noop",
                        "reason": "cleared"}, ev.t_session_ms)
                elif ev.type == "commit":
                    await self._handle_commit(ev)
                elif (ev.type == "hold"
                      and str(ev.data.get("reason", "")).startswith("discard_")):
                    self._reset_utterance()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("editor failed on %s", ev.type)

    def _on_word(self, ev) -> None:
        w = ev.data["w"]
        self._utt_word_count += 1
        self._pending_words.append(w)
        self._last_word_at = self._loop.time()
        self._t_ms = ev.t_session_ms
        norm = w.strip(".,!?").lower()
        if norm in REVERSAL_WORDS:
            self._reversal_seen = True
        if (self.mode.mid_sentence in ("verbatim", "stability")
                and norm not in self.fillers):
            self._overlay_new.append(w)

    def _reset_utterance(self) -> None:
        self._utt_word_count = 0
        self._pending_words = []
        self._overlay_new = []
        self._overlay_all = []
        self._spec = None
        self._reversal_seen = False
        self._eager_fired_words = 0

    # ------------------------------------------------------------------
    # pace loop: everything that fires on time, not on events

    async def _pace(self) -> None:
        while True:
            await asyncio.sleep(0.1)
            try:
                now = self._loop.time()
                gap = (now - self._last_word_at) if self._last_word_at else 1e9

                if (self.mode.mid_sentence == "verbatim" and self._overlay_new
                        and gap < 1.5
                        and (len(self._overlay_new) >= 3
                             or now - self._last_overlay_at > 0.7)):
                    self._send_verbatim_overlay()

                # STET: render mid-sentence only when the clause looks
                # STABLE — enough words, a clause boundary (punctuation or
                # a 350ms breath), and no reversal word seen so far
                if (self.mode.mid_sentence == "stability"
                        and self._overlay_new
                        and not self._reversal_seen
                        and gap < 1.5
                        and len(self._overlay_all) + len(self._overlay_new) >= 4
                        and (gap >= 0.35
                             or (self._pending_words
                                 and self._pending_words[-1][-1:] in ".,;?"))):
                    self._send_verbatim_overlay()

                if (self.mode.mid_sentence == "speculative"
                        and not self._spec_in_flight
                        and not self.live.pending_commit
                        and not self.live.reflex_dirty
                        and self._words_since_spec() >= 3
                        and 0.5 <= gap < 1.5):
                    asyncio.ensure_future(self._speculate())

                if (self.mode.commit_trigger == "clause"
                        and not self._clause_busy
                        and not self.live.pending_commit
                        and len(self._pending_words) >= 3
                        and 0.6 <= gap < 1.5):
                    asyncio.ensure_future(self._fire_clause())

                if (self.mode.idle_critic_s and self._critic_due
                        and not self.live.pending_commit
                        and gap >= self.mode.idle_critic_s
                        and self._frame_count > self._frames_at_commit
                        and self.last_frame_b64):
                    self._critic_due = False
                    asyncio.ensure_future(self._run_critic())
            except Exception:
                log.exception("editor pace loop error")

    def _words_since_spec(self) -> int:
        if self._spec is None and not self._spec_in_flight:
            return len(self._pending_words)
        base = self._spec[0] if self._spec else self._utt_word_count
        return self._utt_word_count - base

    # ------------------------------------------------------------------
    # zero-call drafts (vow): the user's own words steer the canvas

    def _send_verbatim_overlay(self) -> None:
        if not self.timeline.entries:
            return   # no committed root yet — nothing to edit against
        self._overlay_all.extend(self._overlay_new)
        self._overlay_new = []
        self._last_overlay_at = self._loop.time()
        suffix = re.sub(r"[.!?]+$", "", " ".join(self._overlay_all)).strip()
        if not suffix:
            return
        base = self.live.effective_prompt()
        prompt = f"{base}, {suffix}" if base else suffix
        self.bus.publish("prompt_updated", {
            "commit_id": None, "prompt": prompt, "provisional": True,
        }, self._t_ms)
        self.bus.publish("render_target", {
            "prompt": prompt, "strength": 0.25, "source": "overlay",
            "generation": self.live.generation,
            "entities": [e.model_dump()
                         for e in self.live.effective_state().entities],
            "edit_instruction": (f"update the scene to include: {suffix}. "
                                 f"Keep everything else the same."),
        }, self._t_ms)

    # ------------------------------------------------------------------
    # LLM call plumbing

    def _tone_line(self, tone: dict) -> str:
        return ("tone: " + " ".join(f"{k}={v}" for k, v in tone.items())
                if tone else "tone: unknown")

    async def _call(self, state, directive: str, tone: dict, *,
                    purpose: str, unfinished: bool = False,
                    force_vision: bool | None = None,
                    use_history: bool = True,
                    note: str | None = None, t: int = 0,
                    ledger: str | None = None,
                    history_revs: list[int] | None = None):
        attach = (force_vision if force_vision is not None
                  else (getattr(self.strategy, "vision_always", False)
                        and purpose == "commit")
                  or wants_frame(self.mode, directive, self._verify_debt,
                                 self.calls))
        image = self.last_frame_b64 if attach else None
        history = self.history if (use_history and self.history is not None) else None
        patch, latency_ms, retries = await self.client.propose_patch(
            state, self._tone_line(tone), directive,
            max_changed_fields=self.minimality_max_fields,
            below_magnitude=self.minimality_below_magnitude,
            max_retries=self.max_retries if purpose == "commit" else 0,
            unfinished=unfinished, image_b64=image,
            history=history, note=note,
            ledger=ledger, history_revs=history_revs)
        self.calls += 1
        if image:
            self._verify_debt = False
        self.bus.publish("llm_call", {
            "n": self.calls, "vision": bool(image),
            "latency_ms": latency_ms, "purpose": purpose,
            "mode": self.mode.name,
        }, t or self._t_ms)
        return patch, latency_ms, retries

    # ------------------------------------------------------------------
    # drawing mode: freehand ink read by the VLM, routed by kind.
    # An X over a thing = a removal ORDER (ink discarded, proven delete
    # pipeline). A loop around a thing = a SELECTION (existing region
    # machinery). A doodle OF a thing = a CONTENT SKETCH (the marked-up
    # frame conditions the image model, which paints over the strokes).

    def _parse_note(self) -> str | None:
        """Structural context for the parser: active selection and/or
        the freshly placed upload (the salient referent of bare "it")."""
        parts = []
        if self.active_region is not None:
            parts.append(
                f"the user has SELECTED a region of the image with the "
                f"mouse ({self.active_region['name']}); the directive "
                f"refers to that selection. Phrase edit_instruction "
                f"about it; the render is hard-limited to the selected "
                f"pixels.")
        name = getattr(self, "_last_insert_name", None)
        last = getattr(self, "_last_insert_ms", None)
        if (name and last is not None
                and (self._t_ms - last) < 90000):
            eid = re.sub(r"[^a-z0-9]+", "_",
                         name.lower()).strip("_") or "object"
            parts.append(
                f"the user JUST placed their uploaded {name} (entity id "
                f"{eid}) into the image; bare 'it'/'this'/'that' refers "
                f"to it, and positional commands about it (against the "
                f"wall, flush with, next to ...) must MOVE its box per "
                f"rule 4c.")
        return " ".join(parts) or None

    def _publish_tray(self, t: int) -> None:
        self.bus.publish("tray_updated", {"refs": [
            {"i": i + 1, "object": e["object"], "domain": e["domain"],
             "style_desc": e["style_desc"], "role": e["role"],
             "b64": e["b64"]}
            for i, e in enumerate(getattr(self, "tray", []))]}, t)

    async def _run_director(self, utterance: str, t: int) -> None:
        """Ask the director for a reference plan and load it into the
        renderer's style bank. Runs on tray changes and on commits
        that could start a fresh generation or talk about the refs."""
        from .ink import direct_references

        tray = getattr(self, "tray", [])
        if not tray:
            return
        hint = ""
        try:
            hint = render_prompt(self.timeline.current_state)[:160]
        except Exception:
            pass
        plan = await direct_references(
            tray, utterance, hint, self.client.base_url,
            self.client.model)
        if plan is None:
            return
        # a canvas already exists and the user is talking about the
        # refs: the bank only conditions future ROOTS, so the plan's
        # style text must ride THIS edit's instruction to restyle the
        # current image ("make it more like that new image")
        if utterance and self._frame_count > 0:
            sfx = (plan.get("suffix") or "").strip(" ,.")
            if sfx:
                self._ref_edit_clause = sfx
        uses = {u["ref"]: u for u in plan.get("use", [])
                if 1 <= int(u.get("ref", 0)) <= len(tray)}
        # DETERMINISTIC same-subject demotion (measured, round 4): a
        # ref that PICTURES the kind of thing being requested gets
        # copied when conditioned on. The director is told this but
        # sometimes chooses style anyway — enforce it in code.
        if utterance:
            dwords = {re.sub(r"[^a-z]", "", w.lower())
                      for w in utterance.split() if len(w) > 3}
            for i, entry in enumerate(tray, start=1):
                u = uses.get(i)
                if not u or u.get("as") != "style":
                    continue
                rwords = {re.sub(r"[^a-z]", "", w.lower()) for w in
                          (str(entry.get("object", "")) + " "
                           + str(entry.get("domain", ""))).split()
                          if len(w) > 3}
                if len(dwords & rwords) >= 2:
                    u["as"] = "palette"
                    log.info("director override: ref %d pictures the "
                             "requested subject (%s) — demoted to "
                             "palette", i,
                             ", ".join(sorted(dwords & rwords)[:4]))
        style_refs = []
        for i, entry in enumerate(tray, start=1):
            u = uses.get(i)
            if not u:
                continue
            if u.get("sticky"):
                entry["role"] = u["as"]
            if u["as"] == "style":
                style_refs.append(entry["b64"])
        self.style_refs = style_refs[:2]
        self.style_suffix = (plan.get("suffix") or "").strip() or None
        self.bus.publish("style_bank", {
            "refs": self.style_refs,
            "suffix": self.style_suffix}, t)
        self.bus.publish("ref_plan", {
            "use": plan.get("use", []),
            "note": plan.get("note", ""),
            "suffix": self.style_suffix}, t)
        self._publish_tray(t)

    _REF_TALK = re.compile(
        r"\b(image|images|reference|references|ref|refs|picture|"
        r"photo|uploads?|first one|second one|third one|that one|"
        r"those ones?)\b", re.I)

    async def _fetch_upload_url(self, url: str) -> str | None:
        """Bounded server-side fetch for web-page drags: 8s, 15MB,
        image content only, re-encoded to a canvas-sized JPEG."""
        import base64 as _b64
        import io as _io
        import urllib.request

        def grab() -> str | None:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (psirens)"})
            with urllib.request.urlopen(req, timeout=8) as r:
                ctype = r.headers.get("Content-Type", "")
                if not (ctype.startswith("image/") or "octet" in ctype):
                    return None
                data = r.read(15 * 1024 * 1024)
            from PIL import Image
            img = Image.open(_io.BytesIO(data))
            img.load()
            img = img.convert("RGB")
            img.thumbnail((768, 768))
            buf = _io.BytesIO()
            img.save(buf, format="JPEG", quality=88)
            return _b64.b64encode(buf.getvalue()).decode()

        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, grab)
        except Exception as e:
            log.warning("upload url fetch failed (%s): %s", url[:80], e)
            return None

    async def _handle_upload_drop(self, d: dict, t: int) -> None:
        """An image dropped ON the canvas = insert its object at the
        drop point. The decider names it (for SAM + entity tracking);
        the renderer cuts the actual pixels — identity-perfect — and
        the repair render blends lighting/shadow, post-verified."""
        from .ink import distill_style, read_upload

        b64 = d.get("b64") or ""
        url = (d.get("url") or "").strip()
        try:
            nx = float(d.get("nx") if d.get("nx") is not None else 0.5)
            ny = float(d.get("ny") if d.get("ny") is not None else 0.5)
        except (TypeError, ValueError):
            nx = ny = 0.5
        self._t_ms = t
        if not b64 and url:
            # web-page drag: no File object crosses the browser boundary,
            # only the image URL — fetch it here (bounded) and continue as
            # if the pixels were dropped
            b64 = await self._fetch_upload_url(url)
            if not b64:
                self.bus.publish("upload_routed", {
                    "action": "noop",
                    "reason": "couldn't fetch that image"}, t)
                return
        if not b64:
            self.bus.publish("upload_routed", {
                "action": "noop", "reason": "empty upload"}, t)
            return
        reading = await read_upload(b64, self.client.base_url,
                                    self.client.model)
        # spatial intent from the UI wins outright: the references box
        # always banks, the canvas always inserts. The VLM's role guess
        # only decides for clients that sent no intent.
        intent = (d.get("intent") or "").strip()
        if intent == "insert" and self.last_frame_b64 is None:
            self.bus.publish("upload_routed", {
                "action": "noop",
                "reason": "no picture yet — drop it in the references "
                          "box instead"}, t)
            return
        as_reference = (intent == "reference"
                        or (intent != "insert"
                            and ((reading or {}).get("role") != "insert"
                                 or self.last_frame_b64 is None)))
        if as_reference:
            # REFERENCE TRAY: the upload is banked, not acted on. The
            # director (an LLM with the generator's measured playbook)
            # decides per generation which tray refs to condition on,
            # which feed the suffix only, and which to ignore — the
            # user's spoken instructions override and stick.
            entry = {
                "b64": b64,
                "object": ((reading or {}).get("object")
                           or "reference").strip(),
                "domain": ((reading or {}).get("domain") or "").strip(),
                "style_desc": ((reading or {}).get("style_desc")
                               or "").strip(),
                "role": "auto",
            }
            self.tray = (getattr(self, "tray", []) + [entry])[-8:]
            self._publish_tray(t)
            self.bus.publish("upload_routed", {
                "action": "tray", "target": entry["style_desc"],
                "count": len(self.tray)}, t)
            log.info("tray +1: %r (%s) — %d total", entry["object"],
                     entry["domain"], len(self.tray))
            # PINBOARD burst: a designer drops many refs back-to-back —
            # re-planning after every drop churns on a half-full tray.
            # Debounce: one standing plan ~2.5s after the last drop.
            old_task = getattr(self, "_director_debounce", None)
            if old_task is not None:
                old_task.cancel()

            async def _debounced(ts: int = t) -> None:
                try:
                    await asyncio.sleep(2.5)
                    # standing plans exist to pre-arm the FIRST root;
                    # once a canvas exists the bank is (re)planned by
                    # the commits that actually use it — an unprompted
                    # re-plan mid-session second-guesses a scene it
                    # cannot know the user's next intent for
                    if self._frame_count == 0:
                        await self._run_director("", ts)
                except asyncio.CancelledError:
                    pass

            self._director_debounce = asyncio.ensure_future(_debounced())
            return
        # INTERACTIVE PLACEMENT: nothing renders on drop. The renderer
        # SAM-cuts the object and returns a background-removed preview
        # the user positions, sizes, and rotates on the canvas; the
        # browser commits the final placement (idle timer / checkmark),
        # which lands in handle_insert_commit below.
        name = ((reading or {}).get("object") or "object").strip()
        req_id = str(d.get("req_id") or f"ins{t}")
        if not hasattr(self, "_pending_inserts"):
            self._pending_inserts = {}
        self._pending_inserts[req_id] = {"b64": b64, "name": name}
        self._last_insert_name = name
        self._last_insert_ms = t
        self._pending_inserts = dict(
            list(self._pending_inserts.items())[-4:])
        self.bus.publish("cutout_upload", {
            "req_id": req_id, "b64": b64, "name": name}, t)
        self.bus.publish("upload_routed", {
            "action": "preview", "target": name,
            "req_id": req_id}, t)
        log.info("insert preview requested: %r (%s)", name, req_id)
        return

    async def handle_insert_commit(self, d: dict, t: int) -> None:
        """The user finished placing a dropped object: insert it at
        exactly their position/size/rotation via the proven pixel
        pipeline."""
        pend = (getattr(self, "_pending_inserts", {})
                .pop(str(d.get("req_id") or ""), None))
        if pend is None or self.last_frame_b64 is None:
            self.bus.publish("upload_routed", {
                "action": "noop", "reason": "nothing to place"}, t)
            return
        self._t_ms = t
        b64, name = pend["b64"], pend["name"]
        try:
            nx = max(0.0, min(1.0, float(d.get("nx", 0.5))))
            ny = max(0.0, min(1.0, float(d.get("ny", 0.5))))
            wf = max(0.04, min(1.0, float(d.get("w", 0.3))))
            rot = float(d.get("rot", 0.0)) % 360.0
        except (TypeError, ValueError):
            nx = ny = 0.5
            wf = 0.3
            rot = 0.0
        eid = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "object"
        self._last_insert_name = name
        self._last_insert_ms = t
        self.bus.publish("upload_routed", {
            "action": "insert", "target": name}, t)
        to_box = {"x": nx, "y": ny, "w": wf, "h": wf}
        op = {"id": eid, "name": name, "insert": True, "b64": b64,
              "rot": rot, "from_box": dict(to_box), "to_box": to_box}
        state = self.timeline.current_state
        ops = [{"op": "add", "id": eid, "name": name,
                "description": f"the user's uploaded {name}",
                "box": to_box}]
        new_state, _ = state.apply_entity_edits(ops)
        directive = f"[insert] {name}"
        base_rev = (self.timeline.entries[-1]["commit_id"]
                    if self.timeline.entries else 0)
        entry = self.timeline.append(
            kind="edit", state_after=new_state, utterance=directive,
            tone={}, patch={"entity_edits": ops, "insert": True},
            edit_magnitude=0.3)
        self.live.canonical_changed()
        prompt = render_prompt(new_state)
        log.info("upload insert #%d: %r at (%.2f, %.2f)",
                 entry["commit_id"], name, nx, ny)
        self.bus.publish("patch_applied", {
            "commit_id": entry["commit_id"], "kind": "edit",
            "utterance": directive, "patch": {},
            "entity_edits": ops,
            "changed_fields": [f"entity:{eid}"],
            "old_values": {}, "edit_magnitude": 0.3,
            "reasoning": "upload insertion", "latency_ms": 0,
            "retries": 0, "tone": {}}, t)
        self.bus.publish("state_snapshot", {
            "commit_id": entry["commit_id"],
            "state": new_state.model_dump()}, t)
        self.bus.publish("prompt_updated", {
            "commit_id": entry["commit_id"], "prompt": prompt,
            "provisional": False}, t)
        self.bus.publish("render_target", {
            "prompt": prompt, "strength": 0.3, "source": "mouse",
            "generation": self.live.generation,
            "entities": self._sweep_entities(new_state),
            "edit_instruction": None, "lane": "committed",
            "base_rev": base_rev, "commit_rev": entry["commit_id"],
            "snapshot": True, "move_ops": [op],
            "draft": {"policy": "verify", "class": "insert",
                      "requirements": self._mouse_reqs([op])}}, t)

    _NARRATION_FILLER = {
        "put", "place", "drop", "dropping", "add", "adding", "insert",
        "inserting", "out", "in", "into", "on", "here", "there",
        "this", "that", "these", "those", "the", "a", "an", "my",
        "it", "i", "im",
        "want", "wanna", "to", "gonna", "going", "let's", "lets",
        "just", "like", "so", "okay", "ok", "now", "and", "then",
        "right", "we", "will", "can", "you",
    }

    def _is_insert_narration(self, directive: str, t: int) -> bool:
        name = getattr(self, "_last_insert_name", None)
        last = getattr(self, "_last_insert_ms", None)
        if not name or last is None or (t - last) > 8000:
            return False
        words = [re.sub(r"[^a-z']", "", w.lower())
                 for w in directive.split()]
        words = [w for w in words if w]
        if not words or len(words) > 8:
            return False
        name_words = {re.sub(r"[^a-z']", "", w.lower())
                      for w in name.split()}
        if not any(w in name_words for w in words):
            return False
        residue = [w for w in words
                   if w not in name_words
                   and w not in self._NARRATION_FILLER]
        return not residue

    def handle_insert_cancel(self, d: dict, t: int) -> None:
        pend = (getattr(self, "_pending_inserts", {})
                .pop(str(d.get("req_id") or ""), None))
        if pend is not None:
            log.info("insert preview cancelled: %r", pend["name"])
            self.bus.publish("upload_routed", {
                "action": "noop", "reason": "insert cancelled"}, t)

    def _on_ink_stroke(self, ev) -> None:
        import time as _time
        st = (ev.data or {}).get("stroke") or []
        if len(st) < 2:
            return
        self._t_ms = ev.t_session_ms
        self._stroke_buf.append((ev.t_session_ms, st))
        self._ink_busy = True
        self._ink_busy_ts = _time.monotonic()
        if self._stroke_idle_task is not None:
            self._stroke_idle_task.cancel()
        self._stroke_idle_task = asyncio.ensure_future(
            self._stroke_idle())

    async def _stroke_idle(self) -> None:
        """3s of pen idle with NO speech = a solo drawing, routed like
        always. With speech anywhere near, the commit collects the
        strokes instead (put-that-there)."""
        import time as _time
        try:
            await asyncio.sleep(3.0)
        except asyncio.CancelledError:
            return
        if (self._utt_word_count > 0
                or _time.monotonic() - self._last_commit_ts < 2.0):
            return                       # commit path owns the strokes
        strokes = [st for _, st in self._stroke_buf]
        self._stroke_buf = []
        if strokes:
            await self._handle_ink({"strokes": strokes,
                                    "req_id": "idle"}, self._t_ms)

    @staticmethod
    def _group_strokes(buf: list) -> list:
        """Time-gap clustering: strokes of one gesture arrive in quick
        succession; distinct gestures have speech between them."""
        groups: list = []
        for t, st in sorted(buf, key=lambda x: x[0]):
            if groups and t - groups[-1]["times"][-1] <= 1500:
                groups[-1]["strokes"].append(st)
                groups[-1]["times"].append(t)
            else:
                groups.append({"strokes": [st], "times": [t]})
        for g in groups:
            xs = [p[0] for st in g["strokes"] for p in st]
            ys = [p[1] for st in g["strokes"] for p in st]
            g["t_mid"] = g["times"][len(g["times"]) // 2]
            g["box_pct"] = [max(0, int(min(xs) * 100) - 4),
                            max(0, int(min(ys) * 100) - 4),
                            min(100, int(max(xs) * 100) + 4),
                            min(100, int(max(ys) * 100) + 4)]
        return groups

    async def _annotate_directive(self, directive: str, words_t: list,
                                  groups: list) -> str | None:
        """Bind pointing words to gesture groups by TIME and weave the
        groups' identities into the directive as ⟨drawing⟩ annotations
        the parser is taught to treat as ground truth."""
        from .ink import composite_ink_labeled, read_ink_groups

        if not self.last_frame_b64:
            return None
        marked = composite_ink_labeled(
            self.last_frame_b64, [g["strokes"] for g in groups])
        infos = await read_ink_groups(marked, len(groups),
                                      self.client.base_url,
                                      self.client.model,
                                      group_boxes=[g["box_pct"]
                                                   for g in groups])
        if infos is None:
            return None
        by_id = {int(g.get("id", 0)): g for g in infos}
        kind_phrase = {"object_marked": "the marked object:",
                       "empty_area": "an empty spot:",
                       "sketch": "a sketch of"}
        deictic = re.compile(
            r"^(this|that|these|those|here|there)[.,!?]*$", re.I)
        d_times = [w["t"] for w in words_t
                   if deictic.match(str(w.get("w", "")))]
        order = sorted(range(len(groups)),
                       key=lambda i: groups[i]["t_mid"])
        pairs: dict = {}
        used: set = set()
        for k, dt in enumerate(d_times):
            best, bestd = None, 1e18
            for i in order:
                if i in used:
                    continue
                dd = abs(groups[i]["t_mid"] - dt)
                if dd < bestd:
                    best, bestd = i, dd
            if best is not None:
                pairs[k] = best
                used.add(best)
        out, k = [], 0
        for tok in directive.split():
            out.append(tok)
            if deictic.match(tok):
                gi = pairs.get(k)
                k += 1
                if gi is not None:
                    info = by_id.get(gi + 1, {})
                    out.append("⟨drawing %d: %s %s, box %s percent⟩"
                               % (gi + 1,
                                  kind_phrase.get(info.get("kind"),
                                                  ""),
                                  info.get("content", "?"),
                                  groups[gi]["box_pct"]))
        leftovers = []
        for i in order:
            if i in used:
                continue
            info = by_id.get(i + 1, {})
            if info.get("kind") == "object_marked" and not d_times:
                # unnarrated destructive-looking marks with no pointing
                # words at all: drop rather than guess (pool rule)
                continue
            leftovers.append("drawing %d: %s %s, box %s percent"
                             % (i + 1,
                                kind_phrase.get(info.get("kind"), ""),
                                info.get("content", "?"),
                                groups[i]["box_pct"]))
        text = " ".join(out)
        if leftovers:
            text += ("\n[also drawn, place where the sentence implies: "
                     + "; ".join(leftovers) + "]")
        log.info("put-that-there: %d groups, %d pointing words -> %r",
                 len(groups), len(d_times), text[:160])
        return text

    async def _handle_ink(self, d: dict, t: int) -> None:
        from .ink import composite_ink, read_ink, _hints

        strokes = d.get("strokes") or []
        req_id = str(d.get("req_id") or "0")
        self._t_ms = t
        if not strokes or not self.last_frame_b64:
            self.bus.publish("ink_routed", {
                "req_id": req_id, "action": "noop",
                "reason": "no strokes or no canvas yet"}, t)
            return
        import time as _time
        self._ink_busy = True
        self._ink_busy_ts = _time.monotonic()
        marked = composite_ink(self.last_frame_b64, strokes)
        reading = await read_ink(marked, strokes, self.client.base_url,
                                 self.client.model)
        # SPEECH WINS: if the user is talking (or just committed words)
        # around this drawing, the words are the instruction and the ink
        # is only a pointer — it must not issue its own edit. A
        # destructive reading demotes to select so the utterance gets a
        # correctly-identified region ("remove those" once deleted the
        # TV because the circle hadn't resolved yet).
        def speech_near() -> bool:
            return (self._utt_word_count > 0
                    or _time.monotonic() - self._last_commit_ts < 6.0)

        if (reading is not None
                and reading.get("reading") in ("erase", "sketch")):
            # (move readings never demote: a paired utterance like
            # "move the coffee table this way" NAMES what the arrow
            # already moves — the spoken command is consumed below)
            # grace window: STT words lag the microphone by 1-2s, so a
            # user who talks WHILE drawing looks silent at decision
            # time — hold destructive/generative routes briefly and
            # demote if speech shows up
            g0 = _time.monotonic()
            while (not speech_near()
                   and _time.monotonic() - g0 < 2.5):
                await asyncio.sleep(0.1)
            if speech_near():
                log.info("ink demoted %s -> select (concurrent speech)",
                         reading.get("reading"))
                reading = dict(reading, reading="select")
        if reading is None:                    # decider down: sketch is
            reading = {"reading": "sketch",    # the only route needing
                       "target": "the drawing",  # no interpretation
                       "instruction":
                           "Replace the rough red sketch with a "
                           "realistic version of what it depicts, "
                           "matching the scene."}
        action = reading.get("reading") or "sketch"
        self.bus.publish("ink_routed", {
            "req_id": req_id, "action": action,
            "target": reading.get("target"),
            "observation": reading.get("observation")}, t)

        if action != "select":
            self._ink_busy = False
        if action == "move":
            # the arrow's own geometry is the edit: tail = the object,
            # tail->head = the exact displacement (arrow length = the
            # distance — more faithful than 'a little bit'). The
            # renderer SAM-matches the named object inside from_box, so
            # no browser round-trip is needed.
            main = max(strokes, key=lambda st:
                       ((st[-1][0] - st[0][0]) ** 2
                        + (st[-1][1] - st[0][1]) ** 2))
            tail = main[0]
            dx = main[-1][0] - tail[0]
            dy = main[-1][1] - tail[1]
            if (dx * dx + dy * dy) ** 0.5 < 0.04:
                self.bus.publish("ink_routed", {
                    "req_id": req_id, "action": "noop",
                    "reason": "arrow too short"}, t)
                return
            name = str(reading.get("target") or "the object")[:60]
            bw = bh = 0.22
            fb = {"x": min(1.0, max(0.0, tail[0])),
                  "y": min(1.0, max(0.0, tail[1])),
                  "w": bw, "h": bh}
            tb = {"x": min(1.0, max(0.0, fb["x"] + dx)),
                  "y": min(1.0, max(0.0, fb["y"] + dy)),
                  "w": bw, "h": bh}
            import time as _t2
            self._ink_move_ts = _t2.monotonic()
            log.info("ink move: %r by (%.2f, %.2f)", name, dx, dy)
            self.mouse_commit({"moves": [{
                "entity_id": "_click", "name": name,
                "from_box": fb, "to_box": tb,
            }]}, t)
            return

        if action == "erase":
            # the ink NEVER reaches the image model: route the authored
            # removal sentence through the full spoken pipeline (parse ->
            # remove op -> SAM delete -> post-verify)
            instr = (reading.get("instruction")
                     or f"Remove the {reading.get('target', 'marked object')}.")
            self.bus.publish("commit", {
                "utterance_raw": instr, "utterance_clean": instr,
                "repairs": [], "tone": {},
                "reasons": ["via=ink_erase"]}, t)
            return

        if action == "select":
            # SAM-snap inside the largest closed loop; the renderer's
            # segment_result flows to the browser, which builds the
            # selection and reports it back as set_region (same path a
            # lasso used to take). The decider's IDENTIFICATION of the
            # circled thing is stashed so set_region can name the
            # selection — a nameless region made the parser guess what
            # "this" was (it deleted a coffee table under a circled
            # plant)
            import time as _time
            self._ink_sel = {"name": str(reading.get("target")
                                         or "selected area")[:60],
                             "ts": _time.monotonic()}
            # a circle segments as itself; non-enclosing strokes (an X
            # demoted to select) segment inside their bounding box
            def _closed(st):
                if len(st) < 8:
                    return False
                dx = st[0][0] - st[-1][0]
                dy = st[0][1] - st[-1][1]
                return (dx * dx + dy * dy) ** 0.5 < 0.08
            loops = [st for st in strokes if _closed(st)]
            if loops:
                poly = max(loops, key=len)
            else:
                xs = [pt[0] for st in strokes for pt in st]
                ys = [pt[1] for st in strokes for pt in st]
                pad = 0.02
                x0, x1 = max(0, min(xs) - pad), min(1, max(xs) + pad)
                y0, y1 = max(0, min(ys) - pad), min(1, max(ys) + pad)
                poly = [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]
            self.bus.publish("lasso_segment", {
                "req_id": f"ink-{req_id}",
                "points": poly}, t)
            return

        # sketch: one timeline commit whose render is conditioned on the
        # MARKED frame; the clean frame stays the judge's before-image
        instr = (reading.get("instruction")
                 or "Replace the rough red sketch with a realistic "
                    "version of what it depicts, matching the scene.")
        instr = instr.rstrip(".") + (". Remove every red marker stroke "
                                     "completely — none may remain.")
        h = _hints(strokes)
        bb = h.get("bbox") or [0.0, 0.0, 1.0, 1.0]
        pad = 0.06
        tgt_pct = [max(0, int((bb[0] - pad) * 100)),
                   max(0, int((bb[1] - pad) * 100)),
                   min(100, int((bb[2] + pad) * 100)),
                   min(100, int((bb[3] + pad) * 100))]
        base_rev = (self.timeline.entries[-1]["commit_id"]
                    if self.timeline.entries else 0)
        state = self.timeline.current_state
        entry = self.timeline.append(
            kind="edit", state_after=state,
            utterance=f"[drawing] {reading.get('target', 'sketch')}",
            tone={}, patch={"ink": True}, edit_magnitude=0.6)
        self.live.canonical_changed()
        prompt = render_prompt(state)
        self.bus.publish("patch_applied", {
            "commit_id": entry["commit_id"], "kind": "edit",
            "utterance": f"[drawing] {reading.get('target', 'sketch')}",
            "patch": {}, "entity_edits": [], "changed_fields": [],
            "old_values": {}, "edit_magnitude": 0.6,
            "reasoning": f"ink sketch: {reading.get('observation', '')}",
            "latency_ms": 0, "retries": 0, "tone": {}}, t)
        self.bus.publish("render_target", {
            "prompt": prompt, "strength": 0.6, "source": "commit",
            "generation": self.live.generation,
            "entities": self._sweep_entities(state),
            "edit_instruction": instr, "lane": "committed",
            "base_rev": base_rev, "commit_rev": entry["commit_id"],
            "snapshot": True, "camera": False,
            "ink": {"strokes": strokes},
            "draft": {
                "policy": "loop", "class": "sketch", "size": 512,
                "requirements": [
                    f"The requested change is clearly applied: {instr}",
                    "No red marker strokes or hand-drawn lines remain "
                    "anywhere in the image",
                    "Apart from the sketched area, the framing and "
                    "scene content are unchanged",
                ],
                "target": {"n": str(reading.get("target", "sketch"))[:40],
                           "b": tgt_pct},
                "strategy": self.strategy.name,
                "budget_ms": self.strategy.budget_ms,
                "max_iters": self.strategy.max_iters,
                "n_drafts": self.strategy.n_drafts,
                "judge_profile": ("grounded"
                                  if self.strategy.grounded_judge
                                  else None),
                "ink": {"strokes": strokes},
            }}, t)

    # deictic geometric move: an active selection supplies the exact
    # pixel mask that spoken moves always lacked (the reason voice
    # never cut-pastes) — so a pure directional command on a selection
    # may safely use the mouse machinery. Deterministic, zero LLM.
    _MOVE_RE = re.compile(
        r"^(?:move|shift|put|slide|push)\s+(?:this|that|it)?\s*"
        r"(up|down|left|right)\s*(a\s+(?:bit|little)|a\s+lot|slightly)?"
        r"\s*[.!]?$", re.IGNORECASE)

    _DELETE_RE = re.compile(
        r"^(?:remove|delete|erase|get\s+rid\s+of)\s+"
        r"(?:this|that|it)(?:\s+one)?\s*[.!]?$", re.IGNORECASE)

    def _try_deictic_delete(self, directive: str, t: int) -> bool:
        r = self.active_region
        if not r or not r.get("mask_rle"):
            return False
        if not self._DELETE_RE.match(directive.strip()):
            return False
        bb = self._rle_bbox(r["mask_rle"], r["mask_size"])
        if bb is None:
            return False
        log.info("deictic delete: %r -> masked cut of %r", directive,
                 r.get("name"))
        self.mouse_commit({"moves": [{
            "entity_id": "_click", "name": r.get("name") or "selection",
            "delete": True,
            "from_box": {"x": bb[0] + bb[2] / 2, "y": bb[1] + bb[3] / 2,
                         "w": bb[2], "h": bb[3]},
            "mask_rle": r["mask_rle"], "mask_size": r["mask_size"],
        }]}, t)
        self.set_region(None)          # the selection's referent is gone
        return True

    def _try_deictic_move(self, directive: str, t: int) -> bool:
        r = self.active_region
        if not r or not r.get("mask_rle"):
            return False
        m = self._MOVE_RE.match(directive.strip())
        if not m:
            return False
        direction = m.group(1).lower()
        mod = (m.group(2) or "").lower()
        mag = 0.07 if "bit" in mod or "little" in mod or "slightly" in mod \
            else 0.30 if "lot" in mod else 0.15
        bb = self._rle_bbox(r["mask_rle"], r["mask_size"])
        if bb is None:
            return False
        dx = {"left": -mag, "right": mag}.get(direction, 0.0)
        dy = {"up": -mag, "down": mag}.get(direction, 0.0)
        from_box = {"x": bb[0] + bb[2] / 2, "y": bb[1] + bb[3] / 2,
                    "w": bb[2], "h": bb[3]}
        to_box = {"x": max(0.0, min(1.0, from_box["x"] + dx)),
                  "y": max(0.0, min(1.0, from_box["y"] + dy)),
                  "w": bb[2], "h": bb[3]}
        log.info("deictic move: %r %s %.2f on %r", directive, direction,
                 mag, r.get("name"))
        self.mouse_commit({"moves": [{
            "entity_id": "_click", "name": r.get("name") or "selection",
            "from_box": from_box, "to_box": to_box,
            "mask_rle": r["mask_rle"], "mask_size": r["mask_size"],
        }]}, t)
        return True

    @staticmethod
    def _rle_bbox(counts: list, size: list) -> list | None:
        """Normalized [x, y, w, h] of an RLE mask (ROW-major binary RLE,
        the frontend rleEncodeJS format the renderer also decodes)."""
        try:
            import numpy as np
            w, h = int(size[0]), int(size[1])
            a = np.zeros(w * h, dtype=bool)
            pos, val = 0, 0
            for c in counts:
                if val:
                    a[pos:pos + c] = True
                pos += c
                val ^= 1
            m = a.reshape(h, w)
            rows = np.flatnonzero(m.any(axis=1))
            cols = np.flatnonzero(m.any(axis=0))
            if not len(rows) or not len(cols):
                return None
            return [cols[0] / w, rows[0] / h,
                    (cols[-1] - cols[0] + 1) / w,
                    (rows[-1] - rows[0] + 1) / h]
        except Exception:
            return None

    # ------------------------------------------------------------------
    # speculative parse (lossless): the commit gate holds ~1.4s of
    # silence before committing; the detector hands us the would-be
    # utterance at pause start and we run the SAME parse call early.
    # The client memoizes the response keyed by the exact request body,
    # so the commit-time call reuses it only if every input (words,
    # tone, state, ledger, attached frame, history) is byte-identical —
    # otherwise it silently falls through to a normal call.

    def _on_pre_commit(self, ev) -> None:
        d = ev.data or {}
        directive = (d.get("utterance_clean") or "").strip()
        if (not directive or self.mode.commit_trigger == "clause"
                or self.live.pending_commit):
            return
        asyncio.ensure_future(
            self._prewarm_parse(directive, d.get("tone") or {}))

    async def _prewarm_parse(self, directive: str, tone: dict) -> None:
        try:
            state = self.timeline.current_state
            note = self._parse_note()
            ledger = build_ledger(self.timeline.entries)
            if ledger:
                hint, _, _ = inverse_scan(directive,
                                          self.timeline.entries)
                if hint:
                    ledger = f"{ledger}\n{hint}"
            history_revs = [e["commit_id"]
                            for e in self.timeline.entries][-24:]
            attach = (getattr(self.strategy, "vision_always", False)
                      or wants_frame(self.mode, directive,
                                     self._verify_debt, self.calls))
            await self.client.prewarm(
                state, self._tone_line(tone), directive,
                unfinished=False,
                image_b64=self.last_frame_b64 if attach else None,
                history=self.history, note=note, ledger=ledger,
                history_revs=history_revs)
        except Exception:
            log.debug("speculative prewarm failed (ignored)",
                      exc_info=True)

    # ------------------------------------------------------------------
    # commit path

    async def _handle_commit(self, ev) -> None:
        t = ev.t_session_ms
        self._t_ms = t
        d = ev.data
        tone = d.get("tone") or {}
        self._tone = tone

        import time as _time
        self._last_commit_ts = _time.monotonic()
        # TRAILING WINDOW: a final gesture often lands just after the
        # sentence ends — hold 2s (extended per new stroke, capped 4s)
        # before collecting the utterance's drawings
        pinned_words = d.get("words_t") or []
        if self._stroke_buf or self._ink_busy:
            w0 = _time.monotonic()
            last_n = len(self._stroke_buf)
            while _time.monotonic() - w0 < 4.0:
                await asyncio.sleep(0.1)
                n = len(self._stroke_buf)
                if n != last_n:
                    last_n = n
                elif _time.monotonic() - w0 >= 2.0:
                    break
        # collect this utterance's gesture groups
        self._ink_groups = None
        if self._stroke_buf:
            if self._stroke_idle_task is not None:
                self._stroke_idle_task.cancel()
            groups = self._group_strokes(self._stroke_buf)
            self._stroke_buf = []
            if len(groups) == 1:
                # single gesture + speech = today's flow: identify it,
                # demote to a named region, let the words command it
                await self._handle_ink(
                    {"strokes": groups[0]["strokes"],
                     "req_id": "utt"}, t)
            else:
                self._ink_groups = groups
                self._ink_busy = False
        # a solo drawing may still be resolving (decider + SAM +
        # set_region) — bounded wait, stale flags ignored
        waited = 0.0
        while (self._ink_busy and waited < 4.0
               and _time.monotonic() - self._ink_busy_ts < 8.0):
            await asyncio.sleep(0.1)
            waited += 0.1
        if waited:
            log.info("commit waited %.1fs for in-flight ink", waited)
        # let the reflex flush the utterance's last fragment
        waited = 0.0
        while self.live.reflex_pending and waited < 3.0:
            await asyncio.sleep(0.1)
            waited += 0.1
        while self._clause_busy:
            await asyncio.sleep(0.1)

        if self.live.reflex_revert_requested:
            self.live.reflex_revert_requested = False
            self._reset_utterance()
            self._do_reflex_revert(t)
            return

        # reflex commit: the whole utterance was consumed by the grammars.
        # Residue is computed HERE, from the editor's own inputs — relying
        # on the reflex session to have computed it first is a race (the
        # two consume the commit event from separate queues).
        if self.live.reflex_dirty and self.live.reflex_log:
            residue = residue_words(d["utterance_clean"],
                                    self.fillers | BASE_STOPWORDS)
            if not residue:
                self._reset_utterance()
                self._promote_reflex(d["utterance_clean"], tone, t)
                return
            # single-writer rule: the reflex matched only a FRAGMENT of a
            # larger sentence, so its guess was a visual preview, not
            # state. The editor sees the full sentence against CANONICAL
            # state and is the only writer ("make them UNC Blue" once let
            # a mid-sentence color reflex bake a global blue palette into
            # every future render). canonical_changed() after the commit
            # discards the preview overlay.
            log.info("reflex previewed but residue remains (%s) — editor "
                     "owns the full utterance; preview state discarded",
                     residue)

        if self.mode.commit_trigger == "clause":
            directive = " ".join(self._pending_words).strip()
            self._reset_utterance()
            content = [w for w in directive.split()
                       if w.strip(".,!?").lower() not in self.fillers]
            if not content:
                return          # clauses already consumed everything
        else:
            directive = d["utterance_clean"]

        # INSERT NARRATION GUARD: people talk while they drop ("put in
        # this chair") and the ASR garbles it ("out in this chair"); the
        # parser then invents an edit from it (observed: a full-redraw
        # "zoom in on the chair" that erased the scene). Within a short
        # window of placement activity, an utterance that only names the
        # inserted object with placement/deictic filler is narration —
        # never an edit.
        if self._is_insert_narration(directive, t):
            log.info("insert narration suppressed: %r", directive)
            self.bus.publish("hold", {
                "reason": "narrating the drop (no edit)",
                "utterance": directive}, t)
            self._reset_utterance()
            return

        # REFERENCE DIRECTOR: re-plan the tray for this request when it
        # could start a fresh generation (first frame) or when the user
        # is talking about the references themselves. Ordinary edits
        # skip it — their style already lives in the canvas.
        if getattr(self, "tray", None) and (
                self._frame_count == 0
                or self._REF_TALK.search(directive)):
            await self._run_director(directive, t)

        # multi-gesture utterance: weave each drawing's identity into
        # the directive at its time-matched pointing word
        if getattr(self, "_ink_groups", None):
            groups = self._ink_groups
            self._ink_groups = None
            annotated = await self._annotate_directive(
                directive, pinned_words, groups)
            if annotated:
                directive = annotated
            self.bus.publish("ink_routed", {
                "req_id": "multi", "action": "multi",
                "target": f"{len(groups)} drawings"}, t)

        # an arrow just executed a move: a paired utterance narrating it
        # ("move the coffee table this way a bit") must not fire a
        # second, vaguer edit on top
        import time as _t3
        if (_t3.monotonic() - getattr(self, "_ink_move_ts", 0.0) < 8.0
                and re.match(r"^(?:okay[,.\s]+)?(?:now\s+)?"
                             r"(?:move|shift|put|slide|push|drag)\b",
                             directive.strip(), re.IGNORECASE)):
            log.info("spoken move consumed by arrow gesture: %r",
                     directive)
            self._reset_utterance()
            return

        # circled selection + pure directional command = geometric move
        # via the mouse machinery (the selection supplies the exact
        # mask; zero LLM, deterministic delta)
        if self._try_deictic_move(directive, t):
            self._reset_utterance()
            return
        # circled selection + pure removal = masked delete of EXACTLY
        # the circled pixels — never a name-guess ("remove this" over a
        # circled plant once deleted the coffee table)
        if self._try_deictic_delete(directive, t):
            self._reset_utterance()
            return

        # promote a speculative/eager answer instead of re-asking
        if self.mode.mid_sentence == "speculative" or self.mode.eager_eou:
            waited = 0.0
            while self._spec_in_flight and waited < 20.0:
                await asyncio.sleep(0.1)
                waited += 0.1
            if self._spec is not None:
                spec_words, patch, latency_ms = self._spec
                if spec_words == self._utt_word_count and patch is not None:
                    log.info("promoting speculative/eager parse (0 extra calls)")
                    state = self.timeline.current_state
                    self._reset_utterance()
                    self.bus.publish("llm_call", {
                        "n": self.calls, "vision": False, "latency_ms": 0,
                        "purpose": "promote", "mode": self.mode.name,
                    }, t)
                    self._apply_canonical(state, patch, directive, tone, t,
                                          latency_ms, 0)
                    self._arm_critic()
                    return

        state = self.timeline.current_state
        note = self._parse_note()
        self._reset_utterance()

        self.bus.publish("patch_request", {
            "directive": directive, "tone": tone, "commit_seq": ev.seq,
        }, t)
        # the timeline ledger: the model's memory of past frames, plus the
        # deterministic inverse-scan hint ("this may undo #2"). Ink-
        # synthesized commits get NO history machinery at all — a
        # scribble over the fireplace once matched the inverse-scan and
        # time-travelled the session (restore #3 + replay)
        ink_commit = any(str(r).startswith("via=ink")
                         for r in (d.get("reasons") or []))
        ledger = None if ink_commit else build_ledger(self.timeline.entries)
        hint_undone = hint_before = None
        if ledger:
            hint, hint_undone, hint_before = inverse_scan(
                directive, self.timeline.entries)
            if hint:
                ledger = f"{ledger}\n{hint}"
        history_revs = ([] if ink_commit else
                        [e["commit_id"] for e in self.timeline.entries][-24:])
        self.live.pending_commit = True
        try:
            try:
                patch, latency_ms, retries = await self._call(
                    state, directive, tone, purpose="commit", note=note,
                    t=t, ledger=ledger, history_revs=history_revs)
            except EditorError as e:
                # one retry on any editor failure: stalls are usually
                # transient (JIT spike, batching hiccup) and a user
                # utterance must not die on a one-off — the second
                # attempt typically lands in well under a second
                log.warning("editor call failed (%s); retrying once", e)
                patch, latency_ms, retries = await self._call(
                    state, directive, tone, purpose="commit", note=note,
                    t=t, ledger=ledger, history_revs=history_revs)
        except EditorError as e:
            log.warning("editor failed for %r: %s", directive, e)
            self.bus.publish("error", {
                "where": "editor", "message": str(e), "recoverable": True,
                "directive": directive,
            }, t)
            return
        finally:
            self.live.pending_commit = False

        # deterministic inverse upgrade (the user's #1 rule: no model
        # roulette on the motivating case). The inverse-scan flagged that
        # this directive undoes commit H; if the model's own patch removes
        # exactly what H added, both detectors agree — a lossless restore
        # of the pre-H frame is strictly better than a lossy edit.
        if hint_undone is not None and hint_before is not None:
            undone_entry = self.timeline.get(hint_undone)
            added = {op.get("id") for op in
                     (undone_entry.get("patch") or {}).get("entity_edits") or []
                     if op.get("op") == "add"} if undone_entry else set()
            removed = {e.id for e in patch.entity_edits if e.op == "remove"}
            model_agrees = (patch.history_action is not None
                            and patch.history_action.restore_rev == hint_before)
            patch_confirms = bool(added and (added & removed))
            if model_agrees or patch_confirms:
                # NORMALIZE: a targeted inverse ("remove the stickers")
                # restores the pre-frame and REPLAYS the unrelated later
                # commits (capped 2) — the model tends to forget survivors
                # or add redundant edits; code makes the action canonical.
                survivors = [e for e in self.timeline.entries
                             if e["commit_id"] > hint_undone
                             and e["kind"] == "edit"
                             and not e.get("undone_by")]
                if len(survivors) <= 2:
                    from .schema import HistoryAction
                    patch.history_action = HistoryAction(
                        action="replay" if survivors else "restore",
                        restore_rev=hint_before,
                        replay_revs=[e["commit_id"] for e in survivors])
                    patch.entity_edits = []
                    patch.edits = []
                    log.info("deterministic inverse normalize: restore #%d "
                             "(undoes #%d), replay %s", hint_before,
                             hint_undone,
                             [e["commit_id"] for e in survivors])
        if patch.history_action is not None:
            self._apply_history_action(patch, directive, tone, t,
                                       latency_ms, retries)
        else:
            self._apply_canonical(state, patch, directive, tone, t,
                                  latency_ms, retries)
        self._arm_critic()

    # ------------------------------------------------------------------
    # history-aware editing: restore / replay / restore_edit
    #
    # The timeline is a first-class tool: "remove the stickers" restores
    # the pre-sticker keyframe (pixel-exact, 0ms) instead of forward-
    # editing damage away; later good commits replay on top (git rebase
    # for pixels). The committed FIFO guarantees ordering.

    def _state_at(self, commit_id: int):
        from ..scene.state import SceneState
        e = self.timeline.get(commit_id)
        return SceneState(**e["state_after"]) if e else None

    def _apply_history_action(self, patch, directive: str, tone: dict,
                              t: int, latency_ms: int, retries: int) -> None:
        from ..scene.state import SceneState

        ha = patch.history_action
        target = self.timeline.get(ha.restore_rev)
        if target is None:      # schema makes this near-impossible; be safe
            log.warning("history_action names unknown rev %s; editing "
                        "forward instead", ha.restore_rev)
            self._apply_canonical(self.timeline.current_state, patch,
                                  directive, tone, t, latency_ms, retries)
            return
        prev_head = (self.timeline.entries[-1]["commit_id"]
                     if self.timeline.entries else 0)

        # BLAST-RADIUS GUARD: a restore orphans every live edit after the
        # target that isn't replayed. Destroying lots of work is only
        # legitimate when the user actually TALKED about going back — a
        # plain removal ("remove the stuff on the bed") that the model
        # mistook for time travel must fall back to a forward edit.
        # (A one-word false match once silently rewound 13 edits.)
        _TEMPORAL = {"back", "revert", "restore", "undo", "before",
                     "earlier", "was", "again", "previous", "beginning",
                     "start"}
        dwords = {w.strip(".,!?\"'").lower() for w in directive.split()}
        at_risk = [e for e in self.timeline.entries
                   if e["kind"] == "edit" and not e.get("undone_by")
                   and ha.restore_rev < e["commit_id"] <= prev_head
                   and e["commit_id"] not in (ha.replay_revs or [])]
        if len(at_risk) > 3 and not (dwords & _TEMPORAL):
            log.warning("history_action restore #%s would orphan %d live "
                        "edits but %r never mentions going back — editing "
                        "forward instead", ha.restore_rev, len(at_risk),
                        directive)
            patch.history_action = None
            self._apply_canonical(self.timeline.current_state, patch,
                                  directive, tone, t, latency_ms, retries)
            return
        state = self._state_at(ha.restore_rev)

        # GRAFT conflict check: a replay must be a live later edit whose
        # entity ops still make sense against the restored state
        replays: list[dict] = []
        for rev in (ha.replay_revs or []):
            e = self.timeline.get(rev)
            if (e is None or e["kind"] != "edit" or e.get("undone_by")
                    or not (ha.restore_rev < rev <= prev_head)):
                log.info("replay #%s rejected (missing/undone/out of range)",
                         rev)
                continue
            ops = (e.get("patch") or {}).get("entity_edits") or []
            known = {ent.id for ent in state.entities}
            adds = {op.get("id") for op in ops if op.get("op") == "add"}
            conflict = any(op.get("op") in ("update", "remove")
                           and op.get("id") not in known | adds
                           for op in ops)
            if conflict:
                log.info("replay #%s rejected (references entities absent "
                         "after restore)", rev)
                continue
            replays.append(e)
            fields = {k: v for k, v in (e.get("patch") or {}).items()
                      if k not in ("entity_edits", "reflex", "history_action")}
            state, _ = state.apply_patch(fields)
            state, _ = state.apply_entity_edits(ops)

        # the current call's own edits apply last (restore_edit case)
        state, _ = state.apply_patch(patch.field_patch())
        state, _ = state.apply_entity_edits(patch.entity_ops())

        entry = self.timeline.append(
            kind="revert", state_after=state, utterance=directive,
            tone=tone, revert_target=ha.restore_rev,
            patch={"history_action": {
                "action": ha.action, "restore_rev": ha.restore_rev,
                "replay_revs": [e["commit_id"] for e in replays]}})
        new_id = entry["commit_id"]
        for e in self.timeline.entries:
            if ha.restore_rev < e["commit_id"] < new_id:
                if any(r["commit_id"] == e["commit_id"] for r in replays):
                    e["replayed_into"] = new_id
                else:
                    e.setdefault("undone_by", new_id)
        self.live.canonical_changed()
        prompt = render_prompt(state)

        self.bus.publish("patch_applied", {
            "commit_id": new_id, "kind": "revert",
            "revert_target": ha.restore_rev,
            "utterance": directive, "patch": {},
            "entity_edits": [], "changed_fields": [],
            "old_values": {}, "edit_magnitude": patch.edit_magnitude,
            "reasoning": (f"time travel: {ha.action} #{ha.restore_rev}"
                          + (f" + replay {[e['commit_id'] for e in replays]}"
                             if replays else "")),
            "latency_ms": latency_ms, "retries": retries, "tone": tone,
        }, t)
        self.bus.publish("state_snapshot", {
            "commit_id": new_id, "state": state.model_dump()}, t)
        self.bus.publish("prompt_updated", {
            "commit_id": new_id, "prompt": prompt, "provisional": False}, t)

        # render chain, in committed-FIFO order: restore, replays, own edit
        followups = [e for e in replays]
        own_edit = (ha.action == "restore_edit"
                    or bool(patch.field_patch() or patch.entity_ops()))
        n_targets = 1 + len(followups) + (1 if own_edit else 0)
        def _target(i, **kw):
            base = {"prompt": prompt, "strength": 0.6, "source": "commit",
                    "generation": self.live.generation,
                    "entities": self._sweep_entities(state),
                    "base_rev": ha.restore_rev,
                    "lane": "committed"}
            if i == n_targets - 1:
                base.update({"snapshot": True, "commit_rev": new_id})
            base.update(kw)
            return base
        self.bus.publish("render_target",
                         _target(0, restore_rev=ha.restore_rev, reset=True),
                         t)
        for i, e in enumerate(followups):
            p = e.get("patch") or {}
            mouse_ops = (p.get("mouse_ops")
                         or ([p["mouse_op"]] if p.get("mouse_op") else None))
            if mouse_ops:
                # mouse edits replay as exact geometry, never as language
                self.bus.publish("render_target",
                                 _target(1 + i, edit_instruction=None,
                                         move_ops=mouse_ops,
                                         draft={"policy": "verify",
                                                "class": "replay",
                                                "requirements":
                                                self._mouse_reqs(mouse_ops),
                                                "preserve_last": True}), t)
                continue
            instr = (e.get("utterance", "").strip().rstrip(".!?")
                     + ". Keep everything else exactly the same.")
            self.bus.publish("render_target",
                             _target(1 + i, edit_instruction=instr,
                                     draft={"policy": "verify",
                                            "class": "replay",
                                            "requirements": [
                                                "The requested change is "
                                                "clearly applied: " + instr]
                                            }), t)
        if own_edit:
            instr = (directive.strip().rstrip(".!?")
                     + ". Keep everything else exactly the same.")
            self.bus.publish("render_target",
                             _target(n_targets - 1, edit_instruction=instr,
                                     draft={"policy": "verify",
                                            "class": "replay",
                                            "requirements": [
                                                "The requested change is "
                                                "clearly applied: " + instr]
                                            }),
                             t)
        log.info("time travel: %s -> #%d (restore #%d, replay %s, own_edit=%s)",
                 ha.action, new_id, ha.restore_rev,
                 [e["commit_id"] for e in replays], own_edit)

    # ------------------------------------------------------------------
    # speculative call (headstart)

    async def _speculate(self, eager: bool = False) -> None:
        self._spec_in_flight = True
        try:
            words_at = self._utt_word_count
            directive = " ".join(self._pending_words).strip()
            # same single-writer rule as the commit path: speculate
            # against canonical state, never a reflex preview
            state = self.timeline.current_state
            try:
                patch, latency_ms, _ = await self.client.propose_patch(
                    state, self._tone_line(self._tone), directive,
                    max_changed_fields=7, below_magnitude=0.0, max_retries=0,
                    unfinished=not eager,
                    history=list(self.history) if self.history is not None else None)
            except EditorError as e:
                log.debug("speculative call failed (harmless): %s", e)
                return
            self.calls += 1
            self.bus.publish("llm_call", {
                "n": self.calls, "vision": False, "latency_ms": latency_ms,
                "purpose": "speculative", "mode": self.mode.name,
            }, self._t_ms)
            self._spec = (words_at, patch, latency_ms)
            show = (self.mode.mid_sentence == "speculative"
                    and self.mode.spec_display)
            if (show and words_at == self._utt_word_count
                    and not self.live.pending_commit
                    and not self.live.reflex_dirty):
                changed = self.live.set_overlay(patch.field_patch(),
                                                patch.entity_ops())
                if changed:
                    effective = self.live.effective_state()
                    prompt = self.live.effective_prompt()
                    self.bus.publish("draft_patch", {
                        "words": directive,
                        "patch": dict(self.live.overlay),
                        "entity_edits": self.live.overlay_entity_ops,
                        "provisional_fields": changed,
                        "state": effective.model_dump(),
                        "edit_magnitude": patch.edit_magnitude,
                        "latency_ms": latency_ms,
                    }, self._t_ms)
                    self.bus.publish("prompt_updated", {
                        "commit_id": None, "prompt": prompt,
                        "provisional": True,
                    }, self._t_ms)
                    self.bus.publish("render_target", {
                        "prompt": prompt,
                        "strength": round(min(0.3, patch.edit_magnitude), 3),
                        "source": "draft",
                        "generation": self.live.generation,
                        "entities": [e.model_dump() for e in effective.entities],
                        "edit_instruction": phrase_edit(
                            patch.field_patch(), patch.entity_ops(),
                            state.entities),
                    }, self._t_ms)
        finally:
            self._spec_in_flight = False

    # ------------------------------------------------------------------
    # clause commit (oneartist)

    async def _fire_clause(self) -> None:
        self._clause_busy = True
        try:
            words = self._pending_words[:]
            self._pending_words = []
            directive = re.sub(r"[.!?]+$", "", " ".join(words)).strip()
            content = [w for w in directive.split()
                       if w.strip(".,!?").lower() not in self.fillers]
            if not content or self.live.reflex_dirty:
                # nudge clauses belong to the reflex; don't double-handle
                return
            state = self.timeline.current_state
            t = self._t_ms
            self.bus.publish("patch_request", {
                "directive": directive, "tone": self._tone,
                "commit_seq": None,
            }, t)
            self.live.pending_commit = True
            try:
                patch, latency_ms, retries = await self._call(
                    state, directive, self._tone, purpose="clause",
                    unfinished=True, t=t)
            except EditorError as e:
                log.warning("clause call failed for %r: %s", directive, e)
                return
            finally:
                self.live.pending_commit = False
            self._apply_canonical(state, patch, directive, self._tone, t,
                                  latency_ms, retries)
        finally:
            self._clause_busy = False

    # ------------------------------------------------------------------
    # idle critic (easel)

    def _arm_critic(self) -> None:
        if self.mode.idle_critic_s:
            self._critic_due = True
            self._frames_at_commit = self._frame_count

    async def _run_critic(self) -> None:
        state = self.timeline.current_state
        t = self._t_ms
        directive = "(silent canvas check)"
        note = ("You are the critic. Compare the attached canvas image to "
                "the state. If the image contradicts the state — duplicated "
                "objects, missing things, wrong colors or positions — "
                "return corrective edits that make the state unambiguous. "
                "If they match, return empty edits.")
        self.live.pending_commit = True
        try:
            patch, latency_ms, retries = await self._call(
                state, directive, {}, purpose="critic",
                force_vision=True, use_history=False, note=note, t=t)
        except EditorError as e:
            log.debug("critic call failed (harmless): %s", e)
            return
        finally:
            self.live.pending_commit = False
        if not patch.field_patch() and not patch.entity_ops():
            log.info("critic: canvas matches state")
            return
        log.info("critic: repairing drift: %s", patch.field_patch())
        self._apply_canonical(state, patch, directive, {}, t,
                              latency_ms, retries)

    # ------------------------------------------------------------------
    # canonical application (shared by commit / promote / clause / critic)

    def _pick_instruction(self, patch, state, prompt: str,
                          skip_move_ids: frozenset = frozenset(),
                          directive: str | None = None,
                          has_moves: bool = False) -> str | None:
        """Choose the image-edit instruction per mode policy.

        LLM-authored instructions are validated (3-45 words, non-empty) and
        fall back to deterministic phrase_edit — a bad brushstroke must
        never reach the canvas. CONSERVATOR's drift budget replaces every
        Nth delta edit with a refresh edit that re-grounds the whole image.
        """
        fallback = phrase_edit(patch.field_patch(), patch.entity_ops(),
                               state.entities, skip_move_ids=skip_move_ids)
        instruction = fallback
        # constraint ladder: when a magnitude suffix is chosen below, the
        # unsuffixed text + starting rung are also published so the
        # renderer's draft loop can CLIMB (re-suffix looser) instead of
        # being stuck with one blind up-front guess. None on every path
        # that doesn't tier by magnitude.
        self._last_ladder = None
        if (self.mode.instruction_author == "user" and directive
                and not has_moves and not directive.startswith("(")):
            # LLM buffer: prefer the editor's own cleaned imperative
            # (written per-directive with the canvas in context, so
            # references like "the headlights" stay grounded) — raw ASR
            # carries fillers and false starts straight into the image
            # model ("make uh make it UNC blue"). The user's words remain
            # the fallback when the authored sentence fails validation.
            # The SUFFIX tiers by the LLM's edit_magnitude: the keep-suffix
            # that makes small edits surgical actively strangles big ones
            # ("move the camera up. Keep everything else exactly the
            # same." is a contradiction the model resolves by doing
            # nothing) — big edits get permission, structural edits get
            # the whole scene to re-stage from.
            text = directive.strip().rstrip(".!?")
            authored = (patch.edit_instruction or "").strip().rstrip(".!?")
            if re.match(r"(?i)\s*(no change|nothing (to|needs)|"
                        r"it( i|')s already|already )", authored):
                # a no-op instruction re-renders the whole image for
                # nothing — every such pass is pure regression risk
                log.info("no-op instruction suppressed: %r", authored[:60])
                return fallback
            nw = len(authored.split())
            if authored and 3 <= nw <= 45:
                text = authored
            mag = patch.edit_magnitude or 0.0
            # ref-mention edit: make the vague "like that image" concrete
            # by inlining the director's distilled style text, and give
            # the repaint enough magnitude to actually restyle
            clause = getattr(self, "_ref_edit_clause", None)
            if clause:
                self._ref_edit_clause = None
                text = f"{text}, in this style: {clause}"
                if self.active_region is None:
                    mag = max(mag, 0.6)
            # a circled selection guarantees scope in PIXELS (the region
            # splice restores everything outside the circle), so strength
            # policy splits by intent:
            # - modifying what's there: surgical, low strength — a
            #   region wall-edit at 0.6 once rebuilt half the window
            # - ADDING something new inside the circle: full power —
            #   at surgical strength a new in-ground hot tub simply
            #   never materialized through five verify retries
            if self.active_region is not None:
                has_adds = any(op.get("op") == "add"
                               for op in patch.entity_ops())
                mag = (max(mag, 0.75) if has_adds
                       else min(mag, 0.5))
            if mag >= 0.8:
                rung = 2
                instruction = (f"{text}. Redraw the whole image as needed "
                               f"to achieve this, keeping the same scene "
                               f"and subjects: {prompt}.")
            elif mag >= 0.6:
                rung = 1
                instruction = f"{text}. Adjust the rest of the image as needed."
            else:
                rung = 0
                instruction = f"{text}. Keep everything else exactly the same."
            self._last_ladder = {"base": text, "start_rung": rung,
                                 "prompt_full": prompt}
        if self.mode.instruction_author == "llm":
            authored = (patch.edit_instruction or "").strip()
            nwords = len(authored.split())
            if authored and 3 <= nwords <= 45:
                if patch.preserve:
                    keep = ", ".join(patch.preserve[:5])
                    authored += f" Keep unchanged: {keep}."
                instruction = authored
                self._last_ladder = None   # no magnitude suffix to climb
            elif authored:
                log.info("authored instruction rejected (%d words); "
                         "using deterministic phrasing", nwords)
        if (self.mode.drift_budget
                and instruction is not None
                and self._edits_since_refresh + 1 >= self.mode.drift_budget):
            self._edits_since_refresh = -1   # incremented to 0 by caller
            instruction = (f"refine and clean up the image so it precisely "
                           f"matches: {prompt}. Remove any artifacts, "
                           f"blur, or smearing.")
            self._last_ladder = None
            log.info("drift budget reached — refresh edit replaces delta")
        return instruction

    _REMOVAL_TALK = re.compile(
        r"\b(remove|removes|removed|delete|deletes|deleted|get rid|"
        r"gets rid|take out|takes out|took out|erase|erases|clear|"
        r"clears|cleared|without|no more|empty|gone|replace|replaces|"
        r"replaced|instead|swap)\b", re.I)

    def _sanitize_removals(self, patch, state, directive: str) -> None:
        """An entity removal must be LICENSED by the directive — a
        removal verb, or the entity's own name in the user's words.
        (Live failure: 'make the wall on the left just a flat wall'
        was parsed as remove-the-chair.)"""
        ops = getattr(patch, "entity_edits", None)
        if not ops:
            return
        if self._REMOVAL_TALK.search(directive):
            return
        dir_words = {re.sub(r"[^a-z']", "", w.lower())
                     for w in directive.split()}
        ent_by_id = {e.id: e for e in state.entities}
        kept = []
        for e in ops:
            if e.op != "remove":
                kept.append(e)
                continue
            ent = ent_by_id.get(e.id)
            ename = str(e.name or (ent.name if ent else "") or e.id or "")
            nwords = {re.sub(r"[^a-z']", "", w.lower())
                      for w in re.split(r"[\s_]+", ename) if w}
            if nwords & dir_words:
                kept.append(e)
            else:
                log.warning("removal of %r rejected — directive %r "
                            "neither removes nor names it",
                            ename, directive[:60])
        patch.entity_edits = kept

    def _apply_canonical(self, state, patch, directive: str, tone: dict,
                         t: int, latency_ms: int, retries: int) -> None:
        self._sanitize_removals(patch, state, directive)
        base_rev = (self.timeline.entries[-1]["commit_id"]
                    if self.timeline.entries else 0)
        new_state, changed = state.apply_patch(patch.field_patch())
        new_state, ent_changed = new_state.apply_entity_edits(patch.entity_ops())
        changed = changed + ent_changed
        if patch.entity_ops():
            self._verify_debt = True      # next frame-carrying call verifies
        # camera moves are IMAGE-MODEL edits: full redraw tier + a fresh
        # seed at the renderer (the inherited session seed reproduces the
        # old framing deterministically — the reason text-zooms once
        # looked "impossible"). The LLM's reframe field is only a SIGNAL
        # that the directive is a camera move; nothing geometric happens.
        camera = (getattr(patch, "reframe", None) is not None
                  or "composition" in changed)
        if camera and patch.edit_magnitude < 0.85:
            log.info("camera move: magnitude %.1f floored to 0.85",
                     patch.edit_magnitude)
            patch.edit_magnitude = 0.85
        entry = self.timeline.append(
            kind="edit", state_after=new_state, utterance=directive,
            tone=tone, patch={**patch.field_patch(),
                              "entity_edits": patch.entity_ops()},
            edit_magnitude=patch.edit_magnitude)
        self.live.canonical_changed()
        prompt = render_prompt(new_state)
        # VOICE NEVER CUT-PASTES: geometric moves are a mouse gesture.
        # The editor LLM expresses relational edits ("make the slide
        # curve into the pool") as box updates, and the cut-paste path
        # can only translate a sprite — it can never reshape it, and its
        # checklist verified position instead of intent (the slide-into-
        # pool session passed with the slide never touching water).
        # Spoken update ops now update STATE only; the pixels change via
        # the judged image-edit path with the user's words as the
        # requirement. Deletes still composite below (SAM is proven
        # ground truth for removals).
        move_ops: list[dict] = []
        # removals become composite DELETES: prose removal instructions do
        # nothing against Klein's reference conditioning (verified with
        # saved frames — 'remove all the palm trees' at every tier kept
        # every tree), but the mouse layer's SAM-cut + fill + repair path
        # is proven. Route voice removals through the same machinery.
        region = self.active_region
        ent_by_id = {e.id: e for e in state.entities}
        seen_names: set[str] = set()
        region_used_for_delete = False
        regenerate = (getattr(patch, "removal_route", None)
                      == "regenerate")
        if regenerate and any(op.get("op") == "remove"
                              for op in patch.entity_ops()):
            # the model chose REGENERATION (rule 10d): no SAM surgery —
            # the rewritten scene text drives a full redraw with the
            # current image as reference. Force the redraw rung.
            if (patch.edit_magnitude or 0) < 0.8:
                patch.edit_magnitude = 0.8
            log.info("removal route: regenerate (mag %.2f, %d removes)",
                     patch.edit_magnitude,
                     sum(1 for op in patch.entity_ops()
                         if op.get("op") == "remove"))
        for op in patch.entity_ops():
            if op.get("op") != "remove":
                continue
            # regenerate route: the cuts STILL happen (marked regen) —
            # an approved empty draft once regrew furniture at promote
            # because the conditioning still contained it. Removing
            # content from the conditioning generalizes to any large
            # scene rewrite; the redraw then repaints freely.
            if region is not None and not region_used_for_delete:
                # "remove this/that" with a live selection: the mask IS
                # the target — exact pixels, no name-based re-detection
                region_used_for_delete = True
                # the op names WHAT is removed; the region only supplies
                # the exact pixels. Naming the op after the REGION once
                # produced the impossible requirement "no floor area is
                # present any more" — the judge failed a good edit and
                # rolled it back
                op_name = str(op.get("name") or op.get("id")
                              or region["name"]).replace("_", " ")
                move_ops.append({
                    "id": op.get("id") or "selection",
                    "name": op_name, "delete": True, "all": False,
                    "mask_rle": region["mask_rle"],
                    "mask_size": region["mask_size"],
                    "from_box": {"x": 0.5, "y": 0.5, "w": 1.0, "h": 1.0}})
                continue
            ent = ent_by_id.get(op.get("id"))
            name = ((ent.name if ent is not None else None)
                    or op.get("name") or op.get("id") or "")
            base = re.sub(r"[_\s]*\d+$", "", name).replace("_", " ").strip()
            # positional modifiers are for HUMANS, not for SAM: 'plant
            # left/center/right' are the same open-vocabulary query and
            # once segmented the same plants three times with different
            # results while the un-modified sofa went uncut
            base = re.sub(
                r"\b(left|right|center|centre|middle|front|back|upper|"
                r"lower|near|far|first|second|third)\b", " ", base)
            base = re.sub(r"\s+", " ", base).strip()
            base = base or name.replace("_", " ")
            bx = op.get("box") or {}
            dedup_key = (base.lower(),
                         round(float(bx.get("x", -1)), 2),
                         round(float(bx.get("y", -1)), 2))
            # pointed ops (person_1, person_2) share a base but carry
            # DISTINCT boxes — dedup on (name, box), not name alone
            if not base or dedup_key in seen_names:
                continue
            seen_names.add(dedup_key)
            dop = {"id": op.get("id") or base, "name": base,
                   "delete": True, "all": True,
                   "from_box": {"x": 0.5, "y": 0.5, "w": 1.0, "h": 1.0}}
            if op.get("box"):
                # the parser POINTED at this exact instance: targeted
                # cut via box overlap, never open-vocabulary delete-all
                dop["from_box"] = op["box"]
                dop["all"] = False
            if patch.preserve:
                # "remove X, keep Y": the kept things shield their SAM
                # instances from the category cut
                dop["keep"] = list(patch.preserve[:3])
                kbs = getattr(patch, "preserve_boxes", None) or []
                if kbs:
                    dop["keep_boxes"] = [
                        {"x": b.x, "y": b.y, "w": b.w, "h": b.h}
                        for b in kbs[:3]]
            if regenerate:
                dop["regen"] = True
            if ent is not None:
                dop["from_box"] = {"x": ent.box.x, "y": ent.box.y,
                                   "w": ent.box.w, "h": ent.box.h}
                dop["all"] = False      # a tracked entity is ONE instance
            move_ops.append(dop)
        # SPOKEN MOVES execute GEOMETRICALLY, exactly like mouse drags:
        # a positional command now produces a real box delta (rule 4c),
        # and prose alone provably cannot move an object ("flush against
        # the right wall" survived a draft fail, an aimed retry, AND a
        # restage — the chair never left the center; the composite
        # translate is the machinery that actually works). The spoken
        # instruction still rides along, so the repair render blends
        # light/shadow and the judge verifies intent, not just position.
        _vertical_talk = re.search(
            r"\b(up|down|raise|lower|lift|higher|lower|ceiling|"
            r"mount|hang|float|wall-mount)\b", directive, re.I)
        for mv in extract_move_ops(patch.entity_ops(), state.entities):
            ent = ent_by_id.get(mv["id"])
            mv["name"] = ((ent.name if ent is not None else None)
                          or mv["id"]).replace("_", " ")
            if not _vertical_talk:
                mv["floor_lock"] = True
            move_ops.append(mv)

        # removal net #2 — derive deletes from the FIELD DIFF: the LLM
        # sometimes expresses a removal purely as a field rewrite
        # ("street with palm trees" -> "street") with no remove ops at
        # all, and pixels never depend on model op-choice roulette. Any
        # phrase dropped from a field on a below-redraw edit becomes a
        # SAM delete target; if SAM finds no such object the composite
        # skips it, so non-object rephrasings are harmless. Full-redraw
        # edits (>=0.8: relocations, reframes) regenerate from the new
        # prompt anyway and are excluded.
        dwords = {w.strip(".,!?\"'").lower() for w in directive.split()}
        if patch.edit_magnitude < 0.8 and (dwords & REMOVAL_VERBS):
            # the field-diff net exists to catch removals the model
            # expressed only as a field rewrite — it must NOT fire on
            # rephrasings ("logo on the right" -> "logo ending in the
            # pool" once manufactured a delete op for an object called
            # "right", hijacking the route and the judge's checklist)
            for f in changed:
                # only OBJECT-BEARING fields: a palette rewrite dropping
                # "greens" is a style change, not a removal
                if f not in ("subject", "setting"):
                    continue
                for phrase in dropped_phrases(getattr(state, f, "") or "",
                                              getattr(new_state, f, "")
                                              or ""):
                    if phrase.lower() in seen_names:
                        continue
                    seen_names.add(phrase.lower())
                    move_ops.append({
                        "id": phrase.replace(" ", "_"), "name": phrase,
                        "delete": True, "all": True,
                        "from_box": {"x": 0.5, "y": 0.5,
                                     "w": 1.0, "h": 1.0}})
        instruction = self._pick_instruction(
            patch, state, prompt,
            skip_move_ids=frozenset(m["id"] for m in move_ops),
            directive=directive, has_moves=bool(move_ops))
        # geometry-executed edits build their own renderer-side repair/
        # outpaint instruction — but ONLY when the edit is purely
        # geometric. A mixed patch ("remove the trees and make it
        # sunset") keeps the spoken instruction; the composite still cuts
        # the pixels and the renderer appends the geometry clauses.
        delete_ops = [op for op in move_ops if op.get("delete")]
        if delete_ops:
            expected = set(stale_fields_for(
                state, [op.get("name") or "" for op in delete_ops]))
            extra_fields = [f for f in changed
                            if not f.startswith("entity:")
                            and f not in expected]
            extra_ops = [op for op in patch.entity_ops()
                         if op.get("op") != "remove"]
            if not extra_fields and not extra_ops:
                instruction = None  # delete-pure: proven repair path
                # the user's literal words remain a judge requirement
                # even with no instruction (see _draft_policy)
                object.__setattr__(patch, "verify_intent", directive)
                # a delete render must stay in the low-strength MERGE
                # regime: at >=0.7 nothing pins untouched regions to the
                # conditioning and the repaint once rebuilt the entire
                # room around a lamp removal
                if patch.edit_magnitude > 0.4:
                    log.info("delete-pure: magnitude %.1f capped to 0.4",
                             patch.edit_magnitude)
                    patch.edit_magnitude = 0.4
        self._edits_since_refresh += 1
        # scene-text reconciliation for spoken deletes: when the parse
        # left subject/setting asserting removed content, the same
        # background repair the mouse path uses rewrites it (the voice
        # path previously had NO equivalent, so stale prompts kept
        # regenerating deleted furniture on later strong renders)
        _del_labels = [str(op.get("name") or op.get("id") or "")
                       .replace("_", " ")
                       for op in move_ops if op.get("delete")]
        _del_labels = [x for x in _del_labels if x]
        if _del_labels and not any(
                f in changed for f in ("subject", "setting")):
            from ..scene.edits import stale_fields_for as _sff
            _stale = _sff(new_state, _del_labels) or ["subject", "setting"]
            asyncio.ensure_future(self._reconcile_removed(
                entry["commit_id"], _del_labels, _stale, t))

        self.bus.publish("patch_applied", {
            "commit_id": entry["commit_id"],
            "kind": "edit",
            "utterance": directive,
            "patch": patch.field_patch(),
            "entity_edits": patch.entity_ops(),
            "changed_fields": changed,
            "old_values": {f: getattr(state, f) for f in changed
                           if not f.startswith("entity:")},
            "edit_magnitude": patch.edit_magnitude,
            "reasoning": patch.reasoning,
            "latency_ms": latency_ms,
            "retries": retries,
            "tone": tone,
        }, t)
        self.bus.publish("state_snapshot", {
            "commit_id": entry["commit_id"],
            "state": new_state.model_dump(),
        }, t)
        self.bus.publish("prompt_updated", {
            "commit_id": entry["commit_id"], "prompt": prompt,
            "provisional": False,
        }, t)
        if (instruction is not None or move_ops
                or self._frame_count == 0):
            # edit-native: the patch becomes an image EDIT of the current
            # canvas; box deltas travel as move_ops and zooms as reframe
            # (geometry in code, not prose); a patch that changed nothing
            # visual renders nothing
            self.bus.publish("render_target", {
                "prompt": prompt,
                "strength": patch.edit_magnitude,
                "source": "commit",
                "generation": self.live.generation,
                "entities": self._sweep_entities(new_state),
                "edit_instruction": instruction,
                "lane": "committed",
                "base_rev": base_rev,
                "commit_rev": entry["commit_id"],
                "snapshot": True,
                "move_ops": move_ops or None,
                "camera": camera,
                "region": (region if (region is not None and not move_ops
                                      and instruction is not None)
                           else None),
                "preserve": (patch.preserve[:5]
                             if (patch.preserve
                                 and "setting" in changed
                                 and "subject" not in changed
                                 and (patch.edit_magnitude or 0) >= 0.75)
                             else None),
                "draft": self._strategy_plan(
                    patch, changed, move_ops, region, camera, new_state,
                    instruction, ladder=self._last_ladder),
            }, t)

    def set_region(self, data: dict | None) -> None:
        """The user's current canvas selection (lasso/click). While set,
        spoken edits render ONLY inside this mask — enforcement is the
        renderer's free-mask merge, not prompt hope."""
        if data and data.get("mask_rle") and data.get("mask_size"):
            name = str(data.get("name") or "selected area")
            # a fresh ink-circle selection carries no name through the
            # SAM round-trip — restore the decider's identification so
            # the parser knows what "this" IS
            sel = getattr(self, "_ink_sel", None)
            if sel and name in ("selected area", "None"):
                import time as _time
                if _time.monotonic() - sel.get("ts", 0) < 20.0:
                    name = sel["name"]
                self._ink_sel = None
            self.active_region = {
                "mask_rle": data["mask_rle"],
                "mask_size": data["mask_size"],
                "name": name,
            }
            self._ink_busy = False
            log.info("active region set (%s)", self.active_region["name"])
        else:
            if self.active_region is not None:
                log.info("active region cleared")
            self.active_region = None

    @staticmethod
    def _partner_box(contact, target_name: str, new_state) -> list | None:
        """Percent box of the contact partner (the party that is NOT the
        stencil target), matched by word overlap against scene entities.
        None when the partner isn't a tracked entity — the renderer then
        drops the stencil entirely on climb, which is always safe."""
        tw = set(target_name.lower().split())
        for name in (contact.a, contact.b):
            nw = set(name.lower().split())
            if tw & nw:
                continue                     # this party IS the target
            for ent in getattr(new_state, "entities", []) or []:
                ew = set((ent.name or ent.id).lower().split())
                if ew & nw and ent.box is not None:
                    b = ent.box
                    return [max(0, int(b.x * 100) - 8),
                            max(0, int(b.y * 100) - 8),
                            min(100, int((b.x + b.w) * 100) + 8),
                            min(100, int((b.y + b.h) * 100) + 8)]
        return None

    @staticmethod
    def _entity_target(patch, new_state) -> dict | None:
        """Deterministic stencil target when the parser leaves target
        null: the box the patch itself committed to (entity add/update
        ops carry one), else the edited entity's box from scene state.
        Padded ~8%/side — the stencil is a coarse fence, SAM snaps the
        precise mask downstream."""
        def _pct(name: str, bx) -> dict:
            x0 = max(0, int(bx.x * 100) - 8)
            y0 = max(0, int(bx.y * 100) - 8)
            x1 = min(100, int((bx.x + bx.w) * 100) + 8)
            y1 = min(100, int((bx.y + bx.h) * 100) + 8)
            if x1 <= x0 or y1 <= y0:
                return None
            return {"n": name[:40], "b": [x0, y0, x1, y1]}

        boxes = []
        for op in getattr(patch, "entity_edits", []) or []:
            if op.op in ("add", "update"):
                if op.box is not None:
                    boxes.append((op.name or op.id, op.box))
                else:
                    for ent in getattr(new_state, "entities", []) or []:
                        if ent.id == op.id and ent.box is not None:
                            boxes.append((ent.name or ent.id, ent.box))
                            break
        if not boxes:
            return None
        if len(boxes) == 1:
            return _pct(*boxes[0])
        # several entities edited: union fence around all of them
        x0 = min(b.x for _, b in boxes)
        y0 = min(b.y for _, b in boxes)
        x1 = max(b.x + b.w for _, b in boxes)
        y1 = max(b.y + b.h for _, b in boxes)

        class _U:                       # duck-typed box for _pct
            pass
        u = _U(); u.x, u.y, u.w, u.h = x0, y0, x1 - x0, y1 - y0
        return _pct(boxes[0][0], u)

    def _strategy_plan(self, patch, changed, move_ops, region, camera,
                       new_state, instruction, ladder=None) -> dict:
        """Base routing (today's _draft_policy) enriched per the active
        strategy. Absent keys = today's behavior; every enrichment is a
        pure addition the renderer may ignore."""
        base = self._draft_policy(patch, changed, move_ops, region,
                                  camera, new_state,
                                  instruction=instruction)
        st = self.strategy
        name = st.name
        if name == "switchboard":
            # route by the parser's difficulty signal; spatial-word
            # backstop forces the relational tier
            diff = getattr(patch, "edit_difficulty", "standard")
            name = ROUTER.get(diff, "surveyor")
            if getattr(patch, "contact", None) is not None:
                name = ROUTER["relational"] if name == "stencil" else name
            ex = STRATEGIES[name]
        else:
            ex = st
        # constraint ladder rides on EVERY mode (standard included): the
        # magnitude rating is only the STARTING rung; the renderer climbs
        # when the judge says the constraint itself blocked the edit.
        if ladder and base.get("policy") == "loop":
            base = dict(base)
            lad = dict(ladder)
            if (getattr(patch, "contact", None) is not None
                    or getattr(patch, "edit_difficulty", "")
                    == "relational"):
                # relational edits restructure surroundings by definition
                lad["start_rung"] = max(int(lad.get("start_rung", 0)), 1)
            base["ladder"] = lad
            base["max_climb"] = ex.max_climb
        # contact rides on EVERY mode: the renderer zoom-judges the
        # relation on passing drafts (the full-frame judge is measurably
        # lenient on spatial relations at draft resolution)
        ct = getattr(patch, "contact", None)
        if ct is not None and base.get("policy") == "loop":
            base = dict(base)
            base["contact"] = {"a": ct.a, "b": ct.b}
        if st.name == "standard":
            return base
        base = dict(base)
        base["strategy"] = name
        base["budget_ms"] = ex.budget_ms
        base["max_iters"] = ex.max_iters
        base["n_drafts"] = ex.n_drafts
        if ex.grounded_judge:
            base["judge_profile"] = "grounded"
        # stencil target: parser box, only for scoped classes, never
        # for global edits / camera / composites / regions
        tgt = getattr(patch, "target", None)
        if (ex.use_target and not camera and not move_ops
                and region is None and base.get("policy") == "loop"
                and base.get("class") in ("add", "attribute", "replace")):
            if tgt is not None:
                b = list(tgt.b)
                area = (b[2] - b[0]) * (b[3] - b[1])
                # sanity fence: reject unit-confused boxes (0-1 fractions
                # -> ~0% area) and near-full-frame boxes (no scoping value)
                if b[2] > b[0] and b[3] > b[1] and 100 <= area <= 9000:
                    base["target"] = {"n": tgt.n, "b": b}
            if "target" not in base:
                fb = self._entity_target(patch, new_state)
                if fb is not None:
                    base["target"] = fb
        # rung-1 stencil widening: for a contact edit the fence must
        # include BOTH parties plus the ground between them — sinking a
        # hot tub to pool level is a change to the deck, not the tub
        if ct is not None and base.get("target"):
            pb = self._partner_box(ct, base["target"]["n"], new_state)
            if pb is not None:
                t = base["target"]["b"]
                base["widen"] = {
                    "n": f"{ct.a} + {ct.b}",
                    "b": [min(t[0], pb[0]), min(t[1], pb[1]),
                          max(t[2], pb[2]), max(t[3], pb[3])]}
        g = (getattr(patch, "grounding", None) or "").strip()
        if g:
            base["grounding"] = g[:220]
        log.info("strategy plan: %s (routed from %s) target=%s contact=%s "
                 "grounding=%s class=%s", name, st.name,
                 base.get("target"), base.get("contact"),
                 "yes" if base.get("grounding") else "no",
                 base.get("class"))
        return base

    @staticmethod
    def _draft_policy(patch, changed, move_ops, region, camera,
                      new_state, instruction=None) -> dict:
        """Deterministic routing + requirements: EVERY committed edit is
        LLM-verified. Draftable classes (adds, subject/setting swaps,
        camera, attribute/style) pre-verify via the draft loop; pixel-
        mechanical edits (removals, moves, region edits) post-verify on
        the committed result with one corrective retry. Attribute edits
        draft at 512 — 256 was the judge's measured blind spot for
        subtle tints."""
        from ..scene.edits import _pos_phrase

        # pixel-mechanical paths: verify the RESULT
        if move_ops:
            has_pos_moves = any((not op.get("delete"))
                                and op.get("to_box")
                                for op in move_ops)
            # every deletion is ONE aggregated requirement — a [:3]/[:2]
            # truncation here once verified a 6-object furniture removal
            # as "no coffee table, no rug" and honestly PASSED a canvas
            # still full of furniture
            reqs = []
            del_names = [str(op.get("name") or op.get("id") or "object")
                         .replace("_", " ")
                         for op in move_ops if op.get("delete")]
            if del_names:
                reqs.append("None of these are present anywhere in the "
                            "scene any more: " + ", ".join(del_names[:8]))
            for op in move_ops[:4]:
                if op.get("delete"):
                    continue
                name = str(op.get("name") or op.get("id")
                           or "object").replace("_", " ")
                if op.get("to_box"):
                    reqs.append(f"A {name} is present "
                                f"{_pos_phrase(op['to_box'])}")
            reqs = reqs[:3]
            if instruction:
                # mixed edits (geometry + spoken intent): the intent is
                # a requirement too, or the judge scores only the
                # mechanical half
                reqs.append("The requested change is clearly applied: "
                            + str(instruction)[:200])
            elif getattr(patch, "verify_intent", None):
                # delete-pure edits null the instruction, but the user's
                # literal request must still be a question the judge is
                # asked ("Remove all the furniture" was never checked)
                reqs.append("The user's request is clearly satisfied: "
                            + str(patch.verify_intent)[:200])
            if has_pos_moves:
                # 2D sprite translation has no depth sense: a dresser
                # once landed in front of the bed. Plausibility is part
                # of every move's contract, not this scenario's.
                reqs.append("Every moved object is seated plausibly "
                            "(on the floor / against surfaces, correct "
                            "perspective, not overlapping or blocking "
                            "other objects)")
            reqs.append("Apart from those changes, the framing and "
                        "everything else in the scene are unchanged")
            return {"policy": "verify", "class": "composite",
                    "requirements": reqs, "preserve_last": True,
                    "color_guard": True}
        if region is not None:
            rname = region.get("name") or "the selection"
            req = (f"Inside the selected area ({rname}), the requested "
                   f"change is clearly applied: "
                   f"{instruction or 'the edit'}")
            return {"policy": "verify", "class": "region",
                    "requirements": [req, "Outside the selected area, "
                                     "nothing in the scene changed"],
                    "preserve_last": True, "color_guard": True}
        if instruction is None:
            return {"policy": "direct"}

        adds = [op for op in patch.entity_ops() if op.get("op") == "add"]
        sem = [f for f in changed if not f.startswith("entity:")]
        reqs: list[str] = []
        removes = [str(op.get("name") or op.get("id") or "")
                   .replace("_", " ")
                   for op in patch.entity_ops()
                   if op.get("op") == "remove"]
        removes = [r for r in removes if r]
        if removes:
            reqs.append("None of these are present anywhere in the "
                        "scene any more: " + ", ".join(removes[:10]))
        if camera:
            cls = "camera"
            comp = (new_state.composition or "a new framing").strip()
            reqs.append(f"The framing/viewpoint is now: {comp}")
        elif adds:
            cls = "add"
            for op in adds[:2]:
                name = (op.get("name") or op.get("id") or "object")
                name = str(name).replace("_", " ")
                box = op.get("box") or {}
                pos = _pos_phrase(box) if box else ""
                reqs.append(f"A {name} is now present"
                            + (f" {pos}" if pos else "")
                            + " in the scene")
                # adds repaint locally and once grotesquely distorted a
                # neighboring chair while adding a lamp: intactness of
                # everything already present is part of EVERY add
                reqs.append("Every object that was already in the "
                            "scene is still present, unchanged and "
                            "undistorted")
                # the token merge can average a tentative add halfway
                # back into the old pixels (a hot tub once shipped as a
                # translucent ghost in the pool): solidity is part of
                # the contract, ghosts fail and escalate
                reqs.append(f"The {name} is SOLID and fully rendered — "
                            f"not faint, translucent, ghost-like, or "
                            f"half-blended into what was behind it")
                desc = str(op.get("description") or "").strip()
                # the description IS the contract — a checklist that only
                # names the object lets any couch pass for "a supreme red
                # super fuzzy couch with a giant supreme logo" (observed)
                if desc and desc.lower() != name.lower():
                    reqs.append(f"The {name} matches this description: "
                                f"{desc}")
        elif "subject" in sem:
            cls = "replace"
            reqs.append(f"The main subject is now: "
                        f"{new_state.subject.strip()}")
            if new_state.setting.strip():
                # roots and subject swaps must satisfy EVERY element of
                # the description — "a standing desk" once passed as a
                # normal desk because only the summary was checked
                reqs.append("Every specific element of this "
                            "description is clearly present: "
                            f"{new_state.subject.strip()}, "
                            f"{new_state.setting.strip()}")
        elif "setting" in sem:
            cls = "setting"
            reqs.append(f"The scene/setting is now: "
                        f"{new_state.setting.strip()}")
        else:
            # attribute/style class: lighting, palette, mood, medium,
            # entity updates — draft at 512 where the judge can see it
            cls = "attribute"
            labels = {"lighting": "The lighting is now",
                      "palette": "The overall color palette is now",
                      "mood": "The mood/atmosphere is now",
                      "medium": "The image style/medium is now"}
            for f in sem:
                if f in labels:
                    v = str(getattr(new_state, f, "")).strip()
                    if v:
                        reqs.append(f"{labels[f]}: {v}")
            if not reqs:
                reqs.append("The requested change is clearly applied: "
                            + str(instruction))
            reqs = reqs[:3]
            reqs.append("Apart from that change, the framing, objects, "
                        "and scene content are unchanged")
            return {"policy": "loop", "class": cls, "draft_size": 512,
                    "requirements": reqs, "preserve_last": True}
        # class-appropriate preservation: what must SURVIVE the edit.
        # An add must not reframe or restyle; a setting swap must keep
        # the subject; a camera move must keep the scene's contents; a
        # subject replacement must keep the setting. Generalizes "asked
        # for a floaty, got a zoom too" to every class.
        reqs = reqs[:3]
        cguard = False
        if cls == "add":
            cguard = True
            reqs.append("Apart from the added object(s), the framing, "
                        "background, and other objects are unchanged")
        elif cls == "setting":
            subj = new_state.subject.strip()
            reqs.append(("The main subject"
                         + (f" ({subj})" if subj else "")
                         + " is still present and unchanged"))
        elif cls == "camera":
            reqs.append("The scene still contains the same subjects and "
                        "objects — only the viewpoint changed")
        elif cls == "replace":
            setting = new_state.setting.strip()
            reqs.append(("The setting"
                         + (f" ({setting})" if setting else "")
                         + " and framing are unchanged"))
        return {"policy": "loop", "class": cls,
                "requirements": reqs, "preserve_last": True,
                "color_guard": cguard}

    # ------------------------------------------------------------------
    # mouse layer: warm object cache (LLM-enumerated, SAM pre-cut)

    # ------------------------------------------------------------------
    # mouse layer: drag/resize releases from the browser

    @staticmethod
    def _sweep_entities(state) -> list[dict]:
        """Entity list for render targets, with the SUBJECT included as a
        pseudo-entity: most scenes track the main thing in the subject
        FIELD, not as an entity, and it's the #1 thing users grab. The
        renderer's mask sweep segments whatever appears here."""
        out = [e.model_dump() for e in state.entities]
        if state.subject.strip():
            out.append({"id": "_subject", "name": state.subject.strip(),
                        "box": {"x": 0.5, "y": 0.5, "w": 0.6, "h": 0.6}})
        return out

    @staticmethod
    def _mouse_reqs(move_ops: list) -> list:
        """Judge requirements from exact mouse/replay geometry."""
        from ..scene.edits import _pos_phrase

        reqs = []
        # deletions aggregate into ONE requirement so none is silently
        # dropped: a [:3] cap once left a sofa unchecked and the judge
        # honestly passed a failed remove-all-furniture edit
        del_names = [str(op.get("name") or op.get("id") or "object")
                     .replace("_", " ")
                     for op in (move_ops or []) if op.get("delete")]
        if del_names:
            reqs.append("None of these are present in the scene any "
                        "more: " + ", ".join(del_names[:8]))
        for op in (move_ops or [])[:4]:
            if op.get("delete"):
                continue
            nm = str(op.get("name") or op.get("id")
                     or "object").replace("_", " ")
            if op.get("to_box"):
                reqs.append(f"A {nm} is present "
                            f"{_pos_phrase(op['to_box'])}")
        reqs.append("Apart from those changes, everything else in the "
                    "scene is unchanged")
        return reqs[:4]

    @staticmethod
    def _mouse_phrase(name: str, from_box: dict, to_box: dict) -> str:
        """Templated transcript so the ledger reads like a spoken edit and
        'undo that' / 'put the dog back' work verbatim."""
        from ..scene.edits import _pos_phrase

        for art in ("a ", "an ", "the "):
            if name.lower().startswith(art):
                name = name[len(art):]
        parts = []
        if (abs(to_box["x"] - from_box.get("x", to_box["x"])) > 0.01
                or abs(to_box["y"] - from_box.get("y", to_box["y"])) > 0.01):
            parts.append(f"moved the {name} {_pos_phrase(to_box)}")
        r = to_box["w"] / max(from_box.get("w", to_box["w"]), 1e-6)
        if r > 1.15:
            parts.append(f"made the {name} bigger")
        elif r < 0.87:
            parts.append(f"made the {name} smaller")
        return " and ".join(parts) or f"adjusted the {name}"

    def mouse_commit(self, data: dict, t: int) -> None:
        """A drag or resize release becomes a first-class commit: same
        timeline, ledger, keyframe, and FIFO renderer path as a spoken
        edit. The mouse is a second reflex layer — zero LLM calls; the
        exact geometry travels as move_ops and is persisted in the entry
        (patch.mouse_ops) so history replay re-executes the ops instead of
        re-deriving it from the templated sentence."""
        # one message may carry ONE move or a marquee GROUP of moves; the
        # whole group is a single commit, ledger entry, and render
        moves_in = data.get("moves") or [data]
        state = self.timeline.current_state
        move_ops: list[dict] = []
        ops: list[dict] = []
        names: list[str] = []
        n_deletes = 0
        for m in moves_in[:8]:
            eid = m.get("entity_id") or "_click"
            to_box = m.get("to_box") or {}
            from_box = m.get("from_box") or {}
            has_mask = bool(m.get("mask_rle") and m.get("mask_size"))
            is_delete = bool(m.get("delete"))
            if not is_delete and not all(k in to_box
                                         for k in ("x", "y", "w", "h")):
                continue
            ent = next((e for e in state.entities if e.id == eid), None)
            if ent is None and from_box:
                # identity requires SHAPE agreement, never mere containment:
                # a grab landing inside a big entity's box is usually a
                # different, smaller thing in front of it (a cold click on
                # a tree once inherited the car subject's identity and the
                # ledger claimed the car was deleted). Inherit only when
                # the boxes genuinely coincide (IoU >= 0.5).
                best, best_iou = None, 0.0
                for e in state.entities:
                    iou = _box_iou(from_box, {"x": e.box.x, "y": e.box.y,
                                              "w": e.box.w, "h": e.box.h})
                    if iou > best_iou:
                        best, best_iou = e, iou
                if best is not None and best_iou >= 0.5:
                    ent, eid = best, best.id
            if ent is None and not has_mask and not m.get("name"):
                # truly anonymous move (no entity, no mask, no name)
                # has nothing the renderer can match. A NAMED move is
                # valid — the composite SAM-matches the name inside
                # from_box (arrow moves on pixel-only objects died
                # here: 'move the hot tub' arrow was read then dropped)
                continue
            # never guess a name (the old subject fallback labeled every
            # anonymous grab as the main subject); an unnamed selection
            # stays "object" — the mask, not the name, drives the edit
            name = (m.get("name") or (ent.name if ent is not None else "")
                    or "object")
            if not from_box and ent is not None:
                from_box = {"x": ent.box.x, "y": ent.box.y,
                            "w": ent.box.w, "h": ent.box.h}
            # the client's exact mask (what the user saw highlighted)
            # beats any re-detection; name doubles as the SAM text prompt
            op = {"id": eid, "name": name, "from_box": from_box}
            if is_delete:
                op["delete"] = True
                n_deletes += 1
            else:
                to_box = {k: max(0.0, min(1.0, float(to_box[k])))
                          for k in ("x", "y", "w", "h")}
                # dead-zone: sub-perceptual pointer motion is input noise,
                # not an edit — a 1px twitch on release must never commit
                # a render (mirrors the client's guard, protects any client)
                dx = abs(to_box["x"] - from_box.get("x", to_box["x"]))
                dy = abs(to_box["y"] - from_box.get("y", to_box["y"]))
                sw = abs(to_box["w"] / max(from_box.get("w", to_box["w"]),
                                           1e-6) - 1)
                sh = abs(to_box["h"] / max(from_box.get("h", to_box["h"]),
                                           1e-6) - 1)
                if dx + dy < 0.004 and sw < 0.02 and sh < 0.02:
                    continue
                op["to_box"] = to_box
            if has_mask:
                op["mask_rle"] = m["mask_rle"]
                op["mask_size"] = m["mask_size"]
            move_ops.append(op)
            names.append(name)
            if ent is not None:
                ops.append({"op": "remove", "id": eid, "name": None,
                            "description": None, "box": None}
                           if is_delete else
                           {"op": "update", "id": eid, "name": None,
                            "description": None, "box": to_box})
        if not move_ops:
            log.info("mouse_commit ignored (no valid moves in %r)",
                     [m.get("entity_id") for m in moves_in])
            return
        new_state = state
        if ops:
            new_state, _ = state.apply_entity_edits(ops)
        clean = []
        for n in names:
            for art in ("a ", "an ", "the "):
                if n.lower().startswith(art):
                    n = n[len(art):]
            clean.append(n)
        if len(move_ops) == 1 and not n_deletes:
            directive = self._mouse_phrase(
                names[0], move_ops[0]["from_box"], move_ops[0]["to_box"])
        else:
            joined = (f"the {clean[0]}" if len(clean) == 1 else
                      ", ".join(f"the {n}" for n in clean[:-1])
                      + f" and the {clean[-1]}")
            verb = ("removed" if n_deletes == len(move_ops)
                    else "moved" if not n_deletes else "adjusted")
            directive = (f"{verb} {joined}"
                         + (" together" if len(clean) > 1
                            and verb == "moved" else ""))
        base_rev = (self.timeline.entries[-1]["commit_id"]
                    if self.timeline.entries else 0)
        entry = self.timeline.append(
            kind="edit", state_after=new_state, utterance=directive,
            tone={}, patch={"entity_edits": ops, "mouse_ops": move_ops},
            edit_magnitude=0.3)
        self.live.canonical_changed()
        prompt = render_prompt(new_state)
        log.info("mouse commit #%d: %s", entry["commit_id"], directive)

        self.bus.publish("patch_applied", {
            "commit_id": entry["commit_id"], "kind": "edit",
            "utterance": directive, "patch": {},
            "entity_edits": ops,
            "changed_fields": [f"entity:{op['id']}" for op in move_ops],
            "old_values": {}, "edit_magnitude": 0.3,
            "reasoning": "mouse drag", "latency_ms": 0, "retries": 0,
            "tone": {},
        }, t)
        self.bus.publish("state_snapshot", {
            "commit_id": entry["commit_id"],
            "state": new_state.model_dump()}, t)
        self.bus.publish("prompt_updated", {
            "commit_id": entry["commit_id"], "prompt": prompt,
            "provisional": False}, t)
        mreqs = self._mouse_reqs(move_ops)[:-1]
        self.bus.publish("render_target", {
            "prompt": prompt,
            "strength": 0.3,
            "source": "mouse",
            "generation": self.live.generation,
            "entities": self._sweep_entities(new_state),
            "edit_instruction": None,       # moves repair with the fixed prompt
            "lane": "committed",
            "base_rev": base_rev,
            "commit_rev": entry["commit_id"],
            "snapshot": True,
            "move_ops": move_ops,
            "draft": {"policy": "verify", "class": "mouse",
                      "requirements": mreqs[:3] + [
                          "Apart from those changes, everything else in "
                          "the scene is unchanged"],
                      "preserve_last": True, "color_guard": True},
        }, t)
        # consistency guard, mouse edition: a delete changed PIXELS but the
        # scene TEXT may still assert the thing ("with palm trees") — every
        # later render would contradict the delete and a big edit would
        # resurrect it. One background LLM call rewrites the stale field(s).
        deleted = [op["name"] for op in move_ops
                   if op.get("delete") and op.get("name")
                   and op["name"] != "object"]
        if deleted:
            fields = stale_fields_for(new_state, deleted)
            if fields:
                asyncio.ensure_future(self._reconcile_removed(
                    entry["commit_id"], deleted, fields, t))

    async def _reconcile_removed(self, commit_id: int, labels: list[str],
                                 fields: list[str], t: int) -> None:
        """Amend a delete-commit's scene text so it stops asserting the
        removed thing. State-only: the pixels are already correct, so the
        amendment republishes state/prompt for the SAME commit and renders
        nothing. Dropped if the timeline head has already moved on."""
        things = ", ".join(dict.fromkeys(labels))
        directive = (f"(the {things} was just erased from the picture by "
                     f"direct manipulation. Rewrite the "
                     f"{', '.join(fields)} field(s) so the scene text no "
                     f"longer mentions it. Change nothing else; do not "
                     f"add, move, or remove any objects.)")
        state = self.timeline.current_state
        try:
            patch, _, _ = await self._call(
                state, directive, {}, purpose="reconcile",
                use_history=False, force_vision=False, t=t)
        except EditorError as e:
            log.warning("reconcile after delete failed (stale text kept): %s",
                        e)
            return
        entry = self.timeline.entries[-1] if self.timeline.entries else None
        if entry is None or entry["commit_id"] != commit_id:
            log.info("reconcile: head moved past #%d — dropped", commit_id)
            return
        field_patch = {f: v for f, v in patch.field_patch().items()
                       if f in fields}
        if not field_patch:
            return
        new_state, changed = state.apply_patch(field_patch)
        entry["state_after"] = new_state.model_dump()
        self.live.canonical_changed()
        self.bus.publish("state_snapshot", {
            "commit_id": commit_id, "state": new_state.model_dump()}, t)
        self.bus.publish("prompt_updated", {
            "commit_id": commit_id, "prompt": render_prompt(new_state),
            "provisional": False}, t)
        log.info("reconciled stale scene text after delete: %s", changed)

    # ------------------------------------------------------------------
    # reflex promotion + reverts (unchanged semantics from v2)

    def _do_reflex_revert(self, t: int) -> None:
        entries = self.timeline.entries
        if not entries:
            return
        last = entries[-1]
        if last["kind"] == "revert" and last.get("revert_target"):
            idx = next((i for i, e in enumerate(entries)
                        if e["commit_id"] == last["revert_target"]), None)
            target_idx = idx - 1 if idx is not None else None
        else:
            target_idx = len(entries) - 2
        if target_idx is None or target_idx < 0:
            from ..scene.state import SceneState
            entry = self.timeline.append(kind="revert",
                                         state_after=SceneState(),
                                         revert_target=0)
            self.live.canonical_changed()
            new_state = self.timeline.current_state
            self.bus.publish("state_snapshot", {
                "commit_id": entry["commit_id"],
                "state": new_state.model_dump()}, t)
            self.bus.publish("prompt_updated", {
                "commit_id": entry["commit_id"],
                "prompt": render_prompt(new_state),
                "provisional": False}, t)
            self.bus.publish("render_target", {
                "prompt": render_prompt(new_state), "strength": 0.8,
                "source": "commit", "generation": self.live.generation,
                "entities": [], "reset": True}, t)
        else:
            self.revert(entries[target_idx]["commit_id"], t)

    def _promote_reflex(self, directive: str, tone: dict, t: int) -> None:
        base_rev = (self.timeline.entries[-1]["commit_id"]
                    if self.timeline.entries else 0)
        old_state = self.timeline.current_state
        new_state = self.live.effective_state()
        reflex_log = list(self.live.reflex_log)
        promote_ops = [dict(o) for o in self.live.overlay_entity_ops]
        move_ops = extract_move_ops(promote_ops, old_state.entities)
        instruction = phrase_edit(dict(self.live.overlay), promote_ops,
                                  old_state.entities,
                                  skip_move_ids=frozenset(
                                      m["id"] for m in move_ops))
        entry = self.timeline.append(
            kind="edit", state_after=new_state, utterance=directive,
            tone=tone, patch={"reflex": reflex_log}, edit_magnitude=0.3)
        self.live.canonical_changed()
        prompt = render_prompt(new_state)
        changed = [f for f in ("subject", "setting", "composition", "lighting",
                               "palette", "mood", "medium")
                   if getattr(new_state, f) != getattr(old_state, f)]
        changed += [f"entity:{e.id}" for e in new_state.entities
                    if e.model_dump() not in
                    [x.model_dump() for x in old_state.entities]]
        self.bus.publish("patch_applied", {
            "commit_id": entry["commit_id"], "kind": "edit",
            "utterance": directive, "patch": {},
            "entity_edits": [], "changed_fields": changed,
            "old_values": {}, "edit_magnitude": 0.3,
            "reasoning": "reflex: " + "; ".join(reflex_log),
            "latency_ms": 0, "retries": 0, "tone": tone,
        }, t)
        self.bus.publish("state_snapshot", {
            "commit_id": entry["commit_id"], "state": new_state.model_dump(),
        }, t)
        self.bus.publish("prompt_updated", {
            "commit_id": entry["commit_id"], "prompt": prompt,
            "provisional": False,
        }, t)
        labels = {"lighting": "The lighting is now",
                  "palette": "The overall color palette is now",
                  "mood": "The mood/atmosphere is now",
                  "medium": "The image style/medium is now",
                  "subject": "The main subject is now",
                  "setting": "The scene/setting is now",
                  "composition": "The framing is now"}
        rreqs = []
        for f in changed:
            if f in labels:
                v = str(getattr(new_state, f, "")).strip()
                if v:
                    rreqs.append(f"{labels[f]}: {v}")
        if not rreqs:
            rreqs.append("The requested change is clearly applied: "
                         + (instruction or directive))
        rreqs = rreqs[:3]
        rreqs.append("Apart from those changes, the framing and scene "
                     "content are unchanged")
        self.bus.publish("render_target", {
            "prompt": prompt, "strength": 0.3, "source": "commit",
            "generation": self.live.generation,
            "entities": [e.model_dump() for e in new_state.entities],
            "edit_instruction": instruction,
            "lane": "committed",
            "base_rev": base_rev,
            "commit_rev": entry["commit_id"],
            "snapshot": True,
            "move_ops": move_ops or None,
            "draft": {"policy": "verify", "class": "reflex",
                      "requirements": rreqs, "preserve_last": True},
        }, t)
        log.info("reflex commit #%d: %s", entry["commit_id"], reflex_log)

    def undo(self, t: int) -> None:
        """Manual back button: one deterministic step back, zero LLM —
        the same walk the reflex 'undo' voice grammar uses. Restores the
        previous live frame from its keyframe (pixel-exact, instant);
        repeated presses keep walking further back."""
        self._do_reflex_revert(t)

    def redo(self, t: int) -> None:
        """Manual forward button: the exact inverse of undo's walk, zero
        LLM. Only meaningful while sitting on an undo chain — one press
        steps forward to the next originally-live frame (keyframe restore,
        pixel-exact). After a NEW edit, there is nothing to redo (standard
        redo semantics)."""
        entries = self.timeline.entries
        last = entries[-1] if entries else None
        if last is None or last["kind"] != "revert":
            self.bus.publish("error", {
                "where": "redo", "recoverable": True,
                "message": "nothing to go forward to"}, t)
            return
        target = last.get("revert_target") or 0
        if target == 0:
            nxt = entries[0] if entries else None
        else:
            idx = next((i for i, e in enumerate(entries)
                        if e["commit_id"] == target), None)
            nxt = (entries[idx + 1]
                   if idx is not None and idx + 1 < len(entries) else None)
        if nxt is None or nxt["kind"] != "edit":
            self.bus.publish("error", {
                "where": "redo", "recoverable": True,
                "message": "nothing to go forward to"}, t)
            return
        self.revert(nxt["commit_id"], t)

    def revert(self, commit_id: int, t_session_ms: int) -> None:
        old_state = self.timeline.current_state
        entry = self.timeline.revert_to(commit_id)
        if entry is None:
            self.bus.publish("error", {
                "where": "revert", "recoverable": True,
                "message": f"no commit {commit_id}",
            }, t_session_ms)
            return
        self.live.canonical_changed()
        self._verify_debt = True
        new_state = self.timeline.current_state
        changed = [f for f in new_state.model_fields
                   if getattr(new_state, f) != getattr(old_state, f)]
        self.bus.publish("patch_applied", {
            "commit_id": entry["commit_id"],
            "kind": "revert",
            "revert_target": commit_id,
            "utterance": f"revert to #{commit_id}",
            "patch": {}, "changed_fields": changed,
            "old_values": {f: getattr(old_state, f) for f in changed},
            "edit_magnitude": None, "reasoning": None,
            "latency_ms": 0, "retries": 0, "tone": {},
        }, t_session_ms)
        self.bus.publish("state_snapshot", {
            "commit_id": entry["commit_id"],
            "state": new_state.model_dump(),
        }, t_session_ms)
        self.bus.publish("prompt_updated", {
            "commit_id": entry["commit_id"],
            "prompt": render_prompt(new_state),
            "provisional": False,
        }, t_session_ms)
        self.bus.publish("render_target", {
            "prompt": render_prompt(new_state),
            "strength": 0.8,
            "source": "commit",
            "generation": self.live.generation,
            "entities": [e.model_dump() for e in new_state.entities],
            # undo = pixel-exact keyframe restore (0ms, no diffusion);
            # reset stays as the fallback when the keyframe was evicted
            "restore_rev": commit_id,
            "reset": True,
        }, t_session_ms)

    async def close(self) -> None:
        self._task.cancel()
        self._pace_task.cancel()
        self.bus.unsubscribe(self._queue)
