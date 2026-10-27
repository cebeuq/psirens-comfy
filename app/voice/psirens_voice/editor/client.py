"""HTTP client for the delta-editor LLM served by vLLM (OpenAI-compatible).

Structured output: newer vLLM accepts response_format={"type":"json_schema"},
older builds use extra_body={"guided_json": schema}. The working form is
probed once per SERVER (cached by base_url, so the startup warmup's discovery
carries into every session's clients) and only falls through to the other
form on an HTTP 4xx — a server that ANSWERED "I don't know that parameter".
Timeouts are load evidence, not shape evidence, and never trigger a second
probe (that pattern once doubled every failure's cost).

Calls stream via llm_http, so a timeout closes the connection and vLLM
aborts the generation server-side instead of finishing it into the void.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time

from ..llm_http import ChatHTTPError, ChatTimeout, stream_chat_blocking
from ..scene.edits import dropped_phrases, stale_fields_for
from ..scene.state import SceneState

# unambiguous removal verbs, any register — the TRIGGER for the removal-
# completeness retry (the LLM still does all semantic extraction). Kept
# tight on purpose: ambiguous words ("back", "off") caused false trips.
REMOVAL_VERBS = frozenset({"remove", "removes", "removed", "delete",
                           "deletes", "deleted", "erase", "erased", "rid"})
from .prompts import SYSTEM_PROMPT, build_user_message
from .schema import PATCH_JSON_SCHEMA, ScenePatch, patch_schema

log = logging.getLogger("psirens.editor")

# test/verification mode: on a speculative-cache hit, ALSO make the
# fresh HTTP call, log whether the cached and fresh responses agree,
# and use the FRESH one (behavior identical to no-cache production)
SPEC_VERIFY = os.environ.get("PSIRENS_SPEC_VERIFY") == "1"

# structured-output mode discovered per server URL — a server property, so
# it is shared by all client instances (sessions build their own clients)
_MODE_BY_URL: dict[str, str] = {}


class EditorError(Exception):
    pass


class EditorTimeout(EditorError):
    """Deadline exceeded or server unreachable — congestion, not a bad
    request. Callers with a fire-rate (drafts) should back off on this."""


class DeltaEditorClient:
    def __init__(self, base_url: str, model: str, temperature: float = 0.2,
                 max_tokens: int = 700, timeout_s: float = 2.0,
                 system_prompt: str | None = None, lenient: bool = False,
                 message_builder=None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        # lenient mode (fast-loop drafts): skip the minimality and subject
        # guards — drafts are disposable, superseded, and never enter history
        self.lenient = lenient
        # optional (state, tone_line, directive, unfinished) -> str override
        # so callers with their own system prompt (drafts) build user
        # messages in the format that prompt's examples teach
        self.message_builder = message_builder
        # speculative-parse response memo (see _chat): {body_sha: (ts, content)}
        self._memo: dict[str, tuple[float, str]] = {}
        self._prewarm_evt: asyncio.Event | None = None
        self._spec_parts: dict | None = None

    def _body(self, messages: list[dict], mode: str,
              schema: dict | None = None) -> dict:
        schema = schema or PATCH_JSON_SCHEMA
        body: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if mode == "response_format":
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "scene_patch", "schema": schema},
            }
        else:
            body["guided_json"] = schema
        return body

    # ---- speculative-parse memoization ------------------------------
    # The commit gate waits ~1.4s of silence after the last word before
    # committing; the parse can run DURING that dead time. Losslessness
    # is structural: the speculative response is stored keyed by the
    # sha256 of the exact request body (messages incl. any image bytes +
    # schema), and reused only when the commit-time request is byte-
    # identical — so the cached content is literally the response to the
    # same request, never a "close enough" one. Any input drift (new
    # words, new frame, tone change, state change) is a different key
    # and falls through to a normal HTTP call.

    @staticmethod
    def _memo_key(messages, schema) -> str:
        return hashlib.sha256(json.dumps(
            [messages, schema], sort_keys=True,
            separators=(",", ":")).encode()).hexdigest()

    async def _chat(self, messages: list[dict],
                    schema: dict | None = None,
                    memo_put: bool = False) -> str:
        key = self._memo_key(messages, schema)
        if not memo_put:
            if key not in self._memo and self._prewarm_evt is not None \
                    and not self._prewarm_evt.is_set():
                # a speculative call for (probably) this request is in
                # flight — wait for it instead of racing a duplicate
                try:
                    await asyncio.wait_for(self._prewarm_evt.wait(), 2.5)
                except asyncio.TimeoutError:
                    pass
            hit = self._memo.pop(key, None)
            if hit is not None and time.monotonic() - hit[0] < 20.0:
                if SPEC_VERIFY:
                    # semantic three-way check: cached-vs-fresh measured
                    # against the server's OWN fresh-vs-fresh jitter (the
                    # vLLM batcher is nondeterministic even at temp 0, and
                    # emits compact/spaced JSON randomly — byte equality
                    # is the wrong bar). The cache is lossless if its
                    # disagreement rate matches the control's.
                    fresh1 = await self._chat_http(messages, schema)
                    fresh2 = await self._chat_http(messages, schema)

                    def sem(c):
                        try:
                            return ScenePatch.model_validate_json(
                                c).model_dump()
                        except Exception:
                            return c
                    a, b, c = sem(hit[1]), sem(fresh1), sem(fresh2)
                    log.info("speculative parse VERIFY: cached-vs-fresh "
                             "%s | fresh-vs-fresh control %s",
                             "MATCH" if a == b else "DIFF",
                             "MATCH" if b == c else "DIFF")
                    return fresh1
                log.info("speculative parse HIT — commit parse cost 0ms")
                return hit[1]
        content = await self._chat_http(messages, schema)
        if memo_put:
            self._memo = {key: (time.monotonic(), content)}
        return content

    async def _chat_http(self, messages: list[dict],
                         schema: dict | None = None) -> str:
        loop = asyncio.get_running_loop()
        known = _MODE_BY_URL.get(self.base_url)
        modes = [known] if known else ["response_format", "guided_json"]
        for i, mode in enumerate(modes):
            try:
                content = await loop.run_in_executor(
                    None, stream_chat_blocking, self.base_url,
                    self._body(messages, mode, schema), self.timeout_s)
                _MODE_BY_URL[self.base_url] = mode
                return content
            except ChatTimeout as e:
                # busy/unreachable — the request shape is not the problem;
                # never burn a second timeout probing the other mode
                log.warning("editor call timed out (mode=%s): %s", mode, e)
                raise EditorTimeout(str(e)) from None
            except ChatHTTPError as e:
                if 400 <= e.status < 500 and i + 1 < len(modes):
                    log.info("mode %s rejected (HTTP %d); trying %s",
                             mode, e.status, modes[i + 1])
                    continue
                log.warning("editor call failed (mode=%s): %s", mode, e)
                raise EditorError(str(e)) from None
        raise EditorError("no structured-output mode accepted")

    def _first_messages(self, user_msg: str, image_b64: str | None,
                        history: list[dict] | None) -> list[dict]:
        """First-attempt message list — the SINGLE assembly path shared
        by propose_patch and prewarm, so a speculative request is byte-
        identical to the commit-time request by construction."""
        if image_b64:
            user_content: object = [
                {"type": "text",
                 "text": user_msg + "\nnote: the attached image is the "
                         "CURRENT canvas the user is reacting to. The "
                         "timeline block still applies — check whether a "
                         "history_action serves this directive better than "
                         "an edit."},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{image_b64}"}},
            ]
        else:
            user_content = user_msg
        messages = [{"role": "system", "content": self.system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_content})
        return messages

    async def prewarm(self, state: SceneState, tone_line: str,
                      directive: str, *, unfinished: bool = False,
                      image_b64: str | None = None,
                      history: list[dict] | None = None,
                      note: str | None = None,
                      ledger: str | None = None,
                      history_revs: list[int] | None = None) -> None:
        """Speculative first-attempt parse fired during the commit gate's
        silence window. Stores the raw response in the body-keyed memo;
        NO validation, NO history mutation, NO side effects — the real
        propose_patch replays everything from the cached content exactly
        as if the HTTP call had just returned. Never raises."""
        evt = asyncio.Event()
        self._prewarm_evt = evt
        self._spec_parts = {
            "directive": directive, "tone": tone_line,
            "state": hashlib.md5(state.model_dump_json()
                                 .encode()).hexdigest()[:8],
            "image": hashlib.md5((image_b64 or "")
                                 .encode()).hexdigest()[:8],
            "ledger": hashlib.md5((ledger or "")
                                  .encode()).hexdigest()[:8],
            "note": note or "", "revs": tuple(history_revs or ()),
            "ts": time.monotonic(),
        }
        try:
            user_msg = (self.message_builder(state, tone_line, directive,
                                             unfinished)
                        if self.message_builder
                        else build_user_message(state, tone_line, directive,
                                                unfinished=unfinished,
                                                note=note, ledger=ledger))
            messages = self._first_messages(user_msg, image_b64, history)
            t0 = time.monotonic()
            await self._chat(messages, schema=patch_schema(history_revs),
                             memo_put=True)
            log.info("speculative parse warmed in %.0fms (%d words)",
                     (time.monotonic() - t0) * 1000,
                     len(directive.split()))
        except Exception as e:
            log.info("speculative parse failed (ignored): %s", e)
        finally:
            evt.set()
            self._prewarm_evt = None

    async def propose_patch(self, state: SceneState, tone_line: str,
                            directive: str,
                            max_changed_fields: int, below_magnitude: float,
                            max_retries: int = 1,
                            unfinished: bool = False,
                            image_b64: str | None = None,
                            history: list[dict] | None = None,
                            note: str | None = None,
                            ledger: str | None = None,
                            history_revs: list[int] | None = None
                            ) -> tuple[ScenePatch, int, int]:
        """Returns (patch, latency_ms, retries). Raises EditorError.

        image_b64: JPEG of the current canvas — attached so the model sees
        what the user sees (it edits the SCENE, grounded by the image).
        history: append-only conversation (mutated in place on success) so
        repeated calls share a cached prefix instead of re-reading state.
        note: structural side-channel (e.g. "already applied instantly: X")
        kept OUT of the quoted directive so it can't leak into fields.
        """
        user_msg = (self.message_builder(state, tone_line, directive, unfinished)
                    if self.message_builder
                    else build_user_message(state, tone_line, directive,
                                            unfinished=unfinished, note=note,
                                            ledger=ledger))
        # what goes into CONVERSATION MEMORY omits the timeline block: the
        # ledger is regenerated fresh EVERY call and must be the ONLY one the
        # model sees — stale ledgers in old turns contradict it and produce
        # degenerate pattern-completion answers
        history_msg = (user_msg if ledger is None
                       else build_user_message(state, tone_line, directive,
                                               unfinished=unfinished,
                                               note=note, ledger=None))
        call_schema = patch_schema(history_revs)
        messages = self._first_messages(user_msg, image_b64, history)
        # speculation observability: when a recent prewarm exists, log
        # which inputs drifted (the miss reason) — this is how the hit
        # rate gets tuned without guessing
        sp = getattr(self, "_spec_parts", None)
        if sp and time.monotonic() - sp.get("ts", 0) < 20.0:
            cur = {
                "directive": directive, "tone": tone_line,
                "state": hashlib.md5(state.model_dump_json()
                                     .encode()).hexdigest()[:8],
                "image": hashlib.md5((image_b64 or "")
                                     .encode()).hexdigest()[:8],
                "ledger": hashlib.md5((ledger or "")
                                      .encode()).hexdigest()[:8],
                "note": note or "", "revs": tuple(history_revs or ()),
            }
            drift = [k for k in cur if cur[k] != sp.get(k)]
            if drift:
                log.info("speculative parse MISS — drifted inputs: %s",
                         ", ".join(drift))
            self._spec_parts = None
        t0 = time.monotonic()
        retries = 0
        while True:
            content = await self._chat(messages, schema=call_schema)
            try:
                patch = ScenePatch.model_validate_json(content)
                # drop echo no-ops (value identical to current state) before
                # judging minimality — echoing is tolerated, not punished
                effective = {f: v for f, v in patch.field_patch().items()
                             if v != getattr(state, f)}
                patch.edits = [e for e in patch.edits
                               if effective.get(e.field) == e.value]
                # adds without a box are unusable — drop them
                patch.entity_edits = [e for e in patch.entity_edits
                                      if not (e.op == "add" and e.box is None)]
                # ops on ids the STATE doesn't know are NOT bogus in an
                # edit-native pipeline: the object usually exists in PIXELS,
                # painted from prose ("modern kitchen" implies stools). They
                # no-op against state but MUST still phrase into the image
                # edit — "remove the stool" has to reach the canvas even
                # though no entity tracks a stool. (Silently stripping these
                # made removal requests vanish three times in one run.)
                known = {e.id for e in state.entities}
                pixel_only = [(e.op, e.id) for e in patch.entity_edits
                              if e.op in ("update", "remove")
                              and e.id not in known]
                if pixel_only:
                    log.info("pixel-only entity ops (canvas, not state): %s",
                             pixel_only)
                # contradiction: remove + add/update of the SAME id in one
                # patch would delete the object from state (and canvas).
                # The keep-op is the intent; the remove is the mistake.
                keep_ids = {e.id for e in patch.entity_edits
                            if e.op in ("add", "update")}
                patch.entity_edits = [e for e in patch.entity_edits
                                      if not (e.op == "remove"
                                              and e.id in keep_ids)]
                # consistency guard: a removal must not leave the removed
                # thing ASSERTED in scene text — the prompt would
                # contradict the edit and the image model keeps the object
                # ("remove all the palm trees" while setting still read
                # "street with palm trees" failed twice in one session).
                # One retry asks for the field rewrite; on exhaustion the
                # patch is accepted as-is, never dropped.
                ent_names = {e.id: (e.name or e.id) for e in state.entities}
                removed_labels = [
                    (ent_names.get(e.id) or e.id or "").replace("_", " ")
                    for e in patch.entity_edits if e.op == "remove"]
                removed_labels = [x for x in removed_labels if x]
                if removed_labels:
                    stale = stale_fields_for(
                        state, removed_labels,
                        field_overrides=patch.field_patch())
                    # ALL-quantifier removals ("remove all the
                    # furniture") always stale the scene text: the
                    # label-mention test cannot see that "a modern home
                    # interior" asserts furniture, and a prompt that
                    # keeps asserting it regenerates everything on the
                    # next strong render
                    fp = patch.field_patch()
                    # the regenerate route is DEFINED by the rewrite
                    # (rule 10d): choosing it without touching
                    # subject/setting redraws the same furnished
                    # description and changes nothing
                    if (not stale
                            and (getattr(patch, "removal_route", None)
                                 == "regenerate"
                                 or re.search(
                                     r"\b(all|every|everything)\b",
                                     directive, re.I))
                            and not (fp.get("subject")
                                     or fp.get("setting"))):
                        stale = ["subject", "setting"]
                        if retries < max_retries:
                            raise ValueError(
                                "this removes a whole category — the "
                                "scene text still describes the old "
                                "contents. Also rewrite subject and "
                                "setting so they positively describe "
                                "the scene WITHOUT the removed things "
                                "(e.g. 'an empty room, bare floor')")
                    if stale and retries < max_retries:
                        raise ValueError(
                            f"the removal is contradicted by the scene "
                            f"text: the {stale[0]} field still mentions "
                            f"the removed thing. Also rewrite "
                            f"{' and '.join(stale)} so the scene no "
                            f"longer asserts it (keep the rest of the "
                            f"field's content)")
                    if stale:
                        log.warning("removal leaves stale text in %s "
                                    "(accepted after retries)", stale)
                # RELOCATION consistency: "now the same X in <place>"
                # without a setting rewrite produces overlay artifacts
                # (the new weather painted into the old room)
                if (re.search(r"\b(same|now)\b.*\b(in|on|at|inside|"
                              r"outside|under|by)\b", directive, re.I)
                        and re.search(r"\bsame\b", directive, re.I)
                        and not patch.field_patch().get("setting")
                        and retries < max_retries):
                    raise ValueError(
                        "this relocates the same subject to a new "
                        "setting — you MUST rewrite the setting field "
                        "to the new location, keep the subject "
                        "unchanged, set edit_magnitude 0.85+, and list "
                        "the subject in preserve")
                # structural-update consistency: a contact/relational
                # patch that transforms an object ("sink the fire pit")
                # must not leave scene text asserting its OLD build — a
                # prompt reading "flush-mounted fire pit, ... a raised
                # stone fire pit" contradicts itself and the image model
                # resolves toward its prior (the raised one). Same
                # rewrite-retry mechanism as removals.
                if getattr(patch, "contact", None) is not None:
                    upd_labels = [
                        (ent_names.get(e.id) or e.id or "")
                        .replace("_", " ")
                        for e in patch.entity_edits if e.op == "update"]
                    upd_labels = [x for x in upd_labels if x]
                    if upd_labels:
                        # only fields the patch did NOT rewrite: a field
                        # the model already re-authored is trusted (the
                        # mention-based check can't tell old form from
                        # new, so it would never converge otherwise)
                        stale_u = [f for f in stale_fields_for(
                            state, upd_labels,
                            field_overrides=patch.field_patch())
                            if f not in patch.field_patch()]
                        if stale_u and retries < max_retries:
                            raise ValueError(
                                f"this edit changes how the "
                                f"{upd_labels[0]} is built, but the "
                                f"{' and '.join(stale_u)} field(s) still "
                                f"describe its old form. Also rewrite "
                                f"{' and '.join(stale_u)} so the scene "
                                f"text matches the object's NEW form "
                                f"(keep the rest of each field's "
                                f"content)")
                        if stale_u:
                            log.warning(
                                "structural update leaves stale text in "
                                "%s (accepted after retries)", stale_u)
                # removal completeness: a removal directive answered with
                # a patch that removes NOTHING (no remove ops, no dropped
                # field text) silently no-ops — the model sometimes emits
                # an empty patch for "remove the overhead lamp". Force one
                # retry naming the things to remove; accept on exhaustion.
                dwords = {w.strip(".,!?\"'").lower()
                          for w in directive.split()}
                if (not self.lenient and (dwords & REMOVAL_VERBS)
                        and patch.history_action is None
                        and not removed_labels):
                    dropped = []
                    for f, v in patch.field_patch().items():
                        dropped += dropped_phrases(
                            getattr(state, f, "") or "", v)
                    if not dropped and retries < max_retries:
                        raise ValueError(
                            'the directive asks to remove or erase '
                            'something, but this patch removes nothing. '
                            'Name each thing to remove as an entity_edits '
                            'op {"op": "remove", "id": '
                            '"<snake_case_thing>"}')
                    if not dropped:
                        log.warning("removal directive had no removal "
                                    "effect (accepted after retries)")
                # a scene-defining patch on a subject-less scene must name the
                # subject (models occasionally omit it)
                if (not self.lenient and not state.subject
                        and "setting" in effective
                        and "subject" not in effective and retries < max_retries):
                    raise ValueError(
                        "the scene has no subject yet; include a subject edit "
                        "naming the main thing from the directive")
                if (not self.lenient and len(effective) > max_changed_fields
                        and patch.edit_magnitude < below_magnitude):
                    if retries >= max_retries:
                        # broad edit with under-reported magnitude: honor the
                        # user's words, but clamp magnitude up so the renderer
                        # knows the change is big
                        patch.edit_magnitude = below_magnitude
                        patch.reasoning = ((patch.reasoning or "")
                                           + " [magnitude auto-raised]").strip()
                        log.warning("minimality: %d fields at low magnitude; "
                                    "auto-raised to %.2f", len(effective),
                                    below_magnitude)
                    else:
                        raise ValueError(
                            f"patch changes {len(effective)} fields but "
                            f"edit_magnitude is only {patch.edit_magnitude}; "
                            f"either patch fewer fields or raise edit_magnitude")
                latency_ms = int((time.monotonic() - t0) * 1000)
                if history is not None:
                    # append-only conversation memory: cheap text stand-ins
                    # (never the image, never the timeline block)
                    history.append({"role": "user", "content": history_msg})
                    history.append({"role": "assistant", "content": content})
                return patch, latency_ms, retries
            except (ValueError, json.JSONDecodeError) as e:
                if retries >= max_retries:
                    raise EditorError(f"invalid patch after {retries + 1} tries: {e}")
                retries += 1
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user",
                                 "content": f"That patch was invalid: {e}. "
                                            f"Return a corrected JSON patch."})
