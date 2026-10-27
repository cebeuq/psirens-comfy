"""Commit detector state machine.

    LISTENING --(word)--> SPEAKING
    SPEAKING  --(observed pause >= pause_min AND p_eot >= eot_enter)--> MAYBE_DONE
    MAYBE_DONE --(new word/pending)--> SPEAKING
    MAYBE_DONE --(hold signal)--> HOLDING       (trailing filler, rising pitch,
                                                 p_eot dropped below eot_exit)
    MAYBE_DONE --(dwell >= confirm_ms, gates pass)--> COMMIT --> LISTENING
    HOLDING   --(new word)--> SPEAKING
    HOLDING   --(silence >= hold_timeout)--> COMMIT(timeout) --> LISTENING

Timing subtlety: Kyutai finalizes words ~delay_ms (500) behind the audio, so
"observed pause" is measured in model-observed time:
    observed_pause = (t_now - delay_ms) - last_word_end
Any pending partial word counts as activity regardless of pause math.

Every decision carries a human-readable `reasons` list — that is the handle
the replay harness tunes against. The detector is deliberately free of
asyncio/bus dependencies so it can be unit-tested and replayed byte-for-byte;
CommitSession (session_wiring.py) adapts it to the event bus.

Erring conservative per spec §2.2: any single hold signal beats all commit
signals; the hold timeout guarantees nothing is lost forever.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

from .self_repair import apply_self_repair, _norm

log = logging.getLogger("psirens.commit")


@dataclass
class DetectorConfig:
    eot_enter: float = 0.6
    eot_exit: float = 0.4
    pause_min_ms: int = 500
    confirm_ms: int = 550
    hold_timeout_ms: int = 2500
    min_content_words: int = 2
    pitch_rise_hz_per_s: float = 40.0
    delay_ms: int = 500
    fillers: frozenset[str] = frozenset()

    @classmethod
    def from_config(cls, cfg, delay_ms: int) -> "DetectorConfig":
        c = cfg.commit
        return cls(
            eot_enter=c.eot_enter, eot_exit=c.eot_exit,
            pause_min_ms=c.pause_min_ms, confirm_ms=c.confirm_ms,
            hold_timeout_ms=c.hold_timeout_ms,
            min_content_words=c.min_content_words,
            pitch_rise_hz_per_s=c.pitch_rise_hz_per_s,
            delay_ms=delay_ms,
            fillers=frozenset(f.lower() for f in c.fillers),
        )


@dataclass
class Decision:
    """What the state machine wants published this tick."""
    kind: str                 # "commit" | "hold" | "discard" | None
    data: dict = field(default_factory=dict)


class CommitDetector:
    def __init__(self, cfg: DetectorConfig,
                 pitch_slope_fn: Callable[[], float | None] = lambda: None,
                 tone_fn: Callable[[], dict] = lambda: {},
                 spec_fn: Callable[[dict], None] | None = None):
        self.cfg = cfg
        self.pitch_slope_fn = pitch_slope_fn  # tail pitch slope Hz/s, or None
        self.tone_fn = tone_fn                # discrete tone attrs at commit time
        # speculative-parse hook: called once per stable word tail as
        # soon as a pause is observed — the parse then overlaps the
        # commit gate's confirm dwell instead of following it. Purely
        # advisory; the state machine is untouched.
        self.spec_fn = spec_fn
        self._spec_fired: tuple | None = None
        self._spec_tone: dict | None = None
        self.state = "LISTENING"
        self.words: list[dict] = []           # utterance buffer since last commit
        self.pending = ""                     # in-flight partial word
        self._maybe_since: int | None = None
        self._hold_since: int | None = None
        self._hold_reason = ""

    # -- inputs --
    def on_word(self, word: dict) -> None:
        self.words.append(word)
        self.pending = ""
        if self.state in ("MAYBE_DONE", "HOLDING", "LISTENING"):
            self.state = "SPEAKING"
            self._maybe_since = None
            self._hold_since = None

    def on_partial(self, pending: str) -> None:
        self.pending = pending
        if pending and self.state in ("MAYBE_DONE", "HOLDING", "LISTENING"):
            self.state = "SPEAKING"
            self._maybe_since = None
            self._hold_since = None

    def tick(self, t_ms: int, p_eot: float) -> Decision | None:
        """Called on every vad_state frame (12.5Hz). t_ms = session audio clock."""
        c = self.cfg
        if not self.words:
            return None

        last_end = self.words[-1]["t_end_ms"]
        observed_pause = (t_ms - c.delay_ms) - last_end
        active = bool(self.pending) or observed_pause < 0

        # speculative parse: a pause has been observed but the gate has
        # not confirmed yet. Hand the WOULD-BE commit payload (same
        # self-repair, same tone source as _commit) to the hook so the
        # LLM parse runs during the dwell. Fires once per word tail;
        # any new word changes the key and re-arms it.
        if (self.spec_fn is not None and not active
                and observed_pause >= c.pause_min_ms
                and self.state in ("SPEAKING", "MAYBE_DONE", "HOLDING")):
            key = (len(self.words), last_end)
            if key != self._spec_fired:
                self._spec_fired = key
                raw = [w["w"] for w in self.words]
                n_content = sum(1 for w in raw
                                if _norm(w) not in c.fillers)
                if n_content >= c.min_content_words:
                    clean, _ = apply_self_repair(raw)
                    # tone is measured HERE, at pause start — the same
                    # measurement the commit will reuse. Re-measuring at
                    # commit time samples ~1.4s deeper into trailing
                    # silence, which dilutes the energy/tempo windows
                    # and flipped buckets on 5 of 8 test commits (the
                    # sole cause of speculative-parse misses).
                    self._spec_tone = dict(self.tone_fn())
                    try:
                        self.spec_fn({"utterance_clean": clean,
                                      "tone": dict(self._spec_tone)})
                    except Exception:
                        pass

        if self.state == "SPEAKING":
            if not active and observed_pause >= c.pause_min_ms and p_eot >= c.eot_enter:
                self.state = "MAYBE_DONE"
                self._maybe_since = t_ms
            elif (not active
                  and observed_pause >= c.pause_min_ms + c.hold_timeout_ms):
                # liveness guarantee: sustained dead silence after real
                # words MUST commit even when the end-of-turn model never
                # cooperates — a fresh session's semantic VAD outputs
                # ~0.02 for its first ~10 seconds (sometimes for the whole
                # session), which trapped utterances in SPEAKING forever.
                # This extends HOLDING's "nothing is lost forever" promise
                # to the state a cold p_eot could never leave. The window
                # (pause_min + hold_timeout, ~3s) is far longer than the
                # normal gate path, so it only fires when the primary
                # signal is broken.
                return self._commit(t_ms, p_eot, observed_pause,
                                    observed_pause, via="silence_timeout")
            return None

        if self.state == "MAYBE_DONE":
            if active:
                self.state = "SPEAKING"
                self._maybe_since = None
                return None
            hold_reason = self._hold_signal(p_eot)
            if hold_reason:
                self.state = "HOLDING"
                self._hold_since = t_ms
                self._hold_reason = hold_reason
                return Decision("hold", {
                    "reason": hold_reason,
                    "utterance_so_far": self._utterance_raw(),
                })
            dwell = t_ms - (self._maybe_since or t_ms)
            if dwell >= c.confirm_ms:
                return self._commit(t_ms, p_eot, observed_pause, dwell, via="gate")
            return None

        if self.state == "HOLDING":
            if active:
                self.state = "SPEAKING"
                self._hold_since = None
                return None
            held = t_ms - (self._hold_since or t_ms)
            if held >= c.hold_timeout_ms:
                return self._commit(t_ms, p_eot, observed_pause, held, via="timeout")
            return None

        return None

    # -- internals --
    # words a finished sentence essentially never ends on — a pause
    # after one is the speaker THINKING, not done ("fill in the part of
    # the left window that's ..." once committed mid-thought and the
    # tail became a second, garbled edit)
    _DANGLING_TAIL = frozenset({
        "the", "a", "an", "with", "and", "so", "to", "of", "but", "or",
        "which", "is", "are", "was", "were", "then", "because", "if",
        "when", "while", "its", "it's", "that's", "there's", "into",
        "onto", "than",
    })

    def _hold_signal(self, p_eot: float) -> str | None:
        c = self.cfg
        if self.words and _norm(self.words[-1]["w"]) in c.fillers:
            return "trailing_filler"
        if (self.words
                and _norm(self.words[-1]["w"]) in self._DANGLING_TAIL):
            return "dangling_clause"
        slope = self.pitch_slope_fn()
        if slope is not None and slope > c.pitch_rise_hz_per_s:
            return "rising_pitch"
        if p_eot < c.eot_exit:
            return "eot_low"
        return None

    def _utterance_raw(self) -> str:
        return " ".join(w["w"] for w in self.words)

    def _commit(self, t_ms: int, p_eot: float, pause: int, dwell: int,
                via: str) -> Decision:
        c = self.cfg
        # reuse the pause-start tone when this commit is for the same
        # word tail the speculation measured (see the spec_fn block) —
        # the fresher one differs only by extra trailing silence
        tone = None
        if (self._spec_tone is not None and self.words
                and self._spec_fired == (len(self.words),
                                         self.words[-1]["t_end_ms"])):
            tone = self._spec_tone
            fresh = self.tone_fn()
            if fresh != tone:
                log.info("tone: pause-start %s reused over commit-time "
                         "%s (silence dilution)", tone, fresh)
        self._spec_tone = None
        raw_words = [w["w"] for w in self.words]
        words_snapshot = list(self.words)
        n_content = sum(1 for w in raw_words if _norm(w) not in c.fillers)
        t_first = self.words[0]["t_start_ms"]
        t_last = self.words[-1]["t_end_ms"]
        self.words = []
        self.pending = ""
        self.state = "LISTENING"
        self._maybe_since = None
        self._hold_since = None

        if n_content < c.min_content_words:
            return Decision("discard", {
                "reason": "min_content",
                "utterance_raw": " ".join(raw_words),
                "n_content_words": n_content,
            })

        clean, repairs = apply_self_repair(raw_words)
        reasons = [
            f"via={via}",
            f"p_eot={p_eot:.2f}",
            f"observed_pause_ms={int(pause)}",
            f"dwell_ms={int(dwell)}",
        ]
        if via == "timeout":
            reasons.append(f"hold_reason_was={self._hold_reason}")
        slope = self.pitch_slope_fn()
        if slope is not None:
            reasons.append(f"tail_pitch_slope_hz_s={slope:.0f}")
        return Decision("commit", {
            "utterance_raw": " ".join(raw_words),
            "utterance_clean": clean,
            "words_t": [{"w": w["w"], "t": w["t_end_ms"]}
                        for w in words_snapshot],
            "repairs": repairs,
            "tone": tone if tone is not None else self.tone_fn(),
            "t_speech_start_ms": t_first,
            "t_speech_end_ms": t_last,
            "reasons": reasons,
        })
