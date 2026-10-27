"""Adapts CommitDetector to the event bus: consumes word / partial_transcript
/ vad_state events, publishes commit / hold / discard events."""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from .detector import CommitDetector, DetectorConfig

if TYPE_CHECKING:
    from ..config import Config
    from ..events import EventBus
    from ..prosody.engine import ProsodySession

log = logging.getLogger("psirens.commit")


class CommitSession:
    def __init__(self, cfg: "Config", bus: "EventBus",
                 prosody: "ProsodySession | None",
                 loop: asyncio.AbstractEventLoop, delay_ms: int):
        def pitch_slope():
            if prosody and prosody.latest:
                return prosody.latest["pitch"]["slope_hz_s"]
            return None

        def tone():
            if prosody:
                return prosody.tone_attributes()
            return {}

        self._last_t = 0

        def speculate(data: dict) -> None:
            # advisory pre-commit signal for the editor's speculative
            # parse; carries the exact would-be utterance_clean + tone
            bus.publish("pre_commit", data, self._last_t)

        self.detector = CommitDetector(
            DetectorConfig.from_config(cfg, delay_ms),
            pitch_slope_fn=pitch_slope, tone_fn=tone,
            spec_fn=speculate)
        self.bus = bus
        self._queue = bus.subscribe()
        self._task = loop.create_task(self._consume())

    async def _consume(self) -> None:
        det = self.detector
        while True:
            ev = await self._queue.get()
            try:
                if ev.type == "word":
                    det.on_word(ev.data)
                elif ev.type == "partial_transcript":
                    det.on_partial(ev.data.get("pending", ""))
                elif ev.type == "vad_state":
                    self._last_t = ev.t_session_ms
                    decision = det.tick(ev.t_session_ms, ev.data["p_eot"])
                    if decision is not None:
                        etype = {"commit": "commit", "hold": "hold",
                                 "discard": "hold"}[decision.kind]
                        data = dict(decision.data)
                        if decision.kind == "discard":
                            data["reason"] = "discard_" + data.get("reason", "")
                        self.bus.publish(etype, data, ev.t_session_ms)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("commit detector failed on %s", ev.type)

    async def close(self) -> None:
        self._task.cancel()
        self.bus.unsubscribe(self._queue)
