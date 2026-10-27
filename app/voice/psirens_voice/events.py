"""Event model and in-process event bus.

Every stage of the pipeline communicates through Event objects with one
envelope. The same schema is used for three consumers:
  1. the live browser websocket,
  2. the JSONL session recorder,
  3. the offline replay harness.

`t_session_ms` counts milliseconds of *audio consumed* since session start,
which makes event timing deterministic under replay. `t_wall` is wall-clock
and is only used for latency measurement.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

from pydantic import BaseModel

PROTOCOL_VERSION = 1

# Known event types (informational; the envelope is schema-checked, data
# payloads are validated by their producers):
#   session_start, audio_level,
#   partial_transcript, word, vad_state,          (M2)
#   prosody_frame,                                 (M3)
#   commit, hold,                                  (M4)
#   patch_request, patch_applied, state_snapshot, prompt_updated,  (M5)
#   error, session_end


class Event(BaseModel):
    v: int = PROTOCOL_VERSION
    seq: int
    t_wall: int          # wall-clock, ms since epoch
    t_session_ms: int    # ms of audio consumed since session start
    type: str
    data: dict[str, Any]


class EventBus:
    """Fan-out pub/sub. Publishing never blocks: a slow subscriber loses its
    oldest events rather than stalling the audio path."""

    def __init__(self) -> None:
        self._subscribers: list[asyncio.Queue[Event]] = []
        self._seq = 0

    def subscribe(self, maxsize: int = 2000) -> asyncio.Queue[Event]:
        q: asyncio.Queue[Event] = asyncio.Queue(maxsize=maxsize)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[Event]) -> None:
        if q in self._subscribers:
            self._subscribers.remove(q)

    def publish(self, type: str, data: dict[str, Any], t_session_ms: int) -> Event:
        self._seq += 1
        ev = Event(
            seq=self._seq,
            t_wall=int(time.time() * 1000),
            t_session_ms=t_session_ms,
            type=type,
            data=data,
        )
        for q in self._subscribers:
            try:
                q.put_nowait(ev)
            except asyncio.QueueFull:
                try:
                    q.get_nowait()  # drop oldest
                except asyncio.QueueEmpty:
                    pass
                try:
                    q.put_nowait(ev)
                except asyncio.QueueFull:
                    pass
        return ev
