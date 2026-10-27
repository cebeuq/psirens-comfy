"""Per-connection session: owns the event bus, the recorder, and the audio
pipeline. Later milestones register more audio sinks (STT, prosody); M1 only
records and reports level.
"""
from __future__ import annotations

import datetime as dt
import secrets
from pathlib import Path
from typing import Protocol

import numpy as np

from .config import Config
from .events import EventBus
from .recorder import SessionRecorder


def new_session_id() -> str:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{secrets.token_hex(3)}"


class AudioSink(Protocol):
    """Anything that consumes the 24kHz int16 mono stream, frame by frame."""

    async def feed(self, pcm: bytes, t_session_ms: int) -> None: ...
    async def close(self) -> None: ...


class Session:
    def __init__(self, cfg: Config, extra_meta: dict | None = None):
        self.cfg = cfg
        self.id = new_session_id()
        self.sample_rate: int = cfg.audio.sample_rate
        self.bus = EventBus()
        self.t_session_ms = 0
        self.sinks: list[AudioSink] = []

        sessions_root = Path(cfg.paths.data_dir) / "sessions"
        self.recorder = SessionRecorder(
            sessions_root,
            self.id,
            self.sample_rate,
            meta={
                "session_id": self.id,
                "sample_rate": self.sample_rate,
                "frame_ms": cfg.audio.frame_ms,
                "config_digest": cfg.digest(),
                "config": cfg.as_dict(),
                **(extra_meta or {}),
            },
        )

        # level meter throttle
        self._level_interval_ms = 1000 / cfg.ui.audio_level_hz
        self._last_level_ms = -1e9

        # the recorder also subscribes to every event
        self._rec_queue = self.bus.subscribe()

    def start(self) -> None:
        self.bus.publish(
            "session_start",
            {
                "session_id": self.id,
                "sample_rate": self.sample_rate,
                "frame_ms": self.cfg.audio.frame_ms,
                "config_digest": self.cfg.digest(),
            },
            self.t_session_ms,
        )

    async def feed_audio(self, pcm: bytes) -> None:
        """Consume one binary websocket message of int16 mono PCM."""
        if len(pcm) % 2:
            pcm = pcm[:-1]
        if not pcm:
            return
        self.recorder.write_audio(pcm)

        n_samples = len(pcm) // 2
        frame_ms = n_samples * 1000 / self.sample_rate
        self.t_session_ms += round(frame_ms)

        if self.t_session_ms - self._last_level_ms >= self._level_interval_ms:
            self._last_level_ms = self.t_session_ms
            samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(samples**2)))
            dbfs = 20 * np.log10(rms) if rms > 1e-6 else -120.0
            self.bus.publish("audio_level", {"dbfs": round(dbfs, 1)}, self.t_session_ms)

        for sink in self.sinks:
            await sink.feed(pcm, self.t_session_ms)

    def drain_recorder(self) -> None:
        """Write all pending events to the JSONL log (called by the ws loop)."""
        while not self._rec_queue.empty():
            self.recorder.write_event(self._rec_queue.get_nowait())

    async def close(self) -> None:
        import asyncio
        for sink in self.sinks:
            await sink.close()
        # sinks publish from worker threads via call_soon_threadsafe; let those
        # callbacks land before the final drain
        await asyncio.sleep(0.05)
        self.bus.publish("session_end", {"t_total_ms": self.t_session_ms}, self.t_session_ms)
        self.drain_recorder()
        self.recorder.close()
