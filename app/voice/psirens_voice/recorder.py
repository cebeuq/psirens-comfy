"""Session recorder: raw audio (WAV) + full event log (JSONL) + metadata.

The WAV file is written incrementally. The `wave` module patches the header on
close; if the process dies first, `replay.py --fix-wav` repairs the header from
the file length.
"""
from __future__ import annotations

import json
import wave
from pathlib import Path
from typing import Any

from .events import Event

EVENT_FLUSH_EVERY = 25

# never recorded: bulky binary payloads that are reconstructable anyway
EXCLUDED_EVENT_TYPES = {"canvas_frame"}


class SessionRecorder:
    def __init__(self, sessions_root: Path, session_id: str, sample_rate: int,
                 meta: dict[str, Any]):
        self.dir = Path(sessions_root) / session_id
        self.dir.mkdir(parents=True, exist_ok=True)

        self._wav = wave.open(str(self.dir / "audio.wav"), "wb")
        self._wav.setnchannels(1)
        self._wav.setsampwidth(2)  # int16
        self._wav.setframerate(sample_rate)

        self._events = open(self.dir / "events.jsonl", "a", encoding="utf-8")
        self._since_flush = 0

        with open(self.dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        self.closed = False

    def write_audio(self, pcm: bytes) -> None:
        if not self.closed:
            self._wav.writeframes(pcm)

    def write_event(self, ev: Event) -> None:
        if self.closed or ev.type in EXCLUDED_EVENT_TYPES:
            return
        self._events.write(ev.model_dump_json() + "\n")
        self._since_flush += 1
        if self._since_flush >= EVENT_FLUSH_EVERY:
            self._events.flush()
            self._since_flush = 0

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._wav.close()
        self._events.flush()
        self._events.close()
