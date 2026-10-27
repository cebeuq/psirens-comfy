"""Prosody engine: fuses the DSP lane and the SER lane into discrete tone
attributes, published as prosody_frame events every frame_hop_ms.

Also keeps a `latest` snapshot that the commit detector reads synchronously
(tail pitch slope, hesitation inputs) and can compute per-utterance tone at
commit time.
"""
from __future__ import annotations

import asyncio
import logging
import re
from collections import deque
from typing import TYPE_CHECKING

import numpy as np

from .buckets import RobustNorm, bucket_absolute
from .dsp import pitch_stats, rms_dbfs
from .ser import SerEngine

if TYPE_CHECKING:
    from ..config import Config
    from ..events import EventBus

log = logging.getLogger("psirens.prosody")

_PUNCT = re.compile(r"[^\w']+")


def normalize_word(w: str) -> str:
    return _PUNCT.sub("", w.lower())


class ProsodySession:
    """AudioSink + bus consumer (word events)."""

    def __init__(self, cfg: "Config", ser: SerEngine | None, bus: "EventBus",
                 loop: asyncio.AbstractEventLoop, fillers: list[str]):
        p = cfg.prosody
        self.cfg = cfg
        self.ser = ser
        self.bus = bus
        self.loop = loop
        self.fillers = {f.lower() for f in fillers}

        self.sample_rate = cfg.audio.sample_rate
        self.hop_ms = p.frame_hop_ms
        self.energy_window_ms = p.energy_window_ms
        self.pitch_window_ms = p.pitch_window_ms
        self.pitch_tail_ms = p.pitch_tail_ms
        self.tempo_window_ms = p.tempo_window_ms
        self.hesitation_window_ms = p.hesitation_window_ms
        self.warmup_ms = p.warmup_s * 1000
        self.ser_window_s = p.ser.window_s
        self.ser_hop_ms = p.ser.hop_s * 1000
        cold = p.cold_start
        self.cold = {
            "energy": tuple(cold.energy_dbfs),
            "tempo": tuple(cold.tempo_wps),
            "valence": tuple(cold.valence),
            "arousal": tuple(cold.arousal),
        }

        # ~8s ring buffer of float32 audio
        self._chunks: deque[np.ndarray] = deque()
        self._buffered_samples = 0
        self._max_samples = self.sample_rate * 8

        self.t_ms = 0
        self._last_hop_ms = -1e9
        self._last_ser_ms = -1e9
        self._ser_busy = False
        self._ser_result: dict | None = None

        self._energy_norm = RobustNorm()
        self._tempo_norm = RobustNorm()

        # words seen (from the STT lane), for tempo/hesitation
        self._words: deque[dict] = deque(maxlen=400)
        self._word_queue = bus.subscribe()
        self._word_task = loop.create_task(self._consume_words())

        # latest fused state, read synchronously by the commit detector
        self.latest: dict = {}

    async def _consume_words(self) -> None:
        while True:
            ev = await self._word_queue.get()
            if ev.type == "word":
                self._words.append(ev.data)

    # -- AudioSink --
    async def feed(self, pcm: bytes, t_session_ms: int) -> None:
        self.t_ms = t_session_ms
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        self._chunks.append(samples)
        self._buffered_samples += samples.size
        while self._buffered_samples > self._max_samples and len(self._chunks) > 1:
            self._buffered_samples -= self._chunks.popleft().size

        if t_session_ms - self._last_hop_ms >= self.hop_ms:
            self._last_hop_ms = t_session_ms
            self._publish_frame()

        if (self.ser is not None and self.ser.loaded and not self._ser_busy
                and t_session_ms - self._last_ser_ms >= self.ser_hop_ms):
            self._last_ser_ms = t_session_ms
            self._ser_busy = True
            window = self._tail_audio(int(self.ser_window_s * 1000))
            asyncio.get_running_loop().run_in_executor(
                None, self._run_ser, window)

    async def close(self) -> None:
        self._word_task.cancel()
        self.bus.unsubscribe(self._word_queue)

    # -- internals --
    def _tail_audio(self, ms: int) -> np.ndarray:
        need = int(self.sample_rate * ms / 1000)
        if not self._chunks:
            return np.zeros(0, dtype=np.float32)
        audio = np.concatenate(self._chunks)
        return audio[-need:]

    def _run_ser(self, window: np.ndarray) -> None:
        try:
            if window.size >= self.sample_rate // 2:
                self._ser_result = self.ser.infer(window, self.sample_rate)
        except Exception:
            log.exception("ser inference failed")
        finally:
            self._ser_busy = False

    def _recent_words(self, window_ms: int) -> list[dict]:
        cut = self.t_ms - window_ms
        return [w for w in self._words if w["t_end_ms"] >= cut]

    def compute_tempo_wps(self, window_ms: int | None = None) -> float:
        window_ms = window_ms or self.tempo_window_ms
        words = self._recent_words(window_ms)
        if not words:
            return 0.0
        span_ms = max(500, self.t_ms - max(0, self.t_ms - window_ms))
        return len(words) * 1000 / span_ms

    def compute_hesitation(self, pitch_slope: float | None) -> float:
        """0..1 composite: filler rate + intra-speech pause density + pitch rise."""
        words = self._recent_words(self.hesitation_window_ms)
        filler_score = 0.0
        pause_score = 0.0
        if words:
            n_fillers = sum(1 for w in words if normalize_word(w["w"]) in self.fillers)
            filler_score = min(1.0, n_fillers / 3)
            gaps = [max(0, b["t_start_ms"] - a["t_end_ms"])
                    for a, b in zip(words, words[1:])]
            if gaps:
                pause_score = min(1.0, sum(1 for g in gaps if g > 350) / 3)
        rise_score = 1.0 if (pitch_slope is not None
                             and pitch_slope > self.cfg.commit.pitch_rise_hz_per_s) else 0.0
        return round(0.5 * filler_score + 0.3 * pause_score + 0.2 * rise_score, 3)

    def _publish_frame(self) -> None:
        warmed = self.t_ms >= self.warmup_ms

        energy = rms_dbfs(self._tail_audio(self.energy_window_ms))
        self._energy_norm.add(energy)
        energy_bucket = self._energy_norm.bucket(energy, self.cold["energy"], warmed)

        pstats = pitch_stats(self._tail_audio(self.pitch_window_ms),
                             self.sample_rate, self.pitch_tail_ms / 1000)

        wps = self.compute_tempo_wps()
        if wps > 0:
            self._tempo_norm.add(wps)
        tempo_bucket = (self._tempo_norm.bucket(wps, self.cold["tempo"], warmed)
                        if wps > 0 else "low")

        hes = self.compute_hesitation(pstats["slope_hz_s"])
        hes_bucket = bucket_absolute(hes, (0.25, 0.5))

        ser = self._ser_result or {}
        valence = ser.get("valence")
        arousal = ser.get("arousal")

        frame = {
            "energy": {"raw_dbfs": round(energy, 1), "bucket": energy_bucket},
            "pitch": {
                "median_hz": None if pstats["median_hz"] is None else round(pstats["median_hz"], 1),
                "slope_hz_s": None if pstats["slope_hz_s"] is None else round(pstats["slope_hz_s"], 1),
                "voiced_ratio": pstats["voiced_ratio"],
            },
            "tempo": {"wps": round(wps, 2), "bucket": tempo_bucket},
            "hesitation": {"score": hes, "bucket": hes_bucket},
            "valence": {"raw": None if valence is None else round(valence, 3),
                        "bucket": "med" if valence is None
                        else bucket_absolute(valence, self.cold["valence"])},
            "arousal": {"raw": None if arousal is None else round(arousal, 3),
                        "bucket": "med" if arousal is None
                        else bucket_absolute(arousal, self.cold["arousal"])},
            "calibrating": not warmed,
        }
        self.latest = frame
        self.bus.publish("prosody_frame", frame, self.t_ms)

    def tone_attributes(self) -> dict:
        """Discrete attributes for the LLM, from the latest frame."""
        f = self.latest
        if not f:
            return {"energy": "med", "valence": "med", "tempo": "med", "hesitation": "low"}
        return {
            "energy": f["energy"]["bucket"],
            "valence": f["valence"]["bucket"],
            "tempo": f["tempo"]["bucket"],
            "hesitation": f["hesitation"]["bucket"],
        }

    def tone_line(self) -> str:
        t = self.tone_attributes()
        return (f"tone: energy={t['energy']} valence={t['valence']} "
                f"tempo={t['tempo']} hesitation={t['hesitation']}")
