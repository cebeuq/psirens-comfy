"""Streaming speech-to-text via Voxtral-Mini-4B-Realtime on vLLM.

Words come from a vLLM realtime WebSocket (`/v1/realtime`, OpenAI-style
events: input_audio_buffer.append with base64 PCM16@16k up,
transcription.delta down). The end-of-turn signal keeps using the local
Parakeet EOU model (NemotronSttEngine in EOU-only mode) — Voxtral has no
EOU head, and the commit gate's p_eot semantics are already tuned to
Parakeet.

Timestamps: the realtime API does not expose them, but the audio clock
does — we know exactly how many milliseconds of audio we have appended
when a delta arrives, and the model's streaming delay is a fixed
configuration constant. A finalized word's t_end is (audio_sent_ms -
DELAY_MS), clamped to be monotonic; t_start is estimated from word
length. The commit detector only needs t_end at ~100ms precision (its
own delay_ms constant absorbs the rest).
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from typing import TYPE_CHECKING

import numpy as np

from .stt_nemotron import NemotronSttEngine, NemotronSttSession

if TYPE_CHECKING:
    from .events import EventBus

log = logging.getLogger("psirens.stt")

VOXTRAL_WS = os.environ.get("VOXTRAL_WS",
                            "ws://voxtral:8000/v1/realtime")
VOXTRAL_MODEL = os.environ.get(
    "VOXTRAL_MODEL", "mistralai/Voxtral-Mini-4B-Realtime-2602")
# the model's configured streaming text delay (audio-to-word lag)
DELAY_MS = int(os.environ.get("VOXTRAL_DELAY_MS", "480"))
SR_IN = 24000
SR_OUT = 16000
# a trailing word with no delta growth for this long is final
TAIL_FINAL_MS = 550


class VoxtralSttEngine:
    """Engine facade: the transcriber lives in the voxtral container; the
    local piece is the EOU model. load()/unload() manage the EOU weights
    (the voxtral server has its own sleep mode, driven by /gpu)."""

    def __init__(self, hf_repo: str = ""):
        self.eou_engine = NemotronSttEngine(asr_enabled=False)
        self.ws_url = VOXTRAL_WS
        self.sample_rate = SR_IN
        self.frame_size = 1920          # 80ms @24k, the browser frame
        self.delay_ms = DELAY_MS + 150  # commit-gate observed-pause offset

    @property
    def loaded(self) -> bool:
        return self.eou_engine.loaded

    @loaded.setter
    def loaded(self, v: bool) -> None:
        self.eou_engine.loaded = v

    def load(self) -> None:
        self.eou_engine.load()

    def unload(self) -> None:
        self.eou_engine.unload()

    def acquire(self) -> bool:
        return self.eou_engine.acquire()

    def release(self) -> None:
        self.eou_engine.release()


class VoxtralSttSession:
    """AudioSink: fans audio out to (a) the local EOU stream for p_eot
    and (b) the voxtral realtime WebSocket for words."""

    def __init__(self, engine: VoxtralSttEngine, bus: "EventBus", loop,
                 vad_event_hz: float = 4.0):
        self.engine = engine
        self.bus = bus
        self.loop = loop
        self.eou_session = NemotronSttSession(
            engine.eou_engine, bus, loop, vad_event_hz=vad_event_hz)
        self._q: asyncio.Queue = asyncio.Queue(maxsize=400)
        self._t_ms = 0
        self._audio_ms = 0.0          # session audio appended so far
        self._text = ""               # accumulated stream text
        self._words_final = 0
        self._prev_end = 0
        self._last_growth_ms = 0.0    # audio clock at last text growth
        self._closed = False
        self._task = asyncio.ensure_future(self._ws_worker(), loop=loop)

    # -- AudioSink --
    async def feed(self, pcm: bytes, t_session_ms: int) -> None:
        self._t_ms = t_session_ms
        await self.eou_session.feed(pcm, t_session_ms)
        x24 = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        n16 = int(len(x24) * SR_OUT / SR_IN)
        x16 = np.interp(np.linspace(0, len(x24) - 1, n16),
                        np.arange(len(x24)), x24)
        self._audio_ms += len(x24) / SR_IN * 1000.0
        b64 = base64.b64encode(
            x16.astype(np.int16).tobytes()).decode()
        try:
            self._q.put_nowait(b64)
        except asyncio.QueueFull:
            self._publish("error", {
                "where": "stt", "recoverable": True,
                "message": "voxtral sender behind; dropped audio"})

    async def close(self) -> None:
        self._closed = True
        await self.eou_session.close()
        try:
            self._q.put_nowait(None)
        except asyncio.QueueFull:
            pass
        try:
            await asyncio.wait_for(self._task, timeout=8)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            self._task.cancel()

    def _publish(self, type_: str, data: dict) -> None:
        self.bus.publish(type_, data, self._t_ms)

    # -- words from stream text --
    def _on_delta(self, delta: str) -> None:
        if not delta:
            return
        grew = bool(delta.strip())
        self._text += delta
        if grew:
            self._last_growth_ms = self._audio_ms
        words = self._text.split()
        # every word with a successor is final
        while self._words_final < max(0, len(words) - 1):
            self._emit_word(words[self._words_final])
            self._words_final += 1
        pending = " ".join(words[self._words_final:])
        self._publish("partial_transcript", {"pending": pending})

    def _maybe_finalize_tail(self, force: bool = False) -> None:
        words = self._text.split()
        stale = (self._audio_ms - self._last_growth_ms) >= TAIL_FINAL_MS
        if self._words_final < len(words) and (force or stale):
            while self._words_final < len(words):
                self._emit_word(words[self._words_final])
                self._words_final += 1
            self._publish("partial_transcript", {"pending": ""})

    def _emit_word(self, w: str) -> None:
        t_end = int(max(self._prev_end + 40, self._audio_ms - DELAY_MS))
        t_start = int(max(self._prev_end,
                          t_end - min(700, max(120, 80 * len(w)))))
        self._prev_end = t_end
        self._publish("word", {"w": w, "t_start_ms": t_start,
                               "t_end_ms": t_end})

    # -- websocket worker --
    async def _ws_worker(self) -> None:
        """Outer loop: one realtime connection at a time, ROTATED during a
        quiet moment before the server's context window (~13min of audio)
        fills — the mic streams continuously, so a single connection would
        die mid-session otherwise."""
        while not self._closed:
            outcome = await self._run_connection()
            if outcome != "rotate":
                break
            # fresh stream: per-connection text state resets
            self._text = ""
            self._words_final = 0
            self._conn_start_ms = self._audio_ms
            log.info("voxtral connection rotated at %.0fs of audio",
                     self._audio_ms / 1000)

    async def _run_connection(self) -> str:
        import websockets

        self._conn_start_ms = getattr(self, "_conn_start_ms",
                                      self._audio_ms)
        ws = None
        for attempt in range(4):
            try:
                ws = await websockets.connect(
                    self.engine.ws_url, max_size=None, open_timeout=30)
                break
            except Exception as e:
                if self._closed:
                    return
                log.warning("voxtral connect failed (%s), retry %d",
                            e, attempt + 1)
                await asyncio.sleep(1.5 * (attempt + 1))
        if ws is None:
            self._publish("error", {
                "where": "stt", "recoverable": False,
                "message": "voxtral realtime unreachable"})
            return "failed"
        try:
            await ws.send(json.dumps({"type": "session.update",
                                      "model": VOXTRAL_MODEL}))
            # generation only STARTS on a non-final commit (appends just
            # queue audio server-side; nothing transcribes without this)
            await ws.send(json.dumps({"type": "input_audio_buffer.commit",
                                      "final": False}))

            async def sender():
                while True:
                    b64 = await self._q.get()
                    if b64 is None:
                        try:
                            await ws.send(json.dumps({
                                "type": "input_audio_buffer.commit",
                                "final": True}))
                        except Exception:
                            pass
                        return
                    await ws.send(json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": b64}))

            send_task = asyncio.ensure_future(sender())
            outcome = "done"
            while True:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=0.4)
                except asyncio.TimeoutError:
                    self._maybe_finalize_tail()
                    if send_task.done() and self._q.empty():
                        break
                    # rotation: near the context ceiling AND quiet (no
                    # unfinalized words, no recent growth) -> cycle now
                    if (not self._closed
                            and self._audio_ms - self._conn_start_ms
                            > 600_000
                            and self._words_final
                            >= len(self._text.split())
                            and self._audio_ms - self._last_growth_ms
                            > 1500):
                        outcome = "rotate"
                        break
                    continue
                except Exception:
                    break
                try:
                    ev = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                et = ev.get("type", "")
                if et == "transcription.delta":
                    self._on_delta(ev.get("delta") or "")
                    # the server streams a delta event per 80ms frame even
                    # in silence (often empty), so recv() never times out —
                    # tail finalization must run on EVERY event, not only
                    # in the timeout branch, or the trailing word of each
                    # utterance stays pending forever and no turn can end
                    self._maybe_finalize_tail()
                elif et == "transcription.done":
                    final = ev.get("text")
                    if final and len(final) > len(self._text):
                        self._on_delta(final[len(self._text):])
                    self._maybe_finalize_tail(force=True)
                    break
                elif et == "error":
                    log.warning("voxtral error event: %s", ev.get("error"))
                elif et not in ("session.created", "session.updated"):
                    log.debug("voxtral event ignored: %s", et)
            send_task.cancel()
            self._maybe_finalize_tail(force=(outcome != "rotate"))
            return outcome
        except Exception:
            log.exception("voxtral ws worker failed")
            self._publish("error", {
                "where": "stt", "recoverable": False,
                "message": "voxtral stream failed"})
            return "failed"
        finally:
            try:
                await ws.close()
            except Exception:
                pass
