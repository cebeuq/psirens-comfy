"""Streaming speech-to-text via Kyutai STT (moshi PyTorch package).

Model facts (verified on spark1 GB10, kyutai/stt-1b-en_fr-candle):
  - input: 24kHz mono, frame_size 1920 samples (80ms), frame_rate 12.5Hz
  - per-step latency ~15ms on GB10 (budget: 80ms)
  - words arrive audio_delay_seconds (~0.5s) behind the audio
  - token id 0 = word boundary ("end_of_padding"), 3 = padding
  - lm_gen.step_with_extra_heads() returns (text_tokens, 4 VAD heads);
    heads[2][0,0,0] is the end-of-turn probability used by Kyutai's
    reference scripts

The engine loads once per process. Each websocket session gets an SttSession
running a dedicated worker thread (GPU calls are blocking); events are pushed
back onto the asyncio loop via call_soon_threadsafe.
"""
from __future__ import annotations

import logging
import queue
import threading
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .events import EventBus

log = logging.getLogger("psirens.stt")

BOUNDARY_TOKEN = 0
PADDING_TOKEN = 3
EOT_HEAD_INDEX = 2  # head used for end-of-turn probability (reference scripts)
# Kyutai only emits a word's boundary token when the NEXT word begins, which
# can be seconds later. A word whose last piece is this many frames old is
# finalized anyway — piece tokens of one word arrive near-consecutively, so a
# gap this long (720ms >> the model's 500ms delay) means the word is done.
WORD_STABLE_FRAMES = 9


class SttEngine:
    """Holds the loaded model; create one per process, load() once."""

    def __init__(self, hf_repo: str):
        self.hf_repo = hf_repo
        self.loaded = False
        self.mimi = None
        self.tokenizer = None
        self.lm_gen = None
        self.frame_size = 1920
        self.sample_rate = 24000
        self.frame_rate = 12.5
        self.delay_ms = 500
        self._session_lock = threading.Lock()  # one streaming session at a time

    def load(self) -> None:
        import torch
        from moshi.models import LMGen, loaders

        log.info("loading %s ...", self.hf_repo)
        info = loaders.CheckpointInfo.from_hf_repo(self.hf_repo)
        self.mimi = info.get_mimi(device="cuda")
        self.tokenizer = info.get_text_tokenizer()
        lm = info.get_moshi(device="cuda", dtype=torch.bfloat16)
        self.lm_gen = LMGen(lm, temp=0, temp_text=0.0)

        self.sample_rate = int(self.mimi.sample_rate)
        self.frame_size = int(self.mimi.frame_size)
        self.frame_rate = float(self.mimi.frame_rate)
        self.delay_ms = int(info.stt_config.get("audio_delay_seconds", 0.5) * 1000)
        self.loaded = True
        log.info("stt ready: %dHz frame=%d delay=%dms",
                 self.sample_rate, self.frame_size, self.delay_ms)

    def acquire(self) -> bool:
        return self._session_lock.acquire(blocking=False)

    def release(self) -> None:
        try:
            self._session_lock.release()
        except RuntimeError:
            pass


class SttSession:
    """Per-session streaming state + worker thread. AudioSink protocol."""

    def __init__(self, engine: SttEngine, bus: "EventBus", loop, vad_event_hz: float = 4.0):
        self.engine = engine
        self.bus = bus
        self.loop = loop
        self.q: queue.Queue = queue.Queue(maxsize=100)
        self._vad_interval_ms = 1000 / vad_event_hz if vad_event_hz > 0 else 0

        self._buf = b""
        self._frame_idx = 0
        self._t_ms = 0                 # session clock of latest fed audio
        self._word_tokens: list[int] = []
        self._word_start_frame = 0
        self._word_last_piece_frame = 0
        self._last_activity_frame = -100
        self._last_vad_pub_ms = -1e9

        self.thread = threading.Thread(target=self._run, name="stt-worker", daemon=True)
        self.thread.start()

    # -- AudioSink --
    async def feed(self, pcm: bytes, t_session_ms: int) -> None:
        try:
            self.q.put_nowait((pcm, t_session_ms))
        except queue.Full:
            self._publish("error", {"where": "stt", "recoverable": True,
                                    "message": "stt worker behind; dropped audio"})

    async def close(self) -> None:
        self.q.put(None)
        self.thread.join(timeout=10)

    # -- worker --
    def _publish(self, type_: str, data: dict) -> None:
        t = self._t_ms
        self.loop.call_soon_threadsafe(self.bus.publish, type_, data, t)

    def _audio_ms(self, frame_idx: int) -> int:
        """Session-audio time (ms) that a model frame refers to, delay-corrected."""
        return max(0, round(frame_idx * 1000 / self.engine.frame_rate) - self.engine.delay_ms)

    def _decode_word(self) -> str:
        pieces = [self.engine.tokenizer.id_to_piece(t) for t in self._word_tokens]
        return "".join(pieces).replace("▁", " ").strip()

    def _run(self) -> None:
        import torch

        eng = self.engine
        try:
            with eng.mimi.streaming(1), eng.lm_gen.streaming(1):
                flushing = False
                flush_frames_left = 0
                while True:
                    if not flushing:
                        item = self.q.get()
                        if item is None:
                            # flush: push silence to cover the model delay
                            flushing = True
                            flush_frames_left = int(eng.delay_ms / 1000 * eng.frame_rate) + 3
                            continue
                        pcm, self._t_ms = item
                        self._buf += pcm
                    else:
                        if flush_frames_left <= 0:
                            break
                        flush_frames_left -= 1
                        self._buf += b"\x00\x00" * eng.frame_size

                    frame_bytes = eng.frame_size * 2
                    while len(self._buf) >= frame_bytes:
                        chunk_bytes, self._buf = self._buf[:frame_bytes], self._buf[frame_bytes:]
                        samples = np.frombuffer(chunk_bytes, dtype=np.int16)
                        samples = samples.astype(np.float32) / 32768.0
                        chunk = torch.from_numpy(samples).to("cuda").view(1, 1, -1)
                        self._step(chunk)
        except Exception as e:  # worker must never die silently
            log.exception("stt worker crashed")
            self._publish("error", {"where": "stt", "recoverable": False, "message": str(e)})
        finally:
            # emit any trailing word
            if self._word_tokens:
                self._emit_word()

    def _emit_word(self) -> None:
        """End timestamp = the word's last piece token's frame — NOT the frame
        where the next boundary token shows up. The boundary can arrive only
        when the next word begins, which may be seconds later; using it as the
        end would swallow the pause the commit detector needs to measure."""
        word = self._decode_word()
        end_frame = self._word_last_piece_frame + 1
        self._word_tokens = []
        if not word:
            return
        self._publish("word", {
            "w": word,
            "t_start_ms": self._audio_ms(self._word_start_frame),
            "t_end_ms": self._audio_ms(end_frame),
        })
        self._publish("partial_transcript", {"pending": ""})

    def _step(self, chunk) -> None:
        eng = self.engine
        audio_tokens = eng.mimi.encode(chunk)
        text_tokens, vad_heads = eng.lm_gen.step_with_extra_heads(audio_tokens)
        self._frame_idx += 1
        f = self._frame_idx

        token = int(text_tokens[0, 0, 0].item())
        if token == BOUNDARY_TOKEN:
            if self._word_tokens:
                self._emit_word()
        elif token == PADDING_TOKEN:
            if (self._word_tokens
                    and f - self._word_last_piece_frame >= WORD_STABLE_FRAMES):
                self._emit_word()
        else:
            if not self._word_tokens:
                self._word_start_frame = f
            self._word_tokens.append(token)
            self._word_last_piece_frame = f
            self._last_activity_frame = f
            self._publish("partial_transcript", {"pending": self._decode_word()})

        # VAD state: full rate into the bus/recorder; the websocket forwarder
        # throttles what the browser sees.
        heads = [round(float(h[0, 0, 0].item()), 4) for h in vad_heads]
        speaking = (f - self._last_activity_frame) * 1000 / eng.frame_rate < 600
        self._publish("vad_state", {
            "p_eot": heads[EOT_HEAD_INDEX],
            "heads": heads,
            "speaking": speaking,
        })
