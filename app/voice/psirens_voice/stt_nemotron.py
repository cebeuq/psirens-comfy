"""Streaming speech-to-text via NVIDIA Nemotron + Parakeet EOU (NeMo).

Two cache-aware streaming FastConformer models run in lockstep on the same
16kHz feature stream:
  - nemotron-speech-streaming-en-0.6b: the transcriber. 160ms chunks
    (att_context [70,1]), punctuation + capitalization native, ~26ms of
    GPU compute per chunk on the Spark.
  - parakeet_realtime_eou_120m-v1: end-of-utterance detector. Emits token
    id 1024 (<EOU>) when the speaker is done — the semantic end-of-turn
    signal the commit gate consumes as p_eot.

GB10 workarounds (this hardware mis-executes several CUDA libraries):
  - mel/STFT preprocessing runs on CPU (cuFFT "error 50" on GPU).
  - features are computed INCREMENTALLY with 480-sample symmetric context
    and exact frame alignment — verified bit-identical to full-audio
    preprocessing (dither disabled; these models use normalize: NA).

The browser still sends 24kHz PCM (the prosody/SER path wants 24k);
this engine resamples 24k -> 16k internally.
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

ASR_MODEL = "nvidia/nemotron-speech-streaming-en-0.6b"
EOU_MODEL = "nvidia/parakeet_realtime_eou_120m-v1"
EOU_TOKEN_ID = 1024
ATT_CONTEXT = [70, 1]          # 160ms lookahead
SR_IN = 24000                  # from the browser
SR = 16000                     # model rate
HOP = 160                      # 10ms feature hop @16k
CHUNK = 2560                   # 160ms of 16k samples per step
CTX = 480                      # 3-hop STFT context each side (exactness)
FRAMES_PER_STEP = 16
# a trailing word with no text change for this many steps is final (RNNT
# rarely revises; 2 steps = 320ms)
WORD_STABLE_STEPS = 2


class NemotronSttEngine:
    """Holds both loaded models; create one per process, load() once."""

    def __init__(self, hf_repo: str = ASR_MODEL, asr_enabled: bool = True):
        self.hf_repo = hf_repo or ASR_MODEL
        self.asr_enabled = asr_enabled   # False: EOU-only (words come from
        self.loaded = False              # an external transcriber, e.g.
        self.asr = None                  # Voxtral realtime)
        self.eou = None
        self.sample_rate = SR_IN       # what the pipeline feeds us
        self.frame_size = 1920         # 80ms @24k, matches the browser frames
        self.delay_ms = 200            # lookahead (160ms) + margin
        self._session_lock = threading.Lock()

    def _load_one(self, name, ctx=None):
        import nemo.collections.asr as nemo_asr
        from omegaconf import open_dict

        m = nemo_asr.models.ASRModel.from_pretrained(model_name=name)
        m = m.to("cuda").eval()
        if ctx is not None:
            m.encoder.set_default_att_context_size(ctx)
            m.encoder.setup_streaming_params()
        # strip alignment/timestamp preservation from the decoding config:
        # the EOU checkpoint ships with it on, which makes streaming
        # transcription merging crash ('dict' has no attribute 'extend')
        dec = m.cfg.decoding
        with open_dict(dec):
            dec.preserve_alignments = False
            dec.compute_timestamps = False
            if "greedy" in dec:
                dec.greedy.preserve_alignments = False
                # the CUDA-graph label-looping decoder hit an illegal
                # memory access on RTX PRO 6000 (sm_120) + torch 2.8,
                # poisoning the whole process's CUDA context. The plain
                # decoder is a hair slower and does not do that.
                dec.greedy.use_cuda_graph_decoder = False
        m.change_decoding_strategy(dec)
        m.preprocessor.to("cpu")           # GB10: cuFFT broken on GPU
        m.preprocessor.featurizer.dither = 0.0
        return m

    def load(self) -> None:
        if self.loaded:
            return
        if self.asr_enabled:
            log.info("loading %s + %s ...", self.hf_repo, EOU_MODEL)
            self.asr = self._load_one(self.hf_repo, ctx=ATT_CONTEXT)
        else:
            log.info("loading %s (EOU only) ...", EOU_MODEL)
        self.eou = self._load_one(EOU_MODEL)
        self.loaded = True
        log.info("stt ready: %s + EOU detector",
                 "nemotron streaming" if self.asr_enabled else "external ASR")

    def unload(self) -> None:
        """Release GPU memory (the free-GPU control). load() reverses it."""
        if self.asr is None and self.eou is None:
            return
        import gc

        import torch

        self.asr = None
        self.eou = None
        self.loaded = False
        gc.collect()
        torch.cuda.empty_cache()
        log.info("stt unloaded (gpu freed)")

    def acquire(self) -> bool:
        return self._session_lock.acquire(blocking=False)

    def release(self) -> None:
        try:
            self._session_lock.release()
        except RuntimeError:
            pass


class _StreamState:
    """Per-model cache-aware streaming state, replicating the chunk
    protocol of NeMo's CacheAwareStreamingAudioBuffer exactly:

      streaming_cfg for these models is chunk_size=[9, 16] with
      pre_encode_cache_size=[0, 9] — the FIRST chunk is 9 fresh frames
      with no cache; every later chunk is the PREVIOUS 9 frames prepended
      to 16 fresh frames (drop_extra_pre_encoded then discards the
      overlap after the subsampling conv). Feeding bare 16-frame chunks
      (no prepended cache) desyncs the encoder and garbles words — the
      first live-session bug.
    """

    def __init__(self, model):
        self.model = model
        cc, ct, cl = model.encoder.get_initial_cache_state(batch_size=1)
        self.cache = [cc, ct, cl]
        self.prev_hyp = None
        self.pred_out = None
        self.step = 0
        cfg = model.encoder.streaming_cfg
        self.first_chunk = cfg.chunk_size[0]          # 9 frames
        self.chunk = cfg.chunk_size[1]                # 16 frames
        self.pre_cache = cfg.pre_encode_cache_size[1]  # 9 frames
        self.fed = 0            # fresh frames consumed from the feature buf

    def frames_needed(self, available: int) -> bool:
        if self.fed == 0:
            return available >= self.first_chunk
        return available - self.fed >= self.chunk

    def take(self, feat_buf):
        """Slice the next chunk (with prepended cache) from the feature
        buffer (D, T). Returns the tensor to feed, or None."""
        if self.fed == 0:
            chunk = feat_buf[:, :self.first_chunk]
            self.fed = self.first_chunk
        else:
            lo = self.fed - self.pre_cache
            chunk = feat_buf[:, lo:self.fed + self.chunk]
            self.fed += self.chunk
        return chunk[None]      # (1, D, T)

    def step_features(self, feats, return_transcription: bool):
        import torch

        cfg = self.model.encoder.streaming_cfg
        with torch.inference_mode():
            out = self.model.conformer_stream_step(
                processed_signal=feats,
                processed_signal_length=torch.tensor([feats.shape[2]],
                                                     device=feats.device),
                cache_last_channel=self.cache[0],
                cache_last_time=self.cache[1],
                cache_last_channel_len=self.cache[2],
                keep_all_outputs=False,
                previous_hypotheses=self.prev_hyp,
                previous_pred_out=self.pred_out,
                drop_extra_pre_encoded=(0 if self.step == 0
                                        else cfg.drop_extra_pre_encoded),
                return_transcription=return_transcription,
            )
        self.pred_out = out[0]
        self.cache = [out[2], out[3], out[4]]
        if return_transcription:
            self.prev_hyp = out[5]
        self.step += 1
        return out


class NemotronSttSession:
    """Per-session streaming worker. AudioSink protocol (same as Kyutai's)."""

    def __init__(self, engine: NemotronSttEngine, bus: "EventBus", loop,
                 vad_event_hz: float = 4.0):
        self.engine = engine
        self.bus = bus
        self.loop = loop
        self.q: queue.Queue = queue.Queue(maxsize=200)

        self._t_ms = 0
        self._x16 = np.zeros(0, dtype=np.float32)   # 16k ring
        self._processed = 0                          # samples consumed
        self._steps = 0
        # word tracking from incremental transcription diffs
        self._words_final = 0        # count of words already published
        self._word_first_step: dict[int, int] = {}
        self._word_last_step: dict[int, int] = {}
        self._last_text = ""
        self._stable_steps = 0
        self._eou_high = False
        self._eou_count = 0
        self._eou_text_len = 0
        self._last_word_step = -100

        self.thread = threading.Thread(target=self._run, name="stt-worker",
                                       daemon=True)
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

    def _step_audio_ms(self, step: int) -> int:
        """Session-audio ms a step's words refer to (lookahead-corrected).
        Step 1 consumes 90ms (the 9-frame first chunk); every later step
        consumes 160ms."""
        return max(0, 90 + max(0, step - 1) * 160 - 160)

    def _run(self) -> None:
        import torch

        eng = self.engine
        try:
            asr = _StreamState(eng.asr) if eng.asr is not None else None
            eou = _StreamState(eng.eou)
            self._asr_state = asr
            pre_asr = eng.asr.preprocessor if eng.asr is not None else None
            pre_eou = eng.eou.preprocessor
            feat_asr = None      # (D, T) growing CPU feature buffers
            feat_eou = None
            flushing = False
            flush_left = 0
            while True:
                if not flushing:
                    item = self.q.get()
                    if item is None:
                        if len(self._x16) == 0:
                            break   # zero-audio session: nothing to flush
                        flushing = True
                        flush_left = CHUNK * 3 + CTX
                        continue
                    pcm, self._t_ms = item
                    x24 = (np.frombuffer(pcm, dtype=np.int16)
                           .astype(np.float32) / 32768.0)
                    n16 = int(len(x24) * SR / SR_IN)
                    x16 = np.interp(np.linspace(0, len(x24) - 1, n16),
                                    np.arange(len(x24)), x24)
                    self._x16 = np.concatenate([self._x16,
                                                x16.astype(np.float32)])
                else:
                    if flush_left <= 0:
                        break
                    pad = min(CHUNK, flush_left)
                    self._x16 = np.concatenate(
                        [self._x16, np.zeros(pad, dtype=np.float32)])
                    flush_left -= pad

                # 1) extend the feature buffers (incremental, bit-exact)
                while len(self._x16) - self._processed >= CHUNK + CTX:
                    lo = max(0, self._processed - CTX)
                    seg = self._x16[lo:self._processed + CHUNK + CTX]
                    seg_t = torch.from_numpy(seg)[None]
                    ln = torch.tensor([seg.shape[0]])
                    with torch.no_grad():
                        if pre_asr is not None:
                            f_a, _ = pre_asr(input_signal=seg_t, length=ln)
                        f_e, _ = pre_eou(input_signal=seg_t, length=ln)
                    k0 = (self._processed - lo) // HOP
                    ne = f_e[0, :, k0:k0 + FRAMES_PER_STEP]
                    feat_eou = ne if feat_eou is None else torch.cat(
                        [feat_eou, ne], dim=1)
                    if pre_asr is not None:
                        na = f_a[0, :, k0:k0 + FRAMES_PER_STEP]
                        feat_asr = na if feat_asr is None else torch.cat(
                            [feat_asr, na], dim=1)
                    self._processed += CHUNK
                # 2) run model steps per the chunk protocol (9 first, then
                #    9-cache + 16 fresh)
                while (feat_eou is not None
                       and eou.frames_needed(feat_eou.shape[1])
                       and (asr is None
                            or asr.frames_needed(feat_asr.shape[1]))):
                    fa = asr.take(feat_asr).to("cuda") if asr else None
                    fe = eou.take(feat_eou).to("cuda")
                    self._step(asr, eou, fa, fe)
                # keep buffers bounded (drop consumed prefix, keep cache)
                if eou.fed > 4096:
                    keep = eou.pre_cache
                    drop = eou.fed - keep
                    if feat_asr is not None:
                        feat_asr = feat_asr[:, drop:]
                        asr.fed = keep
                    feat_eou = feat_eou[:, drop:]
                    eou.fed = keep
                if self._processed > SR * 30:
                    cut = self._processed - CTX
                    self._x16 = self._x16[cut:]
                    self._processed = CTX
        except Exception as e:
            log.exception("stt worker crashed")
            if "CUDA" in str(e) or "Accelerator" in type(e).__name__:
                # an illegal memory access poisons the whole process's CUDA
                # context — every later session would get a silently dead
                # STT. Flip healthz to 503 so the failure is loud and new
                # sessions are refused instead of broken.
                log.critical("CUDA context poisoned — marking engine down; "
                             "container needs a restart")
                self.engine.loaded = False
            self._publish("error", {"where": "stt", "recoverable": False,
                                    "message": str(e)})
        finally:
            self._finalize_tail(force=True)

    # -- word emission from transcription diffs --

    def _emit_word(self, idx: int, words: list[str]) -> None:
        w = words[idx]
        start = self._word_first_step.get(idx, self._steps)
        end = self._word_last_step.get(idx, self._steps)
        self._publish("word", {
            "w": w,
            "t_start_ms": self._step_audio_ms(start),
            "t_end_ms": self._step_audio_ms(end) + 80,
        })
        self._last_word_step = self._steps

    def _finalize_tail(self, force: bool = False) -> None:
        words = self._last_text.split()
        while self._words_final < len(words):
            if not force and self._words_final == len(words) - 1:
                return   # trailing word may still grow
            self._emit_word(self._words_final, words)
            self._words_final += 1
        if force and self._last_text:
            self._publish("partial_transcript", {"pending": ""})

    def _step_eou_only(self, eou: _StreamState, fe) -> None:
        eou_out = eou.step_features(fe, return_transcription=True)
        etexts = eou_out[1]
        etext = (etexts[0].text if hasattr(etexts[0], "text")
                 else str(etexts[0]))
        n_eou = etext.count("<EOU>")
        if n_eou > self._eou_count:
            self._eou_count = n_eou
            self._eou_high = True
            self._eou_text_len = len(etext)
        elif self._eou_high and len(etext) > self._eou_text_len + 1:
            self._eou_high = False     # speech resumed after the marker
        if len(etext) > getattr(self, "_etext_len", 0):
            self._etext_len = len(etext)
            self._last_word_step = self._steps
        speaking = (self._steps - self._last_word_step) * 160 < 600
        self._publish("vad_state", {
            "p_eot": 1.0 if self._eou_high else 0.02,
            "heads": [],
            "speaking": speaking,
        })

    def _step(self, asr: _StreamState, eou: _StreamState, fa, fe) -> None:
        self._steps += 1
        if asr is None:
            self._step_eou_only(eou, fe)
            return
        out = asr.step_features(fa, return_transcription=True)
        texts = out[1]
        text = texts[0].text if hasattr(texts[0], "text") else str(texts[0])

        words = text.split()
        prev_words = self._last_text.split()
        if text != self._last_text:
            self._stable_steps = 0
            for i in range(len(words)):
                if i >= len(prev_words) or words[i] != prev_words[i]:
                    if i not in self._word_first_step or i >= len(prev_words):
                        self._word_first_step[i] = self._steps
                    self._word_last_step[i] = self._steps
            self._last_text = text
            pending = " ".join(words[self._words_final:])
            self._publish("partial_transcript", {"pending": pending})
        else:
            self._stable_steps += 1

        # finalize: every word with a successor; the trailing word after
        # WORD_STABLE_STEPS of no growth
        target = len(words) - 1 if words else 0
        if self._stable_steps >= WORD_STABLE_STEPS:
            target = len(words)
        while self._words_final < target:
            self._emit_word(self._words_final, words)
            self._words_final += 1

        # EOU stream: same transcription path as the ASR; the detector's
        # transcript literally contains "<EOU>" when the speaker sounds done
        eou_out = eou.step_features(fe, return_transcription=True)
        etexts = eou_out[1]
        etext = (etexts[0].text if hasattr(etexts[0], "text")
                 else str(etexts[0]))
        n_eou = etext.count("<EOU>")
        if n_eou > self._eou_count:
            self._eou_count = n_eou
            self._eou_high = True
            self._eou_text_len = len(etext)
        elif self._eou_high and len(etext) > self._eou_text_len + 1:
            self._eou_high = False     # speech resumed after the marker

        speaking = (self._steps - self._last_word_step) * 160 < 600
        self._publish("vad_state", {
            "p_eot": 1.0 if self._eou_high else 0.02,
            "heads": [],
            "speaking": speaking,
        })
