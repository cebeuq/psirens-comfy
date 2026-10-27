"""Speech-emotion lane: arousal / valence via
audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim.

The model outputs (arousal, dominance, valence) each roughly in [0, 1] and
expects 16kHz mono float input. The regression head below is vendored from the
model card (the checkpoint has no auto-loadable head class).

LICENSE NOTE: this checkpoint is CC-BY-NC-SA (non-commercial). Fine for this
prototype; must be replaced before any commercial use.
"""
from __future__ import annotations

import os

import logging

import numpy as np

log = logging.getLogger("psirens.ser")

SER_SR = 16000


class SerEngine:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.loaded = False
        self._model = None
        self._processor = None

    def load(self) -> None:
        if self.loaded:
            return
        # Deliberately avoids from_pretrained on a custom subclass: the
        # model-card pattern (EmotionModel(Wav2Vec2PreTrainedModel)) breaks on
        # transformers v5 internals. We construct plain modules from the config
        # and load the checkpoint state dict split by prefix instead.
        import safetensors.torch
        import torch
        import torch.nn as nn
        from huggingface_hub import hf_hub_download
        from transformers import Wav2Vec2Config
        from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model

        log.info("loading SER model %s ...", self.model_name)
        config = Wav2Vec2Config.from_pretrained(self.model_name)

        class RegressionHead(nn.Module):
            def __init__(self, cfg):
                super().__init__()
                self.dense = nn.Linear(cfg.hidden_size, cfg.hidden_size)
                self.out_proj = nn.Linear(cfg.hidden_size, cfg.num_labels)

            def forward(self, features):
                return self.out_proj(torch.tanh(self.dense(features)))

        wav2vec2 = Wav2Vec2Model(config)
        head = RegressionHead(config)

        sd = safetensors.torch.load_file(
            hf_hub_download(self.model_name, "model.safetensors"))
        w2v_sd = {k[len("wav2vec2."):]: v for k, v in sd.items()
                  if k.startswith("wav2vec2.")}
        head_sd = {k[len("classifier."):]: v for k, v in sd.items()
                   if k.startswith("classifier.")}
        missing, unexpected = wav2vec2.load_state_dict(w2v_sd, strict=False)
        if unexpected:
            log.warning("ser: unexpected wav2vec2 keys: %s", unexpected[:5])
        real_missing = [k for k in missing if "position" not in k]
        if real_missing:
            raise RuntimeError(f"SER checkpoint missing keys: {real_missing[:5]}")
        head.load_state_dict(head_sd, strict=True)

        model = nn.ModuleDict({"wav2vec2": wav2vec2, "classifier": head}).eval()
        # GB10/sm_121: cuDNN can't find an engine for these conv1d shapes, so
        # cuDNN is disabled for the SER forward (see infer). Verify the GPU
        # path with a probe; fall back to CPU if it still fails.
        # SER_DEVICE=cpu keeps the GPU for STT + the editor LLM on small
        # cards (one 3s window per second is cheap enough on CPU)
        self.device = os.environ.get("SER_DEVICE", "cuda")
        self._model = model.to(self.device)
        try:
            self._probe()
        except RuntimeError as e:
            log.warning("ser on cuda failed (%s); falling back to cpu", e)
            self.device = "cpu"
            self._model = model.to(self.device)
            self._probe()
        self.loaded = True
        log.info("ser ready on %s", self.device)

    def unload(self) -> None:
        """Release GPU memory (the free-GPU control). load() reverses it."""
        if self._model is None:
            return
        import gc

        import torch

        self._model = None
        self._processor = None
        self.loaded = False
        gc.collect()
        torch.cuda.empty_cache()
        log.info("ser unloaded (gpu freed)")

    def _probe(self) -> None:
        self.infer(np.zeros(SER_SR, dtype=np.float32), SER_SR)

    def infer(self, audio: np.ndarray, sr: int) -> dict:
        """audio: float32 [-1,1] mono. Returns arousal/dominance/valence in ~[0,1].
        Blocking (GPU); call from an executor thread."""
        import torch

        if sr != SER_SR:
            n_out = int(round(audio.size * SER_SR / sr))
            audio = np.interp(np.linspace(0, audio.size - 1, n_out),
                              np.arange(audio.size), audio)
        # zero-mean / unit-variance, as the repo's feature extractor does
        audio = (audio - audio.mean()) / (audio.std() + 1e-7)
        x = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad(), torch.backends.cudnn.flags(enabled=False):
            hidden = self._model["wav2vec2"](x)[0].mean(dim=1)
            out = self._model["classifier"](hidden)[0].float().cpu().numpy()
        arousal, dominance, valence = (float(v) for v in out)
        return {"arousal": arousal, "dominance": dominance, "valence": valence}
