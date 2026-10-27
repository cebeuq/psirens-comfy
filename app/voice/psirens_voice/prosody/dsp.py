"""Signal-processing lane of the prosody engine: energy, pitch, pitch slope.

Pure numpy — deliberately no torchaudio/librosa (the NGC image lacks
torchaudio and installing one that matches the NVIDIA torch build is risky).
Pitch uses FFT autocorrelation per 25ms frame with a voicing threshold; that
is plenty for prosody (median F0 and tail slope), which needs trends, not
cent-accurate pitch.
"""
from __future__ import annotations

import numpy as np

PITCH_SR = 8000          # pitch runs on audio decimated to 8kHz
F0_MIN, F0_MAX = 60.0, 400.0
FRAME_S = 0.025
HOP_S = 0.010
VOICING_THRESHOLD = 0.55


def rms_dbfs(samples: np.ndarray) -> float:
    """samples: float32 in [-1, 1]."""
    if samples.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    return 20 * np.log10(rms) if rms > 1e-6 else -120.0


def _resample_to(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out or x.size == 0:
        return x
    n_out = int(round(x.size * sr_out / sr_in))
    return np.interp(np.linspace(0, x.size - 1, n_out), np.arange(x.size), x)


def _frame_f0(frame: np.ndarray, sr: int) -> float | None:
    """Autocorrelation pitch for one frame; None if unvoiced."""
    frame = frame - frame.mean()
    energy = float(np.dot(frame, frame))
    if energy < 1e-6:
        return None
    n = frame.size
    # FFT autocorrelation
    size = 1 << (2 * n - 1).bit_length()
    spec = np.fft.rfft(frame, size)
    ac = np.fft.irfft(spec * np.conj(spec))[:n]
    if ac[0] <= 0:
        return None
    ac = ac / ac[0]
    lag_min = int(sr / F0_MAX)
    lag_max = min(int(sr / F0_MIN), n - 1)
    if lag_max <= lag_min:
        return None
    seg = ac[lag_min:lag_max]
    peak = int(np.argmax(seg))
    if seg[peak] < VOICING_THRESHOLD:
        return None
    return sr / (lag_min + peak)


def pitch_track(samples: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (times_s, f0_hz) for voiced frames only. samples: float32 [-1,1]."""
    x = _resample_to(samples, sr, PITCH_SR)
    flen = int(FRAME_S * PITCH_SR)
    hop = int(HOP_S * PITCH_SR)
    times, f0s = [], []
    for start in range(0, x.size - flen, hop):
        f0 = _frame_f0(x[start:start + flen], PITCH_SR)
        if f0 is not None:
            times.append((start + flen / 2) / PITCH_SR)
            f0s.append(f0)
    return np.array(times), np.array(f0s)


def pitch_stats(samples: np.ndarray, sr: int, tail_s: float) -> dict:
    """Median F0 over the window, slope (Hz/s) over the trailing tail_s of
    voiced frames, and the voiced ratio."""
    times, f0s = pitch_track(samples, sr)
    total_frames = max(1, int((samples.size / sr - FRAME_S) / HOP_S))
    if f0s.size == 0:
        return {"median_hz": None, "slope_hz_s": None, "voiced_ratio": 0.0}

    window_end = samples.size / sr
    tail_mask = times >= (window_end - tail_s)
    slope = None
    if tail_mask.sum() >= 4:
        t, f = times[tail_mask], f0s[tail_mask]
        slope = float(np.polyfit(t, f, 1)[0])

    return {
        "median_hz": float(np.median(f0s)),
        "slope_hz_s": slope,
        "voiced_ratio": round(min(1.0, f0s.size / total_frames), 3),
    }
