"""Discrete low/med/high bucketing.

Two modes per attribute:
  - absolute: fixed [low_edge, high_edge] from config (cold start, and
    permanently for attributes whose raw scale is already calibrated:
    valence, arousal, hesitation)
  - adaptive: per-session robust z-score (median + MAD over a rolling window)
    once warmup has passed (energy, tempo)
"""
from __future__ import annotations

from collections import deque

import numpy as np


def bucket_absolute(x: float, edges: tuple[float, float]) -> str:
    if x < edges[0]:
        return "low"
    if x > edges[1]:
        return "high"
    return "med"


class RobustNorm:
    def __init__(self, window: int = 240, z_edges: tuple[float, float] = (-0.5, 0.5)):
        self.values: deque[float] = deque(maxlen=window)
        self.z_edges = z_edges

    def add(self, x: float) -> None:
        self.values.append(x)

    @property
    def n(self) -> int:
        return len(self.values)

    def bucket(self, x: float, cold_edges: tuple[float, float], warmed_up: bool) -> str:
        if not warmed_up or self.n < 8:
            return bucket_absolute(x, cold_edges)
        arr = np.array(self.values)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        scale = 1.4826 * mad + 1e-6
        z = (x - med) / scale
        if z < self.z_edges[0]:
            return "low"
        if z > self.z_edges[1]:
            return "high"
        return "med"
