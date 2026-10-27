"""Latency summary from a session's events.jsonl.

Measures (p50/p95):
  - word_lag:      wall arrival of a word vs. when its audio ended
                   (includes the model's built-in ~500ms delay)
  - commit_lag:    wall arrival of a commit vs. end of the utterance's audio
  - patch_latency: commit -> patch_applied wall time (also the LLM's own
                   latency_ms as reported by the editor)

Run: python -m psirens_voice.tools.latency_report <events.jsonl>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def pct(xs: list[float], p: float) -> float:
    return float(np.percentile(np.array(xs), p)) if xs else float("nan")


def report(path: Path) -> None:
    events = [json.loads(l) for l in open(path)]
    if not events:
        raise SystemExit("empty event log")

    # wall clock anchor: wall time minus session-audio time, per event; use the
    # median offset as "wall time when session audio position was 0"
    offsets = [e["t_wall"] - e["t_session_ms"] for e in events
               if e["type"] == "audio_level"]
    if not offsets:
        offsets = [events[0]["t_wall"] - events[0]["t_session_ms"]]
    w0 = float(np.median(offsets))

    word_lags = [e["t_wall"] - (w0 + e["data"]["t_end_ms"])
                 for e in events if e["type"] == "word"]
    commit_lags = [e["t_wall"] - (w0 + e["data"]["t_speech_end_ms"])
                   for e in events if e["type"] == "commit"]

    commits = {e["seq"]: e for e in events if e["type"] == "commit"}
    patch_walls, llm_latencies = [], []
    for e in events:
        if e["type"] == "patch_applied" and e["data"].get("kind") == "edit":
            llm_latencies.append(e["data"]["latency_ms"])
    # commit -> patch wall time: pair each patch with the latest prior commit
    commit_walls = sorted((e["t_wall"], e) for e in events if e["type"] == "commit")
    for e in events:
        if e["type"] == "patch_applied" and e["data"].get("kind") == "edit":
            prior = [w for w, _ in commit_walls if w <= e["t_wall"]]
            if prior:
                patch_walls.append(e["t_wall"] - prior[-1])

    def line(name, xs, unit="ms"):
        if not xs:
            print(f"{name:24s} n=0")
            return
        print(f"{name:24s} n={len(xs):<4d} p50={pct(xs,50):6.0f}{unit} "
              f"p95={pct(xs,95):6.0f}{unit} max={max(xs):6.0f}{unit}")

    print(f"session: {path}")
    line("word finalization lag", word_lags)
    line("utterance-end -> commit", commit_lags)
    line("commit -> patch applied", patch_walls)
    line("LLM latency (reported)", llm_latencies)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("events", type=Path)
    args = ap.parse_args()
    report(args.events)


if __name__ == "__main__":
    main()
