"""Editor-LLM latency bench: the real call shape, N times, p50/p95.

Uses the production system prompt, message builder, and guided-JSON patch
schema (including history_action enums), so speculative-decoding configs
are measured under structured output — the combination that actually runs
in production, and the one most likely to silently disable speculation.

  docker compose exec orchestrator python /srv/app/tools/llm_latency.py [N]
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
import urllib.request

from psirens_voice.editor.prompts import SYSTEM_PROMPT, build_user_message
from psirens_voice.editor.schema import patch_schema
from psirens_voice.scene.state import SceneState

BASE = os.environ.get("BENCH_URL", "http://vllm:8000/v1")
MODEL = os.environ.get("EDITOR_MODEL", "nvidia/Qwen3.6-35B-A3B-NVFP4")

STATE = SceneState(subject="a red Lamborghini with a black spoiler",
                   setting="in front of a modern minimalist house",
                   lighting="warm sunset glow", palette="vivid",
                   mood="confident", composition="centered")
LEDGER = ('#1 "A red Lamborghini in front of a modern house" | subject,setting\n'
          '#2 "Add a black spoiler to the car" | update:lamborghini\n'
          '#3 "Make the sky a sunset" | lighting')
DIRECTIVES = [
    "make the headlights glow yellow",
    "add a palm tree on the left side",
    "make the wheels gold",
    "remove the spoiler",
    "make the whole scene slightly darker",
    "put racing stripes on the hood",
]


def call(directive: str) -> tuple[float, int]:
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(
                STATE, "tone: energy=med valence=med tempo=med hesitation=low",
                directive, ledger=LEDGER)},
        ],
        "temperature": 0.2,
        "max_tokens": 256,
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "scene_patch", "schema": patch_schema([1, 2, 3])}},
    }
    req = urllib.request.Request(
        f"{BASE}/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=60) as r:
        out = json.loads(r.read())
    dt = time.monotonic() - t0
    toks = out.get("usage", {}).get("completion_tokens", 0)
    json.loads(out["choices"][0]["message"]["content"])   # must be valid JSON
    return dt, toks


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    call(DIRECTIVES[0])                                    # warmup, uncounted
    times, tokens = [], []
    for i in range(n):
        dt, toks = call(DIRECTIVES[i % len(DIRECTIVES)])
        times.append(dt)
        tokens.append(toks)
        print(f"  call {i+1:2d}: {dt*1000:6.0f}ms  {toks:3d} tok "
              f"({toks/dt:5.1f} tok/s)")
    ms = sorted(t * 1000 for t in times)
    print(f"p50={statistics.median(ms):.0f}ms  "
          f"p95={ms[int(0.95 * (len(ms) - 1))]:.0f}ms  "
          f"mean_tok/s={sum(tokens)/sum(times):.1f}")


if __name__ == "__main__":
    main()
