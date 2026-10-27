"""LLM serving benchmark for the opt-llm investigation.

Measures, against the editor vLLM (the real server, real prompts):
  decode   - pure decode tok/s + TTFT, unstructured (the speed-of-light probe)
  parser   - production parser shape: SYSTEM_PROMPT + user msg, guided JSON
  parser-noschema - same prompt without response_format (guided overhead)
  parser-hist     - parser with N turns of synthetic history (prefix reuse)
  judge    - judge shape: rubric + 2 images (--px) + verdict schema
  prefill  - TTFT vs prompt length (text) at max_tokens=1
  imgfill  - TTFT vs image size at max_tokens=1

Streams (SSE) to separate TTFT from decode; decode tok/s is computed over
the inter-token window (first content chunk -> last), not the total.

  docker compose exec orchestrator python /srv/app/tools/bench_llm.py \
      --mode parser --n 8

Emits one JSON line per call and a summary line, machine-collectable.
"""
from __future__ import annotations

import argparse
import base64
import http.client
import io
import json
import os
import statistics
import sys
import time
import urllib.parse

sys.path.insert(0, "/srv/app")

from psirens_voice.editor.prompts import SYSTEM_PROMPT, build_user_message  # noqa: E402
from psirens_voice.editor.schema import patch_schema                        # noqa: E402
from psirens_voice.scene.state import Entity, SceneState                    # noqa: E402

BASE = os.environ.get("BENCH_URL", "http://vllm:8000/v1")
MODEL = os.environ.get("EDITOR_MODEL", "nvidia/Qwen3.6-35B-A3B-NVFP4")

STATE = SceneState(
    subject="a modern house",
    setting="in Joshua Tree with a pool that has a rock volcano design",
    lighting="warm sunset glow", palette="vivid", mood="confident",
    composition="centered",
    entities=[Entity(id="water_slide", name="water slide",
                     description="a bright red water slide with a "
                                 "Supreme logo",
                     box={"x": 0.68, "y": 0.45, "w": 0.25, "h": 0.3})])
LEDGER = ('#1 "A modern house in Joshua Tree" | subject,setting\n'
          '#2 "Add a red water slide" | entity:water_slide\n'
          '#3 "Make the lighting a warm sunset" | lighting')
DIRECTIVES = [
    "make the water slide go into the pool",
    "add a palm tree on the left side",
    "make the pool tiles bright blue",
    "remove the water slide",
    "make the whole scene slightly darker",
    "put string lights over the patio",
    "add a hot tub next to the pool",
    "turn the volcano into a waterfall",
]

JUDGE_RUBRIC = (
    "You judge whether an image EDIT followed its instruction. The first "
    "image is BEFORE, the second is AFTER (a low-res draft). Judge ONLY "
    "whether each numbered requirement is met. First write analysis: one "
    "short clause per requirement; keep analysis under 35 words. Then "
    "verdict. met[i] answers requirement i in order.")
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "analysis": {"type": "string", "maxLength": 240,
                     "pattern": "^[A-Za-z][A-Za-z0-9 ,.;:'()/-]{7,238}$"},
        "verdict": {"type": "string", "enum": ["pass", "fail"]},
        "met": {"type": "array", "items": {"type": "boolean"},
                "maxItems": 5},
        "what_failed": {"type": "string", "maxLength": 160,
                        "pattern":
                        "^(nothing failed|[A-Za-z][A-Za-z ,.;:'()-]{11,150})$"},
        "corrective_action": {"type": "string",
                              "enum": ["none", "reroll_seed",
                                       "strengthen_edit",
                                       "rephrase_instruction",
                                       "escalate_full_res"]},
        "revised_instruction": {"type": "string", "maxLength": 300},
        "failure_category": {"type": "string",
                             "enum": ["none", "target_missing",
                                      "wrong_region", "too_weak",
                                      "wrong_attribute", "collateral",
                                      "not_judgeable"]},
        "fail_box": {"anyOf": [
            {"type": "array", "minItems": 4, "maxItems": 4,
             "items": {"type": "integer", "minimum": 0, "maximum": 100}},
            {"type": "null"}]},
    },
    "required": ["analysis", "verdict", "met", "what_failed",
                 "corrective_action", "revised_instruction",
                 "failure_category", "fail_box"],
    "additionalProperties": False,
}


def make_jpeg(px: int, seed: int = 0) -> str:
    """Deterministic scene-ish JPEG (b64): sky, ground, house, pool."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (px, px), (135 + seed % 20, 180, 220))
    d = ImageDraw.Draw(img)
    d.rectangle([0, px * 0.55, px, px], fill=(196, 178, 128))
    d.rectangle([px * 0.15, px * 0.30, px * 0.55, px * 0.55],
                fill=(230, 228, 220))
    d.polygon([(px * 0.12, px * 0.30), (px * 0.35, px * 0.18),
               (px * 0.58, px * 0.30)], fill=(120, 90, 70))
    d.ellipse([px * 0.60, px * 0.62, px * 0.92, px * 0.82],
              fill=(60 + seed * 7 % 60, 140, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def sse_call(body: dict, timeout: float = 120.0) -> dict:
    """POST /chat/completions with stream=True; returns timing + usage."""
    body = dict(body)
    body["stream"] = True
    body["stream_options"] = {"include_usage": True}
    u = urllib.parse.urlparse(BASE)
    conn = http.client.HTTPConnection(u.hostname, u.port or 80,
                                      timeout=timeout)
    payload = json.dumps(body)
    t0 = time.monotonic()
    conn.request("POST", u.path + "/chat/completions", payload,
                 {"Content-Type": "application/json"})
    resp = conn.getresponse()
    if resp.status != 200:
        raise RuntimeError(f"HTTP {resp.status}: {resp.read()[:500]!r}")
    t_first = None
    t_last = None
    usage = {}
    text = []
    buf = b""
    while True:
        chunk = resp.read1(65536)
        if not chunk:
            break
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                continue
            ev = json.loads(data)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                delta = ch.get("delta", {}).get("content")
                if delta:
                    now = time.monotonic()
                    if t_first is None:
                        t_first = now
                    t_last = now
                    text.append(delta)
    conn.close()
    t_end = time.monotonic()
    ct = usage.get("completion_tokens", 0)
    decode_s = (t_last - t_first) if (t_first and t_last and ct > 1) else None
    return {
        "total_ms": round((t_end - t0) * 1e3),
        "ttft_ms": round((t_first - t0) * 1e3) if t_first else None,
        "decode_tok_s": round((ct - 1) / decode_s, 1)
        if decode_s and decode_s > 0 else None,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": ct,
        "content": "".join(text),
    }


def body_parser(directive: str, schema: bool = True, history_turns: int = 0,
                max_tokens: int = 700, image_b64: str | None = None,
                revs: list | None = None) -> dict:
    user = build_user_message(
        STATE, "tone: energy=med valence=med tempo=med hesitation=low",
        directive, ledger=LEDGER)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for i in range(history_turns):
        messages.append({"role": "user", "content": build_user_message(
            STATE, "tone: energy=med valence=med tempo=med hesitation=low",
            DIRECTIVES[i % len(DIRECTIVES)], ledger=None)})
        messages.append({"role": "assistant", "content": json.dumps({
            "reasoning": "simple attribute edit", "edits": [
                {"field": "lighting", "value": "warm dusk glow"}],
            "entity_edits": [], "grounding": "the scene lighting",
            "edit_instruction": "warm the lighting toward dusk",
            "edit_magnitude": 0.3, "edit_difficulty": "easy"})})
    if image_b64:
        messages.append({"role": "user", "content": [
            {"type": "text", "text": user},
            {"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{image_b64}"}}]})
    else:
        messages.append({"role": "user", "content": user})
    b = {"model": MODEL, "messages": messages, "temperature": 0.2,
         "max_tokens": max_tokens}
    if schema:
        b["response_format"] = {"type": "json_schema", "json_schema": {
            "name": "scene_patch",
            "schema": patch_schema(revs or [1, 2, 3])}}
    return b


def body_judge(px: int, seed: int, schema: bool = True) -> dict:
    reqs = ("1. The pool is bright blue.\n"
            "2. A red water slide enters the pool.\n"
            "3. Nothing outside the requested change materially changed.")
    content = [{"type": "text",
                "text": "Edit instruction: make the pool bright blue with "
                        "a red water slide entering it.\n"
                        f"Requirements:\n{reqs}"},
               {"type": "image_url", "image_url": {
                   "url": f"data:image/jpeg;base64,{make_jpeg(px, seed)}"}},
               {"type": "image_url", "image_url": {
                   "url": f"data:image/jpeg;base64,"
                          f"{make_jpeg(px, seed + 3)}"}}]
    b = {"model": MODEL, "temperature": 0.1, "max_tokens": 450,
         "messages": [{"role": "system", "content": JUDGE_RUBRIC},
                      {"role": "user", "content": content}]}
    if schema:
        b["response_format"] = {"type": "json_schema", "json_schema": {
            "name": "verdict", "schema": JUDGE_SCHEMA}}
    return b


def body_decode(n_tokens: int = 220) -> dict:
    return {"model": MODEL, "temperature": 0.0, "max_tokens": n_tokens,
            "messages": [{"role": "user", "content":
                          "Write a plain prose description of a desert "
                          "landscape at sunset. No lists. Keep going until "
                          "cut off."}]}


def body_prefill(n_words: int) -> dict:
    filler = " ".join(
        f"item{i} sandstone terrace mesquite shade" for i in range(n_words // 5))
    return {"model": MODEL, "temperature": 0.0, "max_tokens": 1,
            "messages": [{"role": "user", "content":
                          f"Context: {filler}\nSay OK."}]}


def run(mode: str, n: int, px: int, hist: int, warmup: int = 1) -> None:
    def make(i: int) -> dict:
        d = DIRECTIVES[i % len(DIRECTIVES)]
        if mode == "parser":
            return body_parser(d)
        if mode == "parser-noschema":
            return body_parser(d, schema=False)
        if mode == "parser-hist":
            return body_parser(d, history_turns=hist)
        if mode == "parser-img":
            return body_parser(d, image_b64=make_jpeg(1024, i))
        if mode == "parser-newschema":
            # a NEVER-seen revs list per call -> fresh xgrammar compile
            # (production grows history_revs as the timeline advances)
            import random
            revs = sorted(random.sample(range(1, 200), 6))
            return body_parser(d, revs=revs)
        if mode == "judge":
            return body_judge(px, i)
        if mode == "judge-noschema":
            return body_judge(px, i, schema=False)
        if mode == "decode":
            return body_decode()
        raise SystemExit(f"unknown mode {mode}")

    if mode == "prefill":
        for words in (400, 1200, 2400, 3900):
            b = body_prefill(words)
            r0 = sse_call(b)          # cold-ish (may hit prefix cache = off)
            r1 = sse_call(b)          # repeat, same prompt
            print(json.dumps({"mode": "prefill", "words": words,
                              "prompt_tokens": r0["prompt_tokens"],
                              "ttft_ms_first": r0["ttft_ms"],
                              "ttft_ms_repeat": r1["ttft_ms"]}))
        return
    if mode == "imgfill":
        for p in (256, 384, 512, 768, 1024):
            b = body_judge(p, 0)
            b["max_tokens"] = 1
            b.pop("response_format", None)
            r0 = sse_call(b)
            r1 = sse_call(b)
            print(json.dumps({"mode": "imgfill", "px": p,
                              "prompt_tokens": r0["prompt_tokens"],
                              "ttft_ms_first": r0["ttft_ms"],
                              "ttft_ms_repeat": r1["ttft_ms"]}))
        return

    for _ in range(warmup):
        sse_call(make(0))
    rows = []
    for i in range(n):
        r = sse_call(make(i))
        rows.append(r)
        print(json.dumps({k: v for k, v in r.items() if k != "content"}))
        if i == 0 and mode != "decode":
            snippet = (r["content"] or "")[:160].replace("\n", " ")
            print(f"# first content: {snippet}", file=sys.stderr)
    med = lambda k: statistics.median([r[k] for r in rows if r[k] is not None])
    summary = {"mode": mode, "n": n,
               "total_ms_p50": round(med("total_ms")),
               "ttft_ms_p50": round(med("ttft_ms")),
               "decode_tok_s_p50": round(med("decode_tok_s"), 1)
               if any(r["decode_tok_s"] for r in rows) else None,
               "prompt_tokens": rows[0]["prompt_tokens"],
               "completion_tokens_p50": round(med("completion_tokens"))}
    print("SUMMARY " + json.dumps(summary))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="parser")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--px", type=int, default=256)
    ap.add_argument("--hist", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=1)
    a = ap.parse_args()
    run(a.mode, a.n, a.px, a.hist, a.warmup)
