"""Benchmark executor: replays bench/suite.json against the live stack,
one ws session per scenario, saving every step's committed frame.

    python -m psirens_voice.tools.bench_run \
        --suite /srv/data/bench/suite.json --out /srv/data/bench/run1
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import pathlib

import websockets

from .ws_smoke import load_wav_frames

SR, FRAME_MS = 24000, 80
SPF = SR * FRAME_MS // 1000
SILENCE = b"\x00\x00" * SPF


def _b64(path: str) -> str:
    from PIL import Image
    im = Image.open(path).convert("RGB")
    im.thumbnail((768, 768))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def _subst(step: dict, v: dict) -> dict:
    out = {}
    for k, val in step.items():
        out[k] = val.format(**v) if isinstance(val, str) else val
    return out


async def run_scenario(url: str, sc: dict, wav_dir: pathlib.Path,
                       out: pathlib.Path, rep: int = 0) -> list[dict]:
    out.mkdir(parents=True, exist_ok=True)
    frames_seen: list[str] = []
    results = []
    async with websockets.connect(url, max_size=None) as ws:
        await ws.send(json.dumps({"type": "client_hello",
                                  "data": {"client": "bench"}}))

        async def recv():
            try:
                async for m in ws:
                    if isinstance(m, str):
                        e = json.loads(m)
                        d = e.get("data") or {}
                        if (e.get("type") == "canvas_frame"
                                and not d.get("preview")
                                and d.get("lane") != "provisional"):
                            frames_seen.append(d["jpeg_b64"])
            except websockets.ConnectionClosed:
                pass

        rt = asyncio.create_task(recv())
        await asyncio.sleep(1.0)
        varz = (sc.get("vars") or [{}])[rep % max(
            1, len(sc.get("vars") or [{}]))]
        for i, step in enumerate(sc["steps"]):
            step = _subst(step, varz)
            n_before = len(frames_seen)
            if step.get("say"):
                wav = wav_dir / f"{sc['id']}_{i}_r{rep}.wav"
                for f in (load_wav_frames(str(wav), SR, SPF)
                          + [SILENCE] * int(6000 / FRAME_MS)):
                    await ws.send(f)
                    await asyncio.sleep(FRAME_MS / 1000)
            elif step.get("drop_ref"):
                await ws.send(json.dumps({"type": "upload_drop", "data": {
                    "b64": _b64(step["drop_ref"]), "nx": 0.5, "ny": 0.5,
                    "intent": "reference"}}))
                await asyncio.sleep(10)
                results.append({"step": i, "frame": None})
                continue
            elif step.get("drop_insert"):
                rid = f"bench{i}"
                await ws.send(json.dumps({"type": "upload_drop", "data": {
                    "b64": _b64(step["drop_insert"]),
                    "nx": step.get("nx", 0.5), "ny": step.get("ny", 0.5),
                    "intent": "insert", "req_id": rid}}))
                await asyncio.sleep(12)
                await ws.send(json.dumps({"type": "insert_commit",
                                          "data": {"req_id": rid,
                                                   "nx": step.get("nx", 0.5),
                                                   "ny": step.get("ny", 0.5),
                                                   "w": step.get("w", 0.3),
                                                   "rot": 0}}))
            # wait for a new committed frame (up to 120s), then settle
            waited = 0.0
            while len(frames_seen) == n_before and waited < 120:
                await asyncio.sleep(1.0)
                waited += 1.0
            await asyncio.sleep(12.0)   # verify/retry frames settle
            fn = None
            if frames_seen:
                fn = f"step_{i}.jpg"
                (out / fn).write_bytes(
                    base64.b64decode(frames_seen[-1]))
            results.append({"step": i, "frame": fn,
                            "timed_out": len(frames_seen) == n_before})
            print(f"  {sc['id']} step {i}: "
                  f"{'frame ' + str(fn) if fn else 'NO FRAME'}"
                  + (" (TIMEOUT)" if len(frames_seen) == n_before else ""),
                  flush=True)
        await ws.send(json.dumps({"type": "stop"}))
        await asyncio.sleep(0.5)
        rt.cancel()
    return results


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wavs", default="/srv/data/bench/wavs")
    ap.add_argument("--url", default="ws://127.0.0.1:8000/ws/session")
    ap.add_argument("--only", default="")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--first-rep", type=int, default=0)
    a = ap.parse_args()
    suite = json.loads(open(a.suite).read())
    outroot = pathlib.Path(a.out)
    manifest = {}
    for rep in range(a.first_rep, a.reps):
        for sc in suite["scenarios"]:
            if a.only and sc["id"] not in a.only.split(","):
                continue
            print(f"== {sc['id']} rep{rep}", flush=True)
            for attempt in range(20):
                try:
                    res = await run_scenario(
                        a.url, sc, pathlib.Path(a.wavs),
                        outroot / f"rep{rep}" / sc["id"], rep=rep)
                    manifest[f"{sc['id']}/rep{rep}"] = res
                    break
                except websockets.ConnectionClosed:
                    print("  session busy (1013) — waiting", flush=True)
                    await asyncio.sleep(90)
            await asyncio.sleep(8)   # session teardown
    (outroot / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("BENCH RUN DONE")


asyncio.run(main())
