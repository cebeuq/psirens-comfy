"""Region-scope smoke: root gen from a WAV, then apply a saved canvas
selection (mask JSON) and speak a region edit; saves before/after.

    python -m psirens_voice.tools.region_smoke \
        --gen /srv/data/samples/gen_interior.wav \
        --mask /srv/data/samples/region_0.json \
        --edit /srv/data/samples/edit_wall.wav \
        --out /srv/data/samples/regionsmoke
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json

import websockets

from .ws_smoke import load_wav_frames


async def run(url: str, gen: str, mask: str, edit: str, out: str,
              sample_rate: int = 24000, frame_ms: int = 80) -> None:
    spf = sample_rate * frame_ms // 1000
    received: list[dict] = []
    silence = b"\x00\x00" * spf

    async with websockets.connect(url, max_size=None) as ws:
        await ws.send(json.dumps({"type": "client_hello",
                                  "data": {"client": "region_smoke"}}))

        async def recv():
            try:
                async for msg in ws:
                    if isinstance(msg, str):
                        e = json.loads(msg)
                        received.append(e)
                        if e.get("type") in ("commit", "prompt_updated"):
                            print(e["type"],
                                  json.dumps(e.get("data") or {})[:140],
                                  flush=True)
            except websockets.ConnectionClosed:
                pass

        recv_task = asyncio.create_task(recv())
        await asyncio.sleep(1.0)

        async def speak(wav):
            frames = load_wav_frames(wav, sample_rate, spf)
            frames += [silence] * int(6000 / frame_ms)
            for f in frames:
                await ws.send(f)
                await asyncio.sleep(frame_ms / 1000)
            await asyncio.sleep(75.0)

        def save_frame(tag):
            cf = [e for e in received if e.get("type") == "canvas_frame"
                  and not e["data"].get("preview")]
            if cf:
                jpg = base64.b64decode(cf[-1]["data"]["jpeg_b64"])
                open(f"{out}_{tag}.jpg", "wb").write(jpg)
                print("frame saved:", tag, len(jpg) // 1024, "KB",
                      flush=True)
            else:
                print("NO FRAME at", tag, flush=True)

        await speak(gen)
        save_frame("before")
        m = json.loads(open(mask).read())
        await ws.send(json.dumps({"type": "set_region", "data": m}))
        await asyncio.sleep(2.0)
        await speak(edit)
        save_frame("after")
        await ws.send(json.dumps({"type": "stop"}))
        await asyncio.sleep(0.5)
        recv_task.cancel()
    print("REGION SMOKE DONE")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8000/ws/session")
    ap.add_argument("--gen", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--edit", required=True)
    ap.add_argument("--out", default="/srv/data/samples/regionsmoke")
    a = ap.parse_args()
    asyncio.run(run(a.url, a.gen, a.mask, a.edit, a.out))


if __name__ == "__main__":
    main()
