"""Stream a WAV like a browser mic and save every final canvas frame.

python -m psirens_voice.tools.frames_run <wav> <out_dir> [--tail 4]
"""
import argparse
import asyncio
import base64
import json
import os
import time

import numpy as np
import soundfile as sf
import websockets


async def main(wav, out, tail):
    os.makedirs(out, exist_ok=True)
    a, sr = sf.read(wav, dtype="float32", always_2d=True)
    a = a.mean(axis=1)
    if sr != 24000:
        n = int(len(a) * 24000 / sr)
        a = np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a)
    pcm = (np.clip(a, -1, 1) * 32767).astype(np.int16)
    fb = 1920
    frames = [pcm[i:i + fb].tobytes() for i in range(0, len(pcm) - fb + 1, fb)]
    n = 0
    t0 = time.monotonic()
    async with websockets.connect("ws://127.0.0.1:8000/ws/session",
                                  max_size=None) as ws:
        await ws.send(json.dumps({"type": "client_hello", "data": {}}))

        async def recv():
            nonlocal n
            async for m in ws:
                if not isinstance(m, str):
                    continue
                e = json.loads(m)
                t = time.monotonic() - t0
                if e["type"] == "canvas_frame" and not e["data"].get("preview"):
                    n += 1
                    with open(os.path.join(out, f"f{n:02d}.jpg"), "wb") as f:
                        f.write(base64.b64decode(e["data"]["jpeg_b64"]))
                    print(f"{t:6.1f}s FRAME {n} [{e['data'].get('source')}]",
                          flush=True)
                elif e["type"] == "commit":
                    print(f"{t:6.1f}s COMMIT {e['data']['utterance_clean']!r}",
                          flush=True)
                elif e["type"] == "render_target" and e["data"].get("lane") == "committed":
                    print(f"{t:6.1f}s TARGET {(e['data'].get('edit_instruction') or '')[:90]!r}",
                          flush=True)

        task = asyncio.create_task(recv())
        for fr in frames:
            await ws.send(fr)
            await asyncio.sleep(0.08)
        await asyncio.sleep(tail)
        await ws.send(json.dumps({"type": "stop"}))
        await asyncio.sleep(0.5)
        task.cancel()
    print("frames", n)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("out")
    ap.add_argument("--tail", type=float, default=4.0)
    a = ap.parse_args()
    asyncio.run(main(a.wav, a.out, a.tail))
