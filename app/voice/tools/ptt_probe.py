"""Put-that-there e2e probe: streams a multi-part utterance while
sending timed ink strokes, so pointing words and gestures interleave
exactly like a live user drawing while talking.

  docker compose exec orchestrator python /srv/app/tools/ptt_probe.py \
      <root.wav> <multi.wav> [out_dir]
"""
from __future__ import annotations

import asyncio
import base64
import json
import math
import pathlib
import sys
import wave

import websockets

FRAME = 1920  # 80ms @ 24k

# (offset_seconds_into_multi_wav, stroke)


def _ts(n):
    return [i / (n - 1) for i in range(n)]


def timed_strokes():
    scrib = []
    for k in range(3):                      # scribble over the plant
        y = 0.58 + 0.05 * k
        scrib.append([[0.14 + 0.14 * t, y + 0.02 * math.sin(9 * t)]
                      for t in _ts(10)])
    circle_desk = [[0.55 + 0.09 * math.cos(2 * math.pi * t),
                    0.62 + 0.10 * math.sin(2 * math.pi * t)]
                   for t in _ts(30)]
    circle_tv = [[0.76 + 0.09 * math.cos(2 * math.pi * t),
                  0.34 + 0.08 * math.sin(2 * math.pi * t)]
                 for t in _ts(30)]
    return ([(0.9 + 0.15 * i, s) for i, s in enumerate(scrib)]
            + [(2.9, circle_desk)] + [(4.6, circle_tv)])


async def main(root_wav: str, multi_wav: str,
               out_dir: str = "/srv/data/ptt_probe") -> None:
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with wave.open(root_wav, "rb") as w:
        pcm_root = w.readframes(w.getnframes())
    with wave.open(multi_wav, "rb") as w:
        pcm_multi = w.readframes(w.getnframes())

    async with websockets.connect("ws://127.0.0.1:8000/ws/session?mode=scrim",
                                  max_size=None, open_timeout=120) as ws:
        await ws.send(json.dumps({"type": "client_hello",
                                  "data": {"ua": "ptt_probe"}}))
        n = [0]
        root_done = asyncio.Event()

        async def reader():
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue
                ev = json.loads(raw)
                t, d = ev.get("type"), ev.get("data", {})
                if t == "canvas_frame" and not d.get("preview"):
                    n[0] += 1
                    (out / f"f{n[0]:03d}_{d.get('lane', 'x')}.jpg"
                     ).write_bytes(base64.b64decode(d.get("jpeg_b64", "")))
                    print("frame", n[0], d.get("lane"),
                          f"render_s={d.get('render_s')}")
                    if d.get("lane") == "committed":
                        root_done.set()
                elif t == "ink_routed":
                    print("INK_ROUTED", json.dumps(d))
                elif t in ("commit", "patch_applied"):
                    print(t, json.dumps(d)[:300])

        r = asyncio.create_task(reader())
        for i in range(0, len(pcm_root), FRAME * 2):
            await ws.send(pcm_root[i:i + FRAME * 2])
            await asyncio.sleep(0.08)
        try:
            await asyncio.wait_for(root_done.wait(), 60)
        except asyncio.TimeoutError:
            print("no root frame — aborting")
            return
        await asyncio.sleep(2)

        print(">>> multi-part utterance with timed strokes")
        strokes = sorted(timed_strokes(), key=lambda x: x[0])
        si = 0
        t_audio = 0.0
        for i in range(0, len(pcm_multi), FRAME * 2):
            await ws.send(pcm_multi[i:i + FRAME * 2])
            t_audio += 0.08
            while si < len(strokes) and strokes[si][0] <= t_audio:
                await ws.send(json.dumps({"type": "ink_stroke",
                                          "data": {"stroke":
                                                   strokes[si][1]}}))
                print(f">>> stroke {si + 1} at {t_audio:.1f}s")
                si += 1
            await asyncio.sleep(0.08)
        while si < len(strokes):
            await ws.send(json.dumps({"type": "ink_stroke",
                                      "data": {"stroke": strokes[si][1]}}))
            si += 1
        await asyncio.sleep(50)
        r.cancel()
    print(f"saved {n[0]} frames -> {out}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2],
                     sys.argv[3] if len(sys.argv) > 3 else
                     "/srv/data/ptt_probe"))
