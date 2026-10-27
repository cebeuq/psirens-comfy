"""Upload-insertion e2e probe: builds a scene from a WAV, then drops an
uploaded image at a canvas point and reports the insertion flow.

  docker compose exec orchestrator python /srv/app/tools/upload_probe.py \
      <root.wav> <upload.jpg> <nx> <ny> [out_dir]
"""
from __future__ import annotations

import asyncio
import base64
import json
import pathlib
import sys
import wave

import websockets

FRAME = 1920


async def main(root_wav: str, upload_path: str, nx: float, ny: float,
               out_dir: str = "/srv/data/upload_probe") -> None:
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with wave.open(root_wav, "rb") as w:
        pcm = w.readframes(w.getnframes())
    up_b64 = base64.b64encode(
        pathlib.Path(upload_path).read_bytes()).decode()

    async with websockets.connect("ws://127.0.0.1:8000/ws/session?mode=scrim",
                                  max_size=None, open_timeout=120) as ws:
        await ws.send(json.dumps({"type": "client_hello",
                                  "data": {"ua": "upload_probe"}}))
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
                elif t == "upload_routed":
                    print("UPLOAD_ROUTED", json.dumps(d))
                elif t == "patch_applied":
                    print("patch_applied", json.dumps(
                        {k: d.get(k) for k in
                         ("commit_id", "utterance", "entity_edits")})[:220])

        r = asyncio.create_task(reader())
        for i in range(0, len(pcm), FRAME * 2):
            await ws.send(pcm[i:i + FRAME * 2])
            await asyncio.sleep(0.08)
        try:
            await asyncio.wait_for(root_done.wait(), 60)
        except asyncio.TimeoutError:
            print("no root frame — aborting")
            return
        await asyncio.sleep(2)
        print(f">>> dropping upload at ({nx}, {ny})")
        await ws.send(json.dumps({"type": "upload_drop", "data": {
            "b64": up_b64, "nx": nx, "ny": ny}}))
        await asyncio.sleep(35)
        r.cancel()
    print(f"saved {n[0]} frames -> {out}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2],
                     float(sys.argv[3]), float(sys.argv[4]),
                     sys.argv[5] if len(sys.argv) > 5 else
                     "/srv/data/upload_probe"))
