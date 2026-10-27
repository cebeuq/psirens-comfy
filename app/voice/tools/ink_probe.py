"""Drawing-mode e2e probe: builds a scene from a WAV, then plays back
scripted ink strokes and reports how the decider routed them.

  docker compose exec orchestrator python /srv/app/tools/ink_probe.py \
      /srv/data/samples/<root>.wav <gesture> [out_dir]

gestures:
  x       — two crossing diagonal strokes over the canvas center
  circle  — a closed loop around the center-right area
  lamp    — a crude lamp silhouette (sketch content) center-left
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

FRAME = 1920


def strokes_for(gesture: str) -> list:
    if gesture == "x":
        return [[[0.40 + 0.20 * t, 0.40 + 0.20 * t] for t in _ts(24)],
                [[0.60 - 0.20 * t, 0.40 + 0.20 * t] for t in _ts(24)]]
    if gesture == "circle":
        return [[[0.68 + 0.14 * math.cos(2 * math.pi * t),
                  0.55 + 0.16 * math.sin(2 * math.pi * t)]
                 for t in _ts(48)]]
    if gesture == "lamp":
        # a recognizable floor lamp: trapezoid shade, stem, oval base
        import math as _m
        shade = [[0.27, 0.30], [0.37, 0.30], [0.35, 0.22], [0.29, 0.22],
                 [0.27, 0.30]]
        stem = [[0.32, 0.30], [0.32, 0.52]]
        base = [[0.32 + 0.05 * _m.cos(2 * _m.pi * t),
                 0.545 + 0.015 * _m.sin(2 * _m.pi * t)] for t in _ts(20)]
        return [shade, stem, base]
    if gesture == "arrow":
        # shaft from the hot-tub area heading left + arrowhead
        shaft = [[0.72 - 0.32 * t, 0.62] for t in _ts(24)]
        head1 = [[0.40, 0.62], [0.46, 0.57]]
        head2 = [[0.40, 0.62], [0.46, 0.67]]
        return [shaft, head1, head2]
    raise SystemExit(f"unknown gesture {gesture}")


def _ts(n):
    return [i / (n - 1) for i in range(n)]


async def main(wav_path: str, gesture: str,
               out_dir: str = "/srv/data/ink_probe",
               follow_wav: str | None = None,
               follow_delay: float = 8.0) -> None:
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with wave.open(wav_path, "rb") as w:
        pcm = w.readframes(w.getnframes())

    async with websockets.connect("ws://127.0.0.1:8000/ws/session?mode=scrim",
                                  max_size=None, open_timeout=120) as ws:
        await ws.send(json.dumps({"type": "client_hello",
                                  "data": {"ua": "ink_probe"}}))
        n = [0]
        root_done = asyncio.Event()

        async def reader():
            async for raw in ws:
                if isinstance(raw, bytes):
                    if raw[:4] != b"PSF1":
                        continue
                    hlen = int.from_bytes(raw[4:8], "big")
                    ev = json.loads(raw[8:8 + hlen].decode())
                    ev["data"]["_jpeg"] = raw[8 + hlen:]
                else:
                    ev = json.loads(raw)
                t, d = ev.get("type"), ev.get("data", {})
                if t == "canvas_frame" and not d.get("preview"):
                    n[0] += 1
                    jpeg = d.get("_jpeg") or base64.b64decode(
                        d.get("jpeg_b64", ""))
                    (out / f"f{n[0]:03d}_{d.get('lane', 'x')}.jpg"
                     ).write_bytes(jpeg)
                    print("frame", n[0], d.get("lane"),
                          f"render_s={d.get('render_s')}")
                    if d.get("lane") == "committed":
                        root_done.set()
                elif t == "ink_routed":
                    print("INK_ROUTED", json.dumps(d))
                elif t in ("commit", "patch_applied"):
                    print(t, json.dumps(d)[:160])
                elif t == "segment_result":
                    cands = (d.get("seg") or {}).get("candidates") or []
                    print("SEGMENT_RESULT req_id=%s candidates=%d %s"
                          % (d.get("req_id"), len(cands),
                             cands[0].get("name") if cands else ""))
                    # emulate the browser: a drawing-mode selection is
                    # reported back as the active region
                    if (str(d.get("req_id", "")).startswith("ink-")
                            and cands):
                        c = cands[0]
                        await ws.send(json.dumps({
                            "type": "set_region", "data": {
                                "mask_rle": c.get("rle"),
                                "mask_size": c.get("size"),
                                "name": c.get("name")}}))
                        print(">>> set_region sent")

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
        print(">>> sending ink:", gesture)
        await ws.send(json.dumps({"type": "ink", "data": {
            "req_id": "probe1", "strokes": strokes_for(gesture)}}))
        if follow_wav:
            await asyncio.sleep(follow_delay)  # 0 = race the drawing
            print(">>> speaking follow-up:", follow_wav)
            with wave.open(follow_wav, "rb") as w2:
                pcm2 = w2.readframes(w2.getnframes())
            for i in range(0, len(pcm2), FRAME * 2):
                await ws.send(pcm2[i:i + FRAME * 2])
                await asyncio.sleep(0.08)
        await asyncio.sleep(40)
        r.cancel()
    print(f"saved {n[0]} frames -> {out}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2],
                     sys.argv[3] if len(sys.argv) > 3 else
                     "/srv/data/ink_probe",
                     sys.argv[4] if len(sys.argv) > 4 else None,
                     float(sys.argv[5]) if len(sys.argv) > 5 else 8.0))
