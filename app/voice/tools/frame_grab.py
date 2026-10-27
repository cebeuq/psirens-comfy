"""Stream a test WAV through a live session and save every canvas frame.

Runs inside the orchestrator container (websockets is installed there):

  docker compose exec orchestrator python /srv/app/tools/frame_grab.py \
      /srv/data/samples/sticker_b.wav /srv/data/framegrab [mode]

Audio is paced real-time (80ms frames, 24kHz mono s16le — the browser
worklet's format). After the file ends, keeps listening for trailing
renders (commit chains, replays) before exiting.
"""
from __future__ import annotations

import asyncio
import base64
import json
import pathlib
import sys
import wave

import websockets

FRAME = 1920            # 80ms @ 24kHz
TAIL_S = 25.0           # keep listening after audio ends


async def main(wav_path: str, out_dir: str, mode: str = "scrim",
               edit_mode: str | None = None) -> None:
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with wave.open(wav_path, "rb") as w:
        assert w.getframerate() == 24000 and w.getnchannels() == 1, \
            f"expected 24kHz mono, got {w.getframerate()}Hz {w.getnchannels()}ch"
        pcm = w.readframes(w.getnframes())

    url = f"ws://127.0.0.1:8000/ws/session?mode={mode}"
    n = [0]

    async with websockets.connect(url, max_size=None,
                                  open_timeout=120) as ws:
        await ws.send(json.dumps({"type": "client_hello",
                                  "data": {"ua": "frame_grab"}}))
        if edit_mode:
            await ws.send(json.dumps({"type": "set_edit_mode",
                                      "data": {"mode": edit_mode}}))

        import time as _time
        t_start = _time.monotonic()

        async def reader():
            async for raw in ws:
                if isinstance(raw, bytes):
                    # binary frame envelope: PSF1 | u32 len | header | JPEG
                    if raw[:4] != b"PSF1":
                        continue
                    hlen = int.from_bytes(raw[4:8], "big")
                    ev = json.loads(raw[8:8 + hlen].decode())
                    ev["data"]["_jpeg"] = raw[8 + hlen:]
                else:
                    ev = json.loads(raw)
                t = ev.get("type")
                d = ev.get("data", {})
                if t == "canvas_frame":
                    if d.get("preview"):
                        continue                    # keep finals only
                    n[0] += 1
                    name = (f"f{n[0]:03d}_{d.get('source', 'x')}"
                            f"_{d.get('lane', 'x')}.jpg")
                    jpeg = d.get("_jpeg") or base64.b64decode(d["jpeg_b64"])
                    (out / name).write_bytes(jpeg)
                    print("frame", name, f"render_s={d.get('render_s')}",
                          f"arrival_ms={int((_time.monotonic()-t_start)*1000)}",
                          f"wall={_time.time():.3f}",
                          f"bytes={len(jpeg)}",
                          f"restored={d.get('restored')}" if d.get("restored")
                          is not None else "")
                elif t in ("commit", "patch_applied", "prompt_updated"):
                    print(t, json.dumps(d)[:220])

        r = asyncio.create_task(reader())
        for i in range(0, len(pcm), FRAME * 2):
            await ws.send(pcm[i:i + FRAME * 2])
            await asyncio.sleep(0.08)
        print("audio done; waiting for trailing renders ...")
        await asyncio.sleep(TAIL_S)
        r.cancel()
    print(f"saved {n[0]} frames -> {out}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2],
                     sys.argv[3] if len(sys.argv) > 3 else "scrim",
                     sys.argv[4] if len(sys.argv) > 4 else None))
