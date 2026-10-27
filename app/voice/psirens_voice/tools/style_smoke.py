"""Style-bank smoke: drop style refs over the ws like a browser would,
then speak root generations / edits from WAVs and capture the frames.

    python -m psirens_voice.tools.style_smoke \
        --refs /srv/data/samples/ref_a.jpg,/srv/data/samples/ref_b.jpg \
        --wavs /srv/data/samples/gen1.wav,/srv/data/samples/edit1.wav \
        --out /srv/data/samples/stylesmoke
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json

import websockets

from .ws_smoke import load_wav_frames


def ref_b64(path: str) -> str:
    from PIL import Image
    img = Image.open(path).convert("RGB")
    img.thumbnail((768, 768))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


async def run(url: str, refs: list[str], wavs: list[str],
              out: str, sample_rate: int = 24000,
              frame_ms: int = 80, post: list[str] | None = None,
              wavs2: list[str] | None = None) -> None:
    post = post or []
    wavs2 = wavs2 or []
    spf = sample_rate * frame_ms // 1000
    received: list[dict] = []
    async with websockets.connect(url, max_size=None) as ws:
        await ws.send(json.dumps({"type": "client_hello",
                                  "data": {"client": "style_smoke"}}))

        async def recv():
            try:
                async for msg in ws:
                    if isinstance(msg, str):
                        e = json.loads(msg)
                        received.append(e)
                        if e.get("type") in ("upload_routed", "style_bank",
                                             "commit", "prompt_updated"):
                            d = dict(e.get("data") or {})
                            d.pop("refs", None)
                            print(e["type"], json.dumps(d)[:200],
                                  flush=True)
            except websockets.ConnectionClosed:
                pass

        recv_task = asyncio.create_task(recv())
        await asyncio.sleep(1.0)
        for rp in refs:
            await ws.send(json.dumps({"type": "upload_drop", "data": {
                "b64": ref_b64(rp), "nx": 0.5, "ny": 0.5}}))
            await asyncio.sleep(0.1)      # true multi-drop burst
        await asyncio.sleep(6.0)          # decider + debounced director
        silence = b"\x00\x00" * spf
        for wi, wav in enumerate(wavs):
            frames = load_wav_frames(wav, sample_rate, spf)
            frames += [silence] * int(6000 / frame_ms)
            for f in frames:
                await ws.send(f)
                await asyncio.sleep(frame_ms / 1000)
            await asyncio.sleep(75.0)     # commit + render + verify
            cf = [e for e in received if e.get("type") == "canvas_frame"
                  and not e["data"].get("preview")]
            if cf:
                jpg = base64.b64decode(cf[-1]["data"]["jpeg_b64"])
                fn = f"{out}_{wi}.jpg"
                open(fn, "wb").write(jpg)
                print("frame saved:", fn, len(jpg) // 1024, "KB",
                      flush=True)
            else:
                print("NO FINAL FRAME after wav", wi, flush=True)
        for pi, pp in enumerate(post):
            await ws.send(json.dumps({"type": "upload_drop", "data": {
                "b64": ref_b64(pp), "nx": 0.7, "ny": 0.6}}))
            await asyncio.sleep(8.0)
        for wi2, wav in enumerate(wavs2):
            frames = load_wav_frames(wav, sample_rate, spf)
            frames += [silence] * int(6000 / frame_ms)
            for f in frames:
                await ws.send(f)
                await asyncio.sleep(frame_ms / 1000)
            await asyncio.sleep(75.0)
            cf = [e for e in received if e.get("type") == "canvas_frame"
                  and not e["data"].get("preview")]
            if cf:
                jpg = base64.b64decode(cf[-1]["data"]["jpeg_b64"])
                fn = f"{out}_w2_{wi2}.jpg"
                open(fn, "wb").write(jpg)
                print("frame saved:", fn, len(jpg) // 1024, "KB",
                      flush=True)
        for _pi, _pp in enumerate([]):
            await ws.send(json.dumps({"type": "upload_drop", "data": {
                "b64": ref_b64(pp), "nx": 0.7, "ny": 0.6}}))
            await asyncio.sleep(40.0)
            cf = [e for e in received if e.get("type") == "canvas_frame"
                  and not e["data"].get("preview")]
            if cf:
                jpg = base64.b64decode(cf[-1]["data"]["jpeg_b64"])
                fn = f"{out}_p{pi}.jpg"
                open(fn, "wb").write(jpg)
                print("frame saved:", fn, len(jpg) // 1024, "KB",
                      flush=True)
        await ws.send(json.dumps({"type": "stop"}))
        await asyncio.sleep(0.5)
        recv_task.cancel()
    print("STYLE SMOKE DONE")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8000/ws/session")
    ap.add_argument("--refs", default="")
    ap.add_argument("--wavs", default="")
    ap.add_argument("--out", default="/srv/data/samples/stylesmoke")
    ap.add_argument("--post", default="",
                    help="images dropped AFTER the wavs")
    ap.add_argument("--wavs2", default="",
                    help="wavs spoken after the post drops")
    a = ap.parse_args()
    asyncio.run(run(a.url,
                    [p for p in a.refs.split(",") if p],
                    [p for p in a.wavs.split(",") if p], a.out,
                    post=[p for p in a.post.split(",") if p],
                    wavs2=[p for p in a.wavs2.split(",") if p]))


if __name__ == "__main__":
    main()
