"""End-to-end mouse-layer probe: warm cache + cold click + drag.

Streams a WAV to build a scene, then verifies all three mouse paths the
browser uses:
  1. WARM  — LLM-enumerated objects arrive as a masks message
  2. COLD  — a click_segment on an arbitrary point returns a mask
  3. DRAG  — a mouse_move with the mask produces a move render + ledger

  docker compose exec orchestrator python /srv/app/tools/mouse_probe.py \
      /srv/data/samples/livingroom.wav /srv/data/mouseprobe [click_x] [click_y]
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


async def main(wav_path: str, out_dir: str,
               cx: float = 0.5, cy: float = 0.55) -> None:
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with wave.open(wav_path, "rb") as w:
        pcm = w.readframes(w.getnframes())

    warm: list[dict] = []
    seg_results: list[dict] = []
    n = [0]

    async with websockets.connect("ws://127.0.0.1:8000/ws/session?mode=scrim",
                                  max_size=None, open_timeout=120) as ws:
        await ws.send(json.dumps({"type": "client_hello",
                                  "data": {"ua": "mouse_probe"}}))

        async def reader():
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue
                ev = json.loads(raw)
                t, d = ev.get("type"), ev.get("data", {})
                if t == "masks":
                    warm.append(d)
                    print("WARM MASKS:", [(e["id"], e["box"])
                                          for e in d["entities"]])
                elif t == "segment_result":
                    seg_results.append(d)
                    s = d.get("seg") or {}
                    print("SEGMENT_RESULT:",
                          [{"box": c["box"], "frac": c["frac"]}
                           for c in s.get("candidates", [])] or None)
                elif t == "canvas_frame" and not d.get("preview"):
                    n[0] += 1
                    name = f"m{n[0]:02d}_{d.get('source')}_{d.get('lane')}.jpg"
                    (out / name).write_bytes(base64.b64decode(d["jpeg_b64"]))
                    print("frame", name)
                elif t == "patch_applied":
                    print("patch_applied:", d.get("commit_id"),
                          repr(d.get("utterance")), d.get("reasoning"))

        r = asyncio.create_task(reader())
        for i in range(0, len(pcm), FRAME * 2):
            await ws.send(pcm[i:i + FRAME * 2])
            await asyncio.sleep(0.08)
        print("audio done; waiting for warm masks ...")
        for _ in range(40):
            if warm:
                break
            await asyncio.sleep(1)
        print(f"warm cache: {'YES' if warm else 'NO (cold path still works)'}")

        print(f"COLD CLICK at ({cx}, {cy}) ...")
        await ws.send(json.dumps({"type": "click_segment", "data": {
            "req_id": 1, "x": cx, "y": cy}}))
        for _ in range(20):
            if seg_results:
                break
            await asyncio.sleep(0.5)

        moves = []
        if warm and warm[-1]["entities"]:
            # marquee-style GROUP move: first two warm objects together
            for e in warm[-1]["entities"][:2]:
                b = dict(e["box"])
                moves.append({
                    "entity_id": e["id"], "name": e["name"], "from_box": b,
                    "to_box": {"x": min(0.92, b["x"] + 0.15),
                               "y": min(0.92, b["y"] + 0.05),
                               "w": b["w"], "h": b["h"]},
                    "mask_rle": e["rle"], "mask_size": e["size"]})
        elif (seg_results and seg_results[0].get("seg")
              and seg_results[0]["seg"].get("candidates")):
            s = seg_results[0]["seg"]["candidates"][0]
            b = dict(s["box"])
            moves.append({
                "entity_id": "_click", "name": None, "from_box": b,
                "to_box": {"x": min(0.92, b["x"] + 0.2), "y": b["y"],
                           "w": b["w"], "h": b["h"]},
                "mask_rle": s["rle"], "mask_size": s["size"]})
        if not moves:
            print("NOTHING SELECTABLE — no drag test")
        else:
            print(f"GROUP DRAG: {[m['entity_id'] for m in moves]}")
            await ws.send(json.dumps({"type": "mouse_move",
                                      "data": {"moves": moves}}))
            await asyncio.sleep(14)
        r.cancel()
    print(f"saved {n[0]} frames -> {out}")


if __name__ == "__main__":
    args = sys.argv[1:]
    asyncio.run(main(args[0], args[1],
                     float(args[2]) if len(args) > 2 else 0.5,
                     float(args[3]) if len(args) > 3 else 0.55))
