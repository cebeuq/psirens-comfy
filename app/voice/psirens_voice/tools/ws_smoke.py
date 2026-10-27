"""Synthetic websocket client: streams a sine tone (default) or a WAV file as
if a browser mic were talking. Used to verify the pipeline without a human.

Run inside the orchestrator container:
    python -m psirens_voice.tools.ws_smoke --seconds 2
    python -m psirens_voice.tools.ws_smoke --wav /srv/data/samples/speech.wav
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math

import numpy as np
import websockets


def load_wav_frames(path: str, sample_rate: int, samples_per_frame: int) -> list[bytes]:
    import soundfile as sf
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != sample_rate:  # linear resample
        n_out = int(len(audio) * sample_rate / sr)
        audio = np.interp(np.linspace(0, len(audio) - 1, n_out),
                          np.arange(len(audio)), audio)
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()
    fb = samples_per_frame * 2
    return [pcm[i:i + fb] for i in range(0, len(pcm) - fb + 1, fb)]


async def run(url: str, seconds: float, sample_rate: int, frame_ms: int,
              wav: str | None = None, tail_silence: float = 0.0) -> None:
    samples_per_frame = sample_rate * frame_ms // 1000
    received: list[dict] = []

    if wav:
        frames = load_wav_frames(wav, sample_rate, samples_per_frame)
        silence = b"\x00\x00" * samples_per_frame
        frames += [silence] * int(tail_silence * 1000 / frame_ms)
    else:
        frames = []
        phase, step = 0.0, 2 * math.pi * 440 / sample_rate
        for _ in range(int(seconds * 1000 / frame_ms)):
            f = bytearray()
            for _ in range(samples_per_frame):
                f += int(12000 * math.sin(phase)).to_bytes(2, "little", signed=True)
                phase += step
            frames.append(bytes(f))

    async with websockets.connect(url, max_size=None) as ws:
        await ws.send(json.dumps({"type": "client_hello", "data": {"client": "ws_smoke"}}))

        async def recv():
            try:
                async for msg in ws:
                    if isinstance(msg, str):
                        received.append(json.loads(msg))
            except websockets.ConnectionClosed:
                pass

        recv_task = asyncio.create_task(recv())

        for frame in frames:
            await ws.send(frame)
            await asyncio.sleep(frame_ms / 1000)

        # allow STT flush + any in-flight LLM patch to come back before stop
        await asyncio.sleep(6.0 if wav else 1.0)
        await ws.send(json.dumps({"type": "stop"}))
        await asyncio.sleep(0.5)
        recv_task.cancel()

    types: dict[str, int] = {}
    for ev in received:
        types[ev.get("type", "?")] = types.get(ev.get("type", "?"), 0) + 1
    start = next((e for e in received if e.get("type") == "session_start"), None)
    levels = [e for e in received if e.get("type") == "audio_level"]
    words = [e for e in received if e.get("type") == "word"]
    errors = [e for e in received if e.get("type") == "error"]
    print("events by type:", types)
    print("session_id:", start["data"]["session_id"] if start else "MISSING")
    print("last audio_level dbfs:", levels[-1]["data"]["dbfs"] if levels else "NONE")
    if words:
        print("transcript:", " ".join(w["data"]["w"] for w in words))
        print("first word:", words[0]["data"])
    for e in received:
        if e.get("type") == "commit":
            d = e["data"]
            print(f"COMMIT t={e['t_session_ms']}ms: {d['utterance_clean']!r}"
                  + (f"  (raw: {d['utterance_raw']!r})" if d.get("repairs") else "")
                  + f"  tone={d.get('tone')}")
        elif e.get("type") == "hold":
            print(f"HOLD   t={e['t_session_ms']}ms: {e['data'].get('reason')}")
        elif e.get("type") == "draft_patch":
            d = e["data"]
            print(f"DRAFT  t={e['t_session_ms']}ms ({d['latency_ms']}ms): "
                  f"words={d['words']!r} patch={d['patch']}")
        elif e.get("type") == "reflex_action":
            d = e["data"]
            print(f"REFLEX t={e['t_session_ms']}ms ({d['latency_ms']}ms): "
                  f"{d['fragment']!r} -> {d['kind']}: {d['detail']}")
        elif e.get("type") == "render_target":
            d = e["data"]
            print(f"RENDER [{d['source']}] strength={d['strength']}: {d['prompt']!r}")
        elif e.get("type") == "patch_applied":
            d = e["data"]
            print(f"PATCH  #{d['commit_id']} mag={d['edit_magnitude']} "
                  f"changed={d['changed_fields']} latency={d['latency_ms']}ms "
                  f"patch={d['patch']}")
        elif e.get("type") == "prompt_updated":
            print(f"PROMPT #{e['data']['commit_id']}: {e['data']['prompt']!r}")
    frames = [e for e in received if e.get("type") == "canvas_frame"]
    if frames:
        print(f"CANVAS: {len(frames)} frames received "
              f"(last: {frames[-1]['data']['source']} #{frames[-1]['data']['seq']}, "
              f"{len(frames[-1]['data']['jpeg_b64'])//1024}KB b64)")
    for e in errors[:5]:
        print("ERROR EVENT:", e["data"])
    assert start, "no session_start received"
    assert levels, "no audio_level events received"
    print("SMOKE OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8000/ws/session")
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--wav", default=None, help="stream this WAV instead of a sine tone")
    ap.add_argument("--tail-silence", type=float, default=3.0,
                    help="seconds of silence appended after the WAV")
    ap.add_argument("--sample-rate", type=int, default=24000)
    ap.add_argument("--frame-ms", type=int, default=80)
    args = ap.parse_args()
    asyncio.run(run(args.url, args.seconds, args.sample_rate, args.frame_ms,
                    wav=args.wav, tail_silence=args.tail_silence))


if __name__ == "__main__":
    main()
