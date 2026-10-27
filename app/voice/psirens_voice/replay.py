"""Offline replay harness: re-run a recorded session's audio through the
identical pipeline (STT -> prosody -> commit detector) with any config, without
anyone re-speaking. This is the tuning loop for commit/prosody thresholds.

Run inside the orchestrator container:

  # replay one session under an experiment config
  python -m psirens_voice.replay run /srv/data/sessions/<id> \
      --config /srv/app/config/experiment.yaml \
      --out /srv/data/replays/<id>-exp1 [--rtf 1.0] [--with-editor]

  # diff commit decisions between two event logs (original vs replay, or two replays)
  python -m psirens_voice.replay compare /srv/data/sessions/<id>/events.jsonl \
      /srv/data/replays/<id>-exp1/events.jsonl

  # repair a WAV whose header wasn't finalized (crashed session)
  python -m psirens_voice.replay fix-wav /srv/data/sessions/<id>/audio.wav
"""
from __future__ import annotations

import argparse
import asyncio
import json
import struct
import wave
from pathlib import Path


# ---------------------------------------------------------------- run

async def _run_async(session_dir: Path, config_path: str | None,
                     out_dir: Path, rtf: float, with_editor: bool) -> None:
    import os

    from .commit.session_wiring import CommitSession
    from .config import load_config
    from .prosody.engine import ProsodySession
    from .prosody.ser import SerEngine
    from .recorder import SessionRecorder
    from .events import EventBus
    from .stt import SttEngine, SttSession

    cfg = load_config(config_path)
    loop = asyncio.get_running_loop()

    wav = wave.open(str(session_dir / "audio.wav"))
    assert wav.getnchannels() == 1 and wav.getsampwidth() == 2
    sr = wav.getframerate()
    assert sr == cfg.audio.sample_rate, f"wav {sr}Hz != cfg {cfg.audio.sample_rate}"
    frame_samples = sr * cfg.audio.frame_ms // 1000

    print(f"loading models (stt={cfg.stt.model}) ...")
    stt_engine = SttEngine(cfg.stt.model)
    await loop.run_in_executor(None, stt_engine.load)
    ser_engine = None
    if cfg.prosody.enabled and cfg.prosody.ser.model:
        ser_engine = SerEngine(cfg.prosody.ser.model)
        await loop.run_in_executor(None, ser_engine.load)

    bus = EventBus()
    out_dir.mkdir(parents=True, exist_ok=True)
    recorder = SessionRecorder(out_dir.parent, out_dir.name, sr, meta={
        "replay_of": str(session_dir),
        "config_digest": cfg.digest(),
        "config": cfg.as_dict(),
        "rtf": rtf,
    })
    rec_queue = bus.subscribe()

    sinks = [SttSession(stt_engine, bus, loop, vad_event_hz=cfg.ui.vad_event_hz)]
    prosody = None
    if cfg.prosody.enabled:
        prosody = ProsodySession(cfg, ser_engine, bus, loop, cfg.commit.fillers)
        sinks.append(prosody)
    commit = CommitSession(cfg, bus, prosody, loop, delay_ms=stt_engine.delay_ms)
    editor = None
    if with_editor:
        from .editor.service import EditorSession
        editor = EditorSession(cfg, bus, loop)

    n_commits = 0

    async def drain():
        nonlocal n_commits
        while True:
            ev = await rec_queue.get()
            recorder.write_event(ev)
            if ev.type == "commit":
                n_commits += 1
                print(f"  [commit t={ev.t_session_ms}ms] {ev.data['utterance_clean']!r}"
                      f"  ({'; '.join(ev.data['reasons'])})")
            elif ev.type == "hold":
                print(f"  [hold   t={ev.t_session_ms}ms] {ev.data.get('reason')}")
            elif ev.type == "prompt_updated":
                print(f"  [prompt t={ev.t_session_ms}ms] {ev.data['prompt']!r}")

    drain_task = asyncio.create_task(drain())

    t_ms = 0
    frame_bytes_n = frame_samples * 2
    while True:
        pcm = wav.readframes(frame_samples)
        if len(pcm) < frame_bytes_n:
            break
        t_ms += cfg.audio.frame_ms
        for s in sinks:
            await s.feed(pcm, t_ms)
        await asyncio.sleep(cfg.audio.frame_ms / 1000 / rtf)

    for s in sinks:
        await s.close()
    await commit.close()
    if editor is not None:
        # give in-flight LLM calls a moment
        await asyncio.sleep(3)
        await editor.close()
    await asyncio.sleep(0.2)
    drain_task.cancel()
    while not rec_queue.empty():
        recorder.write_event(rec_queue.get_nowait())
    recorder.close()
    print(f"done: {t_ms/1000:.1f}s audio, {n_commits} commits -> {out_dir}")


# ---------------------------------------------------------------- compare

def _load_decisions(path: Path) -> list[dict]:
    out = []
    for line in open(path):
        ev = json.loads(line)
        if ev["type"] in ("commit", "hold"):
            out.append(ev)
    return out


def compare(a_path: Path, b_path: Path, tol_ms: int = 400) -> None:
    a = [e for e in _load_decisions(a_path) if e["type"] == "commit"]
    b = [e for e in _load_decisions(b_path) if e["type"] == "commit"]

    used_b: set[int] = set()
    print(f"A: {len(a)} commits ({a_path})")
    print(f"B: {len(b)} commits ({b_path})\n")
    for ea in a:
        match = None
        for j, eb in enumerate(b):
            if j in used_b:
                continue
            same_text = eb["data"]["utterance_clean"] == ea["data"]["utterance_clean"]
            close = abs(eb["t_session_ms"] - ea["t_session_ms"]) <= tol_ms
            if same_text or close:
                match = (j, eb)
                break
        if match is None:
            print(f"LOST   t={ea['t_session_ms']}ms {ea['data']['utterance_clean']!r}")
            continue
        j, eb = match
        used_b.add(j)
        dt = eb["t_session_ms"] - ea["t_session_ms"]
        if eb["data"]["utterance_clean"] != ea["data"]["utterance_clean"]:
            print(f"TEXT   t={ea['t_session_ms']}ms {ea['data']['utterance_clean']!r}"
                  f" -> {eb['data']['utterance_clean']!r}")
        elif abs(dt) > 0:
            print(f"MOVED  t={ea['t_session_ms']}ms dt={dt:+d}ms"
                  f" {ea['data']['utterance_clean']!r}")
        else:
            print(f"SAME   t={ea['t_session_ms']}ms {ea['data']['utterance_clean']!r}")
    for j, eb in enumerate(b):
        if j not in used_b:
            print(f"GAINED t={eb['t_session_ms']}ms {eb['data']['utterance_clean']!r}")


# ---------------------------------------------------------------- fix-wav

def fix_wav(path: Path) -> None:
    raw = path.read_bytes()
    if len(raw) < 44:
        raise SystemExit("file too small to be a WAV")
    data_len = len(raw) - 44
    fixed = bytearray(raw)
    fixed[4:8] = struct.pack("<I", 36 + data_len)
    fixed[40:44] = struct.pack("<I", data_len)
    path.write_bytes(bytes(fixed))
    print(f"fixed header: {data_len} data bytes")


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("session_dir", type=Path)
    r.add_argument("--config", default=None)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--rtf", type=float, default=1.0,
                   help="real-time factor (1.0 = original pacing)")
    r.add_argument("--with-editor", action="store_true",
                   help="also run the delta-editor LLM (needs vllm service up)")

    c = sub.add_parser("compare")
    c.add_argument("a", type=Path)
    c.add_argument("b", type=Path)
    c.add_argument("--tol-ms", type=int, default=400)

    f = sub.add_parser("fix-wav")
    f.add_argument("wav", type=Path)

    args = ap.parse_args()
    if args.cmd == "run":
        asyncio.run(_run_async(args.session_dir, args.config, args.out,
                               args.rtf, args.with_editor))
    elif args.cmd == "compare":
        compare(args.a, args.b, args.tol_ms)
    elif args.cmd == "fix-wav":
        fix_wav(args.wav)


if __name__ == "__main__":
    main()
