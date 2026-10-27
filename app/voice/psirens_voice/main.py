"""FastAPI app: single duplex websocket + static frontend + health.

Websocket protocol (/ws/session):
  - binary messages: int16 mono PCM audio at cfg.audio.sample_rate, up only
  - text messages: JSON, both directions
      client -> server: {"type": "client_hello", "data": {...}}
                        {"type": "revert", "data": {"commit_id": ...}}   (M5)
                        {"type": "stop"}
      server -> client: Event envelopes (events.py)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from .commit.session_wiring import CommitSession
from .config import load_config
from .editor.service import EditorSession
from .scene.live import LiveScene
from .events import Event
from .prosody.engine import ProsodySession
from .prosody.ser import SerEngine
from .reflex.session import ReflexSession
from .session import Session
from .stt import SttEngine, SttSession
from .stt_nemotron import NemotronSttEngine, NemotronSttSession
from .stt_voxtral import VoxtralSttEngine, VoxtralSttSession

log = logging.getLogger("psirens")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

cfg = load_config()
app = FastAPI(title="psirens-voice")

FRONTEND_DIR = Path(os.environ.get("PSIRENS_FRONTEND", "/srv/frontend"))

AUDIO_QUEUE_MAX = 25  # ~2s of 80ms frames

# throttled event types: browser gets at most N/s, the recorder gets full rate
WS_THROTTLE_MS = {"vad_state": 1000 / cfg.ui.vad_event_hz}

STT_KIND = (os.environ.get("STT_ENGINE")
            or cfg.stt.get("engine", "kyutai")) if cfg.stt.enabled else None
if STT_KIND == "voxtral":
    stt_engine = VoxtralSttEngine()
elif STT_KIND == "nemotron":
    stt_engine = NemotronSttEngine(cfg.stt.model)
elif STT_KIND == "kyutai":
    stt_engine = SttEngine(cfg.stt.get("kyutai_model", cfg.stt.model))
else:
    stt_engine = None
ser_engine: SerEngine | None = SerEngine(cfg.prosody.ser.model) if cfg.prosody.enabled else None

# service-to-service event feed (/ws/feed): downstream services (the
# continuity renderer) subscribe here to receive every session event, and may
# send canvas_frame messages back, which are republished to the browser.
FEED_CLIENTS: set[WebSocket] = set()
CURRENT_SESSION: Session | None = None
# renderer engine state, updated from the renderer's engine_status feed
# events. Engine choice is RENDERER state, not session state: the HTTP
# control below works with or without a live session.
ENGINE_STATUS: dict = {"engine": "klein", "state": "ready"}


def _load_models_blocking():
    # sequential on purpose: two concurrent CUDA context initializations in
    # one process is asking for trouble
    if stt_engine is not None:
        stt_engine.load()
    if ser_engine is not None:
        ser_engine.load()


@app.on_event("startup")
async def load_models():
    import asyncio as aio
    # a restart can land while vLLM is in sleep mode (its healthcheck
    # answers either way) — wake it unconditionally; no-op when awake.
    # Without this, every editor call stalls until timeout.
    aio.ensure_future(_vllm_gpu("load"))
    await aio.get_event_loop().run_in_executor(None, _load_models_blocking)
    if cfg.editor.enabled:
        aio.ensure_future(_warm_editor())


async def _warm_editor():
    """Warmup calls at app startup: compile the guided-json grammar and cache
    the system-prompt prefix on BOTH llm endpoints (commit editor and draft
    editor), so a session's first patch/draft doesn't pay cold-start costs.
    Retries until the services are up."""
    from .editor.client import DeltaEditorClient
    from .scene.state import SceneState

    targets = [("editor", cfg.editor.base_url,
                os.environ.get(cfg.editor.model_env, "google/gemma-4-E4B-it"))]
    if cfg.draft.enabled:
        targets.append(("draft", cfg.draft.base_url,
                        os.environ.get(cfg.draft.model_env, "LiquidAI/LFM2.5-8B-A1B")))

    async def warm(name, base_url, model):
        from .editor.client import EditorError, EditorTimeout

        client = DeltaEditorClient(base_url=base_url, model=model,
                                   temperature=0.0, max_tokens=64,
                                   timeout_s=30, lenient=True)
        # warm until the server answers FAST, not merely answers: a cold
        # vLLM JIT-compiles kernels lazily per shape (guided-JSON bitmask,
        # speculative-decoding kernels, ...) and the spike must land HERE,
        # never on the user's first utterance (which once timed out on it)
        import time as _time
        fast_needed = 2
        fast = 0
        for attempt in range(40):
            t0 = _time.monotonic()
            answered = False
            try:
                await client.propose_patch(
                    SceneState(), "tone: energy=med valence=med tempo=med hesitation=low",
                    f"warmup {attempt}", max_changed_fields=7,
                    below_magnitude=0.0, max_retries=0)
                answered = True
            except EditorTimeout as e:
                log.info("%s llm warmup waiting (attempt %d): %s",
                         name, attempt + 1, e)
            except EditorError as e:
                # the server ANSWERED — grammar compiled, kernels touched.
                # Whether the warmup patch parsed is irrelevant.
                answered = True
            except Exception as e:
                log.info("%s llm warmup error (attempt %d): %s",
                         name, attempt + 1, e)
            if answered:
                dt = _time.monotonic() - t0
                if dt < 3.0:
                    fast += 1
                    if fast >= fast_needed:
                        log.info("%s llm warmed up (attempt %d, %.1fs)",
                                 name, attempt + 1, dt)
                        return
                else:
                    fast = 0
                    log.info("%s llm warm but slow (%.1fs) — JIT still "
                             "compiling, continuing", name, dt)
                continue
            await asyncio.sleep(15)
        log.warning("%s llm warmup never got fast; first call may stall", name)

    await asyncio.gather(*(warm(*t) for t in targets))


def models_ready() -> bool:
    return ((stt_engine is None or stt_engine.loaded)
            and (ser_engine is None or ser_engine.loaded))


@app.get("/healthz")
async def healthz():
    if GPU_FREED:
        return {"ok": True, "gpu": "freed"}      # freed on purpose ≠ unhealthy
    if not models_ready():
        from fastapi.responses import JSONResponse
        return JSONResponse({"ok": False, "reason": "models loading"}, status_code=503)
    return {"ok": True}


# ---- engine control ---------------------------------------------------------
# HTTP, deliberately not websocket: the session ws only exists while the mic
# runs, but engine choice is renderer state and must be switchable any time.
@app.get("/engine")
async def get_engine():
    return ENGINE_STATUS


@app.post("/engine")
async def set_engine(body: dict):
    from fastapi.responses import JSONResponse
    eng = str((body or {}).get("engine", ""))
    if eng not in ("klein", "u15", "h3"):
        return JSONResponse({"ok": False, "error": "unknown engine"},
                            status_code=400)
    if not FEED_CLIENTS:
        return JSONResponse({"ok": False, "error": "renderer not connected"},
                            status_code=503)
    ENGINE_STATUS.update({"engine": eng, "state": "loading", "error": None})
    await _broadcast_to_feed(json.dumps(
        {"type": "set_engine", "data": {"engine": eng}}))
    return {"ok": True, **ENGINE_STATUS}


# ---- free-GPU control -------------------------------------------------------
# The whole stack idles at ~78GB VRAM; the owner shares this GPU with other
# work. "Free" releases everything a running process CAN release without
# dying: STT/SER unload in-process, the renderer unloads FLUX+SAM (feed
# event), and vLLM uses its official sleep mode (level 1: weights parked in
# host RAM, KV cache discarded — wake is seconds). "Load" reverses all three.
GPU_FREED = False
_GPU_LOCK = asyncio.Lock()
VLLM_ROOT = cfg.editor.base_url.rsplit("/v1", 1)[0] if cfg.editor.enabled else None
VOXTRAL_ROOT = (os.environ.get("VOXTRAL_HTTP", "http://voxtral:8000")
                if STT_KIND == "voxtral" else None)


async def _vllm_gpu(action: str) -> str:
    """POST the sleep-mode endpoints of every vLLM server (editor +
    voxtral); returns a status string (never raises — a partial free is
    still a big win and must not abort the rest)."""
    if VOXTRAL_ROOT is not None:
        await _vllm_one(VOXTRAL_ROOT, action)
    return await _vllm_one(VLLM_ROOT, action)


async def _vllm_one(root: str | None, action: str) -> str:
    if root is None:
        return "disabled"
    import httpx
    url = (f"{root}/sleep?level=1" if action == "free"
           else f"{root}/wake_up")
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(url)
            if r.status_code // 100 == 2:
                return "ok"
            return f"http {r.status_code}"
    except Exception as e:
        log.warning("vllm %s failed: %s", action, e)
        return f"error: {e}"


@app.post("/gpu/free")
async def gpu_free():
    from fastapi.responses import JSONResponse
    global GPU_FREED
    async with _GPU_LOCK:
        if CURRENT_SESSION is not None:
            return JSONResponse({"ok": False, "reason": "session active"},
                                status_code=409)
        loop = asyncio.get_running_loop()

        def _unload():
            if stt_engine is not None:
                stt_engine.unload()
            if ser_engine is not None:
                ser_engine.unload()

        await loop.run_in_executor(None, _unload)
        await _broadcast_to_feed(json.dumps(
            {"type": "gpu", "data": {"action": "free"}}))
        vllm_status = await _vllm_gpu("free")
        GPU_FREED = True
        log.info("gpu freed (vllm sleep: %s)", vllm_status)
        return {"ok": True, "vllm": vllm_status}


@app.post("/gpu/load")
async def gpu_load():
    global GPU_FREED
    async with _GPU_LOCK:
        if not GPU_FREED and models_ready():
            return {"ok": True, "already": True}
        vllm_status = await _vllm_gpu("load")
        await _broadcast_to_feed(json.dumps(
            {"type": "gpu", "data": {"action": "load"}}))
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _load_models_blocking)
        GPU_FREED = False
        log.info("gpu loaded (vllm wake: %s)", vllm_status)
        return {"ok": True, "vllm": vllm_status}


@app.get("/gpu")
async def gpu_state():
    return {"freed": GPU_FREED, "ready": models_ready()}


@app.get("/sessions/{session_id}/events.jsonl")
async def get_events(session_id: str):
    from fastapi.responses import FileResponse, PlainTextResponse
    p = Path(cfg.paths.data_dir) / "sessions" / session_id / "events.jsonl"
    if not p.is_file() or ".." in session_id or "/" in session_id:
        return PlainTextResponse("not found", status_code=404)
    return FileResponse(p, media_type="application/jsonl")


@app.websocket("/ws/feed")
async def ws_feed(ws: WebSocket):
    """Downstream-service feed. Sends every event of the active session;
    accepts {"type": "canvas_frame", "data": {...}} messages which are
    published onto the active session's bus (and reach the browser)."""
    await ws.accept()
    FEED_CLIENTS.add(ws)
    log.info("feed client connected (%d total)", len(FEED_CLIENTS))
    # the style bank is STATE carried by a transient event: a client that
    # (re)connects mid-session would silently lose the session's style,
    # so replay the current bank on connect
    try:
        ed = getattr(CURRENT_SESSION, "editor", None)
        refs = getattr(ed, "style_refs", None)
        if refs:
            await ws.send_text(json.dumps({
                "type": "style_bank",
                "data": {"refs": refs,
                         "suffix": getattr(ed, "style_suffix", None)}}))
            log.info("style bank replayed to feed client (%d refs)",
                     len(refs))
    except Exception:
        log.exception("style bank replay failed")
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if (text := msg.get("text")) is not None:
                try:
                    incoming = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if incoming.get("type") == "set_engine":
                    # engine switching must work with no live session
                    # (bench tooling, headless ops): relay feed -> feed
                    await _broadcast_to_feed(text)
                    continue
                if incoming.get("type") == "engine_status":
                    # track renderer engine state globally (GET /engine)
                    d = incoming.get("data") or {}
                    if d.get("engine"):
                        ENGINE_STATUS.update(
                            {"engine": d["engine"],
                             "state": d.get("state", "ready"),
                             "error": d.get("error")})
                if (incoming.get("type") in ("canvas_frame", "segment_result",
                                             "masks", "insert_preview",
                                             "engine_status")
                        and CURRENT_SESSION is not None):
                    CURRENT_SESSION.bus.publish(
                        incoming["type"], incoming.get("data", {}),
                        CURRENT_SESSION.t_session_ms)
    finally:
        FEED_CLIENTS.discard(ws)
        log.info("feed client disconnected (%d left)", len(FEED_CLIENTS))


async def _broadcast_to_feed(payload: str) -> None:
    for client in list(FEED_CLIENTS):
        try:
            await client.send_text(payload)
        except Exception:
            FEED_CLIENTS.discard(client)


@app.websocket("/ws/session")
async def ws_session(ws: WebSocket):
    await ws.accept()
    if GPU_FREED:
        await ws.send_text(json.dumps({"type": "error", "data": {
            "where": "gpu", "message": "GPU is freed — load it first",
            "recoverable": True}}))
        await ws.close(code=1013)
        return
    if not models_ready():
        await ws.send_text(json.dumps({"type": "error", "data": {
            "where": "session", "message": "models still loading", "recoverable": True}}))
        await ws.close(code=1013)
        return
    if stt_engine is not None and not stt_engine.acquire():
        await ws.send_text(json.dumps({"type": "error", "data": {
            "where": "session", "message": "another session is active", "recoverable": True}}))
        await ws.close(code=1013)
        return

    session = Session(cfg)
    global CURRENT_SESSION
    CURRENT_SESSION = session
    log.info("session %s connected", session.id)

    if stt_engine is not None:
        stt_session_cls = (VoxtralSttSession if STT_KIND == "voxtral"
                           else NemotronSttSession
                           if STT_KIND == "nemotron" else SttSession)
        session.sinks.append(stt_session_cls(
            stt_engine, session.bus, asyncio.get_running_loop(),
            vad_event_hz=cfg.ui.vad_event_hz))
    prosody: ProsodySession | None = None
    if cfg.prosody.enabled:
        prosody = ProsodySession(cfg, ser_engine, session.bus,
                                 asyncio.get_running_loop(), cfg.commit.fillers)
        session.sinks.append(prosody)
    commit_session: CommitSession | None = None
    if cfg.commit.enabled:
        commit_session = CommitSession(
            cfg, session.bus, prosody, asyncio.get_running_loop(),
            delay_ms=stt_engine.delay_ms if stt_engine else 500)
    editor_session: EditorSession | None = None
    reflex_session: ReflexSession | None = None
    if cfg.editor.enabled:
        from .modes import DEFAULT_MODE, MODES
        mode_name = ws.query_params.get("mode", DEFAULT_MODE)
        mode = MODES.get(mode_name, MODES[DEFAULT_MODE])
        live = LiveScene()
        editor_session = EditorSession(cfg, session.bus,
                                       asyncio.get_running_loop(), live=live,
                                       mode=mode)
        session.editor = editor_session   # feed replay reads style bank
        # v3: the separate draft-LLM loop is gone (all four designs deleted
        # it); mid-sentence behavior now lives inside EditorSession per mode
        if cfg.reflex.enabled:
            reflex_session = ReflexSession(cfg, session.bus, live,
                                           asyncio.get_running_loop())

    # feed=0: a client that renders by itself (the ComfyUI mic button)
    # gets transcripts + scene edits but never drives the app's renderer
    feed_on = ws.query_params.get("feed", "1") != "0"
    ws_queue = session.bus.subscribe()
    audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=AUDIO_QUEUE_MAX)
    dropped_frames = 0
    last_sent_ms: dict[str, float] = {}

    async def forward_events():
        """bus -> browser, and drain the recorder as a side effect.
        High-rate event types are throttled for the browser only."""
        import time as _time
        while True:
            ev: Event = await ws_queue.get()
            session.drain_recorder()
            throttle = WS_THROTTLE_MS.get(ev.type)
            if throttle is not None:
                now = _time.monotonic() * 1000
                if now - last_sent_ms.get(ev.type, -1e9) < throttle:
                    continue
                last_sent_ms[ev.type] = now
            payload = ev.model_dump_json()
            if feed_on and ev.type not in ("canvas_frame", "segment_result",
                                           "masks", "insert_preview",
                                           "engine_status"):
                # don't echo renderer-produced events back to their producer
                await _broadcast_to_feed(payload)
            try:
                await ws.send_text(payload)
            except Exception:
                # browser gone; keep draining so the recorder still gets events
                pass

    async def consume_audio():
        while True:
            pcm = await audio_queue.get()
            try:
                await session.feed_audio(pcm)
            finally:
                audio_queue.task_done()

    forward_task = asyncio.create_task(forward_events())
    audio_task = asyncio.create_task(consume_audio())

    session.start()
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if (pcm := msg.get("bytes")) is not None:
                try:
                    audio_queue.put_nowait(pcm)
                except asyncio.QueueFull:
                    audio_queue.get_nowait()  # drop oldest
                    audio_queue.task_done()
                    audio_queue.put_nowait(pcm)
                    dropped_frames += 1
                    session.bus.publish(
                        "error",
                        {"where": "audio_queue", "message": "backpressure: dropped oldest frame",
                         "recoverable": True, "dropped_total": dropped_frames},
                        session.t_session_ms,
                    )
            elif (text := msg.get("text")) is not None:
                try:
                    control = json.loads(text)
                except json.JSONDecodeError:
                    continue
                ctype = control.get("type")
                if ctype == "stop":
                    break
                # a malformed control message must NEVER kill the session:
                # one bad field from a stale tab used to crash this whole
                # handler, after which every button spoke to a dead socket
                try:
                    if ctype == "client_hello":
                        log.info("session %s hello: %s", session.id,
                                 control.get("data"))
                    elif ctype == "revert":
                        if editor_session is not None:
                            editor_session.revert(
                                int(control.get("data", {}).get("commit_id")
                                    or -1),
                                session.t_session_ms)
                    elif ctype == "insert_commit":
                        if editor_session is not None:
                            asyncio.ensure_future(
                                editor_session.handle_insert_commit(
                                    control.get("data") or {},
                                    session.t_session_ms))
                    elif ctype == "insert_cancel":
                        if editor_session is not None:
                            editor_session.handle_insert_cancel(
                                control.get("data") or {},
                                session.t_session_ms)
                    elif ctype == "upload_drop":
                        # image dropped on the canvas: insert its object
                        session.bus.publish("upload_drop",
                                            control.get("data", {}),
                                            session.t_session_ms)
                    elif ctype == "ink_stroke":
                        # streamed stroke (put-that-there): server-side
                        # clock stamps it for word-time binding
                        session.bus.publish("ink_stroke",
                                            control.get("data", {}),
                                            session.t_session_ms)
                    elif ctype == "ink_clear":
                        session.bus.publish("ink_clear", {},
                                            session.t_session_ms)
                    elif ctype == "ink":
                        # freehand drawing finished: strokes to the ink
                        # decider (editor session routes by reading)
                        session.bus.publish("ink",
                                            control.get("data", {}),
                                            session.t_session_ms)
                    elif ctype == "undo":
                        # manual back button: deterministic keyframe walk-back
                        if editor_session is not None:
                            editor_session.undo(session.t_session_ms)
                    elif ctype == "redo":
                        # manual forward button: inverse walk, same keyframes
                        if editor_session is not None:
                            editor_session.redo(session.t_session_ms)
                    elif ctype == "mouse_move":
                        # drag/resize release from the canvas — a first-class
                        # commit through the same pipeline as a spoken edit
                        if editor_session is not None:
                            editor_session.mouse_commit(
                                control.get("data", {}), session.t_session_ms)
                    elif ctype == "click_segment":
                        # rides the bus -> /ws/feed -> renderer, which
                        # answers with a segment_result
                        session.bus.publish("click_segment",
                                            control.get("data", {}),
                                            session.t_session_ms)
                    elif ctype == "lasso_segment":
                        # freehand trace -> renderer snaps it to the object
                        session.bus.publish("lasso_segment",
                                            control.get("data", {}),
                                            session.t_session_ms)
                    elif ctype == "set_edit_mode":
                        mode = str(control.get("data", {}).get("mode", ""))
                        from .editor.strategy import STRATEGIES
                        if editor_session is not None and mode in STRATEGIES:
                            from .editor.strategy import resolve
                            editor_session.strategy = resolve(mode)
                            log.info("edit mode -> %s", mode)
                    elif ctype == "set_region":
                        # active selection: spoken edits scope to this mask
                        if editor_session is not None:
                            editor_session.set_region(
                                control.get("data") or None)
                    elif ctype == "set_resolution":
                        size = int(control.get("data", {}).get("size") or 0)
                        if size in (512, 768, 1024):
                            # rides the bus -> /ws/feed -> renderer
                            session.bus.publish("set_resolution",
                                                {"size": size},
                                                session.t_session_ms)
                    elif ctype == "set_model":
                        model = str(control.get("data", {}).get("model", ""))
                        if model in ("fast", "hq"):
                            session.bus.publish("set_model",
                                                {"model": model},
                                                session.t_session_ms)
                    elif ctype == "set_engine":
                        eng = str(control.get("data", {}).get("engine", ""))
                        if eng in ("klein", "u15", "h3"):
                            # rides the bus -> /ws/feed -> renderer, which
                            # swaps weights between renders and answers
                            # with engine_status events
                            session.bus.publish("set_engine",
                                                {"engine": eng},
                                                session.t_session_ms)
                except Exception:
                    log.exception("control %r failed (session continues)",
                                  ctype)
                    session.bus.publish("error", {
                        "where": "control", "recoverable": True,
                        "message": f"{ctype} failed"}, session.t_session_ms)
    except WebSocketDisconnect:
        pass
    finally:
        # let queued audio finish, then tear down
        try:
            await asyncio.wait_for(audio_queue.join(), timeout=0.1)
        except (asyncio.TimeoutError, Exception):
            pass
        audio_task.cancel()
        if commit_session is not None:
            await commit_session.close()
        if reflex_session is not None:
            await reflex_session.close()
        if editor_session is not None:
            await editor_session.close()
        await session.close()
        forward_task.cancel()
        session.bus.unsubscribe(ws_queue)
        if stt_engine is not None:
            stt_engine.release()
        if CURRENT_SESSION is session:
            CURRENT_SESSION = None
        log.info("session %s closed at t=%dms, dropped=%d",
                 session.id, session.t_session_ms, dropped_frames)


class _NoCacheStatic(StaticFiles):
    """Serve frontend files with Cache-Control: no-cache. Without it,
    browsers heuristically reuse cached app.js/style.css for hours after
    a deploy — new HTML then references handlers that don't exist in the
    stale script (observed as dead buttons). no-cache still allows 304
    revalidation, so reloads stay cheap."""

    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "no-cache"
        return resp


if FRONTEND_DIR.is_dir():
    app.mount("/", _NoCacheStatic(directory=str(FRONTEND_DIR), html=True),
              name="frontend")
