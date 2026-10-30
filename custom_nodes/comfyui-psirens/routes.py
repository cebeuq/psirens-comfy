"""Server routes for the mic button.

/psirens/ws bridges the browser's mic stream to the Psirens voice service
(speech-to-text, prosody, commit detection, LLM scene editor) with feed=0,
so the voice service only listens and edits the scene: ComfyUI renders.

/psirens/submit runs each spoken edit through the Comfy Developer Platform:
the Comfy SDK submits the graph to the Comfy API (v2, served by Comfy's
comfy-api-proxy at COMFY_BASE_URL), which queues it on this ComfyUI. The
job's progress, previews and outputs are mirrored to the browser that
pressed Talk, so App Mode and the node editor show them as usual.
"""
import asyncio
import json
import os

import aiohttp
from aiohttp import web
from server import PromptServer

from . import canvas_store

VOICE_URL = os.environ.get("PSIRENS_VOICE_URL",
                           "http://127.0.0.1:8800").rstrip("/")
server = PromptServer.instance
routes = server.routes

# ---- Talk runs via the Comfy SDK -> Comfy API -------------------------------
# ComfyUI addresses a job's events to the client_id that queued it; through
# the Comfy API that is the API job, not the browser. When the API queues a
# Talk run here, remember which browser it belongs to and copy that job's
# events to it.
_talk_waiting: list[str] = []      # browser client ids, oldest first
_mirror: dict[str, str] = {}       # Comfy API job client_id -> browser
_flagged: set[str] = set()         # job client ids we gave preview flags
_DONE = {"execution_success", "execution_error", "execution_interrupted"}
_sdk = None


def _on_prompt(json_data):
    prompt = json_data.get("prompt") or {}
    cid = json_data.get("client_id")
    if cid and _talk_waiting and any(
            isinstance(n, dict) and n.get("class_type") == "PsirensVoicePrompt"
            for n in prompt.values()):
        _mirror[cid] = _talk_waiting.pop(0)
        # live previews: ComfyUI sends the App Mode preview format only to
        # clients that advertise it, so advertise it for the API job
        if cid not in server.sockets_metadata:
            server.sockets_metadata[cid] = {
                "feature_flags": {"supports_preview_metadata": True}}
            _flagged.add(cid)
    return json_data


server.add_on_prompt_handler(_on_prompt)
_send = server.send


async def _send_mirrored(event, data, sid=None):
    await _send(event, data, sid)
    to = _mirror.get(sid) if sid else None
    if to and to != sid:
        await _send(event, data, to)
        if event in _DONE:
            _mirror.pop(sid, None)
            if sid in _flagged:
                _flagged.discard(sid)
                server.sockets_metadata.pop(sid, None)


server.send = _send_mirrored


def _sdk_submit(graph: dict) -> str:
    global _sdk
    from comfy_sdk import Comfy        # COMFY_BASE_URL names the Comfy API
    if _sdk is None:
        _sdk = Comfy()
    return _sdk.submit(_sdk.workflows.from_json(graph)).id


@routes.post("/psirens/submit")
async def talk_submit(request):
    """Queue a Talk run through the Comfy SDK and the Comfy API."""
    body = await request.json()
    graph = body.get("prompt")
    browser = str(body.get("client_id") or "")
    if not isinstance(graph, dict) or not browser:
        return web.json_response({"error": "need prompt and client_id"},
                                 status=400)
    if not os.environ.get("COMFY_BASE_URL"):
        return web.json_response({"error": "COMFY_BASE_URL not set"},
                                 status=501)
    _talk_waiting.append(browser)
    try:
        job_id = await asyncio.to_thread(_sdk_submit, graph)
    except Exception as e:
        if browser in _talk_waiting:
            _talk_waiting.remove(browser)
        return web.json_response({"error": f"comfy api: {e}"}, status=502)
    return web.json_response({"job_id": job_id, "via": "comfy-sdk"})


@routes.get("/psirens/canvas")
async def canvas_slots(request):
    return web.json_response(canvas_store.describe())


@routes.post("/psirens/reset")
async def canvas_reset(request):
    """Forget a session's canvas: the next voice prompt starts a new picture."""
    body = await request.json() if request.can_read_body else {}
    session = str((body or {}).get("session") or "ui")
    canvas_store.drop(session)
    return web.json_response({"reset": session})


@routes.post("/psirens/restore")
async def canvas_restore(request):
    """Undo: bring back the canvas latent saved at keyframe `rev`."""
    body = await request.json()
    ok = canvas_store.restore(str(body.get("session") or "ui"),
                              str(body.get("slot") or "latest"),
                              int(body.get("rev", -1)))
    return web.json_response({"restored": ok}, status=200 if ok else 404)


@routes.get("/psirens/status")
async def voice_status(request):
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(VOICE_URL + "/healthz",
                             timeout=aiohttp.ClientTimeout(total=3)) as r:
                return web.json_response({"voice": r.status == 200,
                                          "url": VOICE_URL})
    except Exception as e:
        return web.json_response({"voice": False, "error": str(e)})


@routes.get("/psirens/ws")
async def voice_bridge(request):
    ws = web.WebSocketResponse(max_msg_size=0)
    await ws.prepare(request)
    up_url = VOICE_URL.replace("http", "ws", 1) + "/ws/session?feed=0"
    try:
        async with aiohttp.ClientSession() as s:
            async with s.ws_connect(up_url, max_msg_size=0) as up:
                async def downstream():
                    async for m in up:
                        if m.type == aiohttp.WSMsgType.TEXT:
                            await ws.send_str(m.data)
                        elif m.type == aiohttp.WSMsgType.BINARY:
                            await ws.send_bytes(m.data)
                        else:
                            break
                    await ws.close()

                task = asyncio.create_task(downstream())
                try:
                    async for m in ws:
                        if m.type == aiohttp.WSMsgType.BINARY:
                            await up.send_bytes(m.data)
                        elif m.type == aiohttp.WSMsgType.TEXT:
                            await up.send_str(m.data)
                        else:
                            break
                finally:
                    task.cancel()
    except Exception as e:
        if not ws.closed:
            await ws.send_str(json.dumps({"type": "error", "data": {
                "where": "psirens", "message": f"voice service: {e}"}}))
    if not ws.closed:
        await ws.close()
    return ws
