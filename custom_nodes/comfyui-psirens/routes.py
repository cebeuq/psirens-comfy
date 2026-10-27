"""Server routes for the mic button.

/psirens/ws bridges the browser's mic stream to the Psirens voice service
(speech-to-text, prosody, commit detection, LLM scene editor) with feed=0,
so the voice service only listens and edits the scene: ComfyUI renders.
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
routes = PromptServer.instance.routes


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
