"""Continuity renderer service.

Modes (RENDERER_MODE env):
  - "flux2": edit-native rendering via FLUX.2 Klein (root generation once,
             everything after is an edit of the persistent canvas)
  - "mock":  color-field morphs (CPU, instant, for development)

Renders are LATEST-TARGET-WINS: while a frame is being generated, newer
render_targets simply replace the pending one — the canvas always moves
toward the most recent intent and never builds a backlog.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import pathlib
import time
from collections import deque

import websockets

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s renderer %(levelname)s %(message)s")
log = logging.getLogger("renderer")

MODE = os.environ.get("RENDERER_MODE", "mock")
# provisional renders (previews of half-finished sentences) are great on a
# dedicated GPU; on ONE shared card they steal time from the editor LLM at
# exactly the moment a commit needs it. PSIRENS_PROVISIONAL=0 drops them.
PROVISIONAL = os.environ.get("PSIRENS_PROVISIONAL", "1") not in ("0", "false",
                                                                 "no")
FEED_URL = os.environ.get("FEED_URL", "ws://orchestrator:8000/ws/feed")
HEARTBEAT = pathlib.Path(os.environ.get("HEARTBEAT_FILE", "/tmp/renderer-heartbeat"))
MOCK_FRAME_INTERVAL_S = 0.3


class RendererApp:
    def __init__(self) -> None:
        self.mode = MODE
        self.mock = None
        self.zimage = None
        # committed edits are deltas: dropping one permanently loses its
        # content from the canvas. They queue FIFO and ALWAYS execute, in
        # order. Only provisional churn (overlay/reflex/draft) is
        # latest-target-wins and preemptable.
        self.commit_q: deque = deque()
        self.pending: dict | None = None      # provisional latest-wins slot
        self.pending_event = asyncio.Event()
        self.gpu_req: str | None = None       # "free"/"load" from the feed
        self.engine = os.environ.get("RENDER_ENGINE", "klein")
        self.engine_req: str | None = None    # pending engine switch
        # session style bank: refs (b64) + distilled suffix from the
        # orchestrator's style_bank event; applied to ROOT renders only
        self.style: dict | None = None
        self.seq = 0
        # single outbound queue: step previews arrive from the render thread
        # while the worker coroutine is blocked in the executor, so every
        # frame goes through one sender task instead of racing ws.send
        self.out_q: asyncio.Queue = asyncio.Queue()

        from .mock import MockCanvas
        self.mock = MockCanvas()
        self.edit_native = self.mode == "flux2"
        if self.mode == "flux2":
            from .engines import free_other_sidecars, make_canvas
            # a restart while a sidecar still holds weights (from a prior
            # switch) would OOM our own load — free everyone else first
            free_other_sidecars(self.engine)
            self.zimage = make_canvas(self.engine)
            self.zimage.load()  # blocking at startup; healthcheck allows for it

    # ---- feed reader ----
    async def read_feed(self, ws) -> None:
        async for raw in ws:
            HEARTBEAT.write_text(str(time.time()))
            try:
                ev = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "session_start":
                self.style = None
                self.mock.__init__()
                if self.zimage:
                    self.zimage.reset_session()
                self.pending = None
                self.commit_q.clear()
            elif ev.get("type") == "set_resolution":
                size = int(ev.get("data", {}).get("size", 0))
                if size in (512, 768, 1024) and hasattr(self.zimage, "set_size"):
                    self.zimage.set_size(size)
            elif ev.get("type") == "set_engine":
                # handled by the render worker between renders, like "gpu":
                # a switch can never yank weights from under a denoise loop
                name = str(ev.get("data", {}).get("engine", ""))
                if name in ("klein", "u15", "h3"):
                    self.engine_req = name
                    self.pending_event.set()
            elif ev.get("type") == "set_model":
                tag = str(ev.get("data", {}).get("model", ""))
                if hasattr(self.zimage, "set_model"):
                    self.zimage.set_model(tag)
            elif ev.get("type") == "click_segment":
                asyncio.create_task(self._click_segment(ev.get("data", {})))
            elif ev.get("type") == "lasso_segment":
                asyncio.create_task(self._lasso_segment(ev.get("data", {})))
            elif ev.get("type") == "cutout_upload":
                asyncio.create_task(self._cutout_upload(
                    ev.get("data", {})))
            elif ev.get("type") == "style_bank":
                d = ev.get("data") or {}
                self.style = d if d.get("refs") else None
                log.info("style bank: %d refs%s",
                         len((d or {}).get("refs") or []),
                         " cleared" if not self.style else "")
            elif ev.get("type") == "gpu":
                # handled by the render worker between renders, so a free
                # can never yank the model out from under an in-flight
                # denoise loop
                action = ev.get("data", {}).get("action")
                if action in ("free", "load"):
                    self.gpu_req = action
                    self.pending_event.set()
            elif ev.get("type") == "render_target":
                d = ev["data"]
                committed = (d.get("lane") == "committed"
                             or (d.get("lane") is None
                                 and d.get("source") == "commit"))
                if committed:
                    self.commit_q.append(d)
                elif not PROVISIONAL:
                    continue
                else:
                    self.pending = d
                self.pending_event.set()

    # ---- outbound sender ----
    async def send_worker(self, ws) -> None:
        while True:
            msg = await self.out_q.get()
            await ws.send(msg)

    def _frame_msg(self, jpeg_b64: str, source, strength: float,
                   **extra) -> str:
        self.seq += 1
        return json.dumps({"type": "canvas_frame", "data": {
            "jpeg_b64": jpeg_b64,
            "seq": self.seq,
            "source": source,
            "strength": strength,
            **extra,
        }})

    async def _maybe_calibrate(self, loop) -> None:
        """Scene-floor calibration in idle GPU time (one no-op render pair
        per root/model/resolution change) — see calibrate_scene_floor."""
        if (self.zimage is None
                or not getattr(self.zimage, "needs_floor_calib", False)
                or self.commit_q or self.pending is not None
                or self.gpu_req is not None):
            return
        try:
            await loop.run_in_executor(
                None, self.zimage.calibrate_scene_floor)
        except Exception:
            log.exception("scene floor calibration failed (defaults stay)")

    async def _maybe_embed(self, loop) -> None:
        """Mouse layer: precompute the click-segmentation image embedding
        while the queue is idle, so the user's first click on a fresh
        frame doesn't pay the ~200ms encoder pass."""
        if (self.zimage is None or not hasattr(self.zimage, "ensure_embedding")
                or self.commit_q or self.pending is not None
                or self.gpu_req is not None):
            return
        try:
            await loop.run_in_executor(None, self.zimage.ensure_embedding)
        except Exception:
            log.exception("embedding precompute failed (clicks still work)")

    async def _cutout_upload(self, d: dict) -> None:
        """Background-removed placement preview for a dropped object —
        the same SAM cut the eventual insert will use, returned fast so
        the user can position it before anything renders."""
        if not hasattr(self.zimage, "cut_object"):
            return
        loop = asyncio.get_running_loop()
        png = await loop.run_in_executor(
            None, lambda: self.zimage.cut_object(
                d.get("b64") or "", d.get("name") or "object"))
        self.out_q.put_nowait(json.dumps({
            "type": "insert_preview", "data": {
                "req_id": d.get("req_id"),
                "png_b64": png, "ok": png is not None}}))

    async def _lasso_segment(self, d: dict) -> None:
        """Answer one freehand trace: snap it to the object it encircles.
        Same off-queue treatment as clicks."""
        loop = asyncio.get_running_loop()
        t0 = time.monotonic()
        seg = None
        try:
            if self.zimage is not None:
                pts = d.get("points") or []
                seg = await loop.run_in_executor(
                    None, lambda: self.zimage.lasso_segment(pts))
        except Exception:
            log.exception("lasso segmentation failed")
        self.out_q.put_nowait(json.dumps({"type": "segment_result", "data": {
            "req_id": d.get("req_id"),
            "seg": seg,
        }}))
        log.info("lasso segment (%d pts) -> %s in %dms",
                 len(d.get("points") or []),
                 "hit" if seg else "miss",
                 int((time.monotonic() - t0) * 1000))

    async def _click_segment(self, d: dict) -> None:
        """Answer one click: which object is at this point? Runs off the
        render queue — a click must never wait behind a 1.6s render."""
        loop = asyncio.get_running_loop()
        t0 = time.monotonic()
        seg = None
        try:
            if self.zimage is not None:
                seg = await loop.run_in_executor(
                    None, lambda: self.zimage.point_segment(
                        float(d.get("x", 0.5)), float(d.get("y", 0.5))))
        except Exception:
            log.exception("click segmentation failed")
        self.out_q.put_nowait(json.dumps({"type": "segment_result", "data": {
            "req_id": d.get("req_id"),
            "x": d.get("x"), "y": d.get("y"),
            "seg": seg,
        }}))
        log.info("click segment (%.2f, %.2f) -> %s in %.0fms",
                 float(d.get("x", 0)), float(d.get("y", 0)),
                 "hit" if seg else "miss",
                 (time.monotonic() - t0) * 1000)

    # ---- render worker ----
    async def render_worker(self, ws) -> None:
        loop = asyncio.get_running_loop()
        while True:
            if self.engine_req and self.zimage is not None:
                req, self.engine_req = self.engine_req, None
                if req != self.engine:
                    await self._switch_engine(loop, req)
                else:
                    # no-op switch still ACKs, so a drifted status view
                    # (orchestrator restart) can never wait forever
                    self.out_q.put_nowait(json.dumps(
                        {"type": "engine_status",
                         "data": {"engine": self.engine,
                                  "state": "ready"}}))
            if self.gpu_req and self.zimage is not None:
                req, self.gpu_req = self.gpu_req, None
                if req == "free":
                    await loop.run_in_executor(None, self.zimage.unload)
                else:                           # pre-warm before the session
                    await loop.run_in_executor(None, self.zimage.load)
            if not self.commit_q and self.pending is None:
                await self.pending_event.wait()
                self.pending_event.clear()
            # committed edits first, strictly in order; provisional only
            # when the committed queue is drained
            if self.commit_q:
                target = self.commit_q.popleft()
                committed = True
            else:
                target = self.pending
                self.pending = None
                committed = False
            if target is None:
                continue
            prompt = target.get("prompt", "")
            strength = float(target.get("strength", 0.3))
            entities = target.get("entities") or []
            source = target.get("source")

            def on_preview(jpeg_b64: str, step: int, steps: int) -> None:
                # called from the executor thread mid-denoise
                msg = self._frame_msg(jpeg_b64, source, strength,
                                      preview=True, step=step, steps=steps,
                                      lane=("committed" if committed
                                            else "provisional"))
                loop.call_soon_threadsafe(self.out_q.put_nowait, msg)

            t0 = time.monotonic()
            if self.zimage is not None:
                from .preview import RenderPreempted
                extra = {}
                if self.edit_native:
                    # provisional churn must NEVER create the root: a
                    # mid-sentence fragment as the anchor of the whole edit
                    # chain poisons every frame after it. Before the first
                    # commit roots the canvas, provisional targets are
                    # dropped.
                    provisional = (target.get("lane") == "provisional"
                                   or (target.get("lane") is None
                                       and source in ("reflex", "overlay",
                                                      "draft")))
                    if provisional and getattr(self.zimage, "canvas", None) is None:
                        log.info("dropping provisional [%s] target — no root yet",
                                 source)
                        continue
                    # undo path: pixel-exact keyframe restore, no diffusion
                    restore_rev = target.get("restore_rev")
                    if restore_rev is not None:
                        jpeg_b64 = self.zimage.restore(int(restore_rev))
                        if jpeg_b64 is not None:
                            if (target.get("snapshot")
                                    and target.get("commit_rev") is not None):
                                import base64 as _b64
                                self.zimage.snapshot(
                                    int(target["commit_rev"]),
                                    _b64.b64decode(jpeg_b64))
                            self.out_q.put_nowait(self._frame_msg(
                                jpeg_b64, source, strength,
                                render_s=0.0, restored=int(restore_rev),
                                lane="committed"))
                            await self._maybe_calibrate(loop)
                            await self._maybe_embed(loop)
                            continue
                        target["reset"] = True   # keyframe evicted: re-anchor
                    sb = self.style or {}
                    extra = {"style_refs": sb.get("refs"),
                             "style_suffix": sb.get("suffix"),
                             "instruction": target.get("edit_instruction"),
                             "reset": bool(target.get("reset")),
                             "lane": target.get("lane"),
                             "commit_rev": target.get("commit_rev"),
                             "do_snapshot": bool(target.get("snapshot")),
                             "move_ops": target.get("move_ops"),
                             "camera": bool(target.get("camera")),
                             "region": target.get("region"),
                             "preserve_names": target.get("preserve"),
                             "ink": target.get("ink")}
                    draft = target.get("draft") or {}
                # committed renders are NEVER cancelled — each carries a
                # delta whose loss would silently erase a user edit from
                # the canvas. Provisional renders abort between steps for
                # anything newer.
                cancel = ((lambda: False) if committed
                          else (lambda: self.pending is not None
                                or bool(self.commit_q)))
                try:
                    use_loop = (committed and draft.get("policy") == "loop"
                                and extra.get("instruction")
                                and not extra.get("move_ops")
                                and not extra.get("region")
                                and not extra.get("reset")
                                and not self.commit_q
                                and hasattr(self.zimage, "render_looped"))
                    if use_loop:
                        lp = dict(extra)
                        instr = lp.pop("instruction")
                        lp.pop("move_ops"); lp.pop("region")
                        lp.pop("reset")
                        cam = lp.pop("camera")
                        jpeg = await loop.run_in_executor(
                            None, lambda: self.zimage.render_looped(
                                prompt, strength, entities, instr,
                                draft.get("requirements") or [],
                                on_preview=on_preview, camera=cam,
                                draft_size=int(draft.get("draft_size")
                                               or 0),
                                preserve_last=bool(
                                    draft.get("preserve_last")),
                                color_guard=bool(
                                    draft.get("color_guard")),
                                plan=draft,
                                should_abort=lambda: bool(self.commit_q),
                                source=source, **lp))
                    else:
                        jpeg = await loop.run_in_executor(
                            None, lambda: self.zimage.render(
                                prompt, strength, entities,
                                on_preview=on_preview,
                                cancel_check=cancel,
                                source=source, **extra))
                    # POST-VERIFY: pixel-mechanical committed edits
                    # (removals, moves, region edits) are judged on their
                    # RESULT vs the pre-edit keyframe; a fail earns one
                    # restore-and-retry with a fresh seed. Every committed
                    # edit passes the LLM one way or the other.
                    if (committed
                            and (draft.get("policy") == "verify"
                                 or draft.get("class") == "add")
                            and draft.get("requirements")
                            and self.zimage is not None
                            and hasattr(self.zimage, "judge_final")
                            and target.get("base_rev") is not None):
                        base_rev = int(target["base_rev"])
                        reqs = draft["requirements"]
                        instr = target.get("edit_instruction")
                        tb = (draft.get("target") or {}).get("b")
                        v = await loop.run_in_executor(
                            None, lambda: self.zimage.judge_final(
                                base_rev, reqs, instr, target_box=tb))
                        if (v is not None and v.get("verdict") == "pass"
                                and strength >= 0.5
                                and hasattr(self.zimage, "rebase_anchor")):
                            # the old appearance is gone for good —
                            # forget it, or later merges resurrect it
                            await loop.run_in_executor(
                                None, self.zimage.rebase_anchor)
                        if v is not None and v.get("verdict") == "fail":
                            log.info("post-verify FAIL (%s) — restore + "
                                     "retry with fresh seed",
                                     str(v.get("what_failed", ""))[:80])
                            restored = await loop.run_in_executor(
                                None,
                                lambda: self.zimage.restore(base_rev))
                            if restored is not None:
                                # scissors failed verification on a
                                # DELETE: architecture (wall niches,
                                # built-ins) isn't an object — cut+fill
                                # paints back lookalike wall. The retry
                                # switches to instruction repaint.
                                extra2 = dict(extra)
                                strength2 = strength
                                dels = [op for op in
                                        (extra.get("move_ops") or [])
                                        if op.get("delete")]
                                if dels:
                                    names = ", ".join(
                                        str(op.get("name") or "it")
                                        for op in dels)
                                    unfound = list(getattr(
                                        self.zimage, "_unfound_deletes",
                                        None) or [])
                                    if unfound:
                                        # structural features SAM cannot
                                        # cut (cuffs, hems): re-describe
                                        # from scene text, no composite
                                        extra2["move_ops"] = None
                                        extra2["instruction"] = (
                                            f"Redraw so the image "
                                            f"exactly matches this "
                                            f"description, with no "
                                            f"{names} anywhere: "
                                            f"{prompt}. Adjust the "
                                            f"affected areas as needed.")
                                        strength2 = max(strength, 0.6)
                                        log.info("post-verify retry via "
                                                 "repaint (unfound "
                                                 "deletes)")
                                    else:
                                        # SAM found and cut the objects:
                                        # KEEP the composite ops on the
                                        # retry — dropping them once
                                        # cleared the delete pins and
                                        # redrew a furnished room from
                                        # the stale scene text
                                        extra2["instruction"] = None
                                        strength2 = min(
                                            max(strength, 0.5), 0.65)
                                        log.info("post-verify retry: "
                                                 "re-cut composite, "
                                                 "fresh seed, pinned")
                                    extra2["fresh_seed"] = True
                                elif any((not op.get("delete"))
                                         and op.get("to_box")
                                         for op in (extra.get("move_ops")
                                                    or [])):
                                    # POSITIONAL move failed verify
                                    # (e.g. "seated implausibly"):
                                    # re-pasting the same sprite at the
                                    # same coordinates reproduces the
                                    # same failure deterministically.
                                    # Geometry gets one shot — the
                                    # retry RESTAGES: remove the object
                                    # and let the model paint it at the
                                    # described position plausibly.
                                    mvs = [op for op in
                                           extra["move_ops"]
                                           if not op.get("delete")
                                           and op.get("to_box")
                                           and not op.get("insert")]
                                    if not mvs:
                                        # INSERT failed verify: never
                                        # prose-paint the object (the
                                        # model paints it onto tables)
                                        # — re-composite, next solver
                                        # slot
                                        setattr(self.zimage,
                                                "_place_attempt", 1)
                                        extra2["instruction"] = (
                                            extra.get("instruction"))
                                        strength2 = strength
                                        log.info("post-verify retry: "
                                                 "re-composite insert, "
                                                 "next slot")
                                    else:
                                        dels2 = [op for op in
                                                 extra["move_ops"]
                                                 if op.get("delete")]
                                        conv = []
                                        phrases = []
                                        for op in mvs:
                                            conv.append({
                                                "id": op.get("id"),
                                                "name": op.get("name"),
                                                "delete": True,
                                                "all": False,
                                                # regen: no delete stencil —
                                                # the model must be free to
                                                # paint at the DESTINATION,
                                                # not just inside the old
                                                # object's hole
                                                "regen": True,
                                                "from_box":
                                                    op.get("from_box")})
                                            tb = op.get("to_box") or {}
                                            x = float(tb.get("x", 0.5))
                                            y = float(tb.get("y", 0.5))
                                            px = ("left" if x < 0.36 else
                                                  "right" if x > 0.64
                                                  else "center")
                                            py = ("upper " if y < 0.36 else
                                                  "lower " if y > 0.64
                                                  else "")
                                            nm = str(op.get("name")
                                                     or "object")
                                            phrases.append(
                                                f"a {nm} stands at the "
                                                f"{py}{px} of the scene, "
                                                f"naturally placed against "
                                                f"the nearest wall or "
                                                f"clear floor, not "
                                                f"overlapping anything")
                                        frees = [{"id": (op.get("id")
                                                         or "obj") + "_dest",
                                                  "free_box":
                                                      op.get("to_box")}
                                                 for op in mvs
                                                 if op.get("to_box")]
                                        extra2["move_ops"] = (dels2 + conv
                                                              + frees)
                                        extra2["instruction"] = (
                                            "Add exactly these, plausibly "
                                            "integrated: "
                                            + "; ".join(phrases)
                                            + ". Keep everything else "
                                            "exactly the same.")
                                        # the retry is judged against ITS
                                        # OWN contract — box-derived
                                        # positional reqs once demanded
                                        # "center" while the instruction
                                        # said "left" and the judge failed
                                        # a correct result
                                        reqs = [
                                            "The requested change is "
                                            "clearly applied: "
                                            + "; ".join(phrases),
                                            "Every moved object is seated "
                                            "plausibly (on the floor, "
                                            "correct perspective, not "
                                            "overlapping or blocking "
                                            "other objects)",
                                            "Apart from those changes, "
                                            "everything else in the scene "
                                            "is unchanged"]
                                        strength2 = max(strength, 0.55)
                                        log.info("post-verify retry via "
                                                 "RESTAGE placement "
                                                 "(%d moved objects)",
                                                 len(conv))
                                else:
                                    # a verified failure means the
                                    # recipe was too weak — climb the
                                    # ladder (looser suffix, more
                                    # strength), never just re-roll the
                                    # seed (live failure: 'add a TV'
                                    # failed twice identically). Region
                                    # edits stay safe: the pixel splice
                                    # guarantees scope regardless of
                                    # strength.
                                    i2 = extra.get("instruction") or ""
                                    i2 = i2.replace(
                                        "Keep everything else exactly "
                                        "the same",
                                        "Adjust the surrounding area "
                                        "as needed to make this change "
                                        "clearly visible")
                                    if i2:
                                        extra2["instruction"] = i2
                                    strength2 = min(
                                        0.8, max(strength + 0.2, 0.55))
                                    log.info("post-verify retry "
                                             "ESCALATED (strength "
                                             "%.2f)", strength2)
                                extra2.setdefault("fresh_seed", True)
                                if (extra2.get("instruction")
                                        and hasattr(self.zimage,
                                                    "render_looped")):
                                    # retries go through the DRAFT-
                                    # VERIFIED loop: up to 3 judged
                                    # attempts instead of one blind
                                    # shot (restaged placements once
                                    # shipped floating furniture on
                                    # their single roll)
                                    lp2 = {k: v for k, v in
                                           extra2.items()
                                           if k != "instruction"}
                                    lp2.pop("camera", None)
                                    jpeg = await loop.run_in_executor(
                                        None, lambda:
                                        self.zimage.render_looped(
                                            prompt, strength2,
                                            entities,
                                            extra2["instruction"],
                                            reqs,
                                            on_preview=on_preview,
                                            camera=bool(
                                                extra.get("camera")),
                                            plan={"strategy": "atelier",
                                                  "budget_ms": 12000,
                                                  "max_iters": 3,
                                                  "n_drafts": 2},
                                            **lp2))
                                else:
                                    jpeg = await loop.run_in_executor(
                                        None, lambda: self.zimage.render(
                                            prompt, strength2, entities,
                                            on_preview=on_preview,
                                            cancel_check=cancel,
                                            source=source, **extra2))
                                v2 = await loop.run_in_executor(
                                    None,
                                    lambda: self.zimage.judge_final(
                                        base_rev, reqs, instr,
                                        target_box=tb))
                                setattr(self.zimage,
                                        "_place_attempt", 0)
                                log.info(
                                    "post-verify retry: %s",
                                    (v2 or {}).get("verdict", "unjudged"))
                                if ((v2 or {}).get("verdict") == "fail"
                                        and any(op.get("to_box")
                                                and not op.get("delete")
                                                for op in
                                                (extra.get("move_ops")
                                                 or []))):
                                    # SECOND diversified attempt for
                                    # placements before settling
                                    setattr(self.zimage,
                                            "_place_attempt", 2)
                                    jpeg = await loop.run_in_executor(
                                        None, lambda:
                                        self.zimage.render(
                                            prompt, strength, entities,
                                            on_preview=on_preview,
                                            cancel_check=cancel,
                                            source=source, **extra))
                                    setattr(self.zimage,
                                            "_place_attempt", 0)
                                    v2 = await loop.run_in_executor(
                                        None,
                                        lambda: self.zimage.judge_final(
                                            base_rev, reqs, instr,
                                            target_box=tb))
                                    log.info("post-verify attempt3: %s",
                                             (v2 or {}).get("verdict",
                                                            "unjudged"))
                                if ((v2 or {}).get("verdict") == "fail"
                                        and extra2.get("move_ops")
                                        != extra.get("move_ops")
                                        and any(op.get("to_box")
                                                and not op.get("delete")
                                                for op in
                                                (extra.get("move_ops")
                                                 or []))):
                                    # the restage ALSO failed — never
                                    # end with the object missing: the
                                    # first attempt (moved but
                                    # imperfect) is strictly better
                                    # than an empty room, and the user
                                    # can nudge from there
                                    log.info("restage failed too — "
                                             "keeping the geometric "
                                             "attempt (imperfect beats "
                                             "missing)")
                                    restored2 = await loop.run_in_executor(
                                        None, lambda:
                                        self.zimage.restore(base_rev))
                                    if restored2 is not None:
                                        jpeg = await loop.run_in_executor(
                                            None, lambda:
                                            self.zimage.render(
                                                prompt, strength,
                                                entities,
                                                on_preview=on_preview,
                                                cancel_check=cancel,
                                                source=source, **extra))
                                if (dels and v2 is not None
                                        and v2.get("verdict") == "fail"):
                                    # LAST RESORT for structural
                                    # features the conditioning keeps
                                    # regenerating (hoodie cuffs came
                                    # back at 100% overlap twice): one
                                    # full-redraw anchored purely to
                                    # the already-updated scene text.
                                    log.info("post-verify: full-redraw "
                                             "escalation from scene "
                                             "description")
                                    extra3 = dict(extra2)
                                    extra3["instruction"] = (
                                        f"Redraw the whole image as "
                                        f"needed so it exactly matches "
                                        f"this description, with no "
                                        f"{names} anywhere: {prompt}.")
                                    extra3["fresh_seed"] = True
                                    jpeg = await loop.run_in_executor(
                                        None,
                                        lambda: self.zimage.render(
                                            prompt, 0.85, entities,
                                            on_preview=on_preview,
                                            cancel_check=cancel,
                                            source=source, **extra3))
                                    v3 = await loop.run_in_executor(
                                        None,
                                        lambda: self.zimage.judge_final(
                                            base_rev, reqs, instr,
                                            target_box=tb))
                                    log.info(
                                        "post-verify escalation: %s",
                                        (v3 or {}).get("verdict",
                                                       "unjudged"))
                        elif v is not None:
                            log.info("post-verify PASS")
                    frames = [jpeg]
                except RenderPreempted:
                    log.info("render [%s] preempted by newer target", source)
                    continue
                except Exception:
                    log.exception("zimage render failed; mock fallback")
                    frames = self.mock.frames_for_target(prompt, strength,
                                                         entities=entities)
            else:
                frames = self.mock.frames_for_target(prompt, strength,
                                                     entities=entities)
            dt = time.monotonic() - t0
            log.info("render [%s] %.1fs strength=%.2f %r",
                     source, dt, strength, prompt[:60])

            tier = ((target.get("draft") or {}).get("strategy")
                    or "standard") if committed else None
            for jpeg_b64 in frames:
                extras = {"lane": ("committed" if committed
                                   else "provisional")}
                if tier:
                    extras["tier"] = tier
                self.out_q.put_nowait(self._frame_msg(
                    jpeg_b64, source, strength, render_s=round(dt, 1),
                    **extras))
                if len(frames) > 1:
                    await asyncio.sleep(MOCK_FRAME_INTERVAL_S)
            if committed:
                await self._maybe_calibrate(loop)
                await self._maybe_embed(loop)

    async def _switch_engine(self, loop, name: str) -> None:
        """Swap the diffusion engine under the session: the pixel canvas,
        revision history, and keyframes carry over; latents never do (each
        engine re-roots its own chain from pixels). Old weights are freed
        BEFORE the new engine loads so both never share VRAM."""
        from .engines import free_other_sidecars, make_canvas
        log.info("engine switch %s -> %s", self.engine, name)
        self.out_q.put_nowait(json.dumps(
            {"type": "engine_status",
             "data": {"engine": name, "state": "loading"}}))
        try:
            new = make_canvas(name)
        except Exception:
            log.exception("engine %r unavailable — keeping %s", name,
                          self.engine)
            self.out_q.put_nowait(json.dumps(
                {"type": "engine_status",
                 "data": {"engine": self.engine, "state": "ready",
                          "error": f"{name} unavailable"}}))
            return
        old = self.zimage
        # carry the SESSION across engines (pixels only, never latents)
        new.canvas = old.canvas
        new.rev = old.rev
        new.edits_since_root = old.edits_since_root
        new.keyframes = old.keyframes
        new.size = old.size
        try:
            await loop.run_in_executor(None, old.unload)
            # free ANY other engine that still holds weights before we load
            # the target — never let two engines' weights coexist in VRAM
            await loop.run_in_executor(None, free_other_sidecars, name)
            await loop.run_in_executor(None, new.load)
        except Exception:
            log.exception("engine %s load failed — reverting to %s",
                          name, self.engine)
            try:
                await loop.run_in_executor(None, new.unload)
                await loop.run_in_executor(None, old.load)
            except Exception:
                log.exception("engine revert load failed")
            # old.unload() cleared its session state; restore the carry
            old.canvas, old.rev = new.canvas, new.rev
            old.edits_since_root = new.edits_since_root
            old.keyframes, old.size = new.keyframes, new.size
            self.out_q.put_nowait(json.dumps(
                {"type": "engine_status",
                 "data": {"engine": self.engine, "state": "ready",
                          "error": f"{name} failed to load"}}))
            return
        self.zimage = new
        self.engine = name
        self.out_q.put_nowait(json.dumps(
            {"type": "engine_status",
             "data": {"engine": name, "state": "ready"}}))
        log.info("engine %s active", name)

    async def _heartbeat(self, ws) -> None:
        """Touch the health file while the feed link is ALIVE — the feed
        reader only touches it on traffic, so an idle stack (no session)
        went 'unhealthy' after 120s despite being fully operational.
        A ws ping proves liveness; a dead link stops the beat and lets
        the healthcheck fail for real."""
        while True:
            try:
                pong = await ws.ping()
                await asyncio.wait_for(pong, timeout=10)
            except Exception:
                return
            HEARTBEAT.write_text(str(time.time()))
            await asyncio.sleep(30)

    async def run_once(self) -> None:
        async with websockets.connect(FEED_URL, max_size=None) as ws:
            log.info("connected to feed %s (mode=%s)", FEED_URL, self.mode)
            # a (re)connection may have missed session_start events — any
            # retained canvas (warmup, previous session) would silently
            # become the edit base for the next session's targets. Every
            # connect starts blank; the first commit roots fresh.
            self.mock.__init__()
            if self.zimage:
                self.zimage.reset_session()
            self.pending = None
            self.commit_q.clear()
            self.out_q = asyncio.Queue()  # drop frames aimed at a dead socket
            # announce the active engine so the orchestrator's GET /engine
            # is truthful across either side's restart
            self.out_q.put_nowait(json.dumps(
                {"type": "engine_status",
                 "data": {"engine": self.engine, "state": "ready"}}))
            reader = asyncio.create_task(self.read_feed(ws))
            worker = asyncio.create_task(self.render_worker(ws))
            sender = asyncio.create_task(self.send_worker(ws))
            beat = asyncio.create_task(self._heartbeat(ws))
            done, pending = await asyncio.wait(
                {reader, worker, sender, beat},
                return_when=asyncio.FIRST_EXCEPTION)
            for t in pending:
                t.cancel()
            for t in done:
                exc = t.exception()
                if exc:
                    raise exc


async def main() -> None:
    HEARTBEAT.write_text(str(time.time()))  # alive during the long model load
    app = RendererApp()
    while True:
        HEARTBEAT.write_text(str(time.time()))
        try:
            await app.run_once()
        except (OSError, websockets.WebSocketException) as e:
            log.info("feed unavailable (%s); retrying in 5s", e)
        except Exception:
            log.exception("renderer loop crashed; retrying in 5s")
        await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(main())
