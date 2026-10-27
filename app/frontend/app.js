// Psirens voice intent engine — single-page UI.
// One duplex websocket: binary = audio up, JSON text = events down (and
// control messages up).

const els = {
  micBtn: document.getElementById("micBtn"),
  waveform: document.getElementById("waveform"),
  connChip: document.getElementById("connChip"),
  sessionChip: document.getElementById("sessionChip"),
  dropChip: document.getElementById("dropChip"),
  calibChip: document.getElementById("calibChip"),
  levelFill: document.getElementById("levelFill"),
  levelVal: document.getElementById("levelVal"),
  eventLog: document.getElementById("eventLog"),
  eventCount: document.getElementById("eventCount"),
  jsonlLink: document.getElementById("jsonlLink"),
  transcriptPanel: document.getElementById("transcriptPanel"),
};

const MAX_BUFFERED = 12288; // ~250ms of 24kHz int16 mono
let ws = null;
let audioCtx = null;
let workletNode = null;
let mediaStream = null;
let live = false;
let paused = false;
let dropped = 0;
let eventCount = 0;
let sessionId = null;

// ---- loop status chips ----
function setLoopChip(id, text) {
  const chip = document.getElementById(id);
  if (!chip) return;   // chip retired (e.g. draft loop)
  chip.textContent = text;
  chip.classList.remove("flash");
  void chip.offsetWidth;
  chip.classList.add("flash", "active");
  clearTimeout(chip._t);
  chip._t = setTimeout(() => chip.classList.remove("flash"), 1500);
}

// ---- waveform ----
const wctx = els.waveform.getContext("2d");
const wavePeaks = new Array(120).fill(0);
function drawWave() {
  const { width: W, height: H } = els.waveform;
  wctx.clearRect(0, 0, W, H);
  wctx.fillStyle = "#52b7ff";
  const bw = W / wavePeaks.length;
  for (let i = 0; i < wavePeaks.length; i++) {
    const h = Math.max(1, wavePeaks[i] * (H - 4));
    wctx.fillRect(i * bw, (H - h) / 2, Math.max(1, bw - 1), h);
  }
}
drawWave();

// ---- event handling ----
const handlers = {
  session_start(ev) {
    sessionId = ev.data.session_id;
    els.sessionChip.textContent = sessionId;
    els.sessionChip.className = "chip on";
    els.jsonlLink.hidden = false;
    els.jsonlLink.href = `/sessions/${sessionId}/events.jsonl`;
  },
  audio_level(ev) {
    const dbfs = ev.data.dbfs;
    const pct = Math.max(0, Math.min(100, (dbfs + 60) / 60 * 100));
    els.levelFill.style.width = pct + "%";
    els.levelVal.textContent = dbfs.toFixed(0) + " dB";
  },
  word(ev) {
    const s = document.getElementById("finalWords");
    s.textContent += (s.textContent ? " " : "") + ev.data.w;
    s.parentElement.classList.remove("holding");
    s.parentElement.title = "";
    s.parentElement.parentElement.scrollTop = 1e9;
  },
  partial_transcript(ev) {
    document.getElementById("partialLine").textContent = ev.data.pending || "";
    if (ev.data.pending) {
      const line = document.getElementById("finalWords").parentElement;
      line.classList.remove("holding");
      line.title = "";
    }
  },
  commit(ev) {
    const d = ev.data;
    const block = document.createElement("div");
    block.className = "committed";
    const tone = d.tone && d.tone.energy
      ? ` <span class="tone-chips">[${d.tone.energy}·${d.tone.valence}·${d.tone.tempo}·${d.tone.hesitation}]</span>` : "";
    const repaired = d.repairs && d.repairs.length
      ? ` <span class="repair-note" title="raw: ${d.utterance_raw}">✂ repaired</span>` : "";
    block.innerHTML = `${d.utterance_clean}${tone}${repaired}`;
    block.title = (d.reasons || []).join("  ");
    document.getElementById("committedBlocks").appendChild(block);
    const s = document.getElementById("finalWords");
    s.textContent = "";
    const line = s.parentElement;
    line.classList.remove("holding");
    line.title = "";
    // "heard you" pulse
    const panel = els.transcriptPanel;
    panel.classList.remove("pulse");
    void panel.offsetWidth;
    panel.classList.add("pulse");
    panel.querySelector(".scroll").scrollTop = 1e9;
  },
  hold(ev) {
    const line = document.getElementById("finalWords").parentElement;
    if (ev.data.reason && ev.data.reason.startsWith("discard_")) {
      document.getElementById("finalWords").textContent = "";
      line.classList.remove("holding");
      return;
    }
    line.classList.add("holding");
    line.title = "hold: " + ev.data.reason;
  },
  vad_state(ev) {
    const p = ev.data.p_eot;
    document.getElementById("eotFill").style.width = (p * 100).toFixed(0) + "%";
    document.getElementById("eotFill").style.background = p > 0.6 ? "#4ade80" : "#52b7ff";
    document.getElementById("eotVal").textContent = p.toFixed(2);
    // "listening" shimmer — the canvas glows while you speak, so a still
    // image never reads as a stalled system
    const wrap = document.querySelector(".canvas-wrap");
    if (wrap) wrap.classList.toggle("listening", !!ev.data.speaking);
  },
  prosody_frame(ev) {
    const d = ev.data;
    const set = (id, pct, label, bucket) => {
      const fill = document.getElementById("m_" + id);
      fill.style.width = Math.max(0, Math.min(100, pct)) + "%";
      fill.style.background =
        bucket === "high" ? "#fbbf24" : bucket === "low" ? "#7b8794" : "#52b7ff";
      document.getElementById("b_" + id).textContent = label;
    };
    set("energy", (d.energy.raw_dbfs + 60) / 60 * 100, d.energy.bucket, d.energy.bucket);
    set("valence", (d.valence.raw ?? 0.5) * 100,
        d.valence.raw == null ? "–" : d.valence.bucket, d.valence.bucket);
    set("arousal", (d.arousal.raw ?? 0.5) * 100,
        d.arousal.raw == null ? "–" : d.arousal.bucket, d.arousal.bucket);
    set("tempo", d.tempo.wps / 5 * 100, d.tempo.wps.toFixed(1) + " w/s", d.tempo.bucket);
    set("hesitation", d.hesitation.score * 100, d.hesitation.bucket, d.hesitation.bucket);
    const hz = d.pitch.median_hz;
    set("pitch", hz == null ? 0 : (hz - 50) / 300 * 100,
        hz == null ? "–" : hz.toFixed(0) + " Hz", "med");
    els.calibChip.hidden = !d.calibrating;
  },
  patch_applied(ev) {
    const d = ev.data;
    const isReflex = (d.reasoning || "").startsWith("reflex") || d.latency_ms === 0;
    setLoopChip("commitChip",
      `✔ commit #${d.commit_id}` +
      (d.edit_magnitude != null ? ` · mag ${d.edit_magnitude.toFixed(1)}` : "") +
      (isReflex ? " · ⚡0ms" : ` · ${d.latency_ms}ms`));
    // timeline entry
    const item = document.createElement("div");
    item.className = "tl-item" + (d.kind === "revert" ? " tl-revert" : "");
    const mag = d.edit_magnitude == null ? "" :
      ` <span class="tl-mag">${d.edit_magnitude.toFixed(1)}</span>`;
    item.innerHTML =
      `<span class="tl-id">#${d.commit_id}</span> ${d.utterance}${mag}
       <button class="tl-revert-btn" title="revert to this state">↩</button>
       <div class="tl-detail" hidden></div>`;
    item.querySelector(".tl-detail").textContent = JSON.stringify(
      { patch: d.patch, reasoning: d.reasoning, tone: d.tone,
        latency_ms: d.latency_ms, retries: d.retries }, null, 1);
    item.addEventListener("click", (e) => {
      if (e.target.classList.contains("tl-revert-btn")) return;
      const det = item.querySelector(".tl-detail");
      det.hidden = !det.hidden;
    });
    item.querySelector(".tl-revert-btn").addEventListener("click", () => {
      if (ws && ws.readyState === WebSocket.OPEN)
        ws.send(JSON.stringify({ type: "revert", data: { commit_id: d.commit_id } }));
    });
    const tl = document.getElementById("timeline");
    tl.appendChild(item);
    tl.scrollTop = 1e9;
    // magnitude meter
    if (d.edit_magnitude != null) {
      document.getElementById("magRow").hidden = false;
      document.getElementById("magFill").style.width = d.edit_magnitude * 100 + "%";
      document.getElementById("magVal").textContent = d.edit_magnitude.toFixed(2);
    }
  },
  canvas_frame(ev) {
    // SCRIM: provisional frames are sketches — they live in the corner
    // PiP, never on the main canvas
    if (mode === "scrim" && ev.data.lane === "provisional") {
      const pip = document.getElementById("sketchPip");
      document.getElementById("sketchImg").src =
        "data:image/jpeg;base64," + ev.data.jpeg_b64;
      pip.hidden = false;
      return;
    }
    if (ev.data.lane === "committed") {
      document.getElementById("sketchPip").hidden = true;
      if (ml.sels.length) { ml.sels = []; paint(); }
    }
    const img = document.getElementById("canvasImg");
    img.src = "data:image/jpeg;base64," + ev.data.jpeg_b64;
    img.hidden = false;
    img.classList.toggle("developing", !!ev.data.preview);
    document.getElementById("canvasEmpty").hidden = true;
    document.getElementById("canvasMeta").textContent = ev.data.preview
      ? `${ev.data.source} · developing… ${ev.data.step}/${ev.data.steps}`
      : `${ev.data.source} · strength ${(ev.data.strength ?? 0).toFixed(2)} · #${ev.data.seq}` +
        (ev.data.render_s != null ? ` · ${ev.data.render_s}s render` : "");
  },
  state_snapshot(ev) {
    renderScene(ev.data.state, []);
  },
  llm_call(ev) {
    const d = ev.data;
    document.getElementById("llmChip").textContent =
      `🧠 ${d.n} call${d.n === 1 ? "" : "s"} · ${d.purpose}` +
      (d.vision ? " 👁" : "") +
      (d.latency_ms ? ` · ${(d.latency_ms / 1000).toFixed(1)}s` : "");
  },
  draft_patch(ev) {
    renderScene(ev.data.state, ev.data.provisional_fields);
    setLoopChip("draftChip", `✏ draft · sketched · ${ev.data.latency_ms}ms`);
  },
  reflex_action(ev) {
    renderScene(ev.data.state, ev.data.provisional_fields);
    setLoopChip("reflexChip",
      `⚡ ${ev.data.kind} · ${ev.data.detail} · ${ev.data.latency_ms}ms`);
  },
  prompt_updated(ev) {
    const box = document.getElementById("promptBox");
    box.textContent = ev.data.prompt || "— empty scene —";
    box.classList.toggle("draft", !!ev.data.provisional);
  },
  error(ev) {
    if (ev.data.where === "editor") {
      const tl = document.getElementById("timeline");
      const item = document.createElement("div");
      item.className = "tl-item tl-failed";
      item.textContent = "✗ " + (ev.data.directive || "") + " — " + ev.data.message;
      tl.appendChild(item);
      tl.scrollTop = 1e9;
    }
    if (ev.data.where === "audio_queue") {
      els.dropChip.hidden = false;
      els.dropChip.className = "chip warn";
      els.dropChip.textContent = "server drops: " + ev.data.dropped_total;
    }
  },
  session_end() {
    els.sessionChip.className = "chip off";
  },
};

function onEvent(ev) {
  eventCount++;
  els.eventCount.textContent = eventCount;
  if (document.getElementById("eventDrawer").open) {
    els.eventLog.textContent =
      (JSON.stringify(ev) + "\n" + els.eventLog.textContent).slice(0, 40000);
  }
  const h = handlers[ev.type];
  if (h) h(ev);
}

// ---- scene state panel ----
const SCENE_FIELDS = ["subject", "setting", "composition", "lighting", "palette", "mood", "medium"];
let prevScene = {};
function renderScene(state, provisionalFields) {
  const prov = new Set(provisionalFields || []);
  const host = document.getElementById("sceneFields");
  if (!host.dataset.built) {
    host.dataset.built = "1";
    host.innerHTML = SCENE_FIELDS.map((f) =>
      `<div class="scene-row" id="sf_${f}"><label>${f}</label><span class="scene-val"></span></div>`
    ).join("");
  }
  for (const f of SCENE_FIELDS) {
    const row = document.getElementById("sf_" + f);
    const span = row.querySelector(".scene-val");
    const val = state[f] || "";
    if (val !== (prevScene[f] || "")) {
      span.innerHTML = prevScene[f]
        ? `<s class="old-val">${prevScene[f]}</s> ${val || "∅"}`
        : (val || "");
      row.classList.remove("flash");
      void row.offsetWidth;
      row.classList.add("flash");
      const captured = val;
      setTimeout(() => { if ((prevScene[f] || "") === captured) span.textContent = captured; }, 4000);
    } else if (!span.innerHTML) {
      span.textContent = val;
    }
    row.classList.toggle("empty", !val);
    row.classList.toggle("provisional", prov.has(f));
  }
  // entity rows
  let ehost = document.getElementById("entityRows");
  if (!ehost) {
    ehost = document.createElement("div");
    ehost.id = "entityRows";
    host.appendChild(ehost);
  }
  const pos = (b) => {
    const col = b.x < 0.34 ? "left" : b.x > 0.66 ? "right" : "center";
    const row = b.y < 0.34 ? "top" : b.y > 0.66 ? "bottom" : "middle";
    return `${row}-${col}`;
  };
  ehost.innerHTML = (state.entities || []).map((e) =>
    `<div class="scene-row entity-row ${prov.has("entity:" + e.id) ? "provisional" : ""}">
       <label>◳ ${e.name}</label>
       <span class="scene-val">${e.description || ""} · ${pos(e.box)}
         <span class="ent-coords">(${e.box.x.toFixed(2)}, ${e.box.y.toFixed(2)}, ${(e.box.w * 100).toFixed(0)}×${(e.box.h * 100).toFixed(0)}%)</span></span>
     </div>`).join("");
  prevScene = { ...state };
}
renderScene({});

// ---- architecture mode tabs ----
const MODE_NAMES = ["scrim"];
let mode = "scrim";   // the shipped architecture

function paintTabs() {
  document.querySelectorAll(".mode-tab").forEach((b) =>
    b.classList.toggle("active", b.dataset.mode === mode));
}
document.querySelectorAll(".mode-tab").forEach((b) =>
  b.addEventListener("click", async () => {
    if (b.dataset.mode === mode) return;
    mode = b.dataset.mode;
    localStorage.setItem("psirens_mode", mode);
    paintTabs();
    document.getElementById("llmChip").textContent = "🧠 0 calls";
    if (live) { stopMic(); await startMic(); }  // fresh session in new mode
  }));
paintTabs();

// ---- render resolution ----
let renderRes = parseInt(localStorage.getItem("psirens_res") || "768", 10);
if (![512, 768, 1024].includes(renderRes)) renderRes = 768;
function paintRes() {
  // scope to [data-res]: model/gpu buttons share the .res-btn class for
  // styling only and must keep their own active state
  document.querySelectorAll(".res-btn[data-res]").forEach((b) =>
    b.classList.toggle("active", parseInt(b.dataset.res, 10) === renderRes));
}
function sendRes() {
  if (![512, 768, 1024].includes(renderRes)) renderRes = 768;
  if (ws && ws.readyState === WebSocket.OPEN)
    ws.send(JSON.stringify({ type: "set_resolution", data: { size: renderRes } }));
}
document.querySelectorAll(".res-btn[data-res]").forEach((b) =>
  b.addEventListener("click", () => {
    renderRes = parseInt(b.dataset.res, 10);
    localStorage.setItem("psirens_res", String(renderRes));
    paintRes();
    sendRes();   // applies live: next render uses the new size
  }));
paintRes();

// ---- render model (fast 4B / quality 9B) ----
let renderModel = localStorage.getItem("psirens_model") || "fast";
if (!["fast", "hq"].includes(renderModel)) renderModel = "fast";
function paintModel() {
  document.querySelectorAll(".model-btn").forEach((b) =>
    b.classList.toggle("active", b.dataset.model === renderModel));
}
function sendModel() {
  if (ws && ws.readyState === WebSocket.OPEN)
    ws.send(JSON.stringify({ type: "set_model", data: { model: renderModel } }));
}
document.querySelectorAll(".model-btn").forEach((b) =>
  b.addEventListener("click", () => {
    renderModel = b.dataset.model;
    localStorage.setItem("psirens_model", renderModel);
    paintModel();
    sendModel();   // applies live: next render uses the chosen model
  }));
paintModel();

// ---- editing engine (klein / u1.5 / h3) ----
// Selecting an engine hot-swaps the diffusion weights on the renderer:
// the old engine's VRAM is freed, the new one loads, the canvas and undo
// history carry over. Engine choice is RENDERER state, not session state,
// so it goes over HTTP (GET/POST /engine) and works with the mic off; the
// engine_status ws event is just the fast path while a session is live.
let renderEngine = "klein";
let engineLoading = null;
let enginePollTimer = null;
function paintEngine() {
  document.querySelectorAll(".engine-btn").forEach((b) => {
    b.classList.toggle("active", b.dataset.engine === renderEngine);
    b.classList.toggle("busy", b.dataset.engine === engineLoading);
    if (b.dataset.engine === engineLoading) b.textContent = "loading…";
    else b.textContent = { klein: "klein", u15: "u1.5", h3: "h3" }[b.dataset.engine];
  });
}
function applyEngineStatus(d) {
  if (!d || !d.engine) return;
  renderEngine = d.engine;
  engineLoading = d.state === "loading" ? d.engine : null;
  if (d.error) console.warn("engine:", d.error);
  paintEngine();
  if (engineLoading && !enginePollTimer) {
    enginePollTimer = setInterval(refreshEngine, 3000);
  } else if (!engineLoading && enginePollTimer) {
    clearInterval(enginePollTimer);
    enginePollTimer = null;
  }
}
async function refreshEngine() {
  try {
    applyEngineStatus(await (await fetch("/engine")).json());
  } catch (e) { /* orchestrator restarting; next poll or click recovers */ }
}
document.querySelectorAll(".engine-btn").forEach((b) =>
  b.addEventListener("click", async () => {
    if (engineLoading || b.dataset.engine === renderEngine) return;
    engineLoading = b.dataset.engine;
    paintEngine();
    try {
      const r = await fetch("/engine", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ engine: b.dataset.engine }),
      });
      applyEngineStatus(await r.json());
      if (!r.ok) refreshEngine();
    } catch (e) {
      console.warn("engine switch failed:", e);
      engineLoading = null;
      refreshEngine();
    }
  }));
handlers.engine_status = (ev) => applyEngineStatus(ev.data || {});
refreshEngine();

// ---- free / load GPU ----
// "free gpu" unloads every model (STT/SER in the orchestrator, FLUX+SAM in
// the renderer, vLLM to sleep mode) so the GPU is free for other work.
// "load gpu" reloads all of it; starting the mic while freed also reloads
// automatically.
const gpuFreeBtn = document.getElementById("gpuFreeBtn");
const gpuLoadBtn = document.getElementById("gpuLoadBtn");
let gpuFreed = false;
function paintGpu(busyText) {
  if (!busyText) {
    gpuFreeBtn.textContent = gpuFreed ? "gpu freed" : "free gpu";
    gpuLoadBtn.textContent = gpuFreed ? "load gpu" : "gpu loaded";
  }
  gpuFreeBtn.classList.toggle("freed", gpuFreed);
  gpuLoadBtn.classList.toggle("active", !gpuFreed);
}
async function refreshGpu() {
  try {
    const j = await (await fetch("/gpu")).json();
    gpuFreed = j.freed;
  } catch {}
  paintGpu();
}
async function gpuLoad() {
  gpuLoadBtn.textContent = "loading models…";
  gpuFreeBtn.disabled = gpuLoadBtn.disabled = true;
  try {
    await fetch("/gpu/load", { method: "POST" });
    for (let i = 0; i < 90; i++) {           // reload can take ~30s
      const j = await (await fetch("/gpu")).json();
      if (!j.freed && j.ready) break;
      await new Promise((r) => setTimeout(r, 2000));
    }
  } finally {
    gpuFreeBtn.disabled = gpuLoadBtn.disabled = false;
    await refreshGpu();
  }
}
async function gpuFree() {
  if (live) stopMic();
  gpuFreeBtn.textContent = "freeing…";
  gpuFreeBtn.disabled = gpuLoadBtn.disabled = true;
  try {
    // session teardown is async server-side; retry through the 409 window
    for (let i = 0; i < 10; i++) {
      const r = await fetch("/gpu/free", { method: "POST" });
      if (r.status !== 409) break;
      await new Promise((res) => setTimeout(res, 400));
    }
  } finally {
    gpuFreeBtn.disabled = gpuLoadBtn.disabled = false;
    await refreshGpu();
  }
}
gpuFreeBtn.addEventListener("click", () => { if (!gpuFreed) gpuFree(); });
gpuLoadBtn.addEventListener("click", () => { if (gpuFreed) gpuLoad(); });
refreshGpu();

// ---- websocket ----
function connect() {
  return new Promise((resolve, reject) => {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws/session?mode=${mode}`);
    ws.binaryType = "arraybuffer";
    ws.onopen = () => {
      els.connChip.textContent = "connected";
      els.connChip.className = "chip on";
      ws.send(JSON.stringify({ type: "client_hello", data: { ua: navigator.userAgent } }));
      sendRes();        // apply the chosen render resolution to this session
      resolve();
    };
    ws.onclose = () => {
      els.connChip.textContent = "disconnected";
      els.connChip.className = "chip off";
      if (live) {
        // unexpected drop (server restart, 1013 session race): retry with
        // backoff instead of leaving a dead mic the user must re-click
        stopMic();
        let attempt = 0;
        const retry = () => {
          if (live || attempt >= 4) return;
          attempt += 1;
          els.connChip.textContent = `reconnecting (${attempt})…`;
          startMic().catch(() => setTimeout(retry, 1500 * attempt));
        };
        setTimeout(retry, 1200);
      }
    };
    ws.onerror = (e) => reject(e);
    ws.onmessage = (m) => {
      if (typeof m.data === "string") onEvent(JSON.parse(m.data));
    };
  });
}

// ---- mic ----
async function startMic() {
  if (!navigator.mediaDevices) {
    // mic API needs a secure context (HTTPS or localhost)
    els.connChip.textContent = "mic blocked: open via HTTPS or an ssh -L localhost tunnel";
    els.connChip.className = "chip err";
    return;
  }
  if (gpuFreed) await gpuLoad();   // freed GPU: reload everything first
  await connect();
  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
  });
  audioCtx = new AudioContext({ sampleRate: 24000 });
  await audioCtx.audioWorklet.addModule("audio-worklet.js");
  const src = audioCtx.createMediaStreamSource(mediaStream);
  workletNode = new AudioWorkletNode(audioCtx, "pcm-capture");
  src.connect(workletNode);

  workletNode.port.onmessage = (m) => {
    if (paused) return;
    wavePeaks.push(m.data.peak);
    wavePeaks.shift();
    drawWave();
    if (ws && ws.readyState === WebSocket.OPEN) {
      if (ws.bufferedAmount > MAX_BUFFERED) {
        dropped++;
        els.dropChip.hidden = false;
        els.dropChip.textContent = "client drops: " + dropped;
        return;
      }
      ws.send(m.data.pcm);
    }
  };

  live = true;
  els.micBtn.textContent = "⏹ stop";
  els.micBtn.classList.add("live");
  paused = false;
  const pb = document.getElementById("pauseBtn");
  pb.hidden = false;
  pb.textContent = "⏸ pause";
  pb.classList.remove("live");
}

function stopMic() {
  live = false;
  if (workletNode) { workletNode.disconnect(); workletNode = null; }
  if (audioCtx) { audioCtx.close(); audioCtx = null; }
  if (mediaStream) { mediaStream.getTracks().forEach((t) => t.stop()); mediaStream = null; }
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "stop" }));
    ws.close();
  }
  els.micBtn.textContent = "🎤 start";
  els.micBtn.classList.remove("live");
  paused = false;
  document.getElementById("pauseBtn").hidden = true;
}

document.getElementById("pauseBtn").onclick = () => {
  if (!live) return;
  paused = !paused;
  const pb = document.getElementById("pauseBtn");
  pb.textContent = paused ? "▶ resume" : "⏸ pause";
  pb.classList.toggle("live", paused);
  // suspend the audio graph for real (no capture while paused) —
  // the session, canvas, and history all stay put
  if (audioCtx) (paused ? audioCtx.suspend() : audioCtx.resume());
  els.connChip.textContent = paused ? "paused" : "connected";
  els.connChip.className = paused ? "chip warn" : "chip on";
};

els.micBtn.onclick = () => (live ? stopMic() : startMic().catch((e) => {
  console.error(e);
  els.connChip.textContent = "error: " + (e.message || e);
  els.connChip.className = "chip err";
}));

// ---- mouse layer: marquee select, click select, drag move, resize ----
// Selection works like a desktop: drag on empty space rubber-bands one or
// many objects; click selects one; shift-click adds; dragging any selected
// object moves the WHOLE selection as one commit and one render. A plain
// click always picks the BIGGEST object under the cursor (the sofa, not
// its pillow); clicking the same spot again DRILLS DOWN through smaller
// objects, extending into on-the-spot SAM candidates when the warm cache
// runs out. Masks come from two tiers — WARM (LLM-enumerated, SAM
// pre-cut, instant) and COLD (point segmentation, ~100-300ms). A drag
// release ships the exact masks back with the move; the commit rides the
// same pipeline as a spoken edit.
const overlay = document.getElementById("overlay");
const octx = overlay.getContext("2d");
const selChip = document.getElementById("selChip");
const canvasImgEl = document.getElementById("canvasImg");
const ACCENT = "#52b7ff";
const HANDLE = 9;                 // half-size of resize handles, mask px

const ml = {
  warm: [],        // [{id,name,rle,size,maskData,mask,tight,box,area}]
  warmSize: null,  // [W,H] of the warm set
  sels: [],        // selected entries (warm or click-time), group-draggable
  drill: null,     // {nx,ny,seq:[entries],idx,coldDone} click-again drilling
  hover: null,
  pending: null,   // outstanding click_segment {reqId,nx,ny}
  drag: null,      // {ents,sprites,mode,startNx,startNy,ndx,ndy,scale,moved}
  ink: null,       // drawing mode {cur:[[nx,ny],...]|null, strokes:[], timer, sent}
  regionSig: "",   // last selection signature sent as set_region
  optimistic: null,
  reqId: 0,
};

function rleDecode(counts, n) {
  const a = new Uint8Array(n);
  let pos = 0, val = 0;
  for (const c of counts) { if (val) a.fill(1, pos, pos + c); pos += c; val ^= 1; }
  return a;
}

function makeEntry(e, source) {
  const [W, H] = e.size;
  const maskData = rleDecode(e.rle, W * H);
  const mask = document.createElement("canvas");
  mask.width = W; mask.height = H;
  const mctx = mask.getContext("2d");
  const img = mctx.createImageData(W, H);
  for (let i = 0; i < maskData.length; i++)
    if (maskData[i]) img.data[i * 4 + 3] = 255;
  mctx.putImageData(img, 0, 0);
  let area = 0;
  for (let i = 1; i < e.rle.length; i += 2) area += e.rle[i];
  return { id: e.id || null, name: e.name || null, rle: e.rle, size: e.size,
           maskData, mask, tight: e.tight_box_px, box: e.box, area, source };
}

function rleEncodeJS(a) {
  // row-major binary RLE, counts alternate 0-runs/1-runs, first is a 0-run
  const counts = [];
  let val = 0, run = 0;
  for (let i = 0; i < a.length; i++) {
    const v = a[i] ? 1 : 0;
    if (v === val) { run++; continue; }
    counts.push(run); val = v; run = 1;
  }
  counts.push(run);
  return counts;
}

function sendRegion() {
  // the current selection IS the voice-edit scope: while something is
  // selected, spoken edits render only inside it (server-enforced)
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  const sig = ml.sels.map((s) => (s.id || "c") + ":" + (s.area || 0))
    .sort().join("|");
  if (sig === ml.regionSig) return;
  ml.regionSig = sig;
  if (!ml.sels.length) {
    ws.send(JSON.stringify({ type: "set_region", data: null }));
    return;
  }
  const [W, H] = ml.sels[0].size;
  const u = new Uint8Array(W * H);
  for (const sel of ml.sels) {
    if (sel.size[0] !== W || sel.size[1] !== H) continue;
    const m = sel.maskData;
    for (let i = 0; i < m.length; i++) if (m[i]) u[i] = 1;
  }
  const name = ml.sels.length === 1
    ? (ml.sels[0].name || "selected area")
    : `${ml.sels.length} selected areas`;
  ws.send(JSON.stringify({ type: "set_region", data: {
    mask_rle: rleEncodeJS(u), mask_size: [W, H], name } }));
}

function paint() { drawOverlay(); updateChip(); sendRegion(); }

handlers.masks = (ev) => {
  const d = ev.data;
  ml.warm = d.entities.map((e) => makeEntry(e, "warm"));
  ml.warmSize = d.size;
  ml.drill = null;
  // re-bind selections to the fresh warm set where the object survived
  ml.sels = ml.sels.map((s) => (s.source === "warm"
    ? (ml.warm.find((x) => x.id === s.id) || s) : s));
  paint();
};

function boxOverlap(a, b) {
  const ix = Math.max(0, Math.min(a.x + a.w / 2, b.x + b.w / 2)
                       - Math.max(a.x - a.w / 2, b.x - b.w / 2));
  const iy = Math.max(0, Math.min(a.y + a.h / 2, b.y + b.h / 2)
                       - Math.max(a.y - a.h / 2, b.y - b.h / 2));
  const inter = ix * iy;
  return inter / Math.max(a.w * a.h + b.w * b.h - inter, 1e-6);
}

handlers.ink_routed = (ev) => {
  const d = ev.data || {};
  const labels = { erase: "✕ erasing", select: "◯ selected",
                   sketch: "✏ painting your sketch",
                   move: "→ moving", multi: "✏🎤 combining your drawings",
                   noop: "" };
  document.getElementById("canvasMeta").textContent =
    (labels[d.action] || d.action || "") +
    (d.target ? ` — ${d.target}` : "");
  // the drawing has been consumed; selection results (if any) arrive
  // separately as a segment_result and highlight normally
  if (ml.ink) { if (ml.ink.timer) clearTimeout(ml.ink.timer); ml.ink = null; }
  drawOverlay();
};

handlers.segment_result = (ev) => {
  const d = ev.data;
  // drawing-mode circles only: the server segments inside the loop and
  // the best candidate becomes the live selection
  if (typeof d.req_id === "string" && d.req_id.startsWith("ink-")) {
    const cands = (d.seg && d.seg.candidates) || [];
    if (cands.length) { ml.sels = [makeEntry(cands[0], "click")]; }
    paint();
  }
};

// ---- image drop = insertion at the drop point (or a style ref) ----
const canvasWrap = document.querySelector(".canvas-wrap");
// a drop that misses the box must never navigate the tab away
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => e.preventDefault());
const setCanvasMeta = (t) => {
  document.getElementById("canvasMeta").textContent = t;
};
function dropPoint(e) {
  // before the first generation the canvas img is hidden (0x0 rect) —
  // fall back to the wrap, then to dead center
  let r = canvasImgEl.hidden ? canvasWrap.getBoundingClientRect()
                             : canvasImgEl.getBoundingClientRect();
  if (!r.width || !r.height) return { nx: 0.5, ny: 0.5 };
  return {
    nx: Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)),
    ny: Math.max(0, Math.min(1, (e.clientY - r.top) / r.height)),
  };
}
canvasWrap.addEventListener("dragover", (e) => e.preventDefault());
canvasWrap.addEventListener("drop", async (e) => {
  e.preventDefault();
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    setCanvasMeta("start the session first, then drop your image");
    return;
  }
  const { nx, ny } = dropPoint(e);
  const files = Array.from(e.dataTransfer.files || [])
    .filter((f) => f.type.startsWith("image/"));
  if ((e.dataTransfer.files || []).length && !files.length) {
    setCanvasMeta("those files aren't images");
    return;
  }
  if (files.length) {
    // spatial intent: ONE image dropped on the picture = insert it
    // there; SEVERAL at once = that's a pinboard, route to references
    const intent = files.length === 1 ? "insert" : "reference";
    const reqId = intent === "insert"
      ? (crypto.randomUUID ? crypto.randomUUID()
                           : "ins" + Date.now()) : null;
    let sent = 0;
    for (const f of files) {
      let bmp;
      try {
        bmp = await createImageBitmap(f);
      } catch {
        setCanvasMeta(`can't read ${f.type || "an image"} — skipped`);
        continue;
      }
      const sc = Math.min(1, 768 / Math.max(bmp.width, bmp.height));
      const c = document.createElement("canvas");
      c.width = Math.max(1, Math.round(bmp.width * sc));
      c.height = Math.max(1, Math.round(bmp.height * sc));
      c.getContext("2d").drawImage(bmp, 0, 0, c.width, c.height);
      const b64 = c.toDataURL("image/jpeg", 0.9).split(",")[1];
      ws.send(JSON.stringify({ type: "upload_drop",
                               data: { b64, nx, ny, intent,
                                       req_id: reqId } }));
      if (intent === "insert") {
        showInsertGhost(reqId, "data:image/jpeg;base64," + b64, nx, ny);
      }
      sent++;
    }
    if (sent) {
      setCanvasMeta(sent === 1 ? "reading your image…"
                    : `reading ${sent} images…`);
    }
    return;
  }
  // dragged from a web page: no File, just a URL — the server fetches it
  let url = e.dataTransfer.getData("text/uri-list")
         || e.dataTransfer.getData("text/plain") || "";
  if (!url) {
    const html = e.dataTransfer.getData("text/html") || "";
    const m = html.match(/<img[^>]+src="([^"]+)"/i);
    if (m) url = m[1];
  }
  url = url.split("\n")[0].trim();
  if (!/^https?:\/\//i.test(url)) {
    setCanvasMeta("couldn't read that drop — try saving the image first");
    return;
  }
  const urlReq = crypto.randomUUID ? crypto.randomUUID()
                                    : "ins" + Date.now();
  ws.send(JSON.stringify({ type: "upload_drop",
                           data: { url, nx, ny, intent: "insert",
                                   req_id: urlReq } }));
  showInsertGhost(urlReq, null, nx, ny);
  setCanvasMeta("fetching that image…");
});

// the references box: everything dropped here is a style reference
const refZone = document.getElementById("refZone");
refZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  refZone.classList.add("dragover");
});
refZone.addEventListener("dragleave",
                         () => refZone.classList.remove("dragover"));
refZone.addEventListener("drop", async (e) => {
  e.preventDefault();
  e.stopPropagation();
  refZone.classList.remove("dragover");
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    setCanvasMeta("start the session first, then drop your image");
    return;
  }
  const files = Array.from(e.dataTransfer.files || [])
    .filter((f) => f.type.startsWith("image/"));
  let sent = 0;
  for (const f of files) {
    let bmp;
    try { bmp = await createImageBitmap(f); } catch { continue; }
    const sc = Math.min(1, 768 / Math.max(bmp.width, bmp.height));
    const c = document.createElement("canvas");
    c.width = Math.max(1, Math.round(bmp.width * sc));
    c.height = Math.max(1, Math.round(bmp.height * sc));
    c.getContext("2d").drawImage(bmp, 0, 0, c.width, c.height);
    const b64 = c.toDataURL("image/jpeg", 0.9).split(",")[1];
    ws.send(JSON.stringify({ type: "upload_drop",
                             data: { b64, nx: 0.5, ny: 0.5,
                                     intent: "reference" } }));
    sent++;
  }
  if (!sent) {
    let url = e.dataTransfer.getData("text/uri-list")
           || e.dataTransfer.getData("text/plain") || "";
    if (!url) {
      const html = e.dataTransfer.getData("text/html") || "";
      const m = html.match(/<img[^>]+src="([^"]+)"/i);
      if (m) url = m[1];
    }
    url = url.split("\n")[0].trim();
    if (/^https?:\/\//i.test(url)) {
      ws.send(JSON.stringify({ type: "upload_drop",
                               data: { url, nx: 0.5, ny: 0.5,
                                       intent: "reference" } }));
      sent = 1;
    }
  }
  setCanvasMeta(sent ? `🎨 adding ${sent} reference${sent > 1 ? "s" : ""}…`
                     : "couldn't read that drop");
});

handlers.upload_routed = (ev) => {
  const d = ev.data || {};
  document.getElementById("canvasMeta").textContent =
    d.action === "insert" ? `⤵ inserting — ${d.target}` :
    d.action === "tray" ? `🎨 added to references (${d.count})` :
    (d.reason || "");
};

let lastRefPlan = null;
handlers.ref_plan = (ev) => {
  lastRefPlan = ev.data || {};
  const uses = (lastRefPlan.use || [])
    .filter((u) => u.as !== "ignore")
    .map((u) => `#${u.ref} ${u.as}`).join(", ");
  document.getElementById("canvasMeta").textContent =
    uses ? `🎬 refs in play: ${uses}` : "🎬 refs: none fit this request";
  renderTray();
};

let trayData = [];
handlers.tray_updated = (ev) => {
  trayData = (ev.data || {}).refs || [];
  renderTray();
};

// ---- interactive insert placement: drop -> cutout preview -> the
// user drags / resizes / rotates -> idle timer (or checkmark) commits.
const igEl = document.getElementById("insertGhost");
const igImg = document.getElementById("insertGhostImg");
const igLabel = document.getElementById("igLabel");
let ins = null;   // {id, cx, cy, wFrac, aspect, rot, committed, timer}

function showInsertGhost(reqId, src, nx, ny) {
  cancelInsertGhost(false);
  ins = { id: reqId, cx: nx, cy: ny, wFrac: 0.3, aspect: 1, rot: 0,
          committed: false, timer: null, interacted: false };
  igEl.hidden = false;
  igEl.classList.add("pending");
  igEl.classList.remove("committed");
  igLabel.textContent = "cutting out…";
  if (src) { igImg.src = src; ins.aspect = 1; }
  else igImg.removeAttribute("src");
  igImg.onload = () => {
    ins.aspect = igImg.naturalHeight / Math.max(1, igImg.naturalWidth);
    layoutInsertGhost();
  };
  layoutInsertGhost();
}

function layoutInsertGhost() {
  if (!ins) return;
  const wr = canvasWrap.getBoundingClientRect();
  const ir = canvasImgEl.hidden ? wr : canvasImgEl.getBoundingClientRect();
  const w = ins.wFrac * ir.width;
  const h = w * (ins.aspect || 1);
  igEl.style.width = w + "px";
  igEl.style.left = (ir.left - wr.left + ins.cx * ir.width - w / 2) + "px";
  igEl.style.top = (ir.top - wr.top + ins.cy * ir.height - h / 2) + "px";
  igEl.style.transform = `rotate(${ins.rot}deg)`;
}

function armInsertTimer() {
  if (!ins || ins.committed) return;
  if (ins.timer) clearTimeout(ins.timer);
  igLabel.textContent = "placing in 3s — keep adjusting to wait";
  ins.timer = setTimeout(commitInsertGhost, 3000);
}

function commitInsertGhost() {
  if (!ins || ins.committed) return;
  ins.committed = true;
  if (ins.timer) clearTimeout(ins.timer);
  igEl.classList.add("committed");
  igLabel.textContent = "placing…";
  ws.send(JSON.stringify({ type: "insert_commit", data: {
    req_id: ins.id, nx: ins.cx, ny: ins.cy,
    w: ins.wFrac, rot: ins.rot } }));
}

function cancelInsertGhost(tellServer = true) {
  if (!ins) return;
  if (ins.timer) clearTimeout(ins.timer);
  if (tellServer && ws && ws.readyState === WebSocket.OPEN
      && !ins.committed) {
    ws.send(JSON.stringify({ type: "insert_cancel",
                             data: { req_id: ins.id } }));
  }
  ins = null;
  igEl.hidden = true;
}

handlers.insert_preview = (ev) => {
  const d = ev.data || {};
  if (!ins || ins.id !== d.req_id) return;
  if (d.ok && d.png_b64) {
    igImg.src = "data:image/png;base64," + d.png_b64;
    igEl.classList.remove("pending");
    igLabel.textContent = "drag to place · corner resizes · ↻ rotates";
  } else {
    igEl.classList.remove("pending");
    igLabel.textContent = "couldn't cut it out — placing as-is";
  }
};

let igMode = null;   // "move" | "resize" | "rotate"
igEl.addEventListener("pointerdown", (e) => {
  if (!ins || ins.committed) return;
  e.preventDefault();
  e.stopPropagation();
  igEl.setPointerCapture(e.pointerId);
  if (e.target.classList.contains("ig-resize")) igMode = "resize";
  else if (e.target.classList.contains("ig-rot")) igMode = "rotate";
  else if (e.target.tagName === "BUTTON") { igMode = null; return; }
  else { igMode = "move"; igEl.classList.add("dragging"); }
  if (ins.timer) clearTimeout(ins.timer);
  ins.interacted = true;
  igLabel.textContent = "";
});
igEl.addEventListener("pointermove", (e) => {
  if (!ins || !igMode || ins.committed) return;
  const ir = canvasImgEl.hidden ? canvasWrap.getBoundingClientRect()
                                : canvasImgEl.getBoundingClientRect();
  const px = (e.clientX - ir.left) / ir.width;
  const py = (e.clientY - ir.top) / ir.height;
  if (igMode === "move") {
    ins.cx = Math.max(0, Math.min(1, px));
    ins.cy = Math.max(0, Math.min(1, py));
  } else if (igMode === "resize") {
    const dx = Math.abs(px - ins.cx) * 2;
    ins.wFrac = Math.max(0.05, Math.min(1, dx));
  } else if (igMode === "rotate") {
    ins.rot = (Math.atan2(py - ins.cy, px - ins.cx) * 180 / Math.PI
               + 90 + 360) % 360;
  }
  layoutInsertGhost();
});
igEl.addEventListener("pointerup", (e) => {
  if (!ins) return;
  igEl.classList.remove("dragging");
  if (igMode) { igMode = null; armInsertTimer(); }
});
document.getElementById("igOk").addEventListener("click", (e) => {
  e.stopPropagation();
  commitInsertGhost();
});
document.getElementById("igX").addEventListener("click", (e) => {
  e.stopPropagation();
  cancelInsertGhost(true);
});
window.addEventListener("resize", layoutInsertGhost);

function renderTray() {
  const tray = document.getElementById("refTray");
  const useBy = {};
  ((lastRefPlan || {}).use || []).forEach((u) => { useBy[u.i || u.ref] = u.as; });
  tray.innerHTML = "";
  trayData.forEach((r) => {
    const chip = document.createElement("div");
    const use = useBy[r.i];
    chip.className = "ref-chip" + (use && use !== "ignore" ? " used" : "");
    const role = r.role !== "auto" ? r.role : (use || "auto");
    chip.innerHTML =
      `<img src="data:image/jpeg;base64,${r.b64}">` +
      `<span class="num">${r.i}</span>` +
      `<span class="badge">${role}</span>` +
      `<span class="cap">${r.style_desc || r.object || ""}</span>`;
    tray.appendChild(chip);
  });
}

function layoutOverlay() {
  if (canvasImgEl.hidden) { overlay.hidden = true; return; }
  const wr = canvasImgEl.parentElement.getBoundingClientRect();
  const ir = canvasImgEl.getBoundingClientRect();
  overlay.style.left = (ir.left - wr.left) + "px";
  overlay.style.top = (ir.top - wr.top) + "px";
  overlay.style.width = ir.width + "px";
  overlay.style.height = ir.height + "px";
  const dpr = window.devicePixelRatio || 1;
  overlay.width = Math.max(1, Math.round(ir.width * dpr));
  overlay.height = Math.max(1, Math.round(ir.height * dpr));
  overlay.hidden = false;
}
window.addEventListener("resize", () => { layoutOverlay(); paint(); });
canvasImgEl.addEventListener("load", () => { layoutOverlay(); paint(); });
// never let the browser's native image-drag hijack a click on the canvas
canvasImgEl.draggable = false;
canvasImgEl.addEventListener("dragstart", (e) => e.preventDefault());

function evToNorm(e) {
  const r = overlay.getBoundingClientRect();
  return [(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height];
}

function hitStack(nx, ny) {
  // every warm object containing the point (12px snap), BIGGEST first —
  // a plain click means the general object, drilling reaches the parts
  const out = [];
  for (const ent of ml.warm) {
    let hit = inMask(ent, nx, ny);
    if (!hit) {
      const [W, H] = ent.size;
      snap: for (let r = 4; r <= 12; r += 4)
        for (const [dx, dy] of [[r, 0], [-r, 0], [0, r], [0, -r]])
          if (inMask(ent, nx + dx / W, ny + dy / H)) { hit = true; break snap; }
    }
    if (hit) out.push(ent);
  }
  return out.sort((a, b) => b.area - a.area);
}

function inMask(ent, nx, ny) {
  const [W, H] = ent.size;
  const xi = Math.round(nx * W), yi = Math.round(ny * H);
  return xi >= 0 && yi >= 0 && xi < W && yi < H
    && !!ent.maskData[yi * W + xi];
}

function handleAt(ent, nx, ny) {
  const [W, H] = ent.size;
  const x = nx * W, y = ny * H;
  const [x0, y0, x1, y1] = ent.tight;
  for (const [hx, hy] of [[x0, y0], [x1, y0], [x0, y1], [x1, y1]])
    if (Math.abs(x - hx) <= HANDLE + 5 && Math.abs(y - hy) <= HANDLE + 5)
      return true;
  return false;
}

function withEntTransform(ent, fn) {
  octx.save();
  octx.setTransform(overlay.width / ent.size[0], 0, 0,
                    overlay.height / ent.size[1], 0, 0);
  fn();
  octx.restore();
}

function tinted(ent, color) {
  const c = document.createElement("canvas");
  c.width = ent.size[0]; c.height = ent.size[1];
  const cc = c.getContext("2d");
  cc.drawImage(ent.mask, 0, 0);
  cc.globalCompositeOperation = "source-in";
  cc.fillStyle = color;
  cc.fillRect(0, 0, c.width, c.height);
  return c;
}

function ring(ent, alpha) {
  const t = ent._ring || (ent._ring = tinted(ent, ACCENT));
  withEntTransform(ent, () => {
    octx.globalAlpha = alpha;
    const o = 2.5;
    for (const [dx, dy] of [[o, 0], [-o, 0], [0, o], [0, -o],
                            [o, o], [o, -o], [-o, o], [-o, -o]])
      octx.drawImage(t, dx, dy);
    octx.globalCompositeOperation = "destination-out";
    octx.globalAlpha = 1;
    octx.drawImage(ent.mask, 0, 0);
  });
}

function fillMask(ent, alpha) {
  const t = ent._fill || (ent._fill = tinted(ent, "#ffffff"));
  withEntTransform(ent, () => {
    octx.globalAlpha = alpha;
    octx.drawImage(t, 0, 0);
  });
}

function drawHandles(ent) {
  const [x0, y0, x1, y1] = ent.tight;
  withEntTransform(ent, () => {
    octx.fillStyle = "#fff";
    octx.strokeStyle = ACCENT;
    octx.lineWidth = 2;
    for (const [x, y] of [[x0, y0], [x1, y0], [x0, y1], [x1, y1]]) {
      octx.beginPath();
      octx.rect(x - HANDLE, y - HANDLE, HANDLE * 2, HANDLE * 2);
      octx.fill(); octx.stroke();
    }
  });
}

function buildSprite(ent) {
  const [x0, y0, x1, y1] = ent.tight;
  const w = x1 - x0, h = y1 - y0;
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  const cc = c.getContext("2d");
  const kx = canvasImgEl.naturalWidth / ent.size[0];
  const ky = canvasImgEl.naturalHeight / ent.size[1];
  cc.drawImage(canvasImgEl, x0 * kx, y0 * ky, w * kx, h * ky, 0, 0, w, h);
  cc.globalCompositeOperation = "destination-in";
  cc.drawImage(ent.mask, x0, y0, w, h, 0, 0, w, h);
  return c;
}

function drawMoved(d) {
  d.ents.forEach((ent, i) => {
    const [x0, y0, x1, y1] = ent.tight;
    const w = x1 - x0, h = y1 - y0;
    const [W, H] = ent.size;
    const g = ent._ghost || (ent._ghost = tinted(ent, "#0a0e14"));
    withEntTransform(ent, () => {
      octx.globalAlpha = 0.45;                 // ghost at the origin
      octx.drawImage(g, 0, 0);
      octx.globalAlpha = 1;
      if (d.deleted) return;             // delete: ghost only, no sprite
      const nw = w * d.scale, nh = h * d.scale;
      const cx = (x0 + x1) / 2 + d.ndx * W, cy = (y0 + y1) / 2 + d.ndy * H;
      octx.shadowColor = "rgba(0,0,0,.5)";
      octx.shadowBlur = 16;
      octx.drawImage(d.sprites[i], cx - nw / 2, cy - nh / 2, nw, nh);
    });
  });
}

function drawOverlay() {
  if (overlay.hidden) return;
  octx.setTransform(1, 0, 0, 1, 0, 0);
  octx.clearRect(0, 0, overlay.width, overlay.height);
  const moved = (ml.drag && ml.drag.moved) ? ml.drag : ml.optimistic;
  if (moved) drawMoved(moved);
  const movedSet = moved ? new Set(moved.ents) : null;
  if (false && ml.hover && !ml.sels.includes(ml.hover) && !ml.drag && !ml.ink
      && !ml.pending)
    ring(ml.hover, 0.4);
  for (const s of ml.sels) {
    if (movedSet && movedSet.has(s)) continue;
    ring(s, 1);
    fillMask(s, 0.12);
  }

  if (ml.ink) {                                // drawing mode ink
    octx.strokeStyle = ml.ink.sent ? "rgba(255,32,32,0.45)"
                                   : "rgba(255,32,32,0.9)";
    octx.lineWidth = Math.max(3, overlay.width * 0.006);
    octx.lineCap = "round";
    octx.lineJoin = "round";
    const paths = [...ml.ink.strokes];
    if (ml.ink.cur) paths.push(ml.ink.cur);
    for (const pts of paths) {
      if (pts.length < 2) continue;
      octx.beginPath();
      octx.moveTo(pts[0][0] * overlay.width, pts[0][1] * overlay.height);
      for (const [px, py] of pts.slice(1))
        octx.lineTo(px * overlay.width, py * overlay.height);
      octx.stroke();
    }
  }
  if (ml.pending) {                          // click-time lookup in flight
    octx.strokeStyle = ACCENT;
    octx.lineWidth = 2;
    octx.beginPath();
    octx.arc(ml.pending.nx * overlay.width, ml.pending.ny * overlay.height,
             14, 0, Math.PI * 2);
    octx.stroke();
  }
}

function updateChip() {
  if (!ml.sels.length || overlay.hidden) { selChip.hidden = true; return; }
  const one = ml.sels.length === 1;
  document.getElementById("selName").textContent =
    one ? (ml.sels[0].name || "object") : `${ml.sels.length} objects`;
  selChip.hidden = false;
  const a = ml.sels[0];
  const wr = canvasImgEl.parentElement.getBoundingClientRect();
  const r = overlay.getBoundingClientRect();
  const kx = r.width / a.size[0], ky = r.height / a.size[1];
  selChip.style.left = Math.max(0, r.left - wr.left + a.tight[0] * kx) + "px";
  selChip.style.top = Math.max(0, r.top - wr.top + a.tight[1] * ky - 28) + "px";
}
document.getElementById("selX").addEventListener("click", () => {
  ml.sels = []; paint();
});

function deleteSelection() {
  if (!ml.sels.length || !ws || ws.readyState !== WebSocket.OPEN) return;
  const moves = ml.sels.map((ent) => ({
    entity_id: ent.id || "_click", name: ent.name, delete: true,
    from_box: ent.box, mask_rle: ent.rle, mask_size: ent.size }));
  ws.send(JSON.stringify({ type: "mouse_move", data: { moves } }));
  // ghost the removed objects until the fill-in render lands
  ml.optimistic = { ents: [...ml.sels], sprites: [], ndx: 0, ndy: 0,
                    scale: 1, deleted: true };
  const what = ml.sels.length === 1
    ? (ml.sels[0].name || "object") : `${ml.sels.length} objects`;
  document.getElementById("canvasMeta").textContent =
    `removing the ${what}… filling in`;
  ml.sels = [];
  ml.drill = null;
  paint();
}
document.getElementById("selDel").addEventListener("click", deleteSelection);

function startDrag(ents, nx, ny, mode) {
  const a = ents[0];
  const [x0, y0, x1, y1] = a.tight;
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  ml.drag = { ents, mode, startNx: nx, startNy: ny, ndx: 0, ndy: 0,
              scale: 1, moved: false,
              sprites: ents.map(buildSprite), cx, cy,
              startDist: Math.hypot(nx * a.size[0] - cx,
                                    ny * a.size[1] - cy) };
}

overlay.addEventListener("pointerdown", (e) => {
  if (e.button !== 0 || overlay.hidden) return;
  e.preventDefault();          // no native drag/select fallthrough, ever
  const [nx, ny] = evToNorm(e);
  overlay.setPointerCapture(e.pointerId);
  // selections are passive scope markers (voice: "make this blue",
  // "move this up", "remove this") — dragging them was removed: a
  // stale outline once turned an accidental drag into a move edit.
  // Every press on the canvas is ink, always.
  // empty space: begin an ink stroke (a tiny release becomes a click)
  if (!ml.ink || ml.ink.sent) ml.ink = { cur: null, strokes: [], timer: null, sent: false };
  if (ml.ink.timer) { clearTimeout(ml.ink.timer); ml.ink.timer = null; }
  ml.ink.cur = [[nx, ny]];
  drawOverlay();
});

overlay.addEventListener("pointermove", (e) => {
  if (overlay.hidden) return;
  const [nx, ny] = evToNorm(e);
  if (ml.drag) {
    const d = ml.drag;
    if (d.mode === "move") { d.ndx = nx - d.startNx; d.ndy = ny - d.startNy; }
    else {
      const a = d.ents[0];
      d.scale = Math.max(0.25, Math.min(4,
        Math.hypot(nx * a.size[0] - d.cx, ny * a.size[1] - d.cy)
        / Math.max(d.startDist, 1)));
    }
    // dead-zone: grabbing a handle or twitching the pointer is not an
    // edit — "moved" requires perceptible displacement or a real resize
    if (!d.moved && (Math.abs(d.ndx) + Math.abs(d.ndy) > 0.004
                     || (d.mode === "scale" && Math.abs(d.scale - 1) > 0.02)))
      d.moved = true;
    if (d.moved) { drawOverlay(); overlay.style.cursor = "grabbing"; }
    return;
  }
  if (ml.ink && ml.ink.cur) {
    const last = ml.ink.cur[ml.ink.cur.length - 1];
    if (Math.hypot(nx - last[0], ny - last[1]) > 0.004)
      ml.ink.cur.push([nx, ny]);
    drawOverlay();
    return;
  }
  if (overlay.style.cursor !== "crosshair")
    overlay.style.cursor = "crosshair";
});

overlay.addEventListener("pointerup", () => {
  if (ml.ink && ml.ink.cur) {
    const pts = ml.ink.cur;
    ml.ink.cur = null;
    const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
    const tiny = (Math.max(...xs) - Math.min(...xs) < 0.008
                  && Math.max(...ys) - Math.min(...ys) < 0.008);
    if (tiny && !ml.ink.strokes.length) {
      ml.ink = null;
      if (ml.sels.length) { ml.sels = []; paint(); }
      else drawOverlay();
      return;
    }
    if (!tiny) {
      ml.ink.strokes.push(pts);
      // stream it: the server stamps each stroke on the session clock
      // and decides grouping (3s solo idle, or collected by the next
      // spoken commit for put-that-there)
      if (ws && ws.readyState === WebSocket.OPEN)
        ws.send(JSON.stringify({ type: "ink_stroke",
                                 data: { stroke: pts } }));
    }
    if (!ml.ink.strokes.length) ml.ink = null;
    drawOverlay();
    return;
  }
  const d = ml.drag;
  ml.drag = null;
  if (!d) return;
  if (!d.moved) {
    // a stationary press on the selection keeps it — nothing else
    drawOverlay();
    return;
  }
  const moves = d.ents.map((ent) => {
    const b = ent.box;
    return {
      entity_id: ent.id || "_click", name: ent.name,
      from_box: b,
      to_box: {
        x: Math.max(0, Math.min(1, b.x + d.ndx)),
        y: Math.max(0, Math.min(1, b.y + d.ndy)),
        w: Math.max(0.01, Math.min(1, b.w * d.scale)),
        h: Math.max(0.01, Math.min(1, b.h * d.scale)),
      },
      mask_rle: ent.rle, mask_size: ent.size,
    };
  }).filter((m) => {
    // sub-perceptual release = input noise, never a render commit
    const df = Math.abs(m.to_box.x - m.from_box.x)
             + Math.abs(m.to_box.y - m.from_box.y);
    const sr = Math.abs(m.to_box.w / Math.max(m.from_box.w, 1e-6) - 1);
    return df > 0.004 || sr > 0.02;
  });
  if (!moves.length) { overlay.style.cursor = "grab"; drawOverlay(); return; }
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "mouse_move", data: { moves } }));
    // the dropped objects stay put (optimistically) until the render lands
    ml.optimistic = d;
    const what = d.ents.length === 1
      ? (d.ents[0].name || "object") : `${d.ents.length} objects`;
    document.getElementById("canvasMeta").textContent =
      `moving the ${what}… refining`;
  }
  overlay.style.cursor = "grab";
  drawOverlay();
});

function goBack() {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "undo" }));
    document.getElementById("canvasMeta").textContent = "going back…";
  }
}
document.getElementById("backBtn").addEventListener("click", goBack);

function goForward() {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "redo" }));
    document.getElementById("canvasMeta").textContent = "going forward…";
  }
}
document.getElementById("fwdBtn").addEventListener("click", goForward);

window.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "z"
      && !/INPUT|TEXTAREA/.test(document.activeElement.tagName)) {
    e.preventDefault();
    if (e.shiftKey) goForward(); else goBack();
    return;
  }
  if ((e.key === "Delete" || e.key === "Backspace") && ml.sels.length
      && !ml.drag && !/INPUT|TEXTAREA/.test(document.activeElement.tagName)) {
    e.preventDefault();
    deleteSelection();
    return;
  }
  if (e.key !== "Escape") return;
  if (ml.drag) { ml.drag = null; drawOverlay(); }
  else if (ml.ink) {
    ml.ink = null;
    if (ws && ws.readyState === WebSocket.OPEN)
      ws.send(JSON.stringify({ type: "ink_clear" }));
    drawOverlay();
  }
  else if (ml.pending) { ml.pending = null; overlay.style.cursor = "default"; drawOverlay(); }
  else if (ml.sels.length) { ml.sels = []; paint(); }
});

// the optimistic drag layer dies the moment a fresh committed frame lands
const _origCanvasFrame = handlers.canvas_frame;
handlers.canvas_frame = (ev) => {
  _origCanvasFrame(ev);
  if (ev.data.lane === "committed" && !ev.data.preview && ml.optimistic) {
    ml.optimistic = null;
    drawOverlay();
  }
  if (ins && ins.committed && ev.data.lane === "committed"
      && !ev.data.preview) {
    cancelInsertGhost(false);
  }
};


// ---- resizable panel layout ----
// Drag the gutters between panels to resize; double-click a gutter to
// reset. Widths persist across reloads. The canvas overlay re-aligns
// live while dragging.
const gridEl = document.querySelector("main.grid");
const DEFAULT_COLS = [3, 0.9, 0.55, 0.7, 0.85];
let colW = null;
try { colW = JSON.parse(localStorage.getItem("psirens_cols")); } catch {}
if (!Array.isArray(colW) || colW.length !== 5
    || colW.some((w) => typeof w !== "number" || !(w > 0)))
  colW = [...DEFAULT_COLS];
function applyCols() {
  colW.forEach((w, i) => gridEl.style.setProperty(`--c${i}`, w + "fr"));
  layoutOverlay(); drawOverlay(); updateChip();
}
applyCols();
document.querySelectorAll(".gutter").forEach((g) => {
  const i = +g.dataset.g;
  g.addEventListener("dblclick", () => {
    colW = [...DEFAULT_COLS];
    localStorage.setItem("psirens_cols", JSON.stringify(colW));
    applyCols();
  });
  g.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    const startX = e.clientX;
    const total = colW[i] + colW[i + 1];
    const w0 = colW[i];
    const panels = gridEl.querySelectorAll("section.panel");
    const pxPerFr = (panels[i].getBoundingClientRect().width
                     + panels[i + 1].getBoundingClientRect().width) / total;
    g.setPointerCapture(e.pointerId);
    g.classList.add("active");
    const move = (ev) => {
      let w = w0 + (ev.clientX - startX) / Math.max(pxPerFr, 1);
      w = Math.max(0.15, Math.min(total - 0.15, w));
      colW[i] = w;
      colW[i + 1] = total - w;
      applyCols();
    };
    const up = () => {
      g.classList.remove("active");
      g.removeEventListener("pointermove", move);
      g.removeEventListener("pointerup", up);
      localStorage.setItem("psirens_cols", JSON.stringify(colW));
    };
    g.addEventListener("pointermove", move);
    g.addEventListener("pointerup", up);
  });
});
