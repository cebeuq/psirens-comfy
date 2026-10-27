// Psirens mic button: talk to any workflow that has a Psirens Voice Prompt.
// Speech -> transcript -> scene editor (Psirens voice service) -> the voice
// prompt's `scene` + `instruction` widgets are filled -> the graph is queued.
import { app } from "../../scripts/app.js";

// The AudioWorklet runs from a Blob URL: ComfyUI loads every .js file in
// web/ as an extension, so the worklet cannot live there as its own file.
const WORKLET_SRC = `// AudioWorklet processor: turns the mic input into 80ms frames of int16 mono
// PCM at 24kHz. We request a 24kHz AudioContext so the browser resamples for
// us; if the browser refused (context runs at another rate), we fall back to
// linear resampling here.

const TARGET_RATE = 24000;
const FRAME_SAMPLES = 1920; // 80ms at 24kHz

class PcmCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buf = new Float32Array(FRAME_SAMPLES);
    this.fill = 0;
    this.ratio = sampleRate / TARGET_RATE; // sampleRate is the worklet global
    this.readPos = 0;
    this.prev = 0;
  }

  pushSample(s) {
    this.buf[this.fill++] = s;
    if (this.fill === FRAME_SAMPLES) {
      const out = new Int16Array(FRAME_SAMPLES);
      let peak = 0;
      for (let i = 0; i < FRAME_SAMPLES; i++) {
        const v = Math.max(-1, Math.min(1, this.buf[i]));
        out[i] = v < 0 ? v * 32768 : v * 32767;
        const a = Math.abs(v);
        if (a > peak) peak = a;
      }
      this.port.postMessage({ pcm: out.buffer, peak }, [out.buffer]);
      this.fill = 0;
    }
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;

    if (this.ratio === 1) {
      for (let i = 0; i < ch.length; i++) this.pushSample(ch[i]);
    } else {
      // linear resample from context rate to 24kHz
      for (let i = 0; i < ch.length; i++) {
        const cur = ch[i];
        while (this.readPos < 1) {
          this.pushSample(this.prev + (cur - this.prev) * this.readPos);
          this.readPos += this.ratio;
        }
        this.readPos -= 1;
        this.prev = cur;
      }
    }
    return true;
  }
}

registerProcessor("pcm-capture", PcmCapture);
`;
const WORKLET = URL.createObjectURL(new Blob([WORKLET_SRC], { type: "application/javascript" }));
const S = { ws: null, ctx: null, stream: null, live: false, scene: "" };
let ui = null;

function api_base() {
  return location.pathname.replace(/\/[^/]*$/, "");
}
function voiceNodes() {
  return (app.graph?._nodes || []).filter((n) => n.type === "PsirensVoicePrompt");
}
function setWidget(node, name, value) {
  const w = node.widgets?.find((w) => w.name === name);
  if (w) w.value = value;
}

function mountUi() {
  const box = document.createElement("div");
  box.style.cssText = "position:fixed;left:16px;bottom:16px;z-index:9999;display:flex;" +
    "align-items:center;gap:10px;max-width:60vw;font:13px system-ui,sans-serif";
  const btn = document.createElement("button");
  btn.textContent = "🎙 Talk";
  btn.title = "Psirens: speak to edit (fills Psirens Voice Prompt nodes and queues)";
  btn.style.cssText = "border:0;border-radius:999px;padding:10px 16px;cursor:pointer;" +
    "font-weight:600;background:#2b2b33;color:#fff;box-shadow:0 2px 10px #0006";
  const line = document.createElement("div");
  line.style.cssText = "color:#ddd;background:#0009;border-radius:8px;padding:6px 10px;" +
    "display:none;white-space:nowrap;overflow:hidden;text-overflow:ellipsis";
  btn.onclick = () => (S.live ? stop() : start().catch((e) => fail(e)));
  box.append(btn, line);
  document.body.append(box);
  ui = { btn, line };
}

function say(text, color = "#ddd") {
  ui.line.style.display = text ? "block" : "none";
  ui.line.style.color = color;
  ui.line.textContent = text;
}
function fail(e) {
  say(String(e?.message || e), "#f88");
  stop();
}

async function start() {
  if (!navigator.mediaDevices) {
    say("Mic needs HTTPS or localhost (e.g. ssh -L 8288:127.0.0.1:8288 host)", "#f88");
    return;
  }
  if (!voiceNodes().length) say("Add a Psirens Voice Prompt node to steer this workflow", "#fc8");
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const base = location.pathname.replace(/\/[^/]*$/, "");
  S.ws = new WebSocket(`${proto}://${location.host}${base}/psirens/ws`);
  S.ws.binaryType = "arraybuffer";
  S.ws.onmessage = (m) => typeof m.data === "string" && onEvent(JSON.parse(m.data));
  S.ws.onclose = () => S.live && stop();
  await new Promise((res, rej) => { S.ws.onopen = res; S.ws.onerror = () => rej(new Error("voice service unreachable")); });
  S.ws.send(JSON.stringify({ type: "client_hello", data: { ua: navigator.userAgent, client: "comfyui" } }));
  S.ctx = new AudioContext({ sampleRate: 24000 });
  await S.ctx.audioWorklet.addModule(WORKLET);
  S.stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
  });
  const node = new AudioWorkletNode(S.ctx, "pcm-capture");
  node.port.onmessage = (m) => S.ws?.readyState === 1 && S.ws.send(m.data.pcm);
  S.ctx.createMediaStreamSource(S.stream).connect(node);
  S.live = true;
  ui.btn.textContent = "■ Stop";
  ui.btn.style.background = "#c0392b";
  say("listening…");
}

function stop() {
  S.live = false;
  try { S.ws?.readyState === 1 && S.ws.send(JSON.stringify({ type: "stop" })); } catch {}
  try { S.ws?.close(); } catch {}
  S.stream?.getTracks().forEach((t) => t.stop());
  S.ctx?.close().catch(() => {});
  S.ws = S.ctx = S.stream = null;
  ui.btn.textContent = "🎙 Talk";
  ui.btn.style.background = "#2b2b33";
}

function widgetValue(node, name, dflt) {
  const w = node.widgets?.find((w) => w.name === name);
  return w ? w.value : dflt;
}

// keyframes: every committed edit keeps its canvas latent as <slot>@<rev>,
// so "no, go back" can restore it exactly (a no-op edit on the restored
// latent; the token freeze makes it pixel-identical)
// whole-image edits (restyle, relight, camera) move every token a little;
// the token freeze would undo them, so it is switched off for those
const GLOBAL_EDIT = /\b(entire|whole) (image|scene|picture|frame|canvas|thing)\b|\b(repaint|redraw|re-render) the (scene|image|picture)\b|\b(convert|turn|transform|restyle) (it|this|everything)\b|\bcamera\b|\bviewpoint\b/i;

function setFreeze(on) {
  for (const n of app.graph._nodes.filter((n) => n.type === "PsirensTokenPassthrough")) {
    n.properties = n.properties || {};
    if (n.properties.psirens_tau == null) n.properties.psirens_tau = widgetValue(n, "tau", 0.35);
    setWidget(n, "tau", on ? n.properties.psirens_tau : 0);
  }
  // local edits keep the graph's seed (faithful repaint); whole-image edits
  // need fresh noise, or the old seed pulls the picture back to what it was
  for (const n of app.graph._nodes.filter((n) => /KSampler/.test(n.type))) {
    n.properties = n.properties || {};
    if (n.properties.psirens_seed == null) n.properties.psirens_seed = widgetValue(n, "seed", 0);
    // a fresh roll becomes the picture's seed for the local edits after it
    if (!on) n.properties.psirens_seed = Math.floor(Math.random() * 2 ** 31);
    setWidget(n, "seed", n.properties.psirens_seed);
  }
}

async function onCommitTarget(d) {
  const nodes = voiceNodes();
  if (!nodes.length) return;
  const instr = d.edit_instruction || "";
  setFreeze(d.restore_rev != null ||
            !(d.camera || (d.strength ?? 0) >= 0.75 || GLOBAL_EDIT.test(instr)));
  const rev = d.commit_rev ?? -1;
  for (const n of app.graph._nodes.filter((n) => n.type === "PsirensCanvasSave")) {
    setWidget(n, "keyframe", rev);
  }
  if (d.restore_rev != null) {
    let restored = false;
    for (const n of nodes) {
      const r = await fetch(api_base() + "/psirens/restore", { method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session: widgetValue(n, "session", "ui"),
                               slot: widgetValue(n, "slot", "latest"),
                               rev: d.restore_rev }) }).catch(() => null);
      restored = restored || (r && r.ok);
    }
    for (const n of nodes) {
      setWidget(n, "scene", d.prompt || S.scene);
      setWidget(n, "instruction", restored ? "Keep everything exactly the same."
        : "Redraw the image to match: " + (d.prompt || S.scene));
    }
    say("↩ back to #" + d.restore_rev, "#9cf");
  } else {
    for (const n of nodes) {
      setWidget(n, "scene", d.prompt || S.scene);
      setWidget(n, "instruction", d.edit_instruction || "");
    }
    say("✎ " + (d.edit_instruction || d.prompt || ""), "#9f9");
  }
  app.graph.setDirtyCanvas(true, true);
  app.queuePrompt(0, 1);
}

function onEvent(ev) {
  const d = ev.data || {};
  switch (ev.type) {
    case "session_start": {
      // pressing Talk starts a new picture in every voice-steered session
      const sessions = new Set(voiceNodes().map((n) =>
        n.widgets?.find((w) => w.name === "session")?.value || "ui"));
      for (const session of sessions) {
        fetch(api_base() + "/psirens/reset", { method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session }) }).catch(() => {});
      }
      break;
    }
    case "partial_transcript":
      if (d.pending) say("… " + d.pending);
      break;
    case "commit":
      say("“" + (d.utterance_clean || "") + "”", "#fff");
      break;
    case "prompt_updated":
      if (!d.provisional) S.scene = d.prompt || S.scene;
      break;
    case "render_target": {
      const committed = d.lane === "committed" || (d.lane == null && d.source === "commit");
      if (!committed) break;
      onCommitTarget(d);
      break;
    }
    case "error":
      say((d.where ? d.where + ": " : "") + (d.message || "error"), "#f88");
      break;
  }
}

app.registerExtension({
  name: "psirens.voice",
  setup() { mountUi(); },
});
