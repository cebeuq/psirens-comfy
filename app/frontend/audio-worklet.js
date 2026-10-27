// AudioWorklet processor: turns the mic input into 80ms frames of int16 mono
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
