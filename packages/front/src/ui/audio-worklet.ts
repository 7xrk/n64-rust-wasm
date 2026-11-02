/** Plays the core's audio. Chunks arrive over a MessagePort straight from the
 * emulation worker and are resampled to the context rate. */

declare const sampleRate: number;
declare function registerProcessor(name: string, ctor: unknown): void;
declare class AudioWorkletProcessor {
  readonly port: MessagePort;
}

const CAPACITY = 1 << 15;
/** Drop old audio when more than this many seconds are queued. */
const MAX_LATENCY = 0.12;

class N64AudioProcessor extends AudioWorkletProcessor {
  private readonly left = new Float32Array(CAPACITY);
  private readonly right = new Float32Array(CAPACITY);
  private read = 0;
  private write = 0;
  private sourceRate = 32000;
  private phase = 0;

  constructor() {
    super();
    this.port.onmessage = (e: MessageEvent<{ port: MessagePort }>) => {
      e.data.port.onmessage = (m: MessageEvent<{ samples: Int16Array; rate: number }>) => this.push(m.data.samples, m.data.rate);
    };
  }

  private get queued(): number {
    return (this.write - this.read + CAPACITY) % CAPACITY;
  }

  private push(samples: Int16Array, rate: number): void {
    this.sourceRate = rate;
    for (let i = 0; i + 1 < samples.length; i += 2) {
      this.left[this.write] = samples[i] / 32768;
      this.right[this.write] = samples[i + 1] / 32768;
      this.write = (this.write + 1) % CAPACITY;
      if (this.write === this.read) this.read = (this.read + 1) % CAPACITY;
    }
    const limit = Math.floor(MAX_LATENCY * rate);
    if (this.queued > limit) this.read = (this.write - limit + CAPACITY) % CAPACITY;
  }

  process(_inputs: Float32Array[][], outputs: Float32Array[][]): boolean {
    const [outL, outR] = outputs[0];
    const step = this.sourceRate / sampleRate;
    for (let i = 0; i < outL.length; i++) {
      if (this.queued < 2) {
        outL[i] = 0;
        if (outR) outR[i] = 0;
        continue;
      }
      const next = (this.read + 1) % CAPACITY;
      outL[i] = this.left[this.read] + (this.left[next] - this.left[this.read]) * this.phase;
      if (outR) outR[i] = this.right[this.read] + (this.right[next] - this.right[this.read]) * this.phase;
      this.phase += step;
      while (this.phase >= 1) {
        this.phase -= 1;
        this.read = (this.read + 1) % CAPACITY;
      }
    }
    return true;
  }
}

registerProcessor('n64-audio', N64AudioProcessor);
