/** Frame-exact video recording with mediabunny. Timestamps come from the
 * emulated time, so the file is smooth regardless of host speed. */

import {
  AudioSample,
  AudioSampleSource,
  BufferTarget,
  Mp4OutputFormat,
  Output,
  QUALITY_HIGH,
  VideoSample,
  VideoSampleSource,
} from 'mediabunny';

const AUDIO_RATE = 48000;

export class Recorder {
  private readonly output = new Output({ format: new Mp4OutputFormat({ fastStart: 'in-memory' }), target: new BufferTarget() });
  private readonly video = new VideoSampleSource({ codec: 'avc', quality: QUALITY_HIGH, keyFrameInterval: 1 });
  private readonly audio = new AudioSampleSource({ codec: 'aac', quality: QUALITY_HIGH, transform: { sampleRate: AUDIO_RATE } });
  private frames = 0;
  private videoTime = 0;
  private audioTime = 0;
  private ready: Promise<void>;

  constructor(private readonly canvas: OffscreenCanvas) {
    this.output.addVideoTrack(this.video);
    this.output.addAudioTrack(this.audio);
    this.ready = this.output.start();
  }

  get frameCount(): number {
    return this.frames;
  }

  /** Adds the frame currently on the canvas, which lasted `duration` emulated
   * seconds, and the audio produced with it. */
  async add(samples: Int16Array, rate: number, duration: number): Promise<void> {
    await this.ready;
    const sample = new VideoSample(this.canvas, { timestamp: this.videoTime, duration });
    this.videoTime += duration;
    await this.video.add(sample);
    sample.close();
    this.frames++;
    if (samples.length > 0) {
      const audio = new AudioSample({ data: samples, format: 's16', numberOfChannels: 2, sampleRate: rate, timestamp: this.audioTime });
      this.audioTime += samples.length / 2 / rate;
      await this.audio.add(audio);
      audio.close();
    }
  }

  async finish(): Promise<ArrayBuffer> {
    await this.ready;
    await this.output.finalize();
    const buffer = (this.output.target as BufferTarget).buffer;
    if (!buffer) throw new Error('Recording produced no data');
    return buffer;
  }
}
