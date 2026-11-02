/** Frame loop running inside the worker: input, emulation, rendering, audio,
 * plus the TAS layers (movie, savestates, snapshots, cheats, recording). */

import type { Core, Renderer } from 'n64core';
import { applyCheats, type Cheat } from './cheats';
import { MemorySearch, readValue, type ValueType } from './memsearch';
import { Movie, type MovieInfo } from './movie';
import { type PinInfo, Pins } from './pins';
import { Recorder } from './recorder';
import { Scenes } from './scenes';
import { type PageState, Snapshots } from './snapshots';

export const FPS = 60;
/** Emulated frames between scene history entries. */
const SCENE_INTERVAL = 120;
const STATUS_INTERVAL_MS = 100;
/** How long an interactive replay runs before reporting progress and taking messages. */
const REPLAY_SLICE_MS = 50;
/** Gap between a request to resume and the game running again, so the player
 * can take the controller after leaving a pause or the scene history. */
const RESUME_DELAY_MS = 500;
/** Longest stretch the frame loop runs before yielding to the event loop. */
const TICK_BUDGET_MS = 12;
/** Frames rendered at the end of a seek. Rendered pixels exist only on the GPU,
 * and restoring a snapshot overwrites them with the CPU copy of RDRAM, so the
 * tail must cover several game frames: a game running at 10-20 fps needs
 * 3-6 VIs per frame to redraw each of its framebuffers. */
const SEEK_RENDER_TAIL = 30;
/** Most frames a seek replays from before its render tail. When the snapshot before the tail is farther back, such
 * as power-on in an emulator that got a state just before the target, the seek starts from the snapshot nearest the
 * target and may show a few stale framebuffers. */
const MAX_TAIL_REPLAY = 300;
/** Renderer slot that holds the screen while other frames are visited. */
const VISIT_SLOT = -1;
/** Input of frames past the end of the movie while visiting them. */
const NO_INPUT = new Uint32Array(4);

export interface Watch {
  addr: number;
  type: ValueType;
  label: string;
}

export interface Status {
  frame: number;
  fps: number;
  running: boolean;
  speed: number;
  lagFrames: number;
  movie: MovieInfo;
  snapshots: number;
  snapshotBytes: number;
  recording: number | null;
  watchValues: number[];
  slots: (number | null)[];
  /** Controller 1 input used for the last emulated frame. */
  input: number;
  /** Frame a seek or a recompute after an edit is heading to, while it runs. */
  seekTarget: number | null;
}

interface Replay {
  /** Restore a snapshot even when emulating on from the current frame would do. */
  recompute?: boolean;
  /** Input of frames past the end of the movie; without it the live input fills them. */
  pastEnd?: Uint32Array;
  /** Show the target in the status and report progress, for replays the user waits on. */
  progress?: boolean;
  /** What a pause in the middle does: emulate the render tail and stop there, or abort at once. */
  onPause?: 'drawAndStop' | 'abort';
}

interface Slot {
  frame: number;
  state: Uint8Array<ArrayBuffer>;
}

export class Emulator {
  running = false;
  speed = 1;
  /** Live controller input per port, set from the main thread. */
  readonly live = new Uint32Array(4);
  readonly movie = new Movie();
  readonly snapshots = new Snapshots();
  readonly search = new MemorySearch();
  readonly scenes: Scenes;
  readonly pins: Pins;
  cheats: Cheat[] = [];
  watches: Watch[] = [];
  audioPort: MessagePort | null = null;
  romName = '';

  private readonly slots: (Slot | null)[] = Array(10).fill(null);
  /** Per-frame lag flags of the current timeline, for the piano roll. */
  private lag = new Uint8Array(1 << 16);
  private recorder: Recorder | null = null;
  private lastInput = 0;
  private loopGeneration = 0;
  private readonly wake = new MessageChannel();
  private replaying = false;
  private seekTarget: number | null = null;
  private readonly yielder = new MessageChannel();
  private nextTime = 0;
  private timer = 0;
  private resumeTimer = 0;
  /** Start of the next scene: the state SEEK_RENDER_TAIL frames before it and the inputs since. */
  private sceneLeadIn: { start: number; state: PageState; inputs: Uint32Array; frames: number } | null = null;
  private fpsFrames = 0;
  private fpsStart = 0;
  private fps = 0;
  private lastStatus = 0;
  private busy: Promise<void> = Promise.resolve();

  constructor(
    readonly core: Core,
    readonly renderer: Renderer,
    private readonly canvas: OffscreenCanvas,
    private readonly onStatus: (s: Status) => void,
    private readonly onError: (message: string) => void,
    onScenes: () => void,
    onPins: (pins: PinInfo[]) => void,
  ) {
    this.scenes = new Scenes(onScenes);
    this.pins = new Pins(onPins);
    this.wake.port1.onmessage = (e: MessageEvent<number>) => this.runTick(e.data);
  }

  // ------------------------------------------------------------ frames

  /** Input for `frame`, or null when playback reached the end of the movie. */
  private inputFor(frame: number): Uint32Array | null {
    const recorded = this.movie.get(frame);
    if (this.replaying) return recorded ?? this.live;
    if (this.movie.mode === 'play') return recorded;
    if (this.movie.mode === 'record') {
      if (!recorded || recorded.some((v, i) => v !== this.live[i]) || frame + 1 < this.movie.length) {
        this.invalidateAfter(frame);
      }
      this.movie.record(frame, this.live);
    }
    return this.live;
  }

  /** Emulates one frame. Resolves once the recorder accepted it. */
  /** Emulates one frame with `forced` input when given, else the movie or live input. */
  /** Emulates one frame. Returns false when playback reached the end of the movie instead. */
  private async step(render = true, forced?: Uint32Array): Promise<boolean> {
    const frame = this.core.frame;
    const input = forced ?? this.inputFor(frame);
    if (!input) {
      // Playback stops at the end so the next frame can be recorded deliberately.
      this.pause();
      return false;
    }
    this.trackSceneLeadIn(frame, input);
    this.lastInput = input[0];
    for (let port = 0; port < 4; port++) this.core.setPad(port, input[port]);
    applyCheats(this.core, this.cheats);
    this.core.setRendering(render);
    const start = this.core.time;
    const polled = this.core.runFrame();
    if (frame >= this.lag.length) {
      const grown = new Uint8Array(this.lag.length * 2);
      grown.set(this.lag);
      this.lag = grown;
    }
    this.lag[frame] = polled ? 0 : 1;
    if (render) this.renderer.frame(this.core);
    else this.core.gpuClear();
    const audio = this.core.takeAudio();
    if (this.core.frame % this.snapshots.interval === 0) this.snapshots.capture(this.core.frame, this.core);
    const lead = this.sceneLeadIn;
    if (lead?.frames === SEEK_RENDER_TAIL) {
      this.sceneLeadIn = null;
      // The capture is queued right after this frame's draw, so the thumbnail
      // shows this frame while the loop goes on without waiting for it.
      if (render) this.scenes.add(this.core.frame, lead.state, lead.inputs, this.renderer.capture()).catch(this.reportError);
    }
    if (render && this.pins.isStale(this.core.frame)) this.pins.capture(this.core.frame, this.renderer.capture()).catch(this.reportError);
    this.fpsFrames++;
    if (this.recorder && render && !this.replaying) {
      await this.recorder.add(audio.samples, audio.rate, this.core.time - start);
    } else if (this.audioPort && this.running && this.speed === 1 && !this.replaying && audio.samples.length > 0) {
      this.audioPort.postMessage(audio, [audio.samples.buffer]);
    }
    return true;
  }

  /** Collects the frames leading to the next scene. Any jump in between, which
   * replays and loads cause, drops the lead-in and with it that scene. */
  private trackSceneLeadIn(frame: number, input: Uint32Array): void {
    if (this.replaying) {
      this.sceneLeadIn = null;
      return;
    }
    if (frame % SCENE_INTERVAL === SCENE_INTERVAL - SEEK_RENDER_TAIL) {
      this.sceneLeadIn = {
        start: frame,
        state: this.snapshots.capture(frame, this.core),
        inputs: new Uint32Array(SEEK_RENDER_TAIL * 4),
        frames: 0,
      };
    }
    const lead = this.sceneLeadIn;
    if (!lead) return;
    if (lead.start + lead.frames !== frame) {
      this.sceneLeadIn = null;
      return;
    }
    lead.inputs.set(input.subarray(0, 4), lead.frames * 4);
    lead.frames++;
  }

  private reportError = (e: unknown): void => {
    this.onError(e instanceof Error ? e.message : String(e));
  };

  /** Whether to draw the frame about to run on the way to `target`. A snapshot
   * restore blanks the framebuffers, so the frames before the target and before
   * any pin awaiting a screenshot are drawn. */
  private renderTowards(target: number): boolean {
    return target - this.core.frame <= SEEK_RENDER_TAIL || this.pins.staleWithin(this.core.frame, SEEK_RENDER_TAIL);
  }

  /** Serializes control operations against the frame loop. */
  exclusive<T>(fn: () => Promise<T> | T): Promise<T> {
    const result = this.busy.then(fn);
    this.busy = result.then(
      () => undefined,
      () => undefined,
    );
    return result;
  }

  start(): void {
    if (this.running) return;
    this.running = true;
    this.loopGeneration++;
    this.nextTime = performance.now();
    this.fpsStart = this.nextTime;
    this.fpsFrames = 0;
    this.schedule(0);
  }

  pause(): void {
    this.running = false;
    this.loopGeneration++;
    clearTimeout(this.timer);
    clearTimeout(this.resumeTimer);
    // What was recorded since resuming, or one frame advance, is one undo step.
    this.movie.endRecording();
    this.emitStatus(true);
  }

  /** Starts the loop after RESUME_DELAY_MS; a pause in between cancels it. */
  resumeSoon(): void {
    clearTimeout(this.resumeTimer);
    this.resumeTimer = setTimeout(() => this.start(), RESUME_DELAY_MS) as unknown as number;
  }

  frameAdvance(): Promise<void> {
    this.pause();
    return this.exclusive(async () => {
      await this.step();
      this.emitStatus(true);
    });
  }

  /** Runs `count` frames with the current input while paused. */
  runFrames(count: number): Promise<void> {
    this.pause();
    return this.exclusive(async () => {
      const target = this.core.frame + count;
      while (this.core.frame < target && (await this.step(this.renderTowards(target))));
      this.emitStatus(true);
    });
  }

  /** Queues the next tick. Zero delays go through a MessageChannel because
   * chained setTimeout calls are clamped to 4 ms. */
  private schedule(delay: number): void {
    const generation = this.loopGeneration;
    if (delay >= 1) this.timer = setTimeout(() => this.runTick(generation), delay) as unknown as number;
    else this.wake.port2.postMessage(generation);
  }

  private runTick(generation: number): void {
    // A pause or restart since scheduling supersedes this tick.
    if (generation !== this.loopGeneration) return;
    this.exclusive(this.tick).catch((e: unknown) => {
      this.pause();
      this.reportError(e);
    });
  }

  private tick = async (): Promise<void> => {
    if (!this.running) return;
    const frameMs = 1000 / FPS / this.speed;
    // Every slice ends by the deadline so the worker can present the canvas
    // and answer messages even when a frame takes longer than real time.
    const deadline = performance.now() + TICK_BUDGET_MS;
    while (this.running && performance.now() < deadline && (this.speed === Infinity || performance.now() >= this.nextTime)) {
      await this.step();
      this.nextTime += frameMs;
      if (this.recorder) break;
    }
    const now = performance.now();
    // Running slower than real time: drop the backlog instead of chasing it.
    if (now - this.nextTime > frameMs) this.nextTime = now;
    if (now - this.fpsStart >= 500) {
      this.fps = (this.fpsFrames * 1000) / (now - this.fpsStart);
      this.fpsStart = now;
      this.fpsFrames = 0;
    }
    this.emitStatus(false);
    if (this.running) this.schedule(this.recorder ? 0 : Math.max(0, this.nextTime - now));
  };

  // ------------------------------------------------------------ power

  loadRom(rom: Uint8Array, name: string): void {
    this.core.loadRom(rom);
    this.romName = name;
    this.afterPowerOn();
  }

  reset(): void {
    this.core.reset();
    this.afterPowerOn();
  }

  /** Power cycle from the given cartridge saves, or erased ones, so a movie replays identically wherever it is loaded. */
  powerOn(save?: Uint8Array): void {
    this.core.powerOn(save);
    this.afterPowerOn();
  }

  private afterPowerOn(): void {
    this.snapshots.clear();
    this.snapshots.capture(0, this.core);
    this.scenes.clear();
    this.pins.clear();
    this.slots.fill(null);
    this.renderer.clearSlots();
  }

  // ------------------------------------------------------------ savestates

  saveSlot(slot: number): void {
    this.slots[slot] = { frame: this.core.frame, state: this.core.saveState() };
    this.renderer.saveSlot(slot);
    this.emitStatus(true);
  }

  loadSlot(slot: number): boolean {
    const s = this.slots[slot];
    if (!s) return false;
    this.restore(s.state);
    this.renderer.loadSlot(slot, this.core);
    this.emitStatus(true);
    return true;
  }

  exportState(): Uint8Array<ArrayBuffer> {
    return this.core.saveState();
  }

  /** Loads a state from a file. The picture stays stale until the next frame. With `asSnapshot`, the state is also
   * kept as the snapshot of its frame, for a state known to lie on the movie (such as a copy of another emulator
   * playing the same movie), so seeks and edits just after it replay from it instead of from power-on. */
  importState(state: Uint8Array, asSnapshot = false): void {
    this.restore(state);
    if (asSnapshot) this.snapshots.captureLoaded(this.core.frame, this.core);
    this.renderer.frame(this.core);
    this.emitStatus(true);
  }

  private restore(state: Uint8Array): void {
    if (!this.core.loadState(state)) throw new Error('This savestate is from a different core version');
    // A load can land on the very frame the lead-in expects next, which the frame check would miss.
    this.sceneLeadIn = null;
    this.countRerecord();
  }

  /** Jumps to a scene of the history by replaying its lead-in with drawing,
   * so the screen shows the scene instead of the blank a restore leaves. */
  async restoreScene(id: number): Promise<void> {
    const scene = this.scenes.get(id);
    if (!scene) throw new Error('This scene is no longer in the history');
    this.snapshots.restore(scene.state, this.core);
    // Play continues from the scene, so what was recorded after it may not
    // match the timeline from here, e.g. without a movie to replay.
    this.invalidateAfter(scene.frame);
    this.countRerecord();
    this.replaying = true;
    try {
      for (let i = 0; i < scene.inputs.length; i += 4) await this.step(true, scene.inputs.subarray(i, i + 4));
    } finally {
      this.replaying = false;
    }
    this.emitStatus(true);
  }

  private countRerecord(): void {
    if (this.movie.mode === 'record') this.movie.rerecords++;
  }

  /** Moves to `target` by restoring the closest snapshot and replaying the movie.
   * `recompute` always restores, so edited inputs before the current frame take effect. */
  async seek(target: number, recompute = false): Promise<void> {
    await this.replayTo(target, { recompute, progress: true, onPause: 'drawAndStop' });
    this.emitStatus(true);
    this.refreshPinsLater();
  }

  /** Returns false when a pause aborted it; the machine is then mid-way and the caller restores it. */
  private async replayTo(target: number, { recompute = false, pastEnd, progress = false, onPause }: Replay = {}): Promise<boolean> {
    const current = this.core.frame;
    // Start a few frames early so the replayed tail redraws the framebuffers.
    let from = Math.max(0, target - SEEK_RENDER_TAIL);
    let nearest = this.snapshots.nearestFrame(from);
    if (nearest === null || target - nearest > MAX_TAIL_REPLAY) {
      from = target;
      nearest = this.snapshots.nearestFrame(target);
    }
    const resume = !recompute && target >= current && (nearest === null || nearest <= current);
    if (!resume && this.snapshots.restoreNearest(from, this.core) === null) {
      throw new Error(`No snapshot at or before frame ${target}`);
    }
    this.replaying = true;
    this.seekTarget = progress ? target : null;
    const generation = this.loopGeneration;
    let sliceEnd = performance.now() + REPLAY_SLICE_MS;
    try {
      while (
        this.core.frame < target &&
        (await this.step(this.renderTowards(target), pastEnd && (this.movie.get(this.core.frame) ?? pastEnd)))
      ) {
        if (!onPause || performance.now() < sliceEnd) continue;
        if (progress) this.emitStatus(true);
        await this.yieldToMessages();
        if (this.loopGeneration !== generation) {
          if (onPause === 'abort') return false;
          target = Math.min(target, this.core.frame + SEEK_RENDER_TAIL);
          onPause = undefined;
        }
        sliceEnd = performance.now() + REPLAY_SLICE_MS;
      }
    } finally {
      this.replaying = false;
      this.seekTarget = null;
    }
    if (this.core.frame === current) this.renderer.frame(this.core);
    return true;
  }

  /** Lets queued messages such as a pause run in the middle of a long replay. */
  private yieldToMessages(): Promise<void> {
    return new Promise((resolve) => {
      this.yielder.port1.onmessage = () => resolve();
      this.yielder.port2.postMessage(null);
    });
  }

  togglePin(frame: number): void {
    if (this.pins.toggle(frame)) this.refreshPinsLater();
  }

  /** Queues screenshots of the pins awaiting one behind the current operation,
   * so that operation answers first. A pause or seek before or during it cancels
   * it, and the pins wait for the next refresh or for being emulated. */
  private refreshPinsLater(): void {
    const generation = this.loopGeneration;
    this.exclusive(async () => {
      if (this.loopGeneration === generation) await this.refreshPins();
    }).catch(this.reportError);
  }

  /** Takes the screenshot of every pin awaiting one, past the current frame too. */
  private async refreshPins(): Promise<void> {
    const frames = this.pins.staleFrames();
    if (frames.length === 0) return;
    await this.visit(
      frames,
      (frame) => {
        // Replaying draws only frames it emulates, so the frame it started on is taken here.
        if (this.pins.isStale(frame)) this.pins.capture(frame, this.renderer.capture()).catch(this.reportError);
      },
      true,
    );
  }

  /** Calls `read` at each of `frames` and returns the results in the same order,
   * leaving the current frame, screen and movie as they were. */
  async peek<T>(frames: number[], read: () => T): Promise<T[]> {
    const values = new Map<number, T>();
    await this.visit(frames, (frame) => values.set(frame, read()));
    return frames.map((frame) => values.get(frame)!);
  }

  /** Replays the movie to each of `frames` and calls `at` there, then puts the
   * machine back exactly as it was. Past the end of the movie no buttons are
   * pressed, so a visit gives the same result whenever it is made. An
   * `abortable` visit stops at the next pause, which a seek also makes. */
  private async visit(frames: number[], at: (frame: number) => void, abortable = false): Promise<void> {
    const running = this.running;
    this.pause();
    const current = this.core.frame;
    const state = this.core.saveState();
    this.renderer.saveSlot(VISIT_SLOT);
    let aborted = false;
    try {
      for (const frame of [...new Set(frames)].sort((a, b) => a - b)) {
        aborted = !(await this.replayTo(frame, { pastEnd: NO_INPUT, onPause: abortable ? 'abort' : undefined }));
        if (aborted) break;
        at(frame);
      }
    } finally {
      this.core.loadState(state);
      this.renderer.loadSlot(VISIT_SLOT, this.core);
      // Without a movie the visited frames are not the timeline the live input will make.
      if (this.movie.mode === 'off') this.snapshots.invalidateAfter(current);
    }
    this.emitStatus(true);
    // The pause that aborted the visit decides whether the loop runs.
    if (running && !aborted) this.start();
  }

  /** Drops the snapshots and scenes after `frame` and marks later pin screenshots
   * stale because the timeline from there changed. */
  private invalidateAfter(frame: number): void {
    this.snapshots.invalidateAfter(frame);
    this.scenes.invalidateAfter(frame);
    this.pins.invalidateAfter(frame);
  }

  lagFlags(start: number, count: number): Uint8Array<ArrayBuffer> {
    return this.lag.slice(start, start + count);
  }

  /** Lag frames of the current timeline before the current frame. */
  private countLag(): number {
    let n = 0;
    for (let f = 0; f < this.core.frame && f < this.lag.length; f++) n += this.lag[f];
    return n;
  }

  /** 1 for each frame in the range that has an automatic snapshot. */
  snapshotFlags(start: number, count: number): Uint8Array<ArrayBuffer> {
    const flags = new Uint8Array(count);
    for (const f of this.snapshots.info().frames) if (f >= start && f < start + count) flags[f - start] = 1;
    return flags;
  }

  /** Applies a movie edit at `frame` and re-emulates up to the current frame
   * when the edit lies in the past, so the screen shows its effect. */
  async afterMovieEdit(frame: number): Promise<void> {
    this.invalidateAfter(frame);
    if (frame < this.core.frame) {
      this.pause();
      await this.replayTo(this.core.frame, { recompute: true, progress: true, onPause: 'drawAndStop' });
    }
    this.emitStatus(true);
    this.refreshPinsLater();
  }

  // ------------------------------------------------------------ recording

  startRecording(): void {
    if (this.recorder) return;
    this.recorder = new Recorder(this.canvas);
    this.emitStatus(true);
  }

  async stopRecording(): Promise<ArrayBuffer> {
    const r = this.recorder;
    if (!r) throw new Error('Not recording');
    this.recorder = null;
    this.emitStatus(true);
    return r.finish();
  }

  // ------------------------------------------------------------ status

  readWatch(w: Watch): number {
    return readValue(this.core.rdram(), w.addr, w.type);
  }

  emitStatus(force: boolean): void {
    const now = performance.now();
    if (!force && now - this.lastStatus < STATUS_INTERVAL_MS) return;
    this.lastStatus = now;
    const snaps = this.snapshots.info();
    this.onStatus({
      frame: this.core.frame,
      fps: this.fps,
      running: this.running,
      speed: this.speed,
      lagFrames: this.countLag(),
      movie: this.movie.info(),
      snapshots: snaps.frames.length,
      snapshotBytes: snaps.bytes,
      recording: this.recorder ? this.recorder.frameCount : null,
      watchValues: this.watches.map((w) => this.readWatch(w)),
      slots: this.slots.map((s) => (s ? s.frame : null)),
      input: this.lastInput,
      seekTarget: this.seekTarget,
    });
  }
}
