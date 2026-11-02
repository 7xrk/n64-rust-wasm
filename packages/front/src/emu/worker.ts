/// <reference lib="webworker" />
/** Worker entry: owns the core, the WebGPU device and the frame loop. */

import wasmUrl from 'n64core/n64core.wasm?url';
import { Core, RDRAM_SIZE, Renderer, readBE, SCALES } from 'n64core';
import type { Cheat } from './cheats';
import { Emulator, type Status, type Watch } from './emulator';
import type { CompareOp, ValueType } from './memsearch';
import { readValue, typeSize } from './memsearch';
import { Movie, type MovieMode } from './movie';
import type { PinInfo } from './pins';
import { expose, transfer } from './rpc';

declare const self: DedicatedWorkerGlobalScope;

export type WorkerEvent =
  | { type: 'status'; status: Status }
  | { type: 'error'; message: string }
  | { type: 'scenes' }
  | { type: 'pins'; pins: PinInfo[] };

const emit = (event: WorkerEvent) => self.postMessage({ event });

let emu: Emulator | null = null;

function need(): Emulator {
  if (!emu) throw new Error('Emulator is not initialized');
  return emu;
}

/** Runs `fn` with the frame loop held off. */
function locked<T>(fn: (e: Emulator) => T | Promise<T>): Promise<T> {
  const e = need();
  return e.exclusive(() => fn(e));
}

function readBytes(e: Emulator, addr: number, length: number): Uint8Array<ArrayBuffer> {
  const rdram = e.core.rdram();
  const base = toPhys(e, addr);
  const out = new Uint8Array(Math.min(length, 4096));
  for (let i = 0; i < out.length; i++) out[i] = readBE(rdram, base + i, 1);
  return out;
}

/** Accepts KSEG0/KSEG1, TLB-mapped virtual or physical addresses. */
function toPhys(e: Emulator, addr: number): number {
  const a = addr >>> 0;
  if (a < RDRAM_SIZE) return a;
  if (a >= 0x8000_0000 && a < 0xc000_0000) return a & 0x1fff_ffff;
  const p = e.core.virtToPhys(a);
  if (p === null) throw new Error(`Address ${a.toString(16)} is not mapped`);
  return p;
}

const api = {
  async init(canvas: OffscreenCanvas) {
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw new Error('WebGPU is not available in this browser');
    const device = await adapter.requestDevice({
      // Frame memory at the highest internal resolution needs large buffers.
      requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
      },
    });
    device.lost.then((info) => emit({ type: 'error', message: `GPU device lost: ${info.message}` }));
    device.addEventListener('uncapturederror', (e) => emit({ type: 'error', message: (e as GPUUncapturedErrorEvent).error.message }));
    const context = canvas.getContext('webgpu');
    if (!context) throw new Error('Failed to create a WebGPU canvas context');
    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format, alphaMode: 'opaque' });
    const core = await Core.load(wasmUrl);
    emu = new Emulator(
      core,
      new Renderer(device, context, format),
      canvas,
      (status) => emit({ type: 'status', status }),
      (message) => emit({ type: 'error', message }),
      () => emit({ type: 'scenes' }),
      (pins) => emit({ type: 'pins', pins }),
    );
  },

  setAudioPort(port: MessagePort) {
    need().audioPort = port;
  },

  async loadRom(rom: ArrayBuffer, name: string) {
    const e = need();
    e.pause();
    await locked(() => e.loadRom(new Uint8Array(rom), name));
    e.movie.mode = 'off';
    e.start();
  },

  setInput(port: number, value: number) {
    need().live[port] = value;
  },

  setConnected(mask: number) {
    need().core.setConnected(mask);
  },

  pause() {
    need().pause();
  },

  resume() {
    need().resumeSoon();
  },

  frameAdvance() {
    return need().frameAdvance();
  },

  runFrames(count: number) {
    return need().runFrames(count);
  },

  reset() {
    return locked((e) => {
      e.reset();
      e.emitStatus(true);
    });
  },

  setSpeed(speed: number) {
    const e = need();
    e.speed = speed;
    e.emitStatus(true);
  },

  // ---------------------------------------------------------------- states

  saveSlot(slot: number) {
    return locked((e) => e.saveSlot(slot));
  },

  loadSlot(slot: number) {
    return locked((e) => e.loadSlot(slot));
  },

  exportState() {
    return locked((e) => {
      const state = e.exportState();
      return transfer(state, [state.buffer]);
    });
  },

  importState(state: ArrayBuffer, asSnapshot = false) {
    return locked((e) => e.importState(new Uint8Array(state), asSnapshot));
  },

  seek(frame: number) {
    const e = need();
    e.pause();
    return locked(() => e.seek(Math.max(0, frame)));
  },

  /** Scene history, oldest first. */
  scenes() {
    return need().scenes.info();
  },

  /** Jumps to a scene of the history and resumes from there after the resume delay. */
  async restoreScene(id: number) {
    const e = need();
    e.pause();
    await locked(() => e.restoreScene(id));
    e.resumeSoon();
  },

  togglePin(frame: number) {
    return locked((e) => e.togglePin(frame));
  },

  /** Sets the internal resolution multiplier, one of SCALES. */
  setScale(scale: number) {
    if (!SCALES.includes(scale)) throw new Error(`Unsupported scale ${scale}`);
    return locked((e) => {
      e.renderer.setScale(scale);
      e.renderer.frame(e.core);
    });
  },

  setSnapshotInterval(frames: number) {
    need().snapshots.interval = Math.max(1, Math.floor(frames));
  },

  // ---------------------------------------------------------------- movie

  /** Starts a fresh recording from power-on, keeping the current cartridge saves in the movie. */
  newMovie(author: string) {
    return locked((e) => {
      const save = e.core.saveData();
      e.powerOn(save);
      e.movie.clear();
      e.movie.save = save;
      e.movie.author = author;
      e.movie.mode = 'record';
      e.emitStatus(true);
    });
  },

  setMovieMode(mode: MovieMode) {
    const e = need();
    e.movie.mode = mode;
    e.emitStatus(true);
  },

  movieSlice(start: number, count: number) {
    return locked((e) => {
      const inputs = e.movie.slice(start, count);
      const lag = e.lagFlags(start, count);
      const snaps = e.snapshotFlags(start, count);
      return transfer({ inputs, lag, snaps }, [inputs.buffer, lag.buffer, snaps.buffer]);
    });
  },

  /** Overwrites one port for consecutive frames as a single undoable edit. */
  editInputs(frame: number, port: number, values: number[]) {
    return locked((e) => {
      e.movie.setRange(frame, port, values);
      return e.afterMovieEdit(frame);
    });
  },

  spliceMovie(frame: number, deleteCount: number, insertCount: number) {
    return locked((e) => {
      e.movie.splice(frame, deleteCount, insertCount);
      return e.afterMovieEdit(frame);
    });
  },

  undoMovie() {
    return locked(async (e) => {
      const frame = e.movie.undo();
      if (frame !== null) await e.afterMovieEdit(frame);
    });
  },

  redoMovie() {
    return locked(async (e) => {
      const frame = e.movie.redo();
      if (frame !== null) await e.afterMovieEdit(frame);
    });
  },

  /** `snapshots` packs the automatic snapshots along. They follow the live input rather than the movie while the
   * movie is off, so they are left out then. */
  exportMovie(snapshots = true) {
    return locked(async (e) => {
      const packed = snapshots && e.movie.mode !== 'off' ? await e.snapshots.pack(e.movie.length, e.core.build) : undefined;
      const bytes = e.movie.serialize(e.romName, packed);
      return transfer(bytes, [bytes.buffer]);
    });
  },

  /** Loads a movie and plays it from power-on with the saves it carries. */
  importMovie(bytes: ArrayBuffer) {
    return locked(async (e) => {
      const { movie, snapshots } = Movie.parse(new Uint8Array(bytes));
      e.powerOn(movie.save ?? undefined);
      if (snapshots) await e.snapshots.unpack(snapshots, e.core.build);
      e.movie.clear();
      e.movie.save = movie.save;
      for (let f = 0; f < movie.length; f++) e.movie.record(f, movie.get(f)!);
      e.movie.rerecords = movie.rerecords;
      e.movie.author = movie.author;
      e.movie.mode = 'play';
      e.emitStatus(true);
    });
  },

  // ---------------------------------------------------------------- recording

  startRecording() {
    need().startRecording();
  },

  async stopRecording() {
    const buffer = await need().stopRecording();
    return transfer(buffer, [buffer]);
  },

  async screenshot() {
    const e = need();
    const image = await e.renderer.capture();
    const canvas = new OffscreenCanvas(image.width, image.height);
    canvas.getContext('2d')!.putImageData(new ImageData(image.data, image.width, image.height), 0, 0);
    const png = await (await canvas.convertToBlob({ type: 'image/png' })).arrayBuffer();
    return transfer(png, [png]);
  },

  // ---------------------------------------------------------------- cheats / memory

  setCheats(cheats: Cheat[]) {
    need().cheats = cheats;
  },

  setWatches(watches: Watch[]) {
    const e = need();
    e.watches = watches.map((w) => ({ ...w, addr: toPhys(e, w.addr) }));
    e.emitStatus(true);
  },

  searchReset(type: ValueType) {
    return locked((e) => e.search.reset(e.core.rdram(), type));
  },

  searchFilter(op: CompareOp, operand?: number) {
    return locked((e) => e.search.filter(e.core.rdram(), op, operand));
  },

  searchResults(limit: number) {
    return locked((e) => e.search.results(e.core.rdram(), limit));
  },

  /** Big-endian bytes starting at `addr`. */
  readMemory(addr: number, length: number) {
    return locked((e) => {
      const out = readBytes(e, addr, length);
      return transfer(out, [out.buffer]);
    });
  },

  /** Values at each of `frames`, read without moving the current frame or screen.
   * Past the end of the movie the frames run with no buttons pressed. */
  peek(frames: number[], reads: { addr: number; type: ValueType }[]) {
    return locked((e) => e.peek(frames, () => reads.map((r) => readValue(e.core.rdram(), toPhys(e, r.addr), r.type))));
  },

  /** readMemory at `frame`, as peek reads it. */
  peekBytes(frame: number, addr: number, length: number) {
    return locked(async (e) => {
      const [out] = await e.peek([frame], () => readBytes(e, addr, length));
      return transfer(out, [out.buffer]);
    });
  },

  /** One value at `addr`, decoded as `type`. */
  readValue(addr: number, type: ValueType) {
    return locked((e) => readValue(e.core.rdram(), toPhys(e, addr), type));
  },

  writeMemory(addr: number, value: number, type: ValueType) {
    return locked((e) => {
      const p = toPhys(e, addr);
      const size = typeSize(type);
      let raw = value;
      if (type === 'f32') raw = new Uint32Array(new Float32Array([value]).buffer)[0];
      e.core.poke(p, raw >>> 0, size);
      e.emitStatus(true);
    });
  },
};

export type WorkerApi = typeof api;

expose(api, self);
