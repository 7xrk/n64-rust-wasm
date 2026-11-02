/** Main-thread handle to the emulation worker plus the state the UI renders. */

import { effect, signal } from '@preact/signals';
import { SCALES } from 'n64core';
import type { Cheat } from '../emu/cheats';
import type { Status, Watch } from '../emu/emulator';
import type { PinInfo } from '../emu/pins';
import { connect, transfer } from '../emu/rpc';
import type { SceneInfo } from '../emu/scenes';
import type { WorkerApi, WorkerEvent } from '../emu/worker';
import { createAudioOutput } from './audio';
import { InputManager, loadMapping } from './input';
import { initNsoN64 } from './nso-n64';

export const status = signal<Status>({
  frame: 0,
  fps: 0,
  running: false,
  speed: 1,
  lagFrames: 0,
  movie: { mode: 'off', length: 0, rerecords: 0, author: '', canUndo: false, canRedo: false, revision: 0 },
  snapshots: 0,
  snapshotBytes: 0,
  recording: null,
  watchValues: [],
  slots: [],
  input: 0,
  seekTarget: null,
});
export const errorMessage = signal<string | null>(null);
export const romName = signal<string | null>(null);
export const watches = signal<Watch[]>([]);
export const cheats = signal<Cheat[]>([]);
/** Scene history, oldest first. */
export const scenes = signal<SceneInfo[]>([]);
/** Pinned timeline frames with their screenshots, in frame order. */
export const pins = signal<PinInfo[]>([]);

const worker = new Worker(new URL('../emu/worker.ts', import.meta.url), { type: 'module' });

export const emu = connect<WorkerApi, WorkerEvent>(worker, (event) => {
  if (event.type === 'status') status.value = event.status;
  else if (event.type === 'scenes') {
    run(emu.scenes()).then((list) => {
      if (list) scenes.value = list;
    });
  } else if (event.type === 'pins') pins.value = event.pins;
  else errorMessage.value = event.message;
});

/** Keyboard and gamepad state forwarded to controller 1. */
export const liveInput = new InputManager(loadMapping(), (pad) => {
  // The worker rejects calls until attachCanvas has initialized it.
  ready?.then(() => run(emu.setInput(0, pad)));
});

initNsoN64();

/** Runs a worker call and shows its error instead of throwing. */
export async function run<T>(p: Promise<T>): Promise<T | undefined> {
  try {
    return await p;
  } catch (e) {
    errorMessage.value = e instanceof Error ? e.message : String(e);
    return undefined;
  }
}

function storageKey(kind: string): string | null {
  return romName.value ? `n64tas.${kind}.${romName.value}` : null;
}

function load<T>(kind: string, fallback: T): T {
  const key = storageKey(kind);
  try {
    const raw = key && localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

function save(kind: string, value: unknown): void {
  const key = storageKey(kind);
  try {
    if (key) localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Persistence is a convenience; the session keeps working without it.
  }
}

effect(() => {
  const list = cheats.value;
  if (!romName.peek()) return;
  save('cheats', list);
  run(emu.setCheats(list));
});

effect(() => {
  const list = watches.value;
  if (!romName.peek()) return;
  save('watches', list);
  run(emu.setWatches(list));
});

const SCALE_STORAGE = 'n64tas.scale';

function loadScale(): number {
  try {
    const n = Number(localStorage.getItem(SCALE_STORAGE));
    return SCALES.includes(n) ? n : 1;
  } catch {
    return 1;
  }
}

/** Internal resolution multiplier, kept in this browser across sessions. */
export const renderScale = signal(loadScale());

export function setRenderScale(scale: number): void {
  renderScale.value = scale;
  try {
    localStorage.setItem(SCALE_STORAGE, String(scale));
  } catch {
    // Persistence is a convenience; the session keeps working without it.
  }
  run(emu.setScale(scale));
}

let ready: Promise<void> | null = null;
let audioStarted = false;

/** Hands the canvas to the worker. */
export function attachCanvas(canvas: HTMLCanvasElement): Promise<void> {
  ready ??= (async () => {
    const offscreen = canvas.transferControlToOffscreen();
    await emu.init(transfer(offscreen, [offscreen]));
    await emu.setScale(renderScale.peek());
  })().catch((e: Error) => {
    errorMessage.value = e.message;
  });
  return ready;
}

/** Audio needs a user gesture, so it starts with the first ROM load. */
async function startAudio(): Promise<void> {
  if (audioStarted) return;
  audioStarted = true;
  const { port } = await createAudioOutput();
  await emu.setAudioPort(transfer(port, [port]));
}

export async function loadRom(file: File): Promise<void> {
  await ready;
  await startAudio();
  const data = await file.arrayBuffer();
  await run(emu.loadRom(transfer(data, [data]), file.name));
  romName.value = file.name;
  errorMessage.value = null;
  cheats.value = load('cheats', []);
  watches.value = load('watches', []);
}

/** A file the app opens: the extensions its picker accepts and its loader. */
export interface FileKind {
  accept: string;
  open: (file: File) => Promise<unknown>;
}

export const ROM_FILE: FileKind = { accept: '.z64,.n64,.v64,.bin', open: loadRom };
export const STATE_FILE: FileKind = { accept: '.st64', open: async (f) => run(emu.importState(await f.arrayBuffer())) };
export const MOVIE_FILE: FileKind = { accept: '.n64tas', open: async (f) => run(emu.importMovie(await f.arrayBuffer())) };

/** Kinds in the order dropped files open, so a ROM is loaded before a state or movie for it. */
const FILE_KINDS = [ROM_FILE, STATE_FILE, MOVIE_FILE];

function kindOf(file: File): FileKind | undefined {
  const ext = file.name.slice(file.name.lastIndexOf('.')).toLowerCase();
  return FILE_KINDS.find((k) => k.accept.split(',').includes(ext));
}

/** Opens dropped files one after another with the loader of each extension. */
export async function openFiles(files: File[]): Promise<void> {
  const unknown = files.filter((f) => !kindOf(f));
  if (unknown.length > 0) errorMessage.value = `Cannot open: ${unknown.map((f) => f.name).join(', ')}`;
  for (const kind of FILE_KINDS) {
    for (const file of files) if (kindOf(file) === kind) await kind.open(file);
  }
}

export function download(data: BlobPart, name: string, type = 'application/octet-stream'): void {
  const url = URL.createObjectURL(new Blob([data], { type }));
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

export function baseName(): string {
  return (romName.value ?? 'n64').replace(/\.[^.]+$/, '');
}

export function hex(addr: number): string {
  return `0x${(addr >>> 0).toString(16).toUpperCase().padStart(8, '0')}`;
}

/** Physical RDRAM address as the KSEG0 address games and cheat codes use. */
export function kseg0(paddr: number): number {
  return (0x8000_0000 | paddr) >>> 0;
}

export function parseAddress(text: string): number {
  const v = Number.parseInt(text.trim().replace(/^0x/i, ''), 16);
  if (!Number.isFinite(v)) throw new Error(`Invalid address: ${text}`);
  return v >>> 0;
}
