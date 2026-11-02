/** `window.n64`: a handle for AI agents driving the app from page scripts. It
 * shares the app's worker and state, and adds what a page script cannot reach
 * on its own: files by URL, the rendered screen, pad encoding and waiting on
 * emulation. `n64.help` lists everything. */

import { BUTTON_BIT, BUTTONS, type ButtonId, packPad, padButtons, stickX, stickY } from 'n64core';
import type { Status } from '../emu/emulator';
import { VALUE_TYPES } from '../emu/memsearch';
import type { WorkerApi } from '../emu/worker';
import { baseName, cheats, download, emu, errorMessage, loadRom, pins, romName, scenes, status, watches } from './client';

/** Signatures of the worker calls worth knowing, keyed by WorkerApi so a renamed call fails to compile. */
const CALLS: Partial<Record<keyof WorkerApi, string>> = {
  pause: '(): Promise<void>',
  resume: '(): Promise<void>',
  frameAdvance: '(): Promise<void> // one frame with the live input',
  runFrames: '(count: number): Promise<void> // while paused; stops at the end of a played movie',
  seek: '(frame: number): Promise<void> // restores the nearest snapshot and replays the movie',
  setSpeed: '(speed: number): Promise<void> // 1 = realtime, Infinity = unlimited',
  setInput: '(port: number, pad: Pad): Promise<void> // live input, used outside movie playback',
  reset: '(): Promise<void> // soft reset; cartridge saves survive',
  saveSlot: '(slot: number): Promise<void>',
  loadSlot: '(slot: number): Promise<boolean> // false when the slot is empty',
  exportState: '(): Promise<Uint8Array>',
  importState:
    '(state: ArrayBuffer, asSnapshot?: boolean): Promise<void> // asSnapshot keeps it as the snapshot of its frame; only for a state on the movie',
  scenes: '(): Promise<{ id: number; frame: number; thumbnail: Blob }[]>',
  restoreScene: '(id: number): Promise<void>',
  togglePin: '(frame: number): Promise<void> // the screenshot follows in the background; the next seek or pause cancels it',
  newMovie: '(author: string): Promise<void> // powers on from the current saves and records',
  setMovieMode: "(mode: 'off' | 'record' | 'play'): Promise<void>",
  movieSlice: '(start: number, count: number): Promise<{ inputs: Uint32Array /* 4 pads per frame */; lag: Uint8Array; snaps: Uint8Array }>',
  editInputs: '(frame: number, port: number, pads: Pad[]): Promise<void> // one undo step; recomputes up to the current frame',
  spliceMovie: '(frame: number, deleteCount: number, insertCount: number): Promise<void>',
  undoMovie: '(): Promise<void>',
  redoMovie: '(): Promise<void>',
  exportMovie: '(snapshots = true): Promise<Uint8Array> // .n64tas bytes; snapshots packs states every 200 frames so a load seeks fast',
  searchReset: '(type: ValueType): Promise<number> // starts a search over all RDRAM; returns the candidate count',
  searchFilter: '(op: CompareOp, operand?: number): Promise<number> // returns the candidates left',
  searchResults: '(limit: number): Promise<{ addr: number /* physical */; value: number; previous: number }[]>',
  readMemory: '(addr: Address, length: number): Promise<Uint8Array> // big-endian, at most 4096 bytes',
  readValue: '(addr: Address, type: ValueType): Promise<number>',
  peek: '(frames: number[], reads: { addr: Address; type: ValueType }[]): Promise<number[][]> // [frame][read] without moving the current frame; past the movie end no buttons are pressed',
  peekBytes: '(frame: number, addr: Address, length: number): Promise<Uint8Array> // readMemory at another frame, as peek',
  writeMemory: '(addr: Address, value: number, type: ValueType): Promise<void>',
  screenshot: '(): Promise<ArrayBuffer> // PNG',
};

const HELP = `// window.n64: API for AI agents driving the app from page scripts.
type Pad = number; // pack with n64.pad(); stick -127..127, up is +y
type ButtonId = ${BUTTONS.map((b) => `'${b.id}'`).join(' | ')};
type Address = number; // KSEG0 (0x80000000 + offset) or physical
type ValueType = ${VALUE_TYPES.map((t) => `'${t}'`).join(' | ')};
type CompareOp = 'eq' | 'ne' | 'lt' | 'gt' | 'le' | 'ge' | 'changed' | 'unchanged' | 'increased' | 'decreased' | 'changedBy';
interface Status {
  frame: number; fps: number; running: boolean; speed: number; lagFrames: number;
  movie: { mode: 'off' | 'record' | 'play'; length: number; rerecords: number; author: string; canUndo: boolean; canRedo: boolean; revision: number };
  snapshots: number; snapshotBytes: number; recording: number | null; watchValues: number[]; slots: (number | null)[]; input: Pad;
  seekTarget: number | null; // where a seek or a recompute after an edit is heading, while it runs
}
declare const n64: {
  help: string;
  // Preact signals: read .value. Assigning watches.value or cheats.value applies them.
  state: {
    status: Signal<Status>; romName: Signal<string | null>; errorMessage: Signal<string | null>;
    scenes: Signal<{ id: number; frame: number; thumbnail: Blob }[]>; pins: Signal<{ frame: number; thumbnail: Blob | null; stale: boolean }[]>;
    watches: Signal<{ addr: Address; type: ValueType; label: string }[]>; cheats: Signal<{ id: string; name: string; code: string; enabled: boolean }[]>;
  };
  loadRom(url: string): Promise<void>;
  loadMovie(url: string): Promise<void>; // plays it from power-on with its saves
  saveMovie(): Promise<void>; // downloads the current movie
  until(test: (s: Status) => boolean, timeoutMs?: number): Promise<Status>; // rejects after timeoutMs (30000)
  pad(buttons: ButtonId[], x?: number, y?: number): Pad;
  unpad(pad: Pad): { buttons: ButtonId[]; x: number; y: number };
  inputs(start: number, count: number, port?: number): Promise<{ buttons: ButtonId[]; x: number; y: number }[]>;
  screenshot(): Promise<string>; // object URL of the current frame as PNG
  screenHash(): Promise<string>; // SHA-256 hex of the current frame, to compare runs
  emu: {
${Object.entries(CALLS)
  .map(([name, sig]) => `    ${name}${sig};`)
  .join('\n')}
  };
};`;

async function fetchFile(url: string): Promise<File> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return new File([await res.blob()], decodeURIComponent(url.split('/').pop() ?? 'file'));
}

const unpad = (pad: number) => ({ buttons: padButtons(pad), x: stickX(pad), y: stickY(pad) });

const n64 = {
  help: HELP,
  emu,
  state: { status, romName, errorMessage, scenes, pins, watches, cheats },

  loadRom: async (url: string) => loadRom(await fetchFile(url)),

  loadMovie: async (url: string) => emu.importMovie(await (await fetchFile(url)).arrayBuffer()),

  saveMovie: async () => download(await emu.exportMovie(), `${baseName()}.n64tas`),

  /** Polls the status the worker reports, rejecting after `timeoutMs`. */
  until: (test: (s: Status) => boolean, timeoutMs = 30_000): Promise<Status> =>
    new Promise((resolve, reject) => {
      const deadline = performance.now() + timeoutMs;
      const timer = setInterval(() => {
        if (test(status.value)) {
          clearInterval(timer);
          resolve(status.value);
        } else if (performance.now() > deadline) {
          clearInterval(timer);
          reject(new Error(`n64.until timed out after ${timeoutMs} ms at frame ${status.value.frame}`));
        }
      }, 50);
    }),

  /** Agents call this from untyped scripts, where a misspelled button would silently press nothing. */
  pad: (buttons: ButtonId[], x = 0, y = 0) => {
    const unknown = buttons.filter((b) => !(b in BUTTON_BIT));
    if (unknown.length > 0) throw new Error(`Unknown buttons: ${unknown.join(', ')}. Use ${BUTTONS.map((b) => b.id).join(' ')}`);
    return packPad(buttons, x, y);
  },

  unpad,

  inputs: async (start: number, count: number, port = 0) => {
    const { inputs } = await emu.movieSlice(start, count);
    return Array.from({ length: inputs.length / 4 }, (_, i) => unpad(inputs[i * 4 + port]));
  },

  screenshot: async () => URL.createObjectURL(new Blob([await emu.screenshot()], { type: 'image/png' })),

  screenHash: async () => {
    const digest = await crypto.subtle.digest('SHA-256', await emu.screenshot());
    return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, '0')).join('');
  },
};

declare global {
  interface Window {
    n64: typeof n64;
  }
}

window.n64 = n64;
