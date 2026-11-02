/** Automatic savestates taken every `interval` frames. RDRAM is kept as 4 KiB
 * pages shared with the previous snapshot when the game did not write them,
 * so a capture copies only what changed. They make seeking, rewinding and
 * branch edits cheap. */

import { type Core, PAGE_SIZE } from 'n64core';

/** Machine state split by capturePages. */
export interface PageState {
  prefix: Uint8Array;
  pages: Uint8Array[];
}

interface Snapshot extends PageState {
  /** Bytes this snapshot added on top of the pages it shares. */
  bytes: number;
}

export interface SnapshotInfo {
  frames: number[];
  bytes: number;
}

/** Snapshots packed with a movie are at least this many frames apart, so a seek after loading it replays at most
 * this far instead of from power-on, while the file grows by about 0.2 MB per snapshot. */
const PACK_SPACING = 200;

const u32Bytes = (...values: number[]) => new Uint8Array(new Uint32Array(values).buffer);

const gzip = (parts: Uint8Array[]) =>
  new Response(new Blob(parts as BlobPart[]).stream().pipeThrough(new CompressionStream('gzip'))).arrayBuffer();

const gunzip = (bytes: Uint8Array) =>
  new Response(new Blob([bytes as BlobPart]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();

export class Snapshots {
  interval = 30;
  budgetBytes = 512 * 1024 * 1024;
  private readonly store = new Map<number, Snapshot>();
  private bytes = 0;
  /** Pages of the state the core's page-write tracking is relative to. */
  private base: Uint8Array[] | null = null;

  /** Returns the snapshot of `frame`, capturing it unless it is already stored. */
  capture(frame: number, core: Core): PageState {
    const stored = this.store.get(frame);
    if (stored) return stored;
    const { prefix, pages, copied } = core.capturePages(this.base);
    const snap = { prefix, pages, bytes: prefix.length + copied * PAGE_SIZE };
    this.add(frame, snap);
    this.base = pages;
    this.enforceBudget();
    return snap;
  }

  /** Captures `frame` right after a state was loaded into the core from outside, replacing a snapshot already stored
   * at `frame`. The load marks every page written, so the capture copies them all. */
  captureLoaded(frame: number, core: Core): void {
    this.drop(frame);
    this.capture(frame, core);
  }

  /** Packs the snapshots up to `last`, PACK_SPACING apart and the newest one, to keep with a movie. Each keeps only
   * the pages that differ from the one packed before it. */
  async pack(last: number, build: string): Promise<Uint8Array<ArrayBuffer>> {
    const frames = [...this.store.keys()].filter((f) => f <= last).sort((a, b) => a - b);
    const kept: number[] = [];
    for (const f of frames) if (!kept.length || f - kept.at(-1)! >= PACK_SPACING) kept.push(f);
    if (kept.at(-1) !== frames.at(-1)) kept.push(frames.at(-1)!);
    const name = new TextEncoder().encode(build);
    const parts: Uint8Array[] = [u32Bytes(name.length), name, u32Bytes(kept.length)];
    let prev: Uint8Array[] = [];
    for (const f of kept) {
      const { prefix, pages } = this.store.get(f)!;
      // Captures share the pages the game did not write, so identity finds the changed ones without comparing bytes.
      const changed = pages.flatMap((page, p) => (page === prev[p] ? [] : [p]));
      parts.push(u32Bytes(f, prefix.length), prefix, u32Bytes(changed.length, ...changed), ...changed.map((p) => pages[p]));
      prev = pages;
    }
    return new Uint8Array(await gzip(parts));
  }

  /** Adds snapshots made by pack, unless another core build packed them. */
  async unpack(packed: Uint8Array, build: string): Promise<void> {
    const bytes = new Uint8Array(await gunzip(packed));
    const view = new DataView(bytes.buffer);
    let at = 0;
    const u32 = () => {
      at += 4;
      return view.getUint32(at - 4, true);
    };
    const take = (length: number) => bytes.slice(at, (at += length));
    if (new TextDecoder().decode(take(u32())) !== build) return;
    let pages: Uint8Array[] = [];
    for (let count = u32(); count > 0; count--) {
      const frame = u32();
      const prefix = take(u32());
      const changed = Array.from({ length: u32() }, u32);
      pages = [...pages];
      for (const p of changed) pages[p] = take(PAGE_SIZE);
      if (!this.store.has(frame)) this.add(frame, { prefix, pages, bytes: prefix.length + changed.length * PAGE_SIZE });
    }
    this.enforceBudget();
  }

  /** Frame of the newest snapshot at or before `frame`. */
  nearestFrame(frame: number): number | null {
    let best: number | null = null;
    for (const f of this.store.keys()) if (f <= frame && (best === null || f > best)) best = f;
    return best;
  }

  /** Restores the newest snapshot at or before `frame` and returns its frame. */
  restoreNearest(frame: number, core: Core): number | null {
    const best = this.nearestFrame(frame);
    const snap = best === null ? undefined : this.store.get(best);
    if (!snap) return null;
    this.restore(snap, core);
    return best;
  }

  /** Restores a state returned by capture, even one dropped from the store since. */
  restore(state: PageState, core: Core): void {
    if (!core.restorePages(state.prefix, state.pages)) throw new Error('Snapshot is from a different core version');
    this.base = state.pages;
  }

  /** Drops snapshots after `frame`, e.g. when the input at `frame` changed. */
  invalidateAfter(frame: number): void {
    for (const f of this.store.keys()) if (f > frame) this.drop(f);
  }

  clear(): void {
    for (const f of this.store.keys()) this.drop(f);
  }

  info(): SnapshotInfo {
    return { frames: [...this.store.keys()].sort((a, b) => a - b), bytes: this.bytes };
  }

  private add(frame: number, snap: Snapshot): void {
    this.store.set(frame, snap);
    this.bytes += snap.bytes;
  }

  private drop(frame: number): void {
    this.bytes -= this.store.get(frame)?.bytes ?? 0;
    this.store.delete(frame);
  }

  /** Removes the snapshot whose neighbours are closest together until under budget. */
  private enforceBudget(): void {
    while (this.bytes > this.budgetBytes && this.store.size > 2) {
      const frames = [...this.store.keys()].sort((a, b) => a - b);
      let victim = frames[1];
      let gap = Infinity;
      for (let i = 1; i < frames.length - 1; i++) {
        const g = frames[i + 1] - frames[i - 1];
        if (g < gap) {
          gap = g;
          victim = frames[i];
        }
      }
      this.drop(victim);
    }
  }
}
