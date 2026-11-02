/** Scene history: a thumbnail of the screen every few seconds, linked to the
 * snapshot of the same frame so the scene can be restored later. Scenes
 * follow the TAS timeline and are dropped with the snapshots after an edit. */

import type { Renderer } from 'n64core';
import type { PageState } from './snapshots';

const MAX_SCENES = 100;
const THUMB_WIDTH = 160;
const THUMB_HEIGHT = 120;

export interface SceneInfo {
  id: number;
  frame: number;
  thumbnail: Blob;
}

interface Scene extends SceneInfo {
  /** State some frames before the scene; replaying `inputs` from it reaches the
   * scene with the screen drawn, which a snapshot alone cannot restore. */
  state: PageState;
  /** Four ports per frame from `state` up to the scene. */
  inputs: Uint32Array;
}

export type Screen = Awaited<ReturnType<Renderer['capture']>>;

export async function encodeThumbnail({ width, height, data }: Screen): Promise<Blob> {
  const bitmap = await createImageBitmap(new ImageData(data, width, height), {
    resizeWidth: THUMB_WIDTH,
    resizeHeight: THUMB_HEIGHT,
    resizeQuality: 'medium',
  });
  const canvas = new OffscreenCanvas(THUMB_WIDTH, THUMB_HEIGHT);
  canvas.getContext('2d')!.drawImage(bitmap, 0, 0);
  bitmap.close();
  const webp = await canvas.convertToBlob({ type: 'image/webp', quality: 0.8 });
  // convertToBlob silently returns PNG for unsupported types; JPEG is far smaller.
  return webp.type === 'image/webp' ? webp : canvas.convertToBlob({ type: 'image/jpeg', quality: 0.8 });
}

export class Scenes {
  private list: Scene[] = [];
  private nextId = 1;
  /** Scenes whose thumbnail is still encoding, so invalidation can reach them too. */
  private readonly pending = new Set<{ frame: number; dropped: boolean }>();

  constructor(private readonly onChange: () => void) {}

  /** Records the scene of `frame` once `screen`, its picture, is encoded. */
  async add(frame: number, state: PageState, inputs: Uint32Array, screen: Promise<Screen>): Promise<void> {
    const id = this.nextId++;
    const job = { frame, dropped: false };
    this.pending.add(job);
    let thumbnail: Blob;
    try {
      thumbnail = await encodeThumbnail(await screen);
    } finally {
      this.pending.delete(job);
    }
    if (job.dropped) return;
    this.list.push({ id, frame, state, inputs, thumbnail });
    if (this.list.length > MAX_SCENES) this.list.shift();
    this.onChange();
  }

  get(id: number): Scene | undefined {
    return this.list.find((s) => s.id === id);
  }

  /** Scenes in the order they were recorded. */
  info(): SceneInfo[] {
    return this.list.map(({ id, frame, thumbnail }) => ({ id, frame, thumbnail }));
  }

  /** Drops scenes after `frame`, e.g. when the input at `frame` changed. */
  invalidateAfter(frame: number): void {
    for (const job of this.pending) if (job.frame > frame) job.dropped = true;
    if (!this.list.some((s) => s.frame > frame)) return;
    this.list = this.list.filter((s) => s.frame <= frame);
    this.onChange();
  }

  clear(): void {
    this.invalidateAfter(-1);
  }
}
