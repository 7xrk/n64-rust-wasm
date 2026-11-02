/** Frames pinned on the timeline to preview how the movie plays out there.
 * Each keeps a screenshot that goes stale when an earlier input changes and is
 * retaken the next time the frame is emulated. */

import { encodeThumbnail, type Screen } from './scenes';

export interface PinInfo {
  frame: number;
  thumbnail: Blob | null;
  /** The screenshot predates an edit before this frame. */
  stale: boolean;
}

interface Pin extends PinInfo {
  /** Bumped when the pin goes stale, so a screenshot still encoding for the old timeline is discarded. */
  version: number;
}

export class Pins {
  private readonly map = new Map<number, Pin>();

  /** `onChange` receives the list itself: the worker cannot answer a query
   * while it replays towards a pin, which is when the list matters most. */
  constructor(private readonly onChange: (pins: PinInfo[]) => void) {}

  /** Pins or unpins `frame` and returns whether it is pinned now. */
  toggle(frame: number): boolean {
    const pinned = !this.map.delete(frame);
    if (pinned) this.map.set(frame, { frame, thumbnail: null, stale: true, version: 0 });
    this.onChange(this.info());
    return pinned;
  }

  isStale(frame: number): boolean {
    return this.map.get(frame)?.stale ?? false;
  }

  /** Whether a pin awaiting a screenshot lies in (frame, frame + span]. */
  staleWithin(frame: number, span: number): boolean {
    for (const p of this.map.values()) if (p.stale && p.frame > frame && p.frame <= frame + span) return true;
    return false;
  }

  staleFrames(): number[] {
    return [...this.map.values()]
      .filter((p) => p.stale)
      .map((p) => p.frame)
      .sort((a, b) => a - b);
  }

  /** Takes `screen`, the picture of `frame`, as the pin's screenshot once encoded. */
  async capture(frame: number, screen: Promise<Screen>): Promise<void> {
    const pin = this.map.get(frame);
    if (!pin) return;
    pin.stale = false;
    const version = pin.version;
    const thumbnail = await encodeThumbnail(await screen);
    if (pin.version !== version || this.map.get(frame) !== pin) return;
    pin.thumbnail = thumbnail;
    this.onChange(this.info());
  }

  /** Marks the screenshots after `frame` stale, e.g. when the input at `frame` changed. */
  invalidateAfter(frame: number): void {
    let changed = false;
    for (const p of this.map.values()) {
      if (p.frame <= frame || p.stale) continue;
      p.stale = true;
      p.version++;
      changed = true;
    }
    if (changed) this.onChange(this.info());
  }

  clear(): void {
    if (this.map.size === 0) return;
    this.map.clear();
    this.onChange(this.info());
  }

  private info(): PinInfo[] {
    return [...this.map.values()].map(({ frame, thumbnail, stale }) => ({ frame, thumbnail, stale })).sort((a, b) => a.frame - b.frame);
  }
}
