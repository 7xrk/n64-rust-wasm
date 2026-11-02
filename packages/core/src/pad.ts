/** N64 controller state packed as the joybus response:
 * byte0 | byte1 << 8 | stickX << 16 | stickY << 24. The same layout is used
 * by .m64 movies, so values convert without remapping. */

export const BUTTONS = [
  { id: 'A', bit: 1 << 7, label: 'A' },
  { id: 'B', bit: 1 << 6, label: 'B' },
  { id: 'Z', bit: 1 << 5, label: 'Z' },
  { id: 'S', bit: 1 << 4, label: 'S' },
  { id: 'dU', bit: 1 << 3, label: '↑' },
  { id: 'dD', bit: 1 << 2, label: '↓' },
  { id: 'dL', bit: 1 << 1, label: '←' },
  { id: 'dR', bit: 1 << 0, label: '→' },
  { id: 'L', bit: 1 << 13, label: 'L' },
  { id: 'R', bit: 1 << 12, label: 'R' },
  { id: 'cU', bit: 1 << 11, label: 'C↑' },
  { id: 'cD', bit: 1 << 10, label: 'C↓' },
  { id: 'cL', bit: 1 << 9, label: 'C←' },
  { id: 'cR', bit: 1 << 8, label: 'C→' },
] as const;

export type ButtonId = (typeof BUTTONS)[number]['id'];

export const BUTTON_BIT = Object.fromEntries(BUTTONS.map((b) => [b.id, b.bit])) as Record<ButtonId, number>;

export const BUTTON_LABEL = Object.fromEntries(BUTTONS.map((b) => [b.id, b.label])) as Record<ButtonId, string>;

/** Stick axes stay within ±STICK_MAX. -128 fits the byte, but games that
 * negate an axis turn it back into -128, flipping the direction. */
export const STICK_MAX = 127;

export function stickX(pad: number): number {
  return (pad << 8) >> 24;
}

export function stickY(pad: number): number {
  return pad >> 24;
}

export function withStick(pad: number, x: number, y: number): number {
  const cx = Math.max(-STICK_MAX, Math.min(STICK_MAX, Math.round(x))) & 0xff;
  const cy = Math.max(-STICK_MAX, Math.min(STICK_MAX, Math.round(y))) & 0xff;
  return ((pad & 0xffff) | (cx << 16) | (cy << 24)) >>> 0;
}

export function packPad(buttons: Iterable<ButtonId>, x = 0, y = 0): number {
  let v = 0;
  for (const b of buttons) v |= BUTTON_BIT[b];
  return withStick(v, x, y);
}

export function padButtons(pad: number): ButtonId[] {
  return BUTTONS.filter((b) => pad & b.bit).map((b) => b.id);
}
