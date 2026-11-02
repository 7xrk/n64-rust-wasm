/** Live input from the keyboard and the Gamepad API. The Gamepad API only
 * exists on the main thread, so the packed state is forwarded to the worker. */

import { BUTTON_BIT, type ButtonId, withStick } from 'n64core';
import { isNsoN64Gamepad, nsoPads } from './nso-n64';

export interface Mapping {
  keys: Record<string, ButtonId | 'up' | 'down' | 'left' | 'right'>;
  /** Standard gamepad button index -> N64 button. */
  pad: Record<number, ButtonId>;
  stickRange: number;
  deadzone: number;
  /** Gamepads without the standard layout often report noise, so they are opt-in. */
  allowNonStandard: boolean;
}

export const DEFAULT_MAPPING: Mapping = {
  keys: {
    ArrowUp: 'up',
    ArrowDown: 'down',
    ArrowLeft: 'left',
    ArrowRight: 'right',
    KeyX: 'A',
    KeyC: 'B',
    Space: 'Z',
    Enter: 'S',
    KeyA: 'L',
    KeyS: 'R',
    KeyI: 'cU',
    KeyK: 'cD',
    KeyJ: 'cL',
    KeyL: 'cR',
    KeyT: 'dU',
    KeyG: 'dD',
    KeyF: 'dL',
    KeyH: 'dR',
  },
  pad: { 0: 'A', 2: 'B', 1: 'cD', 3: 'cU', 4: 'L', 5: 'R', 6: 'Z', 7: 'Z', 9: 'S', 12: 'dU', 13: 'dD', 14: 'dL', 15: 'dR' },
  stickRange: 80,
  deadzone: 0.15,
  allowNonStandard: false,
};

/** Gamepads whose input is read with the given mapping. The NSO N64
 * controller is read through WebHID instead (nso-n64.ts). */
export function usableGamepads(m: Mapping): Gamepad[] {
  return navigator
    .getGamepads()
    .filter((gp): gp is Gamepad => !!gp && !isNsoN64Gamepad(gp) && (gp.mapping === 'standard' || m.allowNonStandard));
}

const STORAGE_KEY = 'n64tas.mapping';

/** Input types whose keys do not type text, so focusing them keeps hotkeys and controller keys working. */
const NON_TEXT_INPUTS = new Set(['checkbox', 'radio', 'button', 'submit', 'reset', 'file', 'range', 'color']);

/** Whether keys pressed in `target` type text and must be left to it. */
export function isTextEntry(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  if (target.isContentEditable || target.tagName === 'TEXTAREA') return true;
  return target instanceof HTMLInputElement && !NON_TEXT_INPUTS.has(target.type);
}

export function loadMapping(): Mapping {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return { ...DEFAULT_MAPPING, ...JSON.parse(raw) };
  } catch {
    // Storage can be unavailable (private mode); fall back to defaults.
  }
  return DEFAULT_MAPPING;
}

export function saveMapping(m: Mapping): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(m));
  } catch {
    // Not persisting is acceptable.
  }
}

export class InputManager {
  private readonly down = new Set<string>();
  private last = -1;
  private raf = 0;
  /** Set while another UI reads the controller; the game then sees a neutral pad. */
  suspended = false;

  constructor(
    public mapping: Mapping,
    private readonly onChange: (pad: number) => void,
  ) {}

  attach(): void {
    window.addEventListener('keydown', this.onKey);
    window.addEventListener('keyup', this.onKey);
    window.addEventListener('blur', this.onBlur);
    const loop = () => {
      this.poll();
      this.raf = requestAnimationFrame(loop);
    };
    loop();
  }

  detach(): void {
    window.removeEventListener('keydown', this.onKey);
    window.removeEventListener('keyup', this.onKey);
    window.removeEventListener('blur', this.onBlur);
    cancelAnimationFrame(this.raf);
  }

  private onKey = (e: KeyboardEvent): void => {
    if (!(e.code in this.mapping.keys)) return;
    if (isTextEntry(e.target)) return;
    e.preventDefault();
    if (e.type === 'keydown') this.down.add(e.code);
    else this.down.delete(e.code);
  };

  private onBlur = (): void => {
    this.down.clear();
  };

  /** Current packed state from all devices. */
  read(): number {
    const m = this.mapping;
    let buttons = 0;
    let x = 0;
    let y = 0;
    for (const code of this.down) {
      const t = m.keys[code];
      if (t === 'up') y += 1;
      else if (t === 'down') y -= 1;
      else if (t === 'left') x -= 1;
      else if (t === 'right') x += 1;
      else if (t) buttons |= BUTTON_BIT[t];
    }
    for (const gp of usableGamepads(m)) {
      gp.buttons.forEach((b, i) => {
        const t = m.pad[i];
        if (t && b.pressed) buttons |= BUTTON_BIT[t];
      });
      const ax = gp.axes[0] ?? 0;
      const ay = gp.axes[1] ?? 0;
      if (Math.hypot(ax, ay) > m.deadzone) {
        x += ax;
        y -= ay;
      }
      const cx = gp.axes[2] ?? 0;
      const cy = gp.axes[3] ?? 0;
      if (cx > 0.5) buttons |= BUTTON_BIT.cR;
      if (cx < -0.5) buttons |= BUTTON_BIT.cL;
      if (cy > 0.5) buttons |= BUTTON_BIT.cD;
      if (cy < -0.5) buttons |= BUTTON_BIT.cU;
    }
    for (const pad of nsoPads.peek()) {
      buttons |= pad.buttons;
      x += pad.x;
      y += pad.y;
    }
    const len = Math.hypot(x, y);
    if (len > 1) {
      x /= len;
      y /= len;
    }
    return withStick(buttons, x * m.stickRange, y * m.stickRange);
  }

  private poll(): void {
    const v = this.suspended ? 0 : this.read();
    if (v !== this.last) {
      this.last = v;
      this.onChange(v);
    }
  }
}
