/** Scene history picker: an overlay on the screen listing the thumbnails the
 * worker records every few seconds. ZR on an NSO N64 controller toggles it,
 * and while it is open the controller drives the picker instead of the game. */

import { effect, signal } from '@preact/signals';
import { useEffect, useRef } from 'preact/hooks';
import { BUTTON_BIT, stickX } from 'n64core';
import { FPS } from '../emu/emulator';
import { emu, liveInput, romName, run, scenes, status } from './client';
import { nsoPads } from './nso-n64';

const REPEAT_DELAY_MS = 400;
const REPEAT_INTERVAL_MS = 80;

export const pickerOpen = signal(false);
const selected = signal(0);

/** Object URLs of the thumbnails, revoked when a scene leaves the history. */
const thumbnails = new Map<number, string>();
effect(() => {
  const ids = new Set(scenes.value.map((s) => s.id));
  for (const [id, url] of thumbnails) {
    if (ids.has(id)) continue;
    URL.revokeObjectURL(url);
    thumbnails.delete(id);
  }
  for (const s of scenes.value) if (!thumbnails.has(s.id)) thumbnails.set(s.id, URL.createObjectURL(s.thumbnail));
});

export function openPicker(): void {
  if (pickerOpen.value || !romName.value) return;
  liveInput.suspended = true;
  selected.value = Math.max(0, scenes.value.length - 1);
  pickerOpen.value = true;
  run(emu.pause());
}

function hide(): void {
  pickerOpen.value = false;
  liveInput.suspended = false;
}

function choose(id: number): void {
  hide();
  run(emu.restoreScene(id));
}

function move(delta: number): void {
  selected.value = Math.max(0, Math.min(scenes.value.length - 1, selected.value + delta));
}

function timeLabel(frame: number): string {
  const seconds = Math.floor(frame / FPS);
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`;
}

/** Polls the controllers every animation frame, like InputManager, and
 * returns a function that stops it. */
export function attachPicker(): () => void {
  let zr = false;
  let pad = 0;
  let direction = 0;
  let repeatAt = 0;
  let raf = 0;
  const loop = (now: number) => {
    raf = requestAnimationFrame(loop);
    const zrNow = nsoPads.value.some((p) => p.zr);
    const zrPressed = zrNow && !zr;
    zr = zrNow;
    const prev = pad;
    pad = liveInput.read();
    if (!romName.value) return;
    // ZR toggles pause; pausing also opens the history to pick a scene from.
    if (zrPressed && status.value.running) {
      openPicker();
      return;
    }
    if (zrPressed) {
      hide();
      run(emu.resume());
      return;
    }
    if (!pickerOpen.value) return;
    const pressed = pad & ~prev;
    const scene = scenes.value[selected.value];
    if (pressed & BUTTON_BIT.A && scene) {
      choose(scene.id);
      return;
    }
    if (pressed & BUTTON_BIT.B) {
      hide();
      return;
    }
    const threshold = liveInput.mapping.stickRange / 2;
    const left = pad & BUTTON_BIT.dL || stickX(pad) < -threshold;
    const right = pad & BUTTON_BIT.dR || stickX(pad) > threshold;
    const next = left === right ? 0 : left ? -1 : 1;
    if (next !== direction) {
      direction = next;
      repeatAt = now + REPEAT_DELAY_MS;
      move(direction);
    } else if (direction !== 0 && now >= repeatAt) {
      repeatAt = now + REPEAT_INTERVAL_MS;
      move(direction);
    }
  };
  const onKey = (e: KeyboardEvent) => {
    if (e.code !== 'Escape' || !pickerOpen.value) return;
    e.preventDefault();
    hide();
  };
  raf = requestAnimationFrame(loop);
  window.addEventListener('keydown', onKey);
  return () => {
    cancelAnimationFrame(raf);
    window.removeEventListener('keydown', onKey);
  };
}

export function ScenePicker() {
  const strip = useRef<HTMLDivElement>(null);
  const open = pickerOpen.value;
  const index = selected.value;

  useEffect(() => {
    strip.current?.children[index]?.scrollIntoView({ block: 'nearest', inline: 'center' });
  }, [open, index]);

  if (!open) return null;
  const list = scenes.value;
  return (
    <div class="scene-picker">
      <p class="scene-help">Scene history: ←→ to select / A to resume / B to close</p>
      <div class="scene-strip" ref={strip}>
        {list.length === 0 && <p class="muted">No scenes yet</p>}
        {list.map((s, i) => (
          <button type="button" class={`scene ${i === index ? 'selected' : ''}`} onClick={() => choose(s.id)}>
            <img src={thumbnails.get(s.id)} alt="" width={160} height={120} />
            <span>
              F{s.frame} · {timeLabel(s.frame)}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}
