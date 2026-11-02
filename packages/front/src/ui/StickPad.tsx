/** Circular analog stick editor. Shows a value and reports a new one on release. */

import { useState } from 'preact/hooks';
import { STICK_MAX } from 'n64core';

const SIZE = 128;
const RADIUS = SIZE / 2 - 6;

interface Props {
  x: number;
  y: number;
  disabled?: boolean;
  /** Reports the value under the pointer while dragging, then null on release. */
  onDrag?: (value: { x: number; y: number } | null) => void;
  onCommit?: (x: number, y: number) => void;
}

export function StickPad({ x, y, disabled = false, onDrag, onCommit }: Props) {
  const [drag, setDragState] = useState<{ x: number; y: number } | null>(null);
  const shown = drag ?? { x, y };

  const setDrag = (value: { x: number; y: number } | null) => {
    setDragState(value);
    onDrag?.(value);
  };

  const valueAt = (e: PointerEvent) => {
    const rect = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
    const dx = e.clientX - rect.left - rect.width / 2;
    const dy = e.clientY - rect.top - rect.height / 2;
    const clamp = (v: number) => Math.max(-STICK_MAX, Math.min(STICK_MAX, Math.round((v / RADIUS) * STICK_MAX)));
    return { x: clamp(dx), y: clamp(-dy) };
  };

  const cx = SIZE / 2 + (shown.x / STICK_MAX) * RADIUS;
  const cy = SIZE / 2 - (shown.y / STICK_MAX) * RADIUS;

  return (
    <svg
      class={`stick-pad ${disabled ? 'disabled' : ''}`}
      width={SIZE}
      height={SIZE}
      viewBox={`0 0 ${SIZE} ${SIZE}`}
      onPointerDown={(e) => {
        if (disabled || !onCommit) return;
        e.currentTarget.setPointerCapture(e.pointerId);
        setDrag(valueAt(e));
      }}
      onPointerMove={(e) => drag && setDrag(valueAt(e))}
      onPointerUp={() => {
        if (!drag) return;
        onCommit?.(drag.x, drag.y);
        setDrag(null);
      }}
    >
      <circle cx={SIZE / 2} cy={SIZE / 2} r={RADIUS} class="ring" />
      <line x1={SIZE / 2} y1={6} x2={SIZE / 2} y2={SIZE - 6} class="axis" />
      <line x1={6} y1={SIZE / 2} x2={SIZE - 6} y2={SIZE / 2} class="axis" />
      <line x1={SIZE / 2} y1={SIZE / 2} x2={cx} y2={cy} class="arm" />
      <circle cx={cx} cy={cy} r={6} class="knob" />
      <text x={6} y={SIZE - 6} class="readout">
        {shown.x}, {shown.y}
      </text>
    </svg>
  );
}
