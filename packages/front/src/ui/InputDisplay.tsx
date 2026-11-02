/** Shows the controller 1 input of the last emulated frame under the screen. */

import { BUTTON_BIT, BUTTON_LABEL, type ButtonId, stickX, stickY } from 'n64core';
import { status } from './client';
import { StickPad } from './StickPad';

const GROUPS: ButtonId[][] = [
  ['A', 'B', 'Z', 'S'],
  ['L', 'R'],
  ['cU', 'cD', 'cL', 'cR'],
  ['dU', 'dD', 'dL', 'dR'],
];

const MODE_LABEL = { off: 'Free', record: '● Record', play: '▶ Play' } as const;

export function InputDisplay() {
  const s = status.value;
  const pad = s.input;
  return (
    <div class="input-display">
      <StickPad x={stickX(pad)} y={stickY(pad)} disabled />
      <div class="input-buttons">
        {GROUPS.map((group) => (
          <div class="input-group">
            {group.map((id) => (
              <span class={`chip ${pad & BUTTON_BIT[id] ? 'on' : ''}`}>{BUTTON_LABEL[id]}</span>
            ))}
          </div>
        ))}
      </div>
      <div class="input-meta">
        <span class="big">{s.frame}</span>
        <span class="muted small">
          of {s.movie.length} frames / lag {s.lagFrames}
        </span>
        <span class={`mode-badge ${s.movie.mode}`}>{MODE_LABEL[s.movie.mode]}</span>
      </div>
    </div>
  );
}
