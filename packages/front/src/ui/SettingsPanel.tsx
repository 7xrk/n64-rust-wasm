/** Key / gamepad mapping and emulation settings. */

import { useEffect, useState } from 'preact/hooks';
import { BUTTONS, type ButtonId, padButtons, SCALES } from 'n64core';
import { emu, liveInput, renderScale, run, setRenderScale } from './client';
import { DEFAULT_MAPPING, type Mapping, saveMapping, usableGamepads } from './input';
import { isNsoN64Gamepad, nsoError, nsoPads, requestNsoN64, webHidAvailable } from './nso-n64';

type Target = ButtonId | 'up' | 'down' | 'left' | 'right';
const TARGETS: Target[] = ['up', 'down', 'left', 'right', ...BUTTONS.map((b) => b.id)];
const LABEL: Record<string, string> = { up: 'Stick up', down: 'Stick down', left: 'Stick left', right: 'Stick right' };

function isButton(t: Target): t is ButtonId {
  return !(t in LABEL);
}

export function SettingsPanel() {
  const [mapping, setMapping] = useState<Mapping>(liveInput.mapping);
  const [capture, setCapture] = useState<{ target: Target; device: 'key' | 'pad' } | null>(null);
  const [controllers, setControllers] = useState(1);
  const [interval, setIntervalFrames] = useState(30);

  const apply = (m: Mapping) => {
    setMapping(m);
    liveInput.mapping = m;
    saveMapping(m);
  };

  useEffect(() => {
    if (!capture) return;
    if (capture.device === 'key') {
      const onKey = (e: KeyboardEvent) => {
        e.preventDefault();
        e.stopPropagation();
        const keys = Object.fromEntries(Object.entries(mapping.keys).filter(([, t]) => t !== capture.target));
        if (e.code !== 'Escape') keys[e.code] = capture.target;
        apply({ ...mapping, keys });
        setCapture(null);
      };
      window.addEventListener('keydown', onKey, { capture: true });
      return () => window.removeEventListener('keydown', onKey, { capture: true });
    }
    const target = capture.target;
    if (!isButton(target)) return;
    // Gamepad buttons can only be polled. Only a released -> pressed edge counts,
    // so buttons that already read as pressed are not captured.
    let raf = 0;
    const baseline = new Map(usableGamepads(mapping).map((gp) => [gp.index, gp.buttons.map((b) => b.pressed)]));
    const poll = () => {
      for (const gp of usableGamepads(mapping)) {
        const before = baseline.get(gp.index) ?? [];
        const index = gp.buttons.findIndex((b, i) => b.pressed && !before[i]);
        if (index >= 0) {
          const pad = Object.fromEntries(Object.entries(mapping.pad).filter(([, t]) => t !== capture.target));
          pad[index] = target;
          apply({ ...mapping, pad });
          setCapture(null);
          return;
        }
      }
      raf = requestAnimationFrame(poll);
    };
    poll();
    return () => cancelAnimationFrame(raf);
  }, [capture]);

  const keyFor = (t: Target) => Object.entries(mapping.keys).find(([, v]) => v === t)?.[0] ?? '-';
  const padFor = (t: Target) => Object.entries(mapping.pad).find(([, v]) => v === t)?.[0] ?? '-';

  return (
    <section class="panel">
      <h3>Input mapping</h3>
      <p class="muted small">The left stick always drives the analog stick. The right stick maps to the C buttons.</p>
      <table class="grid">
        <thead>
          <tr>
            <th>N64</th>
            <th>Keyboard</th>
            <th>Gamepad</th>
          </tr>
        </thead>
        <tbody>
          {TARGETS.map((t) => (
            <tr>
              <td>{LABEL[t] ?? t}</td>
              <td>
                <button type="button" onClick={() => setCapture({ target: t, device: 'key' })}>
                  {capture?.target === t && capture.device === 'key' ? 'Press a key…' : keyFor(t)}
                </button>
              </td>
              <td>
                {isButton(t) && (
                  <button type="button" onClick={() => setCapture({ target: t, device: 'pad' })}>
                    {capture?.target === t && capture.device === 'pad' ? 'Press a button…' : padFor(t)}
                  </button>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div class="panel-bar">
        <label>
          Stick maximum
          <input
            class="narrow"
            type="number"
            min={1}
            max={127}
            value={mapping.stickRange}
            onChange={(e) => apply({ ...mapping, stickRange: Number(e.currentTarget.value) })}
          />
        </label>
        <button type="button" onClick={() => apply(DEFAULT_MAPPING)}>
          Restore defaults
        </button>
      </div>
      <h3>Gamepads</h3>
      <GamepadList />
      <label class="check">
        <input
          type="checkbox"
          checked={mapping.allowNonStandard}
          onChange={(e) => apply({ ...mapping, allowNonStandard: e.currentTarget.checked })}
        />
        Also use gamepads without the standard mapping
      </label>
      <h3>NSO N64 Controller</h3>
      <NsoPanel />
      <h3>Emulation</h3>
      <div class="panel-bar">
        <label>
          Controllers
          <select
            value={controllers}
            onChange={(e) => {
              const n = Number(e.currentTarget.value);
              setControllers(n);
              run(emu.setConnected((1 << n) - 1));
            }}
          >
            {[1, 2, 3, 4].map((n) => (
              <option value={n}>{n}</option>
            ))}
          </select>
        </label>
        <label>
          Internal resolution
          <select value={renderScale.value} onChange={(e) => setRenderScale(Number(e.currentTarget.value))}>
            {SCALES.map((n) => (
              <option value={n}>{n}×</option>
            ))}
          </select>
        </label>
        <label>
          Auto snapshot interval
          <input
            class="narrow"
            type="number"
            min={1}
            value={interval}
            onChange={(e) => {
              const n = Number(e.currentTarget.value);
              setIntervalFrames(n);
              run(emu.setSnapshotInterval(n));
            }}
          />
          frames
        </label>
      </div>
      <h3>Hotkeys</h3>
      <ul class="muted small">
        <li>P: Pause / resume</li>
        <li>. : Advance one frame</li>
        <li>, : Step back one frame</li>
        <li>` : Toggle fast-forward</li>
        <li>1–9: Load slot / Shift+1–9: Save to slot</li>
        <li>R: Toggle record / play</li>
        <li>⌘Z / ⌘⇧Z: Undo / redo movie edits</li>
        <li>In the piano roll: ↑↓ to select, Enter to go, Delete to delete, ⌘I to insert, ⌘C / ⌘V to copy / paste</li>
      </ul>
    </section>
  );
}

function GamepadList() {
  const [pads, setPads] = useState<{ id: string; note: string }[]>([]);
  useEffect(() => {
    const note = (gp: Gamepad) =>
      isNsoN64Gamepad(gp) ? 'Used via WebHID' : gp.mapping === 'standard' ? 'Standard mapping' : 'Non-standard mapping';
    const update = () => setPads(navigator.getGamepads().flatMap((gp) => (gp ? [{ id: gp.id, note: note(gp) }] : [])));
    update();
    const timer = setInterval(update, 1000);
    return () => clearInterval(timer);
  }, []);
  if (pads.length === 0) return <p class="muted small">No gamepads connected. Press a button to have one detected.</p>;
  return (
    <ul class="small">
      {pads.map((p) => (
        <li>
          {p.id} <span class="muted">{p.note}</span>
        </li>
      ))}
    </ul>
  );
}

/** Connects the controller over WebHID and shows its live state. */
function NsoPanel() {
  const [, setTick] = useState(0);
  const pads = nsoPads.value;
  useEffect(() => {
    if (pads.length === 0) return;
    let raf = 0;
    const loop = () => {
      setTick((t) => t + 1);
      raf = requestAnimationFrame(loop);
    };
    loop();
    return () => cancelAnimationFrame(raf);
  }, [pads.length]);

  if (!webHidAvailable) return <p class="muted small">This browser does not support WebHID. Use desktop Chrome or Edge.</p>;
  return (
    <>
      <div class="panel-bar">
        <button
          type="button"
          onClick={() =>
            requestNsoN64().catch((e: Error) => {
              nsoError.value = e.message;
            })
          }
        >
          Connect controller
        </button>
        <span class="muted small">Connect it by USB or Bluetooth first.</span>
      </div>
      {nsoError.value && <p class="error small">{nsoError.value}</p>}
      <ul class="small">
        {pads.map((p) => (
          <li class="mono">
            {p.device.productName} X {p.x.toFixed(2)} Y {p.y.toFixed(2)} {padButtons(p.buttons).join(' ')}
          </li>
        ))}
      </ul>
    </>
  );
}
