import { signal } from '@preact/signals';
import { useEffect, useRef, useState } from 'preact/hooks';
import { AiPanel } from './AiPanel';
import {
  attachCanvas,
  baseName,
  download,
  emu,
  errorMessage,
  liveInput,
  openFiles,
  ROM_FILE,
  romName,
  run,
  STATE_FILE,
  status,
} from './client';
import { isTextEntry } from './input';
import { InputDisplay } from './InputDisplay';
import { CheatPanel, SearchPanel, WatchPanel } from './MemoryPanel';
import { PianoRoll } from './PianoRoll';
import { attachPicker, openPicker, pickerOpen, ScenePicker } from './ScenePicker';
import { SettingsPanel } from './SettingsPanel';

const SPEEDS = [0.25, 0.5, 1, 2, 4, Infinity];
const TABS = [
  { id: 'search', label: 'Memory search', view: SearchPanel },
  { id: 'watch', label: 'Watch', view: WatchPanel },
  { id: 'cheat', label: 'PAR', view: CheatPanel },
  { id: 'ai', label: 'AI', view: AiPanel },
  { id: 'settings', label: 'Settings', view: SettingsPanel },
] as const;

/** Emulator hotkeys. Keys bound to the controller are left to the input manager. */
function onHotkey(e: KeyboardEvent): void {
  if (isTextEntry(e.target) || !romName.value || pickerOpen.value) return;
  const s = status.value;
  if ((e.metaKey || e.ctrlKey) && e.code === 'KeyZ') {
    e.preventDefault();
    run(e.shiftKey ? emu.redoMovie() : emu.undoMovie());
    return;
  }
  if (e.code in liveInput.mapping.keys) return;
  const slot = /^Digit([1-9])$/.exec(e.code)?.[1];
  if (slot) {
    e.preventDefault();
    if (e.shiftKey) run(emu.saveSlot(Number(slot)));
    else run(emu.loadSlot(Number(slot)));
    return;
  }
  const actions: Record<string, () => void> = {
    KeyP: () => run(s.running ? emu.pause() : emu.resume()),
    Period: () => run(emu.frameAdvance()),
    Comma: () => run(emu.seek(s.frame - 1)),
    Backquote: () => run(emu.setSpeed(s.speed === Infinity ? 1 : Infinity)),
    KeyR: () => run(emu.setMovieMode(s.movie.mode === 'record' ? 'play' : 'record')),
  };
  const action = actions[e.code];
  if (!action) return;
  e.preventDefault();
  action();
}

/** True while files are dragged over the window. */
const dropping = signal(false);

const carriesFiles = (e: DragEvent) => !!e.dataTransfer?.types.includes('Files');

function onDragOver(e: DragEvent): void {
  if (!carriesFiles(e)) return;
  // Without preventDefault the browser navigates to the dropped file.
  e.preventDefault();
  dropping.value = true;
}

function onDragLeave(e: DragEvent): void {
  // relatedTarget is null only when the pointer leaves the window.
  if (!e.relatedTarget) dropping.value = false;
}

function onDrop(e: DragEvent): void {
  if (!carriesFiles(e)) return;
  e.preventDefault();
  dropping.value = false;
  openFiles([...e.dataTransfer!.files]);
}

export function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [tab, setTab] = useState<(typeof TABS)[number]['id']>('search');

  useEffect(() => {
    if (canvasRef.current) attachCanvas(canvasRef.current);
    liveInput.attach();
    const detachPicker = attachPicker();
    window.addEventListener('keydown', onHotkey);
    window.addEventListener('dragover', onDragOver);
    window.addEventListener('dragleave', onDragLeave);
    window.addEventListener('drop', onDrop);
    return () => {
      liveInput.detach();
      detachPicker();
      window.removeEventListener('keydown', onHotkey);
      window.removeEventListener('dragover', onDragOver);
      window.removeEventListener('dragleave', onDragLeave);
      window.removeEventListener('drop', onDrop);
    };
  }, []);

  const s = status.value;
  const loaded = !!romName.value;
  const View = TABS.find((t) => t.id === tab)!.view;

  return (
    <div class={`app ${dropping.value ? 'dropping' : ''}`}>
      <header class="toolbar">
        <h1>N64 TAS</h1>
        <label class="button">
          Open ROM
          <input
            type="file"
            accept={ROM_FILE.accept}
            hidden
            onChange={(e) => {
              const file = e.currentTarget.files?.[0];
              if (file) ROM_FILE.open(file);
            }}
          />
        </label>
        <button type="button" onClick={() => run(s.running ? emu.pause() : emu.resume())} disabled={!loaded}>
          {s.running ? 'Pause' : 'Resume'}
        </button>
        <button type="button" onClick={() => run(emu.frameAdvance())} disabled={!loaded}>
          Frame advance
        </button>
        <button type="button" onClick={() => run(emu.reset())} disabled={!loaded}>
          Reset
        </button>
        <select onChange={(e) => run(emu.setSpeed(Number(e.currentTarget.value)))} value={String(s.speed)}>
          {SPEEDS.map((v) => (
            <option value={String(v)}>{v === Infinity ? 'Unlimited' : `${v}x`}</option>
          ))}
        </select>
        {s.recording === null ? (
          <button type="button" onClick={() => run(emu.startRecording())} disabled={!loaded}>
            Start recording
          </button>
        ) : (
          <button
            type="button"
            class="active"
            onClick={async () => {
              const mp4 = await run(emu.stopRecording());
              if (mp4) download(mp4, `${baseName()}.mp4`, 'video/mp4');
            }}
          >
            Stop recording ({s.recording})
          </button>
        )}
        <button
          type="button"
          disabled={!loaded}
          onClick={async () => {
            const png = await run(emu.screenshot());
            if (png) download(png, `${baseName()}-${s.frame}.png`, 'image/png');
          }}
        >
          Screenshot
        </button>
        <button type="button" onClick={openPicker} disabled={!loaded}>
          Scene history
        </button>
        {s.seekTarget !== null && (
          <>
            <span class="stat">
              Seeking {s.frame} → {s.seekTarget}
            </span>
            <button type="button" onClick={() => run(emu.pause())}>
              Cancel
            </button>
          </>
        )}
        <span class="stat">{s.fps.toFixed(1)} fps</span>
      </header>
      {errorMessage.value && (
        <p class="error" onClick={() => (errorMessage.value = null)}>
          {errorMessage.value}
        </p>
      )}
      <div class="workspace">
        <div class="left">
          <main class="screen">
            <canvas ref={canvasRef} width={1280} height={960} />
            {!loaded && <p class="hint">Choose a ROM file</p>}
            <ScenePicker />
          </main>
          <InputDisplay />
          <div class="slots">
            {Array.from({ length: 9 }, (_, i) => i + 1).map((slot) => (
              <div class="slot">
                <span class="muted">{slot}</span>
                <button type="button" disabled={!loaded} onClick={() => run(emu.saveSlot(slot))}>
                  Save
                </button>
                <button
                  type="button"
                  disabled={s.slots[slot] == null}
                  onClick={() => run(emu.loadSlot(slot))}
                  title={s.slots[slot] != null ? `frame ${s.slots[slot]}` : ''}
                >
                  {s.slots[slot] != null ? `F${s.slots[slot]}` : 'Empty'}
                </button>
              </div>
            ))}
            <button
              type="button"
              disabled={!loaded}
              onClick={async () => {
                const state = await run(emu.exportState());
                if (state) download(state, `${baseName()}-${s.frame}.st64`);
              }}
            >
              Export state
            </button>
            <label class="button">
              Import state
              <input
                type="file"
                accept={STATE_FILE.accept}
                hidden
                onChange={(e) => {
                  const file = e.currentTarget.files?.[0];
                  if (file) STATE_FILE.open(file);
                }}
              />
            </label>
          </div>
          <section class="tools">
            <nav class="tabs">
              {TABS.map((t) => (
                <button type="button" class={t.id === tab ? 'active' : ''} onClick={() => setTab(t.id)}>
                  {t.label}
                </button>
              ))}
            </nav>
            <View />
          </section>
        </div>
        <aside class="side">
          <PianoRoll />
        </aside>
      </div>
    </div>
  );
}
