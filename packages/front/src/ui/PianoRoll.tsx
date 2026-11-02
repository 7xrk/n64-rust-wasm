/** TAS input editor: one row per frame, one column per button. Edits go to the
 * worker as single undoable operations; edits before the current frame are
 * re-emulated there so the screen always matches the movie. */

import { useEffect, useRef, useState } from 'preact/hooks';
import { BUTTONS, STICK_MAX, stickX, stickY, withStick } from 'n64core';
import type { MovieMode } from '../emu/movie';
import { BlobImage } from './BlobImage';
import { baseName, download, emu, MOVIE_FILE, pins, romName, run, status } from './client';
import { StickPad } from './StickPad';

const ROW = 22;
const PORTS = 4;
/** Rows shown past the movie so inputs can be written ahead of time. */
const LOOKAHEAD = 120;

const MODES: { mode: MovieMode; label: string; title: string }[] = [
  { mode: 'off', label: 'Free', title: 'Inputs are not written to the movie' },
  { mode: 'record', label: 'Record', title: 'Writes controller input to the current frame' },
  { mode: 'play', label: 'Play', title: 'Runs on the movie inputs and pauses at the end' },
];

interface Rows {
  start: number;
  inputs: Uint32Array;
  lag: Uint8Array;
  snaps: Uint8Array;
}

interface Paint {
  bit: number;
  on: boolean;
  from: number;
  to: number;
}

let clipboard: number[] = [];

export function PianoRoll() {
  const s = status.value;
  // A seek reports the frames it passes; the timeline stays on where it is heading.
  const playhead = s.seekTarget ?? s.frame;
  const loaded = !!romName.value;
  const scroller = useRef<HTMLDivElement>(null);
  const [scrollTop, setScrollTop] = useState(0);
  const [height, setHeight] = useState(400);
  const [port, setPort] = useState(0);
  const [anchor, setAnchor] = useState<number | null>(null);
  const [focus, setFocus] = useState<number | null>(null);
  const [paint, setPaint] = useState<Paint | null>(null);
  const [follow, setFollow] = useState(true);
  /** Row operation whose result is previewed on the selection while its button is hovered. */
  const [rowPreview, setRowPreview] = useState<'insert' | 'delete' | null>(null);
  const [stickPreview, setStickPreview] = useState<{ x: number; y: number } | null>(null);
  const [rows, setRows] = useState<Rows>({ start: 0, inputs: new Uint32Array(), lag: new Uint8Array(), snaps: new Uint8Array() });

  const total = Math.max(s.movie.length, playhead) + 1 + LOOKAHEAD;
  const first = Math.floor(scrollTop / ROW);
  const count = Math.ceil(height / ROW) + 2;
  const selLo = anchor === null || focus === null ? null : Math.min(anchor, focus);
  const selHi = anchor === null || focus === null ? null : Math.max(anchor, focus);

  useEffect(() => {
    const el = scroller.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setHeight(el.clientHeight));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const el = scroller.current;
    if (!follow || !el) return;
    const target = playhead * ROW - el.clientHeight / 3;
    if (Math.abs(el.scrollTop - target) > ROW * 4) el.scrollTop = Math.max(0, target);
  }, [playhead, follow]);

  useEffect(() => {
    if (!loaded) return;
    let live = true;
    run(emu.movieSlice(first, count)).then((r) => {
      if (live && r) setRows({ start: first, ...r });
    });
    return () => {
      live = false;
    };
  }, [loaded, first, count, playhead, s.movie.rerecords, s.movie.revision]);

  const stored = (frame: number): number => {
    const i = frame - rows.start;
    if (i < 0 || i * PORTS + port >= rows.inputs.length) return 0;
    return rows.inputs[i * PORTS + port];
  };

  /** Value shown for `frame`, including a paint stroke in progress. */
  const shown = (frame: number): number => {
    const v = stored(frame);
    if (!paint || frame < Math.min(paint.from, paint.to) || frame > Math.max(paint.from, paint.to)) return v;
    return paint.on ? v | paint.bit : v & ~paint.bit;
  };

  /** Writes edited values into the loaded rows. The worker answers an edit only
   * after re-emulating up to the current frame, so waiting for it to reload the
   * rows would leave the old values on screen until then. */
  const showEdit = (frame: number, values: number[]) =>
    setRows((r) => {
      const end = Math.min(frame + values.length, r.start + count);
      const inputs = new Uint32Array(Math.max(r.inputs.length, (end - r.start) * PORTS));
      inputs.set(r.inputs);
      for (let f = Math.max(frame, r.start); f < end; f++) inputs[(f - r.start) * PORTS + port] = values[f - frame] >>> 0;
      return { ...r, inputs };
    });

  const edit = (frame: number, values: number[]) => {
    showEdit(frame, values);
    return run(
      emu.editInputs(
        frame,
        port,
        values.map((v) => v >>> 0),
      ),
    );
  };

  const frameAt = (clientY: number): number => {
    const el = scroller.current!;
    return Math.max(0, Math.floor((clientY - el.getBoundingClientRect().top + el.scrollTop) / ROW));
  };

  const endPaint = () => {
    if (!paint) return;
    const lo = Math.min(paint.from, paint.to);
    const hi = Math.max(paint.from, paint.to);
    const values = [];
    for (let f = lo; f <= hi; f++) values.push(shown(f));
    setPaint(null);
    edit(lo, values);
  };

  const select = (frame: number, extend: boolean) => {
    if (!extend || anchor === null) setAnchor(frame);
    setFocus(frame);
  };

  /** Values of the edited port on the selected frames, or null without a selection.
   * They are fetched because the selection can extend past the loaded rows. */
  const selectedValues = async (): Promise<number[] | null> => {
    if (selLo === null || selHi === null) return null;
    const length = selHi - selLo + 1;
    const r = await run(emu.movieSlice(selLo, length));
    if (!r) return null;
    return Array.from({ length }, (_, i) => r.inputs[i * PORTS + port] ?? 0);
  };

  /** Sets the given axes on every selected frame and keeps the other axis of each. */
  const setStick = async (x: number | null, y: number | null) => {
    if (selLo === null || selHi === null) return;
    const apply = (v: number) => withStick(v, x ?? stickX(v), y ?? stickY(v));
    // Fetching the selection waits behind an edit still re-emulating, so the
    // loaded rows show the result first.
    showEdit(
      selLo,
      Array.from({ length: selHi - selLo + 1 }, (_, i) => apply(stored(selLo + i))),
    );
    const values = await selectedValues();
    if (values) edit(selLo, values.map(apply));
  };

  const insertRows = () => {
    if (selLo !== null && selHi !== null) run(emu.spliceMovie(selLo, 0, selHi - selLo + 1));
  };

  const deleteRows = () => {
    if (selLo !== null && selHi !== null) run(emu.spliceMovie(selLo, selHi - selLo + 1, 0));
  };

  const onKeyDown = (e: KeyboardEvent) => {
    const mod = e.metaKey || e.ctrlKey;
    const handled = () => {
      e.preventDefault();
      e.stopPropagation();
    };
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      handled();
      const next = Math.max(0, (focus ?? playhead) + (e.key === 'ArrowDown' ? 1 : -1));
      select(next, e.shiftKey);
      const el = scroller.current!;
      if (next * ROW < el.scrollTop) el.scrollTop = next * ROW;
      if ((next + 1) * ROW > el.scrollTop + el.clientHeight) el.scrollTop = (next + 1) * ROW - el.clientHeight;
    } else if (e.key === 'Enter' && focus !== null) {
      handled();
      run(emu.seek(focus));
    } else if ((e.key === 'Delete' || e.key === 'Backspace') && selLo !== null) {
      handled();
      deleteRows();
    } else if (mod && e.code === 'KeyI' && selLo !== null) {
      handled();
      insertRows();
    } else if (mod && e.code === 'KeyC' && selLo !== null) {
      handled();
      selectedValues().then((values) => {
        if (values) clipboard = values;
      });
    } else if (mod && e.code === 'KeyV' && selLo !== null && clipboard.length > 0) {
      handled();
      edit(selLo, clipboard);
    }
  };

  const pinned = new Set(pins.value.map((p) => p.frame));
  const list = [];
  for (let frame = first; frame < Math.min(total, first + count); frame++) {
    const i = frame - rows.start;
    // Previewing an insert draws the following rows pushed down by the selection length.
    const pushed = rowPreview === 'insert' && selLo !== null && selHi !== null && frame > selHi;
    const v = pushed ? stored(frame - (selHi - selLo + 1)) : shown(frame);
    const inMovie = frame < s.movie.length;
    const lagged = rows.lag[i] === 1 && frame < playhead;
    const selected = selLo !== null && selHi !== null && frame >= selLo && frame <= selHi;
    const stick = selected && stickPreview ? stickPreview : { x: stickX(v), y: stickY(v) };
    const cls = [
      'row',
      frame === playhead ? 'current' : '',
      frame < playhead ? 'past' : '',
      selected ? 'selected' : '',
      selected && rowPreview ? rowPreview : '',
      lagged ? 'lagged' : '',
      inMovie ? '' : 'beyond',
    ]
      .filter(Boolean)
      .join(' ');
    list.push(
      <div class={cls} style={{ top: `${frame * ROW}px` }} key={frame}>
        <span
          class="cell frame"
          title="Click to select, double-click to go to this frame"
          onPointerDown={(e) => select(frame, e.shiftKey)}
          onDblClick={() => run(emu.seek(frame))}
        >
          {frame}
        </span>
        <span
          class={`cell mark ${pinned.has(frame) ? 'pinned' : ''}`}
          title={lagged ? 'Lag frame' : rows.snaps[i] ? 'Has a snapshot' : ''}
          onClick={() => run(emu.togglePin(frame))}
        >
          {pinned.has(frame) ? '📌' : lagged ? 'L' : rows.snaps[i] ? '•' : ''}
        </span>
        {BUTTONS.map((b) => {
          const on = (v & b.bit) !== 0;
          return (
            <span
              class={`cell btn ${on ? 'on' : ''}`}
              onPointerDown={(e) => {
                if (e.button !== 0) return;
                (e.currentTarget.parentElement!.parentElement as HTMLElement).setPointerCapture(e.pointerId);
                setPaint({ bit: b.bit, on: !on, from: frame, to: frame });
              }}
            >
              {b.label}
            </span>
          );
        })}
        <span class={`cell axis ${selected && stickPreview ? 'previewing' : ''}`}>{inMovie || v ? stick.x : ''}</span>
        <span class={`cell axis ${selected && stickPreview ? 'previewing' : ''}`}>{inMovie || v ? stick.y : ''}</span>
      </div>,
    );
  }

  const focusValue = focus === null ? 0 : stored(focus);

  return (
    <section class="panel piano">
      <div class="panel-bar">
        <div class="segmented" role="group" aria-label="Movie mode">
          {MODES.map((m) => (
            <button
              type="button"
              title={m.title}
              class={s.movie.mode === m.mode ? 'active' : ''}
              onClick={() => run(emu.setMovieMode(m.mode))}
            >
              {m.label}
            </button>
          ))}
        </div>
        <button type="button" disabled={!s.movie.canUndo} onClick={() => run(emu.undoMovie())} title="Undo (⌘Z)">
          ↶
        </button>
        <button type="button" disabled={!s.movie.canRedo} onClick={() => run(emu.redoMovie())} title="Redo (⌘⇧Z)">
          ↷
        </button>
        <select value={port} onChange={(e) => setPort(Number(e.currentTarget.value))} title="Controller to edit">
          {[0, 1, 2, 3].map((p) => (
            <option value={p}>P{p + 1}</option>
          ))}
        </select>
        <label class="check">
          <input type="checkbox" checked={follow} onChange={(e) => setFollow(e.currentTarget.checked)} />
          Follow
        </label>
        <span class="spacer" />
        <button
          type="button"
          disabled={!loaded}
          onClick={() => {
            if (
              confirm(
                'Power-cycles with the current saves and starts recording an empty movie. The saves are stored in the movie. The current movie is discarded.',
              )
            )
              run(emu.newMovie(''));
          }}
        >
          New
        </button>
        <button type="button" disabled={!loaded} onClick={async () => download((await run(emu.exportMovie()))!, `${baseName()}.n64tas`)}>
          Save
        </button>
        <label class="button">
          Load
          <input
            type="file"
            accept={MOVIE_FILE.accept}
            hidden
            onChange={(e) => {
              const file = e.currentTarget.files?.[0];
              if (file) MOVIE_FILE.open(file);
            }}
          />
        </label>
      </div>
      <div class="panel-bar muted small">
        <span>{s.movie.length} frames</span>
        <span>Rerecords {s.movie.rerecords}</span>
        <span>
          Snapshots {s.snapshots} / {(s.snapshotBytes / 1048576).toFixed(0)} MB
        </span>
      </div>
      <div class="row header">
        <span class="cell frame">frame</span>
        <span class="cell mark" />
        {BUTTONS.map((b) => (
          <span class="cell btn">{b.label}</span>
        ))}
        <span class="cell axis">X</span>
        <span class="cell axis">Y</span>
      </div>
      <div
        class="roll"
        ref={scroller}
        tabIndex={0}
        onScroll={(e) => setScrollTop(e.currentTarget.scrollTop)}
        onKeyDown={onKeyDown}
        onPointerMove={(e) => paint && setPaint({ ...paint, to: frameAt(e.clientY) })}
        onPointerUp={endPaint}
        onPointerCancel={endPaint}
      >
        <div style={{ height: `${total * ROW}px`, position: 'relative' }}>
          {list}
          {selLo !== null && selHi !== null && (
            <>
              <button
                type="button"
                class="row-insert"
                style={{ top: `${selLo * ROW}px` }}
                onPointerEnter={() => setRowPreview('insert')}
                onPointerLeave={() => setRowPreview(null)}
                onClick={insertRows}
              >
                +
              </button>
              <button
                type="button"
                class="row-delete"
                style={{ top: `${selLo * ROW}px`, height: `${(selHi - selLo + 1) * ROW}px` }}
                onPointerEnter={() => setRowPreview('delete')}
                onPointerLeave={() => setRowPreview(null)}
                onClick={deleteRows}
              >
                ×
              </button>
            </>
          )}
        </div>
      </div>
      {pins.value.length > 0 && (
        <div class="pin-strip">
          {pins.value.map((p) => {
            const rendering = p.stale || !p.thumbnail;
            return (
              <div class={`pin ${rendering ? 'rendering' : ''} ${p.frame === playhead ? 'current' : ''}`} key={p.frame}>
                <button type="button" class="pin-shot" onClick={() => run(emu.seek(p.frame))}>
                  <span class="pin-image">{p.thumbnail && <BlobImage blob={p.thumbnail} width={128} height={96} />}</span>
                  <span>F{p.frame}</span>
                </button>
                {rendering ? (
                  <span class="pin-spinner" />
                ) : (
                  <button type="button" class="pin-remove" onClick={() => run(emu.togglePin(p.frame))}>
                    ×
                  </button>
                )}
              </div>
            );
          })}
        </div>
      )}
      <div class="stick-editor">
        <StickPad
          x={stickX(focusValue)}
          y={stickY(focusValue)}
          disabled={selLo === null}
          onDrag={setStickPreview}
          onCommit={(x, y) => setStick(x, y)}
        />
        <div class="stick-fields">
          <span class="muted small">
            {selLo === null ? 'Click a frame number to select' : selLo === selHi ? `Frame ${selLo}` : `Frames ${selLo}–${selHi}`}
          </span>
          <label>
            X
            <input
              class="narrow"
              type="number"
              min={-STICK_MAX}
              max={STICK_MAX}
              disabled={selLo === null}
              value={stickX(focusValue)}
              onChange={(e) => setStick(Number(e.currentTarget.value), null)}
            />
          </label>
          <label>
            Y
            <input
              class="narrow"
              type="number"
              min={-STICK_MAX}
              max={STICK_MAX}
              disabled={selLo === null}
              value={stickY(focusValue)}
              onChange={(e) => setStick(null, Number(e.currentTarget.value))}
            />
          </label>
          <button type="button" disabled={selLo === null} onClick={() => setStick(0, 0)}>
            Center
          </button>
        </div>
      </div>
    </section>
  );
}
