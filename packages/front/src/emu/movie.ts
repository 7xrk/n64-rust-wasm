/** TAS input movie: one packed pad value per port per VI frame. */

export type MovieMode = 'off' | 'record' | 'play';

const PORTS = 4;
const MAGIC = 'N64TAS1';
const HISTORY_LIMIT = 200;

/** The rows of an edited range before and after one edit. */
interface Edit {
  frame: number;
  before: Uint32Array;
  after: Uint32Array;
  lenBefore: number;
  lenAfter: number;
}

/** A recording run in progress; it becomes one undo step when it ends. */
interface Run {
  frame: number;
  next: number;
  before: Uint32Array;
  lenBefore: number;
}

export interface MovieInfo {
  mode: MovieMode;
  length: number;
  rerecords: number;
  author: string;
  canUndo: boolean;
  canRedo: boolean;
  /** Changes whenever any input changes, so views know to reload rows. */
  revision: number;
}

export class Movie {
  mode: MovieMode = 'off';
  rerecords = 0;
  author = '';
  /** Cartridge saves the movie powers on from; null means erased saves. */
  save: Uint8Array | null = null;
  private data = new Uint32Array(PORTS * 1024);
  private len = 0;
  private revision = 0;
  private readonly undoStack: Edit[] = [];
  private readonly redoStack: Edit[] = [];
  private run: Run | null = null;

  get length(): number {
    return this.len;
  }

  private ensure(frames: number): void {
    if (frames * PORTS <= this.data.length) return;
    const grown = new Uint32Array(Math.max(frames * PORTS, this.data.length * 2));
    grown.set(this.data);
    this.data = grown;
  }

  /** The four port values for `frame`, or null past the end. */
  get(frame: number): Uint32Array | null {
    if (frame >= this.len) return null;
    return this.data.subarray(frame * PORTS, frame * PORTS + PORTS);
  }

  /** Runs an edit of `span` rows from `frame` (null: to the end) and records it for undo. */
  private change(frame: number, span: number | null, apply: () => void): void {
    this.endRecording();
    const rows = (len: number) => (frame + (span ?? Math.max(0, len - frame))) * PORTS;
    const lenBefore = this.len;
    const before = this.data.slice(frame * PORTS, rows(lenBefore));
    apply();
    const after = this.data.slice(frame * PORTS, rows(this.len));
    this.pushUndo({ frame, before, after, lenBefore, lenAfter: this.len });
    this.revision++;
  }

  private pushUndo(edit: Edit): void {
    this.undoStack.push(edit);
    if (this.undoStack.length > HISTORY_LIMIT) this.undoStack.shift();
    this.redoStack.length = 0;
  }

  /** Ends the recording run in progress as one undo step, unless it changed nothing. */
  endRecording(): void {
    const run = this.run;
    if (!run) return;
    this.run = null;
    const after = this.data.slice(run.frame * PORTS, this.len * PORTS);
    if (this.len === run.lenBefore && after.every((v, i) => v === run.before[i])) return;
    this.pushUndo({ frame: run.frame, before: run.before, after, lenBefore: run.lenBefore, lenAfter: this.len });
  }

  /** Sets the length; rows past a shrunk end are zeroed so a later extension starts clean. */
  private setLength(len: number): void {
    if (len < this.len) this.data.fill(0, len * PORTS, this.len * PORTS);
    this.len = len;
  }

  private restore(frame: number, rows: Uint32Array, len: number): void {
    this.ensure(frame + rows.length / PORTS);
    this.data.set(rows, frame * PORTS);
    this.setLength(len);
    this.revision++;
  }

  /** Overwrites one port for consecutive frames starting at `frame`. */
  setRange(frame: number, port: number, values: number[]): void {
    this.change(frame, values.length, () => {
      this.ensure(frame + values.length);
      values.forEach((v, i) => (this.data[(frame + i) * PORTS + port] = v >>> 0));
      this.len = Math.max(this.len, frame + values.length);
    });
  }

  /** Reverts the last edit. Returns the first frame it touched. */
  undo(): number | null {
    this.endRecording();
    const e = this.undoStack.pop();
    if (!e) return null;
    this.restore(e.frame, e.before, e.lenBefore);
    this.redoStack.push(e);
    return e.frame;
  }

  redo(): number | null {
    this.endRecording();
    const e = this.redoStack.pop();
    if (!e) return null;
    this.restore(e.frame, e.after, e.lenAfter);
    this.undoStack.push(e);
    return e.frame;
  }

  /** Records all ports at `frame` and drops everything after it. Consecutive
   * frames form one run, which endRecording turns into a single undo step. */
  record(frame: number, pads: Uint32Array): void {
    if (this.run?.next !== frame) {
      this.endRecording();
      this.run = { frame, next: frame, before: this.data.slice(frame * PORTS, this.len * PORTS), lenBefore: this.len };
    }
    this.run.next = frame + 1;
    this.ensure(frame + 1);
    this.data.set(pads.subarray(0, PORTS), frame * PORTS);
    this.setLength(frame + 1);
    this.revision++;
  }

  /** Inserts or removes frames at `frame`, shifting the rest. */
  splice(frame: number, deleteCount: number, insertCount: number): void {
    this.change(frame, null, () => {
      const tail = this.data.slice((frame + deleteCount) * PORTS, this.len * PORTS);
      const len = Math.max(frame, this.len - deleteCount + insertCount);
      this.ensure(Math.max(len, this.len));
      this.data.fill(0, frame * PORTS, (frame + insertCount) * PORTS);
      this.data.set(tail, (frame + insertCount) * PORTS);
      this.setLength(len);
    });
  }

  /** Copies `count` frames starting at `start` for the piano roll. */
  slice(start: number, count: number): Uint32Array {
    const end = Math.min(this.len, start + count);
    return this.data.slice(start * PORTS, Math.max(start, end) * PORTS);
  }

  info(): MovieInfo {
    return {
      mode: this.mode,
      length: this.len,
      rerecords: this.rerecords,
      author: this.author,
      canUndo: this.undoStack.length > 0 || this.run !== null,
      canRedo: this.redoStack.length > 0,
      revision: this.revision,
    };
  }

  clear(): void {
    this.setLength(0);
    this.rerecords = 0;
    this.save = null;
    this.undoStack.length = 0;
    this.redoStack.length = 0;
    this.run = null;
    this.revision++;
  }

  /** Serialized form: a JSON header line, little-endian input words, the power-on saves, then the snapshots packed
   * by Snapshots.pack. The inputs alone define the movie; the snapshots only spare replaying it. */
  serialize(romName: string, snapshots = new Uint8Array(0)): Uint8Array<ArrayBuffer> {
    const header = new TextEncoder().encode(
      `${JSON.stringify({ magic: MAGIC, rom: romName, frames: this.len, ports: PORTS, rerecords: this.rerecords, author: this.author, save: this.save?.length ?? 0, snapshots: snapshots.length })}\n`,
    );
    const body = new Uint8Array(this.data.buffer, 0, this.len * PORTS * 4);
    const save = this.save ?? new Uint8Array(0);
    const out = new Uint8Array(header.length + body.length + save.length + snapshots.length);
    out.set(header);
    out.set(body, header.length);
    out.set(save, header.length + body.length);
    out.set(snapshots, header.length + body.length + save.length);
    return out;
  }

  static parse(bytes: Uint8Array): { movie: Movie; snapshots: Uint8Array | null } {
    const nl = bytes.indexOf(10);
    if (nl < 0) throw new Error('Not a movie file');
    const header = JSON.parse(new TextDecoder().decode(bytes.subarray(0, nl)));
    if (header.magic !== MAGIC) throw new Error('Not a movie file');
    const bodyEnd = nl + 1 + header.frames * PORTS * 4;
    const body = bytes.slice(nl + 1, bodyEnd);
    const movie = new Movie();
    movie.data = new Uint32Array(body.buffer, 0, header.frames * PORTS);
    movie.len = header.frames;
    movie.rerecords = header.rerecords ?? 0;
    movie.author = header.author ?? '';
    const saveEnd = bodyEnd + (header.save ?? 0);
    movie.save = header.save ? bytes.slice(bodyEnd, saveEnd) : null;
    movie.ensure(movie.len + 1024);
    return { movie, snapshots: header.snapshots ? bytes.subarray(saveEnd, saveEnd + header.snapshots) : null };
  }
}
