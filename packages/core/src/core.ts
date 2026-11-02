/** Typed wrapper around the Rust core's flat C ABI (crate/src/lib.rs). */

interface Exports {
  memory: WebAssembly.Memory;
  __indirect_function_table: WebAssembly.Table;
  n64_rsp_jit_op(instr: number): void;
  n64_rom_alloc(len: number): number;
  n64_load_rom(): void;
  n64_reset(): void;
  n64_power_on(): void;
  n64_save_ptr(): number;
  n64_save_size(): number;
  n64_run_frame(): number;
  n64_set_pad(port: number, value: number): void;
  n64_set_connected(mask: number): void;
  n64_frame(): number;
  n64_time(): number;
  n64_state_ptr(): number;
  n64_state_size(): number;
  n64_state_loaded(): number;
  n64_rdram_ptr(): number;
  n64_rdram_size(): number;
  n64_poke8(paddr: number, v: number): void;
  n64_poke16(paddr: number, v: number): void;
  n64_poke32(paddr: number, v: number): void;
  n64_virt_to_phys(vaddr: number): number;
  n64_dirty_ptr(): number;
  n64_dirty_clear(): void;
  n64_snap_dirty_ptr(): number;
  n64_snap_dirty_clear(): void;
  n64_state_rdram_offset(): number;
  n64_vi_regs_ptr(): number;
  n64_gpu_enable(on: number): void;
  n64_gpu_clear(): void;
  n64_gpu_states_ptr(): number;
  n64_gpu_states_len(): number;
  n64_gpu_prims_ptr(): number;
  n64_gpu_prims_len(): number;
  n64_gpu_batches_ptr(): number;
  n64_gpu_batches_len(): number;
  n64_gpu_tmem_ptr(): number;
  n64_gpu_tmem_len(): number;
  n64_audio_ptr(): number;
  n64_audio_len(): number;
  n64_audio_rate(): number;
  n64_audio_clear(): void;
}

/** Compiles an RSP block module built by crate/src/rsp_jit.rs and adds its function to the core's table. Returns the
 * table slot, or 0 so the core interprets that code instead. */
function compileRspBlock(x: Exports, ptr: number, len: number): number {
  try {
    const module = new WebAssembly.Module(new Uint8Array(x.memory.buffer, ptr, len));
    const instance = new WebAssembly.Instance(module, { env: { memory: x.memory, op: x.n64_rsp_jit_op } });
    const table = x.__indirect_function_table;
    const slot = table.grow(1);
    table.set(slot, instance.exports.run);
    return slot;
  } catch (e) {
    console.error('RSP block compile failed', e);
    return 0;
  }
}

export const RDRAM_SIZE = 0x80_0000;
export const PAGE_SIZE = 4096;

export class Core {
  /** SHA-256 of the wasm, in hex. A state kept across sessions is valid only for the build that made it, since a
   * rebuilt core may emulate the same inputs differently even when the state layout is unchanged. */
  readonly build: string;

  private constructor(
    private readonly x: Exports,
    digest: ArrayBuffer,
  ) {
    this.build = Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, '0')).join('');
  }

  static async load(url: string): Promise<Core> {
    const wasm = await (await fetch(url)).arrayBuffer();
    let x: Exports | undefined;
    const env = { rsp_jit_compile: (ptr: number, len: number) => compileRspBlock(x!, ptr, len) };
    const [{ instance }, digest] = await Promise.all([WebAssembly.instantiate(wasm, { env }), crypto.subtle.digest('SHA-256', wasm)]);
    x = instance.exports as unknown as Exports;
    return new Core(x, digest);
  }

  // Views are recreated on every access because memory.grow detaches them.
  private u32(ptr: number, len: number): Uint32Array<ArrayBuffer> {
    return new Uint32Array(this.x.memory.buffer, ptr, len);
  }

  loadRom(rom: Uint8Array): void {
    const ptr = this.x.n64_rom_alloc(rom.length);
    new Uint8Array(this.x.memory.buffer, ptr, rom.length).set(rom);
    this.x.n64_load_rom();
  }

  /** Soft reset; the cartridge saves survive. */
  reset(): void {
    this.x.n64_reset();
  }

  /** Copy of the cartridge saves (EEPROM then SRAM). */
  saveData(): Uint8Array<ArrayBuffer> {
    return new Uint8Array(this.x.memory.buffer, this.x.n64_save_ptr(), this.x.n64_save_size()).slice();
  }

  /** Power cycle from `save` as returned by saveData, or from erased saves. */
  powerOn(save?: Uint8Array): void {
    if (!save) {
      this.x.n64_power_on();
      return;
    }
    if (save.length !== this.x.n64_save_size()) throw new Error('Save data size does not match this core');
    new Uint8Array(this.x.memory.buffer, this.x.n64_save_ptr(), save.length).set(save);
    this.x.n64_reset();
  }

  /** Runs one VI field. Returns true when the game polled controller 1. */
  runFrame(): boolean {
    return this.x.n64_run_frame() !== 0;
  }

  get frame(): number {
    return this.x.n64_frame();
  }

  /** Emulated seconds since power-on. */
  get time(): number {
    return this.x.n64_time();
  }

  setPad(port: number, value: number): void {
    this.x.n64_set_pad(port, value);
  }

  setConnected(mask: number): void {
    this.x.n64_set_connected(mask);
  }

  saveState(): Uint8Array<ArrayBuffer> {
    return new Uint8Array(this.x.memory.buffer, this.x.n64_state_ptr(), this.x.n64_state_size()).slice();
  }

  /** Returns false when the state is from an incompatible core version. */
  loadState(state: Uint8Array): boolean {
    if (state.length !== this.x.n64_state_size()) return false;
    new Uint8Array(this.x.memory.buffer, this.x.n64_state_ptr(), state.length).set(state);
    return this.x.n64_state_loaded() === 0;
  }

  /** Splits the state into the non-RDRAM prefix and RDRAM pages. Pages not
   * written since the last call are taken from `base` instead of copied. */
  capturePages(base: Uint8Array[] | null): { prefix: Uint8Array<ArrayBuffer>; pages: Uint8Array[]; copied: number } {
    const mem = new Uint8Array(this.x.memory.buffer);
    const ptr = this.x.n64_state_ptr();
    const rdram = ptr + this.x.n64_state_rdram_offset();
    const dirty = this.u32(this.x.n64_snap_dirty_ptr(), RDRAM_SIZE / PAGE_SIZE / 32);
    const pages: Uint8Array[] = [];
    let copied = 0;
    for (let p = 0; p < RDRAM_SIZE / PAGE_SIZE; p++) {
      const changed = (dirty[p >>> 5] >>> (p & 31)) & 1;
      if (base && !changed) {
        pages.push(base[p]);
        continue;
      }
      pages.push(mem.slice(rdram + p * PAGE_SIZE, rdram + (p + 1) * PAGE_SIZE));
      copied++;
    }
    this.x.n64_snap_dirty_clear();
    return { prefix: mem.slice(ptr, rdram), pages, copied };
  }

  /** Loads a state produced by capturePages; it becomes the base for the next capture. */
  restorePages(prefix: Uint8Array, pages: Uint8Array[]): boolean {
    const mem = new Uint8Array(this.x.memory.buffer);
    const ptr = this.x.n64_state_ptr();
    mem.set(prefix, ptr);
    const rdram = ptr + this.x.n64_state_rdram_offset();
    pages.forEach((page, p) => mem.set(page, rdram + p * PAGE_SIZE));
    const ok = this.x.n64_state_loaded() === 0;
    this.x.n64_snap_dirty_clear();
    return ok;
  }

  /** RDRAM words in host order; byte `a` is at bit `(3 - (a & 3)) * 8` of word `a >> 2`. */
  rdram(): Uint32Array<ArrayBuffer> {
    return this.u32(this.x.n64_rdram_ptr(), RDRAM_SIZE / 4);
  }

  poke(paddr: number, value: number, bytes: 1 | 2 | 4): void {
    if (bytes === 1) this.x.n64_poke8(paddr, value);
    else if (bytes === 2) this.x.n64_poke16(paddr, value);
    else this.x.n64_poke32(paddr, value);
  }

  /** Physical address for a CPU virtual address, or null when unmapped. */
  virtToPhys(vaddr: number): number | null {
    const p = this.x.n64_virt_to_phys(vaddr >>> 0) >>> 0;
    return p === 0xffff_ffff ? null : p;
  }

  dirtyPages(): Uint32Array<ArrayBuffer> {
    return this.u32(this.x.n64_dirty_ptr(), RDRAM_SIZE / PAGE_SIZE / 32);
  }

  clearDirty(): void {
    this.x.n64_dirty_clear();
  }

  viRegs(): Uint32Array<ArrayBuffer> {
    return this.u32(this.x.n64_vi_regs_ptr(), 14);
  }

  setRendering(on: boolean): void {
    this.x.n64_gpu_enable(on ? 1 : 0);
  }

  gpuStates(): Uint32Array<ArrayBuffer> {
    return this.u32(this.x.n64_gpu_states_ptr(), this.x.n64_gpu_states_len());
  }

  gpuPrims(): Uint32Array<ArrayBuffer> {
    return this.u32(this.x.n64_gpu_prims_ptr(), this.x.n64_gpu_prims_len());
  }

  gpuBatches(): Uint32Array<ArrayBuffer> {
    return this.u32(this.x.n64_gpu_batches_ptr(), this.x.n64_gpu_batches_len());
  }

  gpuTmem(): Uint16Array<ArrayBuffer> {
    return new Uint16Array(this.x.memory.buffer, this.x.n64_gpu_tmem_ptr(), this.x.n64_gpu_tmem_len());
  }

  gpuClear(): void {
    this.x.n64_gpu_clear();
  }

  /** Drains the interleaved stereo samples produced since the last call. */
  takeAudio(): { samples: Int16Array; rate: number } {
    const len = this.x.n64_audio_len();
    const samples = new Int16Array(this.x.memory.buffer, this.x.n64_audio_ptr(), len).slice();
    this.x.n64_audio_clear();
    return { samples, rate: this.x.n64_audio_rate() };
  }
}

/** Reads a big-endian value at a physical RDRAM address. */
export function readBE(rdram: Uint32Array, paddr: number, bytes: 1 | 2 | 4): number {
  const w = rdram[(paddr >>> 2) & (RDRAM_SIZE / 4 - 1)];
  if (bytes === 4) return w >>> 0;
  if (bytes === 2) return (paddr & 2 ? w : w >>> 16) & 0xffff;
  return (w >>> ((3 - (paddr & 3)) * 8)) & 0xff;
}
