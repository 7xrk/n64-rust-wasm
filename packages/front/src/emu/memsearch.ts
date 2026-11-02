/** RAM search over RDRAM (cheat-finder style), plus value reads for watches.
 * Addresses are physical; the UI shows them as KSEG0 (0x80000000 | addr). */

import { RDRAM_SIZE, readBE } from 'n64core';

export const VALUE_TYPES = ['u8', 's8', 'u16', 's16', 'u32', 's32', 'f32'] as const;

export type ValueType = (typeof VALUE_TYPES)[number];

export type CompareOp = 'eq' | 'ne' | 'lt' | 'gt' | 'le' | 'ge' | 'changed' | 'unchanged' | 'increased' | 'decreased' | 'changedBy';

export interface SearchResult {
  addr: number;
  value: number;
  previous: number;
}

export function typeSize(type: ValueType): 1 | 2 | 4 {
  return type === 'u8' || type === 's8' ? 1 : type === 'u16' || type === 's16' ? 2 : 4;
}

const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);

export function readValue(rdram: Uint32Array, addr: number, type: ValueType): number {
  const raw = readBE(rdram, addr, typeSize(type));
  switch (type) {
    case 's8':
      return (raw << 24) >> 24;
    case 's16':
      return (raw << 16) >> 16;
    case 's32':
      return raw | 0;
    case 'f32':
      u32[0] = raw;
      return f32[0];
    default:
      return raw;
  }
}

export class MemorySearch {
  type: ValueType = 'u8';
  /** null means every aligned address is still a candidate. */
  private candidates: Uint32Array | null = null;
  private previous: Uint32Array | null = null;

  /** Starts a new search and remembers the current RAM as the baseline. */
  reset(rdram: Uint32Array, type: ValueType): number {
    this.type = type;
    this.candidates = null;
    this.previous = rdram.slice();
    return this.count;
  }

  get count(): number {
    return this.candidates ? this.candidates.length : RDRAM_SIZE / typeSize(this.type);
  }

  /** Keeps the candidates matching `op`, then makes the current RAM the baseline. */
  filter(rdram: Uint32Array, op: CompareOp, operand?: number): number {
    const prev = this.previous ?? rdram;
    const size = typeSize(this.type);
    const test = (now: number, before: number): boolean => {
      const v = operand ?? before;
      switch (op) {
        case 'eq':
          return now === v;
        case 'ne':
          return now !== v;
        case 'lt':
          return now < v;
        case 'gt':
          return now > v;
        case 'le':
          return now <= v;
        case 'ge':
          return now >= v;
        case 'changed':
          return now !== before;
        case 'unchanged':
          return now === before;
        case 'increased':
          return now > before;
        case 'decreased':
          return now < before;
        case 'changedBy':
          return now - before === (operand ?? 0);
      }
    };
    const out: number[] = [];
    const visit = (addr: number) => {
      if (test(readValue(rdram, addr, this.type), readValue(prev, addr, this.type))) out.push(addr);
    };
    if (this.candidates) {
      for (const addr of this.candidates) visit(addr);
    } else {
      for (let addr = 0; addr < RDRAM_SIZE; addr += size) visit(addr);
    }
    this.candidates = Uint32Array.from(out);
    this.previous = rdram.slice();
    return this.count;
  }

  results(rdram: Uint32Array, limit: number): SearchResult[] {
    if (!this.candidates) return [];
    const prev = this.previous ?? rdram;
    return [...this.candidates.subarray(0, limit)].map((addr) => ({
      addr,
      value: readValue(rdram, addr, this.type),
      previous: readValue(prev, addr, this.type),
    }));
  }
}
