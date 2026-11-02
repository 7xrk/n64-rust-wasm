/** Pro Action Replay / GameShark codes for N64, applied once per frame.
 *
 * Supported code types (address is the low 24 bits of a KSEG0 address):
 *   80/A0 aaaaaa 00vv  write byte          81/A1 aaaaaa vvvv  write halfword
 *   88/89 aaaaaa ....  write on GS button (treated as always on)
 *   D0/D1 aaaaaa ....  next code only if byte/halfword equals the value
 *   D2/D3 aaaaaa ....  next code only if byte/halfword differs from the value
 *   F0/F1 aaaaaa ....  boot-time write; applied every frame because IPL3
 *                      loads the game over RAM after power-on
 *   50 00nnss vvvv     repeat the next code n times, address += ss, value += vvvv
 *   DE / EE / FF       entry point, expansion pak, enabler: ignored */

import { type Core, readBE } from 'n64core';

export interface Cheat {
  id: string;
  name: string;
  code: string;
  enabled: boolean;
}

interface Line {
  op: number;
  addr: number;
  value: number;
}

export function parseCode(code: string): Line[] {
  const lines: Line[] = [];
  for (const raw of code.split(/\n/)) {
    const hex = raw.replace(/[^0-9a-fA-F]/g, '');
    if (hex.length === 0) continue;
    if (hex.length !== 12) throw new Error(`Invalid code line: "${raw.trim()}"`);
    const word = parseInt(hex.slice(0, 8), 16);
    lines.push({ op: word >>> 24, addr: word & 0xffffff, value: parseInt(hex.slice(8), 16) });
  }
  return lines;
}

function write(core: Core, op: number, addr: number, value: number): void {
  const paddr = addr & 0x7fffff;
  if (op === 0x80 || op === 0xa0 || op === 0x88 || op === 0xf0) core.poke(paddr, value & 0xff, 1);
  else core.poke(paddr & ~1, value & 0xffff, 2);
}

const WRITES = new Set([0x80, 0x81, 0xa0, 0xa1, 0x88, 0x89, 0xf0, 0xf1]);

/** Applies every enabled cheat. */
export function applyCheats(core: Core, cheats: Cheat[]): void {
  const rdram = core.rdram();
  for (const cheat of cheats) {
    if (!cheat.enabled) continue;
    let lines: Line[];
    try {
      lines = parseCode(cheat.code);
    } catch {
      continue;
    }
    for (let i = 0; i < lines.length; i++) {
      const { op, addr, value } = lines[i];
      if (op >= 0xd0 && op <= 0xd3) {
        const bytes = op & 1 ? 2 : 1;
        const current = readBE(rdram, addr & 0x7fffff & (bytes === 2 ? ~1 : ~0), bytes);
        const expected = bytes === 2 ? value : value & 0xff;
        const pass = op < 0xd2 ? current === expected : current !== expected;
        if (!pass) i++;
        continue;
      }
      if (op === 0x50) {
        const next = lines[++i];
        if (!next) break;
        const count = (addr >>> 8) & 0xff;
        const step = addr & 0xff;
        for (let k = 0; k < count; k++) write(core, next.op, next.addr + k * step, next.value + k * value);
        continue;
      }
      if (WRITES.has(op)) write(core, op, addr, value);
    }
  }
}
