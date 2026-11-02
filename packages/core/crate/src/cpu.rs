//! VR4300 interpreter: integer unit, COP0 (exceptions, timer, TLB) and the
//! COP1 floating point unit. Addresses are treated as 32-bit, which is what
//! every commercial game uses.

use crate::state::*;
use crate::sys::{System, CYCLES_PER_INSTR, TLB_DIRTY, TLB_MATCH, TLB_VALID};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TlbEntry {
    pub mask: u64,
    pub hi: u64,
    pub lo0: u64,
    pub lo1: u64,
}

#[repr(C)]
pub struct Cpu {
    pub gpr: [u64; 32],
    pub hi: u64,
    pub lo: u64,
    pub cop0: [u64; 32],
    pub fpr: [u64; 32],
    pub tlb: [TlbEntry; 32],
    pub pc: u32,
    pub next_pc: u32,
    pub cur_pc: u32,
    pub fcr31: u32,
    pub llbit: u32,
    pub branch_pending: u32,
    pub delay_slot: u32,
    /// Count = (cycles >> 1) + count_adj.
    pub count_adj: u32,
}

const C0_INDEX: usize = 0;
const C0_RANDOM: usize = 1;
const C0_ENTRYLO0: usize = 2;
const C0_ENTRYLO1: usize = 3;
const C0_CONTEXT: usize = 4;
const C0_PAGEMASK: usize = 5;
const C0_WIRED: usize = 6;
const C0_BADVADDR: usize = 8;
const C0_COUNT: usize = 9;
const C0_ENTRYHI: usize = 10;
const C0_COMPARE: usize = 11;
const C0_STATUS: usize = 12;
const C0_CAUSE: usize = 13;
const C0_EPC: usize = 14;
const C0_LLADDR: usize = 17;
const C0_XCONTEXT: usize = 20;
const C0_ERROREPC: usize = 30;

const EXC_INT: u32 = 0;
const EXC_MOD: u32 = 1;
const EXC_TLBL: u32 = 2;
const EXC_TLBS: u32 = 3;
const EXC_ADEL: u32 = 4;
const EXC_ADES: u32 = 5;
const EXC_SYS: u32 = 8;
const EXC_BP: u32 = 9;
const EXC_RI: u32 = 10;
const EXC_CPU: u32 = 11;
const EXC_OV: u32 = 12;
const EXC_TR: u32 = 13;

const STATUS_FR: u64 = 1 << 26;
const STATUS_CU1: u64 = 1 << 29;
const CAUSE_IP7: u64 = 1 << 15;
const FCR31_C: u32 = 1 << 23;

#[inline(always)]
fn sext32(v: u32) -> u64 {
    v as i32 as i64 as u64
}

#[inline(always)]
fn simm(instr: u32) -> u64 {
    instr as u16 as i16 as i64 as u64
}

type Fault = ();

impl System {
    // ------------------------------------------------------------ main loop

    pub fn cpu_run(&mut self, until: u64) {
        while self.s.cycles < until && self.s.cycles < self.next_event {
            self.cpu_step();
            self.s.cycles += CYCLES_PER_INSTR;
            if self.idle {
                // Nothing but an interrupt can leave the loop: skip to the next event.
                self.idle = false;
                let target = until.min(self.next_event);
                if target > self.s.cycles {
                    self.s.cycles += (target - self.s.cycles).div_ceil(CYCLES_PER_INSTR) * CYCLES_PER_INSTR;
                }
            }
        }
    }

    fn cpu_step(&mut self) {
        let pc = self.s.cpu.pc;
        self.s.cpu.cur_pc = pc;
        self.s.cpu.delay_slot = self.s.cpu.branch_pending;
        if self.irq_pending {
            self.exception(EXC_INT, 0x180);
            return;
        }
        self.s.cpu.branch_pending = 0;
        let Ok(instr) = self.fetch(pc) else { return };
        self.s.cpu.pc = self.s.cpu.next_pc;
        self.s.cpu.next_pc = self.s.cpu.next_pc.wrapping_add(4);
        self.execute(instr);
        self.s.cpu.gpr[0] = 0;
    }

    #[inline(always)]
    fn fetch(&mut self, pc: u32) -> Result<u32, Fault> {
        if pc & 3 != 0 {
            self.address_error(pc, false);
            return Err(());
        }
        let pa = self.translate(pc, false)?;
        if (pa as usize) < RDRAM_SIZE {
            return Ok(self.s.rdram[(pa >> 2) as usize]);
        }
        Ok(self.phys_read32(pa))
    }

    // ------------------------------------------------------------ exceptions

    fn exception(&mut self, code: u32, offset: u32) {
        let c = &mut self.s.cpu;
        let status = c.cop0[C0_STATUS];
        let mut cause = c.cop0[C0_CAUSE] & !(0x7C | (1 << 31) | (3 << 28));
        cause |= (code as u64) << 2;
        let mut offset = offset;
        if status & 2 == 0 {
            let in_delay = c.delay_slot != 0;
            let epc = if in_delay { c.cur_pc.wrapping_sub(4) } else { c.cur_pc };
            c.cop0[C0_EPC] = sext32(epc);
            if in_delay {
                cause |= 1 << 31;
            }
            c.cop0[C0_STATUS] |= 2;
        } else {
            offset = 0x180;
        }
        c.cop0[C0_CAUSE] = cause;
        let base: u32 = if status & (1 << 22) != 0 { 0xBFC0_0200 } else { 0x8000_0000 };
        c.pc = base.wrapping_add(offset);
        c.next_pc = c.pc.wrapping_add(4);
        c.branch_pending = 0;
        self.update_irq();
    }

    fn cop_unusable(&mut self, cop: u64) {
        self.exception(EXC_CPU, 0x180);
        self.s.cpu.cop0[C0_CAUSE] |= cop << 28;
    }

    fn address_error(&mut self, va: u32, write: bool) {
        self.s.cpu.cop0[C0_BADVADDR] = sext32(va);
        self.exception(if write { EXC_ADES } else { EXC_ADEL }, 0x180);
    }

    fn tlb_exception(&mut self, va: u32, write: bool, lut: u32) {
        let c = &mut self.s.cpu;
        c.cop0[C0_BADVADDR] = sext32(va);
        let vpn2 = (va >> 13) as u64;
        c.cop0[C0_CONTEXT] = (c.cop0[C0_CONTEXT] & !0x7F_FFF0) | ((vpn2 << 4) & 0x7F_FFF0);
        c.cop0[C0_XCONTEXT] = (c.cop0[C0_XCONTEXT] & !0x1_FFFF_FFF0) | ((vpn2 << 4) & 0x7_FFFF_FFF0);
        c.cop0[C0_ENTRYHI] = (c.cop0[C0_ENTRYHI] & 0xFF) | (sext32(va) & !0x1FFF);
        let (code, offset) = if lut & TLB_MATCH == 0 {
            (if write { EXC_TLBS } else { EXC_TLBL }, 0)
        } else if lut & TLB_VALID == 0 {
            (if write { EXC_TLBS } else { EXC_TLBL }, 0x180)
        } else {
            (EXC_MOD, 0x180)
        };
        self.exception(code, offset);
    }

    // ------------------------------------------------------------ memory

    #[inline(always)]
    pub fn translate(&mut self, va: u32, write: bool) -> Result<u32, Fault> {
        if va & 0xC000_0000 == 0x8000_0000 {
            return Ok(va & 0x1FFF_FFFF);
        }
        let e = self.tlb_lut[(va >> 12) as usize];
        let need = if write { TLB_MATCH | TLB_VALID | TLB_DIRTY } else { TLB_MATCH | TLB_VALID };
        if e & need == need {
            return Ok((e & !0xFFF) | (va & 0xFFF));
        }
        self.tlb_exception(va, write, e);
        Err(())
    }

    /// Debugger/cheat access: translates without raising exceptions.
    pub fn translate_quiet(&self, va: u32) -> Option<u32> {
        if va & 0xC000_0000 == 0x8000_0000 {
            return Some(va & 0x1FFF_FFFF);
        }
        let e = self.tlb_lut[(va >> 12) as usize];
        (e & (TLB_MATCH | TLB_VALID) == TLB_MATCH | TLB_VALID).then_some((e & !0xFFF) | (va & 0xFFF))
    }

    fn load8(&mut self, va: u32) -> Result<u8, Fault> {
        let pa = self.translate(va, false)?;
        if (pa as usize) < RDRAM_SIZE {
            return Ok(self.rdram_read8(pa));
        }
        Ok((self.phys_read32(pa & !3) >> ((3 - (pa & 3)) * 8)) as u8)
    }

    fn load16(&mut self, va: u32) -> Result<u16, Fault> {
        if va & 1 != 0 {
            self.address_error(va, false);
            return Err(());
        }
        let pa = self.translate(va, false)?;
        if (pa as usize) < RDRAM_SIZE {
            return Ok(self.rdram_read16(pa));
        }
        let w = self.phys_read32(pa & !3);
        Ok(if pa & 2 == 0 { (w >> 16) as u16 } else { w as u16 })
    }

    fn load32(&mut self, va: u32) -> Result<u32, Fault> {
        if va & 3 != 0 {
            self.address_error(va, false);
            return Err(());
        }
        let pa = self.translate(va, false)?;
        if (pa as usize) < RDRAM_SIZE {
            return Ok(self.s.rdram[(pa >> 2) as usize]);
        }
        Ok(self.phys_read32(pa))
    }

    fn load64(&mut self, va: u32) -> Result<u64, Fault> {
        if va & 7 != 0 {
            self.address_error(va, false);
            return Err(());
        }
        let pa = self.translate(va, false)?;
        let hi = self.phys_read32(pa) as u64;
        let lo = self.phys_read32(pa + 4) as u64;
        Ok(hi << 32 | lo)
    }

    fn store8(&mut self, va: u32, v: u8) -> Result<(), Fault> {
        let pa = self.translate(va, true)?;
        if (pa as usize) < RDRAM_SIZE {
            self.rdram_write8(pa, v);
            return Ok(());
        }
        let shift = (3 - (pa & 3)) * 8;
        self.phys_write32(pa & !3, (v as u32) << shift, 0xFF << shift);
        Ok(())
    }

    fn store16(&mut self, va: u32, v: u16) -> Result<(), Fault> {
        if va & 1 != 0 {
            self.address_error(va, true);
            return Err(());
        }
        let pa = self.translate(va, true)?;
        if (pa as usize) < RDRAM_SIZE {
            self.rdram_write16(pa, v);
            return Ok(());
        }
        let shift = if pa & 2 == 0 { 16 } else { 0 };
        self.phys_write32(pa & !3, (v as u32) << shift, 0xFFFF << shift);
        Ok(())
    }

    fn store32(&mut self, va: u32, v: u32) -> Result<(), Fault> {
        self.store32_masked(va, v, u32::MAX)
    }

    fn store32_masked(&mut self, va: u32, v: u32, mask: u32) -> Result<(), Fault> {
        if va & 3 != 0 {
            self.address_error(va, true);
            return Err(());
        }
        let pa = self.translate(va, true)?;
        self.phys_write32(pa, v, mask);
        Ok(())
    }

    fn store64_masked(&mut self, va: u32, v: u64, mask: u64) -> Result<(), Fault> {
        if va & 7 != 0 {
            self.address_error(va, true);
            return Err(());
        }
        let pa = self.translate(va, true)?;
        self.phys_write32(pa, (v >> 32) as u32, (mask >> 32) as u32);
        self.phys_write32(pa + 4, v as u32, mask as u32);
        Ok(())
    }

    // ------------------------------------------------------------ helpers

    #[inline(always)]
    fn rs(&self, i: u32) -> u64 {
        self.s.cpu.gpr[((i >> 21) & 31) as usize]
    }

    #[inline(always)]
    fn rt(&self, i: u32) -> u64 {
        self.s.cpu.gpr[((i >> 16) & 31) as usize]
    }

    #[inline(always)]
    fn set_rt(&mut self, i: u32, v: u64) {
        self.s.cpu.gpr[((i >> 16) & 31) as usize] = v;
    }

    #[inline(always)]
    fn set_rd(&mut self, i: u32, v: u64) {
        self.s.cpu.gpr[((i >> 11) & 31) as usize] = v;
    }

    #[inline(always)]
    fn vaddr(&self, i: u32) -> u32 {
        self.rs(i).wrapping_add(simm(i)) as u32
    }

    fn branch(&mut self, i: u32, cond: bool, likely: bool) {
        if cond {
            let target = self.s.cpu.cur_pc.wrapping_add(4).wrapping_add((simm(i) << 2) as u32);
            self.s.cpu.next_pc = target;
            self.s.cpu.branch_pending = 1;
        } else if likely {
            self.s.cpu.pc = self.s.cpu.next_pc;
            self.s.cpu.next_pc = self.s.cpu.pc.wrapping_add(4);
        }
    }

    fn jump(&mut self, target: u32) {
        if self.s.boot_done == 0 && target == self.s.entry_point {
            self.boot_finished();
        }
        self.s.cpu.next_pc = target;
        self.s.cpu.branch_pending = 1;
    }

    fn link(&mut self, reg: usize) {
        self.s.cpu.gpr[reg] = sext32(self.s.cpu.cur_pc.wrapping_add(8));
    }

    fn trap(&mut self, cond: bool) {
        if cond {
            self.exception(EXC_TR, 0x180);
        }
    }

    // ------------------------------------------------------------ execute

    fn execute(&mut self, i: u32) {
        let op = i >> 26;
        match op {
            0x00 => self.special(i),
            0x01 => self.regimm(i),
            0x02 | 0x03 => {
                if op == 0x03 {
                    self.link(31);
                }
                let target = (self.s.cpu.cur_pc.wrapping_add(4) & 0xF000_0000) | ((i & 0x03FF_FFFF) << 2);
                self.jump(target);
            }
            0x04 => {
                // `b .` with a nop delay slot is an idle loop waiting for an interrupt.
                if i == 0x1000_FFFF && self.translate_quiet(self.s.cpu.pc).is_some_and(|pa| self.phys_read32(pa) == 0) {
                    self.idle = true;
                }
                self.branch(i, self.rs(i) == self.rt(i), false);
            }
            0x05 => self.branch(i, self.rs(i) != self.rt(i), false),
            0x06 => self.branch(i, self.rs(i) as i64 <= 0, false),
            0x07 => self.branch(i, self.rs(i) as i64 > 0, false),
            0x08 => match (self.rs(i) as i32).checked_add(simm(i) as i32) {
                Some(v) => self.set_rt(i, v as i64 as u64),
                None => self.exception(EXC_OV, 0x180),
            },
            0x09 => self.set_rt(i, sext32(self.rs(i).wrapping_add(simm(i)) as u32)),
            0x0A => self.set_rt(i, ((self.rs(i) as i64) < simm(i) as i64) as u64),
            0x0B => self.set_rt(i, (self.rs(i) < simm(i)) as u64),
            0x0C => self.set_rt(i, self.rs(i) & (i & 0xFFFF) as u64),
            0x0D => self.set_rt(i, self.rs(i) | (i & 0xFFFF) as u64),
            0x0E => self.set_rt(i, self.rs(i) ^ (i & 0xFFFF) as u64),
            0x0F => self.set_rt(i, sext32(i << 16)),
            0x10 => self.cop0_op(i),
            0x11 => self.cop1_op(i),
            0x12 => self.cop_unusable(2),
            0x14 => self.branch(i, self.rs(i) == self.rt(i), true),
            0x15 => self.branch(i, self.rs(i) != self.rt(i), true),
            0x16 => self.branch(i, self.rs(i) as i64 <= 0, true),
            0x17 => self.branch(i, self.rs(i) as i64 > 0, true),
            0x18 => match (self.rs(i) as i64).checked_add(simm(i) as i64) {
                Some(v) => self.set_rt(i, v as u64),
                None => self.exception(EXC_OV, 0x180),
            },
            0x19 => self.set_rt(i, self.rs(i).wrapping_add(simm(i))),
            0x1A => self.ldl(i),
            0x1B => self.ldr(i),
            0x20 => {
                if let Ok(v) = self.load8(self.vaddr(i)) {
                    self.set_rt(i, v as i8 as i64 as u64);
                }
            }
            0x21 => {
                if let Ok(v) = self.load16(self.vaddr(i)) {
                    self.set_rt(i, v as i16 as i64 as u64);
                }
            }
            0x22 => self.lwl(i),
            0x23 => {
                if let Ok(v) = self.load32(self.vaddr(i)) {
                    self.set_rt(i, sext32(v));
                }
            }
            0x24 => {
                if let Ok(v) = self.load8(self.vaddr(i)) {
                    self.set_rt(i, v as u64);
                }
            }
            0x25 => {
                if let Ok(v) = self.load16(self.vaddr(i)) {
                    self.set_rt(i, v as u64);
                }
            }
            0x26 => self.lwr(i),
            0x27 => {
                if let Ok(v) = self.load32(self.vaddr(i)) {
                    self.set_rt(i, v as u64);
                }
            }
            0x28 => {
                let _ = self.store8(self.vaddr(i), self.rt(i) as u8);
            }
            0x29 => {
                let _ = self.store16(self.vaddr(i), self.rt(i) as u16);
            }
            0x2A => self.swl(i),
            0x2B => {
                let _ = self.store32(self.vaddr(i), self.rt(i) as u32);
            }
            0x2C => self.sdl(i),
            0x2D => self.sdr(i),
            0x2E => self.swr(i),
            0x2F => {} // CACHE
            0x30 => {
                let va = self.vaddr(i);
                if let Ok(v) = self.load32(va) {
                    self.set_rt(i, sext32(v));
                    self.s.cpu.llbit = 1;
                    self.s.cpu.cop0[C0_LLADDR] = (self.translate_quiet(va).unwrap_or(0) >> 4) as u64;
                }
            }
            0x31 => self.lwc1(i),
            0x34 => {
                let va = self.vaddr(i);
                if let Ok(v) = self.load64(va) {
                    self.set_rt(i, v);
                    self.s.cpu.llbit = 1;
                    self.s.cpu.cop0[C0_LLADDR] = (self.translate_quiet(va).unwrap_or(0) >> 4) as u64;
                }
            }
            0x35 => self.ldc1(i),
            0x37 => {
                if let Ok(v) = self.load64(self.vaddr(i)) {
                    self.set_rt(i, v);
                }
            }
            0x38 => {
                if self.s.cpu.llbit != 0 {
                    if self.store32(self.vaddr(i), self.rt(i) as u32).is_ok() {
                        self.set_rt(i, 1);
                    }
                } else {
                    self.set_rt(i, 0);
                }
            }
            0x39 => self.swc1(i),
            0x3C => {
                if self.s.cpu.llbit != 0 {
                    if self.store64_masked(self.vaddr(i), self.rt(i), u64::MAX).is_ok() {
                        self.set_rt(i, 1);
                    }
                } else {
                    self.set_rt(i, 0);
                }
            }
            0x3D => self.sdc1(i),
            0x3F => {
                let _ = self.store64_masked(self.vaddr(i), self.rt(i), u64::MAX);
            }
            _ => self.exception(EXC_RI, 0x180),
        }
    }

    fn special(&mut self, i: u32) {
        let sa = (i >> 6) & 31;
        let rs = self.rs(i);
        let rt = self.rt(i);
        match i & 0x3F {
            0x00 => self.set_rd(i, sext32((rt as u32) << sa)),
            0x02 => self.set_rd(i, sext32((rt as u32) >> sa)),
            0x03 => self.set_rd(i, sext32(((rt as i64) >> sa) as u32)),
            0x04 => self.set_rd(i, sext32((rt as u32) << (rs & 31))),
            0x06 => self.set_rd(i, sext32((rt as u32) >> (rs & 31))),
            0x07 => self.set_rd(i, sext32(((rt as i64) >> (rs & 31)) as u32)),
            0x08 => self.jump(rs as u32),
            0x09 => {
                self.link(((i >> 11) & 31) as usize);
                self.jump(rs as u32);
            }
            0x0C => self.exception(EXC_SYS, 0x180),
            0x0D => self.exception(EXC_BP, 0x180),
            0x0F => {} // SYNC
            0x10 => self.set_rd(i, self.s.cpu.hi),
            0x11 => self.s.cpu.hi = rs,
            0x12 => self.set_rd(i, self.s.cpu.lo),
            0x13 => self.s.cpu.lo = rs,
            0x14 => self.set_rd(i, rt << (rs & 63)),
            0x16 => self.set_rd(i, rt >> (rs & 63)),
            0x17 => self.set_rd(i, ((rt as i64) >> (rs & 63)) as u64),
            0x18 => {
                let p = (rs as i32 as i64) * (rt as i32 as i64);
                self.s.cpu.lo = sext32(p as u32);
                self.s.cpu.hi = sext32((p >> 32) as u32);
            }
            0x19 => {
                let p = (rs as u32 as u64) * (rt as u32 as u64);
                self.s.cpu.lo = sext32(p as u32);
                self.s.cpu.hi = sext32((p >> 32) as u32);
            }
            0x1A => {
                let (n, d) = (rs as i32, rt as i32);
                let (q, r) = if d == 0 {
                    (if n >= 0 { -1 } else { 1 }, n)
                } else {
                    (n.wrapping_div(d), n.wrapping_rem(d))
                };
                self.s.cpu.lo = q as i64 as u64;
                self.s.cpu.hi = r as i64 as u64;
            }
            0x1B => {
                let (n, d) = (rs as u32, rt as u32);
                let (q, r) = if d == 0 { (u32::MAX, n) } else { (n / d, n % d) };
                self.s.cpu.lo = sext32(q);
                self.s.cpu.hi = sext32(r);
            }
            0x1C => {
                let p = (rs as i64 as i128) * (rt as i64 as i128);
                self.s.cpu.lo = p as u64;
                self.s.cpu.hi = (p >> 64) as u64;
            }
            0x1D => {
                let p = (rs as u128) * (rt as u128);
                self.s.cpu.lo = p as u64;
                self.s.cpu.hi = (p >> 64) as u64;
            }
            0x1E => {
                let (n, d) = (rs as i64, rt as i64);
                let (q, r) = if d == 0 {
                    (if n >= 0 { -1 } else { 1 }, n)
                } else {
                    (n.wrapping_div(d), n.wrapping_rem(d))
                };
                self.s.cpu.lo = q as u64;
                self.s.cpu.hi = r as u64;
            }
            0x1F => {
                let (q, r) = if rt == 0 { (u64::MAX, rs) } else { (rs / rt, rs % rt) };
                self.s.cpu.lo = q;
                self.s.cpu.hi = r;
            }
            0x20 => match (rs as i32).checked_add(rt as i32) {
                Some(v) => self.set_rd(i, v as i64 as u64),
                None => self.exception(EXC_OV, 0x180),
            },
            0x21 => self.set_rd(i, sext32(rs.wrapping_add(rt) as u32)),
            0x22 => match (rs as i32).checked_sub(rt as i32) {
                Some(v) => self.set_rd(i, v as i64 as u64),
                None => self.exception(EXC_OV, 0x180),
            },
            0x23 => self.set_rd(i, sext32(rs.wrapping_sub(rt) as u32)),
            0x24 => self.set_rd(i, rs & rt),
            0x25 => self.set_rd(i, rs | rt),
            0x26 => self.set_rd(i, rs ^ rt),
            0x27 => self.set_rd(i, !(rs | rt)),
            0x2A => self.set_rd(i, ((rs as i64) < (rt as i64)) as u64),
            0x2B => self.set_rd(i, (rs < rt) as u64),
            0x2C => match (rs as i64).checked_add(rt as i64) {
                Some(v) => self.set_rd(i, v as u64),
                None => self.exception(EXC_OV, 0x180),
            },
            0x2D => self.set_rd(i, rs.wrapping_add(rt)),
            0x2E => match (rs as i64).checked_sub(rt as i64) {
                Some(v) => self.set_rd(i, v as u64),
                None => self.exception(EXC_OV, 0x180),
            },
            0x2F => self.set_rd(i, rs.wrapping_sub(rt)),
            0x30 => self.trap(rs as i64 >= rt as i64),
            0x31 => self.trap(rs >= rt),
            0x32 => self.trap((rs as i64) < rt as i64),
            0x33 => self.trap(rs < rt),
            0x34 => self.trap(rs == rt),
            0x36 => self.trap(rs != rt),
            0x38 => self.set_rd(i, rt << sa),
            0x3A => self.set_rd(i, rt >> sa),
            0x3B => self.set_rd(i, ((rt as i64) >> sa) as u64),
            0x3C => self.set_rd(i, rt << (sa + 32)),
            0x3E => self.set_rd(i, rt >> (sa + 32)),
            0x3F => self.set_rd(i, ((rt as i64) >> (sa + 32)) as u64),
            _ => self.exception(EXC_RI, 0x180),
        }
    }

    fn regimm(&mut self, i: u32) {
        let rs = self.rs(i) as i64;
        let imm = simm(i) as i64;
        match (i >> 16) & 31 {
            0x00 => self.branch(i, rs < 0, false),
            0x01 => self.branch(i, rs >= 0, false),
            0x02 => self.branch(i, rs < 0, true),
            0x03 => self.branch(i, rs >= 0, true),
            0x08 => self.trap(rs >= imm),
            0x09 => self.trap(rs as u64 >= imm as u64),
            0x0A => self.trap(rs < imm),
            0x0B => self.trap((rs as u64) < imm as u64),
            0x0C => self.trap(rs == imm),
            0x0E => self.trap(rs != imm),
            0x10 => {
                self.link(31);
                self.branch(i, rs < 0, false);
            }
            0x11 => {
                self.link(31);
                self.branch(i, rs >= 0, false);
            }
            0x12 => {
                self.link(31);
                self.branch(i, rs < 0, true);
            }
            0x13 => {
                self.link(31);
                self.branch(i, rs >= 0, true);
            }
            _ => self.exception(EXC_RI, 0x180),
        }
    }

    // ------------------------------------------------------------ unaligned

    fn lwl(&mut self, i: u32) {
        let va = self.vaddr(i);
        let Ok(w) = self.load32(va & !3) else { return };
        let shift = (va & 3) * 8;
        let keep = ((1u64 << shift) - 1) as u32;
        let v = (w << shift) | (self.rt(i) as u32 & keep);
        self.set_rt(i, sext32(v));
    }

    fn lwr(&mut self, i: u32) {
        let va = self.vaddr(i);
        let Ok(w) = self.load32(va & !3) else { return };
        let shift = (3 - (va & 3)) * 8;
        let keep = !(u32::MAX >> shift);
        let v = (w >> shift) | (self.rt(i) as u32 & keep);
        self.set_rt(i, sext32(v));
    }

    fn swl(&mut self, i: u32) {
        let va = self.vaddr(i);
        let shift = (va & 3) * 8;
        let _ = self.store32_masked(va & !3, (self.rt(i) as u32) >> shift, u32::MAX >> shift);
    }

    fn swr(&mut self, i: u32) {
        let va = self.vaddr(i);
        let shift = (3 - (va & 3)) * 8;
        let _ = self.store32_masked(va & !3, (self.rt(i) as u32) << shift, u32::MAX << shift);
    }

    fn ldl(&mut self, i: u32) {
        let va = self.vaddr(i);
        let Ok(w) = self.load64(va & !7) else { return };
        let shift = (va & 7) * 8;
        let keep = if shift == 0 { 0 } else { (1u64 << shift) - 1 };
        self.set_rt(i, (w << shift) | (self.rt(i) & keep));
    }

    fn ldr(&mut self, i: u32) {
        let va = self.vaddr(i);
        let Ok(w) = self.load64(va & !7) else { return };
        let shift = (7 - (va & 7)) * 8;
        let keep = if shift == 0 { 0 } else { !(u64::MAX >> shift) };
        self.set_rt(i, (w >> shift) | (self.rt(i) & keep));
    }

    fn sdl(&mut self, i: u32) {
        let va = self.vaddr(i);
        let shift = (va & 7) * 8;
        let _ = self.store64_masked(va & !7, self.rt(i) >> shift, u64::MAX >> shift);
    }

    fn sdr(&mut self, i: u32) {
        let va = self.vaddr(i);
        let shift = (7 - (va & 7)) * 8;
        let _ = self.store64_masked(va & !7, self.rt(i) << shift, u64::MAX << shift);
    }

    // ------------------------------------------------------------ COP0

    pub fn read_count(&self) -> u32 {
        ((self.s.cycles >> 1) as u32).wrapping_add(self.s.cpu.count_adj)
    }

    pub fn reschedule_compare(&mut self) {
        let delta = (self.s.cpu.cop0[C0_COMPARE] as u32).wrapping_sub(self.read_count());
        let delta = if delta == 0 { 1u64 << 32 } else { delta as u64 };
        self.schedule(EV_COMPARE, delta * 2);
    }

    pub fn compare_event(&mut self) {
        self.s.cpu.cop0[C0_CAUSE] |= CAUSE_IP7;
        self.update_irq();
        self.reschedule_compare();
    }

    fn random(&self) -> u64 {
        let wired = (self.s.cpu.cop0[C0_WIRED] & 31) as u64;
        let span = 32 - wired;
        wired + ((self.s.cycles / CYCLES_PER_INSTR) % span)
    }

    fn cop0_read(&self, reg: usize) -> u64 {
        match reg {
            C0_COUNT => self.read_count() as u64,
            C0_RANDOM => self.random(),
            _ => self.s.cpu.cop0[reg],
        }
    }

    fn cop0_write(&mut self, reg: usize, v: u64) {
        let c0 = &mut self.s.cpu.cop0;
        match reg {
            C0_INDEX => c0[reg] = v & 0x8000_003F,
            C0_RANDOM => {}
            C0_ENTRYLO0 | C0_ENTRYLO1 => c0[reg] = v & 0x3FFF_FFFF,
            C0_CONTEXT => c0[reg] = (c0[reg] & 0x7F_FFF0) | (v & !0x7F_FFFF),
            C0_PAGEMASK => c0[reg] = v & 0x01FF_E000,
            C0_WIRED => c0[reg] = v & 63,
            C0_COUNT => {
                self.s.cpu.count_adj = (v as u32).wrapping_sub((self.s.cycles >> 1) as u32);
                self.reschedule_compare();
            }
            C0_ENTRYHI => {
                let old_asid = c0[reg] & 0xFF;
                c0[reg] = v & 0xC000_00FF_FFFF_E0FF;
                if old_asid != v & 0xFF {
                    self.tlb_rebuild();
                }
            }
            C0_COMPARE => {
                c0[reg] = v & 0xFFFF_FFFF;
                c0[C0_CAUSE] &= !CAUSE_IP7;
                self.reschedule_compare();
                self.update_irq();
            }
            C0_STATUS => {
                c0[reg] = v & 0xFF57_FFFF;
                self.update_irq();
            }
            C0_CAUSE => {
                c0[reg] = (c0[reg] & !0x300) | (v & 0x300);
                self.update_irq();
            }
            C0_BADVADDR | 15 => {}
            C0_XCONTEXT => c0[reg] = (c0[reg] & 0x1_FFFF_FFF0) | (v & !0x1_FFFF_FFFF),
            _ => c0[reg] = v,
        }
    }

    fn cop0_op(&mut self, i: u32) {
        let rd = ((i >> 11) & 31) as usize;
        match (i >> 21) & 31 {
            0x00 => self.set_rt(i, sext32(self.cop0_read(rd) as u32)),
            0x01 => self.set_rt(i, self.cop0_read(rd)),
            0x04 => self.cop0_write(rd, sext32(self.rt(i) as u32)),
            0x05 => self.cop0_write(rd, self.rt(i)),
            0x10..=0x1F => match i & 0x3F {
                0x01 => self.tlbr(),
                0x02 => self.tlbw((self.s.cpu.cop0[C0_INDEX] & 31) as usize),
                0x06 => self.tlbw(self.random() as usize),
                0x08 => self.tlbp(),
                0x18 => self.eret(),
                _ => {}
            },
            _ => {}
        }
    }

    fn eret(&mut self) {
        let c = &mut self.s.cpu;
        if c.cop0[C0_STATUS] & 4 != 0 {
            c.pc = c.cop0[C0_ERROREPC] as u32;
            c.cop0[C0_STATUS] &= !4;
        } else {
            c.pc = c.cop0[C0_EPC] as u32;
            c.cop0[C0_STATUS] &= !2;
        }
        c.next_pc = c.pc.wrapping_add(4);
        c.llbit = 0;
        self.update_irq();
    }

    // ------------------------------------------------------------ TLB

    fn tlbr(&mut self) {
        let e = self.s.cpu.tlb[(self.s.cpu.cop0[C0_INDEX] & 31) as usize];
        let g = (e.hi >> 12) & 1;
        let c0 = &mut self.s.cpu.cop0;
        c0[C0_PAGEMASK] = e.mask;
        c0[C0_ENTRYHI] = e.hi & !(1 << 12);
        c0[C0_ENTRYLO0] = e.lo0 | g;
        c0[C0_ENTRYLO1] = e.lo1 | g;
    }

    fn tlbw(&mut self, index: usize) {
        let c0 = &self.s.cpu.cop0;
        let mask = c0[C0_PAGEMASK];
        let g = c0[C0_ENTRYLO0] & c0[C0_ENTRYLO1] & 1;
        self.s.cpu.tlb[index] = TlbEntry {
            mask,
            hi: (c0[C0_ENTRYHI] & !(mask | 0x1F00)) | (g << 12),
            lo0: c0[C0_ENTRYLO0] & !1,
            lo1: c0[C0_ENTRYLO1] & !1,
        };
        self.tlb_rebuild();
    }

    fn tlbp(&mut self) {
        let hi = self.s.cpu.cop0[C0_ENTRYHI];
        let asid = hi & 0xFF;
        let found = self.s.cpu.tlb.iter().position(|e| {
            let vmask = !(e.mask | 0x1FFF) & 0xC000_00FF_FFFF_E000;
            (e.hi & vmask) == (hi & vmask) && ((e.hi >> 12) & 1 != 0 || (e.hi & 0xFF) == asid)
        });
        self.s.cpu.cop0[C0_INDEX] = match found {
            Some(idx) => idx as u64,
            None => 0x8000_0000,
        };
    }

    /// Rebuilds the virtual page lookup table from all TLB entries.
    pub fn tlb_rebuild(&mut self) {
        for page in self.tlb_pages.drain(..) {
            self.tlb_lut[page as usize] = 0;
        }
        let asid = self.s.cpu.cop0[C0_ENTRYHI] & 0xFF;
        for idx in 0..32 {
            let e = self.s.cpu.tlb[idx];
            if (e.hi >> 12) & 1 == 0 && (e.hi & 0xFF) != asid {
                continue;
            }
            let page_size = ((e.mask + 0x2000) >> 1) as u32;
            let base = (e.hi as u32) & !((e.mask as u32) | 0x1FFF);
            for (half, lo) in [e.lo0, e.lo1].into_iter().enumerate() {
                let vstart = base.wrapping_add(half as u32 * page_size);
                if (vstart & 0xC000_0000) == 0x8000_0000 {
                    continue;
                }
                let pstart = (((lo >> 6) & 0xF_FFFF) << 12) as u32;
                let flags = TLB_MATCH | if lo & 2 != 0 { TLB_VALID } else { 0 } | if lo & 4 != 0 { TLB_DIRTY } else { 0 };
                for k in 0..(page_size >> 12) {
                    let vpage = (vstart >> 12).wrapping_add(k) as usize & 0xF_FFFF;
                    self.tlb_lut[vpage] = (pstart.wrapping_add(k << 12)) | flags;
                    self.tlb_pages.push(vpage as u32);
                }
            }
        }
    }

    // ------------------------------------------------------------ COP1

    #[inline(always)]
    fn fr(&self) -> bool {
        self.s.cpu.cop0[C0_STATUS] & STATUS_FR != 0
    }

    fn cop1_usable(&mut self) -> bool {
        if self.s.cpu.cop0[C0_STATUS] & STATUS_CU1 == 0 {
            self.cop_unusable(1);
            return false;
        }
        true
    }

    fn fget32(&self, r: usize) -> u32 {
        if self.fr() || r & 1 == 0 {
            self.s.cpu.fpr[r] as u32
        } else {
            (self.s.cpu.fpr[r & !1] >> 32) as u32
        }
    }

    fn fset32(&mut self, r: usize, v: u32) {
        if self.fr() || r & 1 == 0 {
            let f = &mut self.s.cpu.fpr[r];
            *f = (*f & !0xFFFF_FFFF) | v as u64;
        } else {
            let f = &mut self.s.cpu.fpr[r & !1];
            *f = (*f & 0xFFFF_FFFF) | (v as u64) << 32;
        }
    }

    fn fget64(&self, r: usize) -> u64 {
        self.s.cpu.fpr[if self.fr() { r } else { r & !1 }]
    }

    fn fset64(&mut self, r: usize, v: u64) {
        let r = if self.fr() { r } else { r & !1 };
        self.s.cpu.fpr[r] = v;
    }

    fn fs(&self, r: usize) -> f32 {
        f32::from_bits(self.fget32(r))
    }

    fn fd(&self, r: usize) -> f64 {
        f64::from_bits(self.fget64(r))
    }

    fn lwc1(&mut self, i: u32) {
        if !self.cop1_usable() {
            return;
        }
        if let Ok(v) = self.load32(self.vaddr(i)) {
            self.fset32(((i >> 16) & 31) as usize, v);
        }
    }

    fn ldc1(&mut self, i: u32) {
        if !self.cop1_usable() {
            return;
        }
        if let Ok(v) = self.load64(self.vaddr(i)) {
            self.fset64(((i >> 16) & 31) as usize, v);
        }
    }

    fn swc1(&mut self, i: u32) {
        if !self.cop1_usable() {
            return;
        }
        let v = self.fget32(((i >> 16) & 31) as usize);
        let _ = self.store32(self.vaddr(i), v);
    }

    fn sdc1(&mut self, i: u32) {
        if !self.cop1_usable() {
            return;
        }
        let v = self.fget64(((i >> 16) & 31) as usize);
        let _ = self.store64_masked(self.vaddr(i), v, u64::MAX);
    }

    fn round_mode(&self, v: f64) -> f64 {
        match self.s.cpu.fcr31 & 3 {
            0 => v.round_ties_even(),
            1 => v.trunc(),
            2 => v.ceil(),
            _ => v.floor(),
        }
    }

    fn cop1_op(&mut self, i: u32) {
        if !self.cop1_usable() {
            return;
        }
        let fmt = (i >> 21) & 31;
        let ft = ((i >> 16) & 31) as usize;
        let fs = ((i >> 11) & 31) as usize;
        let fd = ((i >> 6) & 31) as usize;
        match fmt {
            0x00 => self.set_rt(i, sext32(self.fget32(fs))),
            0x01 => self.set_rt(i, self.fget64(fs)),
            0x02 => {
                let v = match fs {
                    0 => 0xA00,
                    31 => self.s.cpu.fcr31,
                    _ => 0,
                };
                self.set_rt(i, sext32(v));
            }
            0x04 => self.fset32(fs, self.rt(i) as u32),
            0x05 => self.fset64(fs, self.rt(i)),
            0x06 => {
                if fs == 31 {
                    self.s.cpu.fcr31 = self.rt(i) as u32 & 0x0183_FFFF;
                }
            }
            0x08 => {
                let c = self.s.cpu.fcr31 & FCR31_C != 0;
                match ft & 3 {
                    0 => self.branch(i, !c, false),
                    1 => self.branch(i, c, false),
                    2 => self.branch(i, !c, true),
                    _ => self.branch(i, c, true),
                }
            }
            0x10 => self.fpu_single(i, fs, ft, fd),
            0x11 => self.fpu_double(i, fs, ft, fd),
            0x14 => {
                let v = self.fget32(fs) as i32;
                match i & 0x3F {
                    0x20 => self.fset32(fd, (v as f32).to_bits()),
                    0x21 => self.fset64(fd, (v as f64).to_bits()),
                    _ => {}
                }
            }
            0x15 => {
                let v = self.fget64(fs) as i64;
                match i & 0x3F {
                    0x20 => self.fset32(fd, (v as f32).to_bits()),
                    0x21 => self.fset64(fd, (v as f64).to_bits()),
                    _ => {}
                }
            }
            _ => {}
        }
    }

    fn fpu_convert_int(&mut self, funct: u32, v: f64, fd: usize) {
        let r = match funct & 3 {
            0 => v.round_ties_even(),
            1 => v.trunc(),
            2 => v.ceil(),
            _ => v.floor(),
        };
        if funct < 0x0C {
            self.fset64(fd, r as i64 as u64);
        } else {
            self.fset32(fd, r as i32 as u32);
        }
    }

    fn fpu_compare(&mut self, funct: u32, a: f64, b: f64) {
        let unordered = a.is_nan() || b.is_nan();
        let c = (unordered && funct & 1 != 0) || (!unordered && ((a == b && funct & 2 != 0) || (a < b && funct & 4 != 0)));
        if c {
            self.s.cpu.fcr31 |= FCR31_C;
        } else {
            self.s.cpu.fcr31 &= !FCR31_C;
        }
    }

    fn fpu_single(&mut self, i: u32, fs: usize, ft: usize, fd: usize) {
        let a = self.fs(fs);
        let b = self.fs(ft);
        let funct = i & 0x3F;
        let out = |s: &mut Self, v: f32| s.fset32(fd, v.to_bits());
        match funct {
            0x00 => out(self, a + b),
            0x01 => out(self, a - b),
            0x02 => out(self, a * b),
            0x03 => out(self, a / b),
            0x04 => out(self, a.sqrt()),
            0x05 => out(self, a.abs()),
            0x06 => self.fset32(fd, self.fget32(fs)),
            0x07 => out(self, -a),
            0x08..=0x0F => self.fpu_convert_int(funct, a as f64, fd),
            0x21 => self.fset64(fd, (a as f64).to_bits()),
            0x24 => {
                let r = self.round_mode(a as f64);
                self.fset32(fd, r as i32 as u32);
            }
            0x25 => {
                let r = self.round_mode(a as f64);
                self.fset64(fd, r as i64 as u64);
            }
            0x30..=0x3F => self.fpu_compare(funct, a as f64, b as f64),
            _ => {}
        }
    }

    fn fpu_double(&mut self, i: u32, fs: usize, ft: usize, fd: usize) {
        let a = self.fd(fs);
        let b = self.fd(ft);
        let funct = i & 0x3F;
        let out = |s: &mut Self, v: f64| s.fset64(fd, v.to_bits());
        match funct {
            0x00 => out(self, a + b),
            0x01 => out(self, a - b),
            0x02 => out(self, a * b),
            0x03 => out(self, a / b),
            0x04 => out(self, a.sqrt()),
            0x05 => out(self, a.abs()),
            0x06 => self.fset64(fd, self.fget64(fs)),
            0x07 => out(self, -a),
            0x08..=0x0F => self.fpu_convert_int(funct, a, fd),
            0x20 => self.fset32(fd, (a as f32).to_bits()),
            0x24 => {
                let r = self.round_mode(a);
                self.fset32(fd, r as i32 as u32);
            }
            0x25 => {
                let r = self.round_mode(a);
                self.fset64(fd, r as i64 as u64);
            }
            0x30..=0x3F => self.fpu_compare(funct, a, b),
            _ => {}
        }
    }
}
