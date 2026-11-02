//! RSP: scalar unit, vector unit (COP2), SP registers and SP DMA.
//! Vector semantics follow the hardware-verified behaviour documented by
//! ares and n64-systemtest.

use crate::state::*;
use crate::sys::System;

#[repr(C)]
pub struct Rsp {
    pub gpr: [u32; 32],
    pub vr: [[u16; 8]; 32],
    /// 48-bit accumulator per lane.
    pub acc: [u64; 8],
    /// Low byte: carry per lane. High byte: not-equal per lane.
    pub vco: u16,
    /// Low byte: compare per lane. High byte: clip per lane.
    pub vcc: u16,
    pub vce: u16,
    pub div_in: u32,
    pub div_out: u32,
    pub div_dp: u32,
    pub pc: u32,
    pub next_pc: u32,
    pub branch_pending: u32,
    /// SP_STATUS in its read layout.
    pub status: u32,
    pub mem_addr: u32,
    pub dram_addr: u32,
    pub rd_len: u32,
    pub wr_len: u32,
    pub semaphore: u32,
    pub cycle_rem: u64,
}

const ST_HALT: u32 = 1 << 0;
const ST_BROKE: u32 = 1 << 1;
pub(crate) const ST_SSTEP: u32 = 1 << 5;
const ST_INTR_BREAK: u32 = 1 << 6;

const ACC_MASK: u64 = (1 << 48) - 1;

/// Lane permutation for each element selector.
pub(crate) const ELEMENTS: [[usize; 8]; 16] = {
    let mut t = [[0usize; 8]; 16];
    let mut e = 0;
    while e < 16 {
        let mut n = 0;
        while n < 8 {
            t[e][n] = match e {
                0 | 1 => n,
                2 | 3 => (n & !1) | (e & 1),
                4..=7 => (n & !3) | (e & 3),
                _ => e & 7,
            };
            n += 1;
        }
        e += 1;
    }
    t
};

pub struct DivTables {
    rcp: [u16; 512],
    rsq: [u16; 512],
}

impl DivTables {
    pub fn new() -> DivTables {
        let mut rcp = [0u16; 512];
        let mut rsq = [0u16; 512];
        for i in 0..512u64 {
            // The ROM stores the mantissa below an implicit 0x10000; entry 0
            // (exactly 2.0) saturates to 0xFFFF on hardware.
            let b = (1u64 << 34) / (i + 512);
            rcp[i as usize] = ((b + 1) >> 8).min(0x1_FFFF) as u16;
            let a = (i + 512) >> (i & 1);
            // Largest b with a * (b + 1)^2 < 2^44, seeded from a float sqrt.
            let mut b = ((1u64 << 44) as f64 / a as f64).sqrt() as u64;
            while a * (b + 1) * (b + 1) >= 1 << 44 {
                b -= 1;
            }
            while a * (b + 2) * (b + 2) < 1 << 44 {
                b += 1;
            }
            rsq[i as usize] = (b >> 1) as u16;
        }
        DivTables { rcp, rsq }
    }
}

#[inline(always)]
fn sext48(v: u64) -> i64 {
    ((v << 16) as i64) >> 16
}

#[inline(always)]
fn clamp16(v: i64) -> u16 {
    v.clamp(-32768, 32767) as i16 as u16
}

/// Accumulator middle slice, clamped to a signed 16-bit range.
#[inline(always)]
fn sat_mid(acc: i64) -> u16 {
    clamp16(acc >> 16)
}

/// Accumulator low slice, clamped to 0..0xFFFF when it overflows 32 bits.
#[inline(always)]
fn sat_low(acc: i64) -> u16 {
    if acc < -(1 << 31) {
        0
    } else if acc > (1 << 31) - 1 {
        0xFFFF
    } else {
        acc as u16
    }
}

#[inline(always)]
fn sat_unsigned(acc: i64) -> u16 {
    let h = (acc >> 32) as i16;
    let m = (acc >> 16) as i16;
    if h < 0 {
        0
    } else if (h ^ m) < 0 {
        0xFFFF
    } else {
        m as u16
    }
}

impl System {
    // ------------------------------------------------------------ SP registers

    pub fn rsp_running(&self) -> bool {
        self.s.rsp.status & ST_HALT == 0
    }

    pub fn sp_reg_read(&mut self, a: u32) -> u32 {
        if a & 0x000C_0000 == 0x0008_0000 {
            return if a & 0xF == 0 { self.s.rsp.pc & 0xFFC } else { 0 };
        }
        let r = &mut self.s.rsp;
        match (a >> 2) & 7 {
            0 => r.mem_addr,
            1 => r.dram_addr,
            2 => r.rd_len,
            3 => r.wr_len,
            4 => r.status,
            5 | 6 => 0,
            _ => {
                let v = r.semaphore;
                r.semaphore = 1;
                v
            }
        }
    }

    pub fn sp_reg_write(&mut self, a: u32, v: u32) {
        if a & 0x000C_0000 == 0x0008_0000 {
            if a & 0xF == 0 {
                self.s.rsp.pc = v & 0xFFC;
                self.s.rsp.next_pc = (v + 4) & 0xFFC;
                self.s.rsp.branch_pending = 0;
            }
            return;
        }
        match (a >> 2) & 7 {
            0 => self.s.rsp.mem_addr = v & 0x1FF8,
            1 => self.s.rsp.dram_addr = v & 0xFF_FFF8,
            2 => {
                self.s.rsp.rd_len = v;
                self.sp_dma(v, false);
            }
            3 => {
                self.s.rsp.wr_len = v;
                self.sp_dma(v, true);
            }
            4 => self.sp_status_write(v),
            7 => self.s.rsp.semaphore = 0,
            _ => {}
        }
    }

    fn sp_status_write(&mut self, v: u32) {
        let was_running = self.rsp_running();
        let st = &mut self.s.rsp.status;
        let mut set_clear = |clear_bit: u32, set_bit: u32, flag: u32| {
            if v & (1 << clear_bit) != 0 {
                *st &= !flag;
            }
            if v & (1 << set_bit) != 0 {
                *st |= flag;
            }
        };
        set_clear(0, 1, ST_HALT);
        set_clear(5, 6, ST_SSTEP);
        set_clear(7, 8, ST_INTR_BREAK);
        for k in 0..8 {
            set_clear(9 + 2 * k, 10 + 2 * k, 1 << (7 + k));
        }
        if v & (1 << 2) != 0 {
            *st &= !ST_BROKE;
        }
        if v & (1 << 3) != 0 {
            self.clear_mi(MI_SP);
        }
        if v & (1 << 4) != 0 {
            self.raise_mi(MI_SP);
        }
        if !was_running && self.rsp_running() {
            self.s.rsp.cycle_rem = 0;
            self.break_cpu_loop();
        }
    }

    /// DMA between RDRAM and SP memory. `to_dram` selects SP -> RDRAM.
    fn sp_dma(&mut self, v: u32, to_dram: bool) {
        let len = ((v & 0xFFF) | 7) + 1;
        let count = ((v >> 12) & 0xFF) + 1;
        let skip = (v >> 20) & 0xFF8;
        let mut mem = self.s.rsp.mem_addr;
        let mut dram = self.s.rsp.dram_addr;
        let bank = mem & 0x1000;
        for _ in 0..count {
            if !to_dram && bank != 0 {
                self.rsp_jit.invalidate(mem & 0xFFF, len);
            }
            for i in (0..len).step_by(4) {
                let m = (bank | ((mem + i) & 0xFFF)) as usize;
                let d = (dram + i) & 0xFF_FFFC;
                if to_dram {
                    let w = u32::from_be_bytes(self.s.sp_mem[m..m + 4].try_into().unwrap());
                    self.rdram_write32(d, w);
                } else {
                    let w = self.s.rdram[((d as usize) >> 2) & (RDRAM_SIZE / 4 - 1)];
                    self.s.sp_mem[m..m + 4].copy_from_slice(&w.to_be_bytes());
                }
            }
            mem = bank | ((mem + len) & 0xFFF);
            dram = (dram + len + skip) & 0xFF_FFF8;
        }
        self.s.rsp.mem_addr = mem;
        self.s.rsp.dram_addr = dram;
        let done = 0xFF8 | (v & 0xFF80_0000);
        if to_dram {
            self.s.rsp.wr_len = done;
        } else {
            self.s.rsp.rd_len = done;
        }
    }

    // ------------------------------------------------------------ execution

    pub fn rsp_catch_up(&mut self, cpu_cycles: u64) {
        let total = cpu_cycles * 2 + self.s.rsp.cycle_rem;
        let mut n = total / 3;
        self.s.rsp.cycle_rem = total % 3;
        while n > 0 && self.rsp_running() {
            n -= match self.rsp_jit_run(n) {
                Some(steps) => steps,
                None => {
                    self.rsp_step();
                    1
                }
            };
        }
    }

    #[inline(always)]
    fn dmem(&self, a: u32) -> u8 {
        self.s.sp_mem[(a & 0xFFF) as usize]
    }

    #[inline(always)]
    fn set_dmem(&mut self, a: u32, v: u8) {
        self.s.sp_mem[(a & 0xFFF) as usize] = v;
    }

    fn dmem32(&self, a: u32) -> u32 {
        u32::from_be_bytes([self.dmem(a), self.dmem(a + 1), self.dmem(a + 2), self.dmem(a + 3)])
    }

    fn rsp_step(&mut self) {
        let pc = self.s.rsp.pc;
        let o = 0x1000 | (pc & 0xFFC) as usize;
        let m = &self.s.sp_mem;
        let i = u32::from_be_bytes([m[o], m[o | 1], m[o | 2], m[o | 3]]);
        let in_delay = self.s.rsp.branch_pending != 0;
        self.s.rsp.branch_pending = 0;
        self.s.rsp.pc = self.s.rsp.next_pc;
        self.s.rsp.next_pc = (self.s.rsp.next_pc + 4) & 0xFFC;
        self.rsp_execute(i, pc, in_delay);
        self.s.rsp.gpr[0] = 0;
        if self.s.rsp.status & ST_SSTEP != 0 {
            self.s.rsp.status |= ST_HALT;
        }
    }

    fn rsp_branch(&mut self, pc: u32, i: u32, cond: bool) {
        if cond {
            let off = ((i as u16 as i16 as i32) << 2) as u32;
            self.s.rsp.next_pc = pc.wrapping_add(4).wrapping_add(off) & 0xFFC;
            self.s.rsp.branch_pending = 1;
        }
    }

    fn rsp_jump(&mut self, target: u32) {
        self.s.rsp.next_pc = target & 0xFFC;
        self.s.rsp.branch_pending = 1;
    }

    fn rsp_break(&mut self) {
        self.s.rsp.status |= ST_HALT | ST_BROKE;
        if self.s.rsp.status & ST_INTR_BREAK != 0 {
            self.raise_mi(MI_SP);
        }
    }

    pub(crate) fn rsp_execute(&mut self, i: u32, pc: u32, _in_delay: bool) {
        let rs_i = ((i >> 21) & 31) as usize;
        let rt_i = ((i >> 16) & 31) as usize;
        let rd_i = ((i >> 11) & 31) as usize;
        let rs = self.s.rsp.gpr[rs_i];
        let rt = self.s.rsp.gpr[rt_i];
        let imm = i as u16 as i16 as i32 as u32;
        let g = &mut self.s.rsp.gpr;
        match i >> 26 {
            0x00 => {
                let sa = (i >> 6) & 31;
                match i & 0x3F {
                    0x00 => g[rd_i] = rt << sa,
                    0x02 => g[rd_i] = rt >> sa,
                    0x03 => g[rd_i] = ((rt as i32) >> sa) as u32,
                    0x04 => g[rd_i] = rt << (rs & 31),
                    0x06 => g[rd_i] = rt >> (rs & 31),
                    0x07 => g[rd_i] = ((rt as i32) >> (rs & 31)) as u32,
                    0x08 => self.rsp_jump(rs),
                    0x09 => {
                        g[rd_i] = (pc + 8) & 0xFFC;
                        self.rsp_jump(rs);
                    }
                    0x0D => self.rsp_break(),
                    0x20 | 0x21 => g[rd_i] = rs.wrapping_add(rt),
                    0x22 | 0x23 => g[rd_i] = rs.wrapping_sub(rt),
                    0x24 => g[rd_i] = rs & rt,
                    0x25 => g[rd_i] = rs | rt,
                    0x26 => g[rd_i] = rs ^ rt,
                    0x27 => g[rd_i] = !(rs | rt),
                    0x2A => g[rd_i] = ((rs as i32) < (rt as i32)) as u32,
                    0x2B => g[rd_i] = (rs < rt) as u32,
                    _ => {}
                }
            }
            0x01 => {
                let cond = match rt_i & 0x11 {
                    0x00 | 0x10 => (rs as i32) < 0,
                    _ => (rs as i32) >= 0,
                };
                if rt_i & 0x10 != 0 {
                    g[31] = (pc + 8) & 0xFFC;
                }
                self.rsp_branch(pc, i, cond);
            }
            0x02 => self.rsp_jump((i & 0x3FF) << 2),
            0x03 => {
                g[31] = (pc + 8) & 0xFFC;
                self.rsp_jump((i & 0x3FF) << 2);
            }
            0x04 => self.rsp_branch(pc, i, rs == rt),
            0x05 => self.rsp_branch(pc, i, rs != rt),
            0x06 => self.rsp_branch(pc, i, (rs as i32) <= 0),
            0x07 => self.rsp_branch(pc, i, (rs as i32) > 0),
            0x08 | 0x09 => g[rt_i] = rs.wrapping_add(imm),
            0x0A => g[rt_i] = ((rs as i32) < (imm as i32)) as u32,
            0x0B => g[rt_i] = (rs < imm) as u32,
            0x0C => g[rt_i] = rs & (i & 0xFFFF),
            0x0D => g[rt_i] = rs | (i & 0xFFFF),
            0x0E => g[rt_i] = rs ^ (i & 0xFFFF),
            0x0F => g[rt_i] = i << 16,
            0x10 => match rs_i {
                0x00 => {
                    let v = if rd_i < 8 {
                        self.sp_reg_read(0x0404_0000 + (rd_i as u32) * 4)
                    } else {
                        self.dp_reg_read(rd_i as u32 - 8)
                    };
                    self.s.rsp.gpr[rt_i] = v;
                }
                0x04 => {
                    if rd_i < 8 {
                        self.sp_reg_write(0x0404_0000 + (rd_i as u32) * 4, rt);
                    } else {
                        self.dp_reg_write(rd_i as u32 - 8, rt);
                    }
                }
                _ => {}
            },
            0x12 => self.cop2(i),
            0x20 => {
                let v = self.dmem(rs.wrapping_add(imm)) as i8 as i32 as u32;
                self.s.rsp.gpr[rt_i] = v;
            }
            0x21 => {
                let a = rs.wrapping_add(imm);
                let v = u16::from_be_bytes([self.dmem(a), self.dmem(a + 1)]) as i16 as i32 as u32;
                self.s.rsp.gpr[rt_i] = v;
            }
            0x23 | 0x27 => {
                let v = self.dmem32(rs.wrapping_add(imm));
                self.s.rsp.gpr[rt_i] = v;
            }
            0x24 => {
                let v = self.dmem(rs.wrapping_add(imm)) as u32;
                self.s.rsp.gpr[rt_i] = v;
            }
            0x25 => {
                let a = rs.wrapping_add(imm);
                let v = u16::from_be_bytes([self.dmem(a), self.dmem(a + 1)]) as u32;
                self.s.rsp.gpr[rt_i] = v;
            }
            0x28 => self.set_dmem(rs.wrapping_add(imm), rt as u8),
            0x29 => {
                let a = rs.wrapping_add(imm);
                self.set_dmem(a, (rt >> 8) as u8);
                self.set_dmem(a + 1, rt as u8);
            }
            0x2B => {
                let a = rs.wrapping_add(imm);
                for (k, b) in rt.to_be_bytes().into_iter().enumerate() {
                    self.set_dmem(a + k as u32, b);
                }
            }
            0x32 => self.lwc2(i),
            0x3A => self.swc2(i),
            _ => {}
        }
    }

    // ------------------------------------------------------------ vector regs

    #[inline(always)]
    fn vbyte(&self, v: usize, b: usize) -> u8 {
        let lane = self.s.rsp.vr[v][(b >> 1) & 7];
        if b & 1 == 0 { (lane >> 8) as u8 } else { lane as u8 }
    }

    #[inline(always)]
    fn set_vbyte(&mut self, v: usize, b: usize, x: u8) {
        let lane = &mut self.s.rsp.vr[v][(b >> 1) & 7];
        if b & 1 == 0 {
            *lane = (*lane & 0x00FF) | (x as u16) << 8;
        } else {
            *lane = (*lane & 0xFF00) | x as u16;
        }
    }

    fn cop2(&mut self, i: u32) {
        if i & (1 << 25) == 0 {
            let rt = ((i >> 16) & 31) as usize;
            let vs = ((i >> 11) & 31) as usize;
            let e = ((i >> 7) & 15) as usize;
            match (i >> 21) & 31 {
                0x00 => {
                    let hi = self.vbyte(vs, e);
                    let lo = self.vbyte(vs, (e + 1) & 15);
                    self.s.rsp.gpr[rt] = u16::from_be_bytes([hi, lo]) as i16 as i32 as u32;
                }
                0x02 => {
                    let v = match vs & 3 {
                        0 => self.s.rsp.vco,
                        1 => self.s.rsp.vcc,
                        _ => self.s.rsp.vce,
                    };
                    self.s.rsp.gpr[rt] = v as i16 as i32 as u32;
                }
                0x04 => {
                    let v = self.s.rsp.gpr[rt];
                    self.set_vbyte(vs, e, (v >> 8) as u8);
                    if e != 15 {
                        self.set_vbyte(vs, e + 1, v as u8);
                    }
                }
                0x06 => {
                    let v = self.s.rsp.gpr[rt] as u16;
                    match vs & 3 {
                        0 => self.s.rsp.vco = v,
                        1 => self.s.rsp.vcc = v,
                        _ => self.s.rsp.vce = v & 0xFF,
                    }
                }
                _ => {}
            }
            return;
        }
        self.vector_op(i);
    }

    fn vector_op(&mut self, i: u32) {
        let e = ((i >> 21) & 15) as usize;
        let vt = ((i >> 16) & 31) as usize;
        let vs = ((i >> 11) & 31) as usize;
        let vd = ((i >> 6) & 31) as usize;
        let funct = i & 0x3F;
        let a = self.s.rsp.vr[vs];
        let src = self.s.rsp.vr[vt];
        // Selectors 0 and 1 use the lanes as they are, which is the common case.
        let b = if e < 2 { src } else { ELEMENTS[e].map(|lane| src[lane & 7]) };
        let r = &mut self.s.rsp;
        let mut out = [0u16; 8];
        match funct {
            0x00..=0x0F => out = multiply(&mut r.acc, funct, vs, &a, &b),
            0x10 | 0x11 => {
                for n in 0..8 {
                    let carry = ((r.vco >> n) & 1) as i32;
                    let v = if funct == 0x10 {
                        a[n] as i16 as i32 + b[n] as i16 as i32 + carry
                    } else {
                        a[n] as i16 as i32 - b[n] as i16 as i32 - carry
                    };
                    set_acc_low(&mut r.acc[n], v as u16);
                    out[n] = clamp16(v as i64);
                }
                r.vco = 0;
            }
            0x13 => {
                for n in 0..8 {
                    let s = a[n] as i16;
                    let t = b[n] as i16;
                    let low = if s < 0 { t.wrapping_neg() } else if s == 0 { 0 } else { t };
                    set_acc_low(&mut r.acc[n], low as u16);
                    out[n] = if s < 0 && t == i16::MIN { 0x7FFF } else { low as u16 };
                }
            }
            0x14 => {
                let mut vco = 0;
                for n in 0..8 {
                    let v = a[n] as u32 + b[n] as u32;
                    set_acc_low(&mut r.acc[n], v as u16);
                    out[n] = v as u16;
                    vco |= ((v >> 16) as u16 & 1) << n;
                }
                r.vco = vco;
            }
            0x15 => {
                let mut vco = 0;
                for n in 0..8 {
                    let v = a[n] as i32 - b[n] as i32;
                    set_acc_low(&mut r.acc[n], v as u16);
                    out[n] = v as u16;
                    vco |= ((v < 0) as u16) << n;
                    vco |= ((v != 0) as u16) << (n + 8);
                }
                r.vco = vco;
            }
            0x1D => {
                for n in 0..8 {
                    out[n] = match e {
                        8 => (r.acc[n] >> 32) as u16,
                        9 => (r.acc[n] >> 16) as u16,
                        10 => r.acc[n] as u16,
                        _ => 0,
                    };
                }
            }
            0x20..=0x23 => {
                let mut vcc = 0;
                for n in 0..8 {
                    let s = a[n] as i16;
                    let t = b[n] as i16;
                    let carry = (r.vco >> n) & 1 != 0;
                    let ne = (r.vco >> (n + 8)) & 1 != 0;
                    let flag = match funct {
                        0x20 => s < t || (s == t && carry && ne),
                        0x21 => s == t && !ne,
                        0x22 => s != t || ne,
                        _ => s > t || (s == t && !(carry && ne)),
                    };
                    vcc |= (flag as u16) << n;
                    let v = if flag { a[n] } else { b[n] };
                    set_acc_low(&mut r.acc[n], v);
                    out[n] = v;
                }
                r.vcc = vcc;
                r.vco = 0;
            }
            0x24 => {
                let (mut vccl, mut vcch) = ((r.vcc & 0xFF) as u8, (r.vcc >> 8) as u8);
                for n in 0..8 {
                    let carry = (r.vco >> n) & 1 != 0;
                    let ne = (r.vco >> (n + 8)) & 1 != 0;
                    let ce = (r.vce >> n) & 1 != 0;
                    let s = a[n];
                    let t = b[n];
                    let v = if carry {
                        if ne {
                            if (vccl >> n) & 1 != 0 { t.wrapping_neg() } else { s }
                        } else {
                            let sum = s as u32 + t as u32;
                            let c = sum > 0xFFFF;
                            let z = sum as u16 == 0;
                            let flag = if ce { z || !c } else { z && !c };
                            set_bit(&mut vccl, n, flag);
                            if flag { t.wrapping_neg() } else { s }
                        }
                    } else if ne {
                        if (vcch >> n) & 1 != 0 { t } else { s }
                    } else {
                        let flag = s as i32 - t as i32 >= 0;
                        set_bit(&mut vcch, n, flag);
                        if flag { t } else { s }
                    };
                    set_acc_low(&mut r.acc[n], v);
                    out[n] = v;
                }
                r.vcc = (vcch as u16) << 8 | vccl as u16;
                r.vco = 0;
                r.vce = 0;
            }
            0x25 => {
                let (mut vco, mut vcc, mut vce) = (0u16, 0u16, 0u16);
                for n in 0..8 {
                    let s = a[n] as i16;
                    let t = b[n] as i16;
                    let v;
                    if (s ^ t) < 0 {
                        let sum = s as i32 + t as i32;
                        v = if sum <= 0 { t.wrapping_neg() as u16 } else { s as u16 };
                        vcc |= ((sum <= 0) as u16) << n | ((t < 0) as u16) << (n + 8);
                        vco |= 1 << n | ((sum != 0 && s as u16 != !(t as u16)) as u16) << (n + 8);
                        vce |= ((sum == -1) as u16) << n;
                    } else {
                        let diff = s as i32 - t as i32;
                        v = if diff >= 0 { t as u16 } else { s as u16 };
                        vcc |= ((t < 0) as u16) << n | ((diff >= 0) as u16) << (n + 8);
                        vco |= ((diff != 0 && s as u16 != !(t as u16)) as u16) << (n + 8);
                    }
                    set_acc_low(&mut r.acc[n], v);
                    out[n] = v;
                }
                r.vco = vco;
                r.vcc = vcc;
                r.vce = vce;
            }
            0x26 => {
                let mut vcc = 0u16;
                for n in 0..8 {
                    let s = a[n] as i16;
                    let t = b[n] as i16;
                    let v;
                    if (s ^ t) < 0 {
                        let le = s as i32 + t as i32 + 1 <= 0;
                        vcc |= ((t < 0) as u16) << (n + 8) | (le as u16) << n;
                        v = if le { !(t as u16) } else { s as u16 };
                    } else {
                        let ge = s as i32 - t as i32 >= 0;
                        vcc |= ((t < 0) as u16) << n | (ge as u16) << (n + 8);
                        v = if ge { t as u16 } else { s as u16 };
                    }
                    set_acc_low(&mut r.acc[n], v);
                    out[n] = v;
                }
                r.vcc = vcc;
                r.vco = 0;
                r.vce = 0;
            }
            0x27 => {
                for n in 0..8 {
                    let v = if (r.vcc >> n) & 1 != 0 { a[n] } else { b[n] };
                    set_acc_low(&mut r.acc[n], v);
                    out[n] = v;
                }
                r.vco = 0;
            }
            0x28..=0x2D => {
                for n in 0..8 {
                    let v = match funct {
                        0x28 => a[n] & b[n],
                        0x29 => !(a[n] & b[n]),
                        0x2A => a[n] | b[n],
                        0x2B => !(a[n] | b[n]),
                        0x2C => a[n] ^ b[n],
                        _ => !(a[n] ^ b[n]),
                    };
                    set_acc_low(&mut r.acc[n], v);
                    out[n] = v;
                }
            }
            0x30..=0x37 => {
                self.vector_divide(funct, e, vt, vs & 7, vd, &b);
                return;
            }
            _ => {
                for n in 0..8 {
                    set_acc_low(&mut r.acc[n], a[n].wrapping_add(b[n]));
                }
            }
        }
        self.s.rsp.vr[vd] = out;
    }

    /// VRCP, VRCPL, VRCPH, VMOV, VRSQ, VRSQL, VRSQH, VNOP.
    fn vector_divide(&mut self, funct: u32, e: usize, vt: usize, de: usize, vd: usize, vte: &[u16; 8]) {
        let r = &mut self.s.rsp;
        for n in 0..8 {
            set_acc_low(&mut r.acc[n], vte[n]);
        }
        let elem = r.vr[vt][e & 7];
        let result = match funct {
            0x30 | 0x31 | 0x34 | 0x35 => {
                let long = funct & 1 != 0;
                let input: i32 = if long && r.div_dp != 0 {
                    ((r.div_in << 16) | elem as u32) as i32
                } else {
                    elem as i16 as i32
                };
                let rsq = funct >= 0x34;
                let tables = div_tables();
                let mask = input >> 31;
                let mut data = input ^ mask;
                if input > -32768 {
                    data = data.wrapping_sub(mask);
                }
                let out: u32 = if data == 0 {
                    0x7FFF_FFFF
                } else if input == -32768 {
                    0xFFFF_0000
                } else {
                    let shift = (data as u32).leading_zeros();
                    let index = (((data as u64) << shift) & 0x7FC0_0000) >> 22;
                    if rsq {
                        let v = tables.rsq[((index & 0x1FE) | (shift as u64 & 1)) as usize] as u32;
                        let v = (0x10000 | v) << 14;
                        (v >> ((31 - shift) >> 1)) ^ mask as u32
                    } else {
                        let v = tables.rcp[index as usize] as u32;
                        let v = (0x10000 | v) << 14;
                        (v >> (31 - shift)) ^ mask as u32
                    }
                };
                r.div_dp = 0;
                r.div_out = out >> 16;
                out as u16
            }
            0x32 | 0x36 => {
                r.div_dp = 1;
                r.div_in = elem as u32;
                r.div_out as u16
            }
            0x33 => vte[de],
            _ => return,
        };
        r.vr[vd][de] = result;
    }

    // ------------------------------------------------------------ vector memory

    fn lwc2(&mut self, i: u32) {
        let base = self.s.rsp.gpr[((i >> 21) & 31) as usize];
        let vt = ((i >> 16) & 31) as usize;
        let op = (i >> 11) & 31;
        let e = ((i >> 7) & 15) as usize;
        let off = ((i & 0x7F) as i32) << 25 >> 25;
        let size: i32 = match op {
            0 => 1,
            1 => 2,
            2 => 4,
            3 | 6 | 7 => 8,
            _ => 16,
        };
        let mut addr = base.wrapping_add((off * size) as u32);
        match op {
            0..=3 => {
                for k in 0..size as usize {
                    if e + k < 16 {
                        let v = self.dmem(addr + k as u32);
                        self.set_vbyte(vt, e + k, v);
                    }
                }
            }
            4 => {
                let end = (16 + e).saturating_sub((addr & 15) as usize).min(16);
                for b in e..end {
                    let v = self.dmem(addr);
                    self.set_vbyte(vt, b, v);
                    addr += 1;
                }
            }
            5 => {
                // Nothing is loaded when the element lies past the address offset.
                let start = (16 + e).saturating_sub((addr & 15) as usize);
                addr &= !15;
                for b in start..16 {
                    let v = self.dmem(addr);
                    self.set_vbyte(vt, b & 15, v);
                    addr += 1;
                }
            }
            6 | 7 => {
                let index = ((addr & 7) as usize).wrapping_sub(e);
                addr &= !7;
                let shift = if op == 6 { 8 } else { 7 };
                for n in 0..8 {
                    let v = self.dmem(addr + ((index.wrapping_add(n)) & 15) as u32) as u16;
                    self.s.rsp.vr[vt][n] = v << shift;
                }
            }
            8 => {
                let index = ((addr & 7) as usize).wrapping_sub(e);
                addr &= !7;
                for n in 0..8 {
                    let v = self.dmem(addr + ((index.wrapping_add(n * 2)) & 15) as u32) as u16;
                    self.s.rsp.vr[vt][n] = v << 7;
                }
            }
            9 => {
                let index = ((addr & 7) as usize).wrapping_sub(e);
                addr &= !7;
                let mut tmp = [0u16; 8];
                for n in 0..4 {
                    tmp[n] = (self.dmem(addr + ((index.wrapping_add(n * 4)) & 15) as u32) as u16) << 7;
                    tmp[n + 4] = (self.dmem(addr + ((index.wrapping_add(n * 4 + 8)) & 15) as u32) as u16) << 7;
                }
                for b in e..(e + 8).min(16) {
                    let byte = if b & 1 == 0 { (tmp[b >> 1] >> 8) as u8 } else { tmp[b >> 1] as u8 };
                    self.set_vbyte(vt, b, byte);
                }
            }
            10 => {
                for b in (16 - e)..(16 + 16 - e).min(32) {
                    let v = self.dmem(addr);
                    self.set_vbyte(vt, b & 15, v);
                    addr += 4;
                }
            }
            11 => {
                let begin = addr & !7;
                let mut a = begin + (((e as u32) + (addr & 8)) & 15);
                let vtbase = vt & !7;
                let mut vtoff = e >> 1;
                for n in 0..8 {
                    let hi = self.dmem(a);
                    a += 1;
                    if a == begin + 16 {
                        a = begin;
                    }
                    let lo = self.dmem(a);
                    a += 1;
                    if a == begin + 16 {
                        a = begin;
                    }
                    self.s.rsp.vr[vtbase + vtoff][n] = u16::from_be_bytes([hi, lo]);
                    vtoff = (vtoff + 1) & 7;
                }
            }
            _ => {}
        }
    }

    fn swc2(&mut self, i: u32) {
        let base = self.s.rsp.gpr[((i >> 21) & 31) as usize];
        let vt = ((i >> 16) & 31) as usize;
        let op = (i >> 11) & 31;
        let e = ((i >> 7) & 15) as usize;
        let off = ((i & 0x7F) as i32) << 25 >> 25;
        let size: i32 = match op {
            0 => 1,
            1 => 2,
            2 => 4,
            3 | 6 | 7 => 8,
            _ => 16,
        };
        let mut addr = base.wrapping_add((off * size) as u32);
        match op {
            0..=3 => {
                for k in 0..size as usize {
                    let v = self.vbyte(vt, (e + k) & 15);
                    self.set_dmem(addr + k as u32, v);
                }
            }
            4 => {
                let end = e + (16 - (addr & 15) as usize);
                for b in e..end {
                    let v = self.vbyte(vt, b & 15);
                    self.set_dmem(addr, v);
                    addr += 1;
                }
            }
            5 => {
                let end = e + (addr & 15) as usize;
                let base_b = 16 - (addr & 15) as usize;
                addr &= !15;
                for b in e..end {
                    let v = self.vbyte(vt, (b + base_b) & 15);
                    self.set_dmem(addr, v);
                    addr += 1;
                }
            }
            6 | 7 => {
                for b in e..e + 8 {
                    let packed = (b & 15) < 8;
                    let v = if packed == (op == 6) {
                        self.vbyte(vt, (b & 7) << 1)
                    } else {
                        (self.s.rsp.vr[vt][b & 7] >> 7) as u8
                    };
                    self.set_dmem(addr, v);
                    addr += 1;
                }
            }
            8 => {
                let index = addr & 7;
                addr &= !7;
                for n in 0..8 {
                    let b = e + n * 2;
                    let v = (self.vbyte(vt, b & 15) << 1) | (self.vbyte(vt, (b + 1) & 15) >> 7);
                    self.set_dmem(addr + ((index + n as u32 * 2) & 15), v);
                }
            }
            9 => {
                let index = addr & 7;
                addr &= !7;
                let lanes: [usize; 4] = match e {
                    0 | 15 => [0, 1, 2, 3],
                    1 => [6, 7, 4, 5],
                    4 => [1, 2, 3, 0],
                    5 => [7, 4, 5, 6],
                    8 => [4, 5, 6, 7],
                    11 => [3, 0, 1, 2],
                    12 => [5, 6, 7, 4],
                    _ => {
                        for n in 0..4 {
                            self.set_dmem(addr + ((index + n * 4) & 15), 0);
                        }
                        return;
                    }
                };
                for (n, lane) in lanes.into_iter().enumerate() {
                    let v = (self.s.rsp.vr[vt][lane] >> 7) as u8;
                    self.set_dmem(addr + ((index + n as u32 * 4) & 15), v);
                }
            }
            10 => {
                let mut index = addr & 7;
                addr &= !7;
                for b in e..e + 16 {
                    let v = self.vbyte(vt, b & 15);
                    self.set_dmem(addr + (index & 15), v);
                    index += 1;
                }
            }
            11 => {
                let start = vt & !7;
                let mut element = 16 - (e & !1);
                let mut index = ((addr & 7) as usize).wrapping_sub(e & !1);
                addr &= !7;
                for reg in start..start + 8 {
                    for _ in 0..2 {
                        let v = self.vbyte(reg, element & 15);
                        self.set_dmem(addr + (index & 15) as u32, v);
                        element += 1;
                        index = index.wrapping_add(1);
                    }
                }
            }
            _ => {}
        }
    }
}

/// Applies one multiply-group operation lane by lane. Each arm gets its own
/// loop so the per-lane work carries no dispatch.
fn multiply(acc: &mut [u64; 8], funct: u32, vs: usize, a: &[u16; 8], b: &[u16; 8]) -> [u16; 8] {
    #[inline(always)]
    fn lanes(acc: &mut [u64; 8], a: &[u16; 8], b: &[u16; 8], step: impl Fn(i64, i64, i64, i64, i64) -> i64, result: impl Fn(i64) -> u16) -> [u16; 8] {
        let mut out = [0u16; 8];
        for n in 0..8 {
            let v = step(a[n] as i16 as i64, b[n] as i16 as i64, a[n] as i64, b[n] as i64, sext48(acc[n])) as u64 & ACC_MASK;
            acc[n] = v;
            out[n] = result(sext48(v));
        }
        out
    }
    let quarter = |acc: i64| clamp16((acc >> 16) as i32 as i64 >> 1) & !15;
    match funct {
        0x00 => lanes(acc, a, b, |s, t, _, _, _| s * t * 2 + 0x8000, sat_mid),
        0x01 => lanes(acc, a, b, |s, t, _, _, _| s * t * 2 + 0x8000, sat_unsigned),
        0x03 => lanes(
            acc,
            a,
            b,
            |s, t, _, _, _| {
                let p = s * t;
                (if p < 0 { p + 31 } else { p }) << 16
            },
            quarter,
        ),
        0x04 => lanes(acc, a, b, |_, _, su, tu, _| (su * tu) >> 16, |acc| acc as u16),
        0x05 => lanes(acc, a, b, |s, _, _, tu, _| s * tu, sat_mid),
        0x06 => lanes(acc, a, b, |_, t, su, _, _| su * t, sat_low),
        0x07 => lanes(acc, a, b, |s, t, _, _, _| (s * t) << 16, sat_mid),
        0x08 => lanes(acc, a, b, |s, t, _, _, old| old + s * t * 2, sat_mid),
        0x09 => lanes(acc, a, b, |s, t, _, _, old| old + s * t * 2, sat_unsigned),
        0x0B => lanes(
            acc,
            a,
            b,
            |_, _, _, _, old| {
                let p = (old >> 16) as i32;
                let p = if p & 32 != 0 {
                    p
                } else if p < 0 {
                    p + 32
                } else {
                    p - 32
                };
                (p as i64) << 16 | (old & 0xFFFF)
            },
            quarter,
        ),
        0x0C => lanes(acc, a, b, |_, _, su, tu, old| old + ((su * tu) >> 16), sat_low),
        0x0D => lanes(acc, a, b, |s, _, _, tu, old| old + s * tu, sat_mid),
        0x0E => lanes(acc, a, b, |_, t, su, _, old| old + su * t, sat_low),
        0x0F => lanes(acc, a, b, |s, t, _, _, old| old + ((s * t) << 16), sat_mid),
        // VRNDP / VRNDN
        _ => {
            let positive = funct == 0x02;
            lanes(
                acc,
                a,
                b,
                |_, t, _, _, old| {
                    let p = if vs & 1 != 0 { t << 16 } else { t };
                    if (old >= 0) == positive { old + p } else { old }
                },
                sat_mid,
            )
        }
    }
}

#[inline(always)]
fn set_acc_low(acc: &mut u64, v: u16) {
    *acc = (*acc & !0xFFFF) | v as u64;
}

#[inline(always)]
fn set_bit(bits: &mut u8, n: usize, v: bool) {
    if v {
        *bits |= 1 << n;
    } else {
        *bits &= !(1 << n);
    }
}

fn div_tables() -> &'static DivTables {
    static TABLES: std::sync::OnceLock<DivTables> = std::sync::OnceLock::new();
    TABLES.get_or_init(DivTables::new)
}
