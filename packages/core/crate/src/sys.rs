//! System glue: scheduler, physical bus and the simple RCP peripherals
//! (MI, VI, AI, PI, RI, SI).

use crate::rdp::GpuOut;
use crate::rsp_jit::RspJit;
use crate::state::*;

pub const CPU_HZ: u64 = 93_750_000;
pub const VI_HZ: u64 = 48_681_812;
/// CPU cycles charged per interpreted instruction.
pub const CYCLES_PER_INSTR: u64 = 2;
/// CPU cycles the CPU runs ahead of the RSP before the RSP catches up.
const RSP_SLICE: u64 = 512;

pub struct System {
    pub s: Box<State>,
    /// Cartridge ROM as big-endian bytes.
    pub rom: Vec<u8>,
    /// Per 4 KiB virtual page: physical page base | TLB_* flags. Rebuilt from
    /// the TLB, so it is not part of the savestate.
    pub tlb_lut: Box<[u32; 1 << 20]>,
    /// Pages currently filled in `tlb_lut`, so a rebuild clears only those.
    pub tlb_pages: Vec<u32>,
    pub next_event: u64,
    pub irq_pending: bool,
    /// Set by the CPU when it entered an idle loop.
    pub idle: bool,
    pub gpu: GpuOut,
    /// Interleaved stereo samples produced since the last drain.
    pub audio: Vec<i16>,
    /// CPU cycles per sample of `audio`, which sets its rate.
    pub audio_cycles_per_sample: u64,
    /// While the AI is idle, the cycle up to which `audio` holds silence;
    /// u64::MAX while it plays. The silence keeps the stream as long as the
    /// emulated time, so a recording stays in sync across quiet stretches.
    pub audio_silent_from: u64,
    /// RDRAM pages written by anything but the RDP since the last GPU upload.
    pub dirty: [u32; PAGE_COUNT / 32],
    /// RDRAM pages written since the last automatic snapshot.
    pub snap_dirty: [u32; PAGE_COUNT / 32],
    pub eeprom_16k: bool,
    /// Translated RSP code. It is derived from IMEM, so it is not part of the savestate.
    pub rsp_jit: RspJit,
}

pub const TLB_MATCH: u32 = 1;
pub const TLB_VALID: u32 = 2;
pub const TLB_DIRTY: u32 = 4;

impl System {
    pub fn new() -> Box<System> {
        let s = unsafe {
            let layout = std::alloc::Layout::new::<State>();
            Box::from_raw(std::alloc::alloc_zeroed(layout) as *mut State)
        };
        Box::new(System {
            s,
            rom: Vec::new(),
            tlb_lut: vec![0; 1 << 20].into_boxed_slice().try_into().unwrap(),
            tlb_pages: Vec::new(),
            next_event: 0,
            irq_pending: false,
            idle: false,
            gpu: GpuOut::new(),
            audio: Vec::with_capacity(8192),
            audio_cycles_per_sample: CPU_HZ / 44100,
            audio_silent_from: u64::MAX,
            dirty: [u32::MAX; PAGE_COUNT / 32],
            snap_dirty: [u32::MAX; PAGE_COUNT / 32],
            eeprom_16k: false,
            rsp_jit: RspJit::new(),
        })
    }

    // ---------------------------------------------------------------- frame

    pub fn run_frame(&mut self) {
        self.s.frame_done = 0;
        self.s.input.polled = 0;
        while self.s.frame_done == 0 {
            let rsp_running = self.rsp_running();
            let mut until = self.next_event;
            if rsp_running {
                until = until.min(self.s.cycles + RSP_SLICE);
            }
            let start = self.s.cycles;
            self.cpu_run(until);
            if rsp_running {
                self.rsp_catch_up(self.s.cycles - start);
            }
            self.process_events();
        }
        self.ai_fill_silence();
        self.s.frame += 1;
    }

    // ------------------------------------------------------------ scheduler

    pub fn schedule(&mut self, ev: usize, delay: u64) {
        self.s.events[ev] = self.s.cycles + delay;
        self.update_next_event();
    }

    pub fn schedule_at(&mut self, ev: usize, at: u64) {
        self.s.events[ev] = at;
        self.update_next_event();
    }

    pub fn update_next_event(&mut self) {
        self.next_event = self.s.events.iter().copied().min().unwrap_or(u64::MAX);
    }

    /// Makes the CPU loop return so the main loop re-evaluates its slice.
    pub fn break_cpu_loop(&mut self) {
        self.next_event = 0;
    }

    fn process_events(&mut self) {
        if self.next_event > self.s.cycles {
            return;
        }
        loop {
            let now = self.s.cycles;
            let Some(ev) = (0..EV_COUNT).find(|&e| self.s.events[e] <= now) else { break };
            self.s.events[ev] = u64::MAX;
            match ev {
                EV_VI_INTR => self.raise_mi(MI_VI),
                EV_FIELD => self.vi_field_end(),
                EV_COMPARE => self.compare_event(),
                EV_AI => self.ai_end_of_dma(),
                EV_PI => {
                    self.s.pi.regs[4] &= !3;
                    self.raise_mi(MI_PI);
                }
                EV_SI => {
                    self.s.si.status &= !3;
                    self.s.si.status |= 1 << 12;
                    self.raise_mi(MI_SI);
                }
                EV_SP => self.raise_mi(MI_SP),
                EV_DP => self.raise_mi(MI_DP),
                _ => unreachable!(),
            }
        }
        self.update_next_event();
    }

    // ------------------------------------------------------------ interrupts

    pub fn raise_mi(&mut self, bit: u32) {
        self.s.mi.intr |= bit;
        self.update_irq();
    }

    pub fn clear_mi(&mut self, bit: u32) {
        self.s.mi.intr &= !bit;
        self.update_irq();
    }

    pub fn update_irq(&mut self) {
        const IP2: u64 = 1 << 10;
        if self.s.mi.intr & self.s.mi.mask != 0 {
            self.s.cpu.cop0[13] |= IP2;
        } else {
            self.s.cpu.cop0[13] &= !IP2;
        }
        let status = self.s.cpu.cop0[12];
        let cause = self.s.cpu.cop0[13];
        self.irq_pending = status & 1 != 0 && status & 6 == 0 && status & cause & 0xFF00 != 0;
        if self.irq_pending {
            self.break_cpu_loop();
        }
    }

    // ------------------------------------------------------------ RDRAM

    #[inline(always)]
    pub fn mark_dirty(&mut self, paddr: u32) {
        let page = (paddr >> PAGE_SHIFT) as usize;
        self.dirty[page >> 5] |= 1 << (page & 31);
        self.snap_dirty[page >> 5] |= 1 << (page & 31);
    }

    pub fn mark_dirty_range(&mut self, paddr: u32, len: u32) {
        if len == 0 {
            return;
        }
        let first = paddr >> PAGE_SHIFT;
        let last = (paddr + len - 1).min(RDRAM_SIZE as u32 - 1) >> PAGE_SHIFT;
        for page in first..=last {
            self.mark_dirty(page << PAGE_SHIFT);
        }
    }

    #[inline(always)]
    pub fn rdram_read8(&self, a: u32) -> u8 {
        let w = self.s.rdram[((a >> 2) as usize) & (RDRAM_SIZE / 4 - 1)];
        (w >> ((3 - (a & 3)) * 8)) as u8
    }

    #[inline(always)]
    pub fn rdram_read16(&self, a: u32) -> u16 {
        let w = self.s.rdram[((a >> 2) as usize) & (RDRAM_SIZE / 4 - 1)];
        if a & 2 == 0 { (w >> 16) as u16 } else { w as u16 }
    }

    #[inline(always)]
    pub fn rdram_write8(&mut self, a: u32, v: u8) {
        if a as usize >= RDRAM_SIZE {
            return;
        }
        let shift = (3 - (a & 3)) * 8;
        let w = &mut self.s.rdram[(a >> 2) as usize];
        *w = (*w & !(0xFF << shift)) | ((v as u32) << shift);
        self.mark_dirty(a);
    }

    #[inline(always)]
    pub fn rdram_write16(&mut self, a: u32, v: u16) {
        if a as usize >= RDRAM_SIZE {
            return;
        }
        let shift = if a & 2 == 0 { 16 } else { 0 };
        let w = &mut self.s.rdram[(a >> 2) as usize];
        *w = (*w & !(0xFFFF << shift)) | ((v as u32) << shift);
        self.mark_dirty(a);
    }

    #[inline(always)]
    pub fn rdram_write32(&mut self, a: u32, v: u32) {
        if a as usize >= RDRAM_SIZE {
            return;
        }
        self.s.rdram[(a >> 2) as usize] = v;
        self.mark_dirty(a);
    }

    // ------------------------------------------------------------ physical bus

    fn rom_read32(&self, offset: u32) -> u32 {
        let o = offset as usize;
        if o + 4 <= self.rom.len() {
            u32::from_be_bytes([self.rom[o], self.rom[o + 1], self.rom[o + 2], self.rom[o + 3]])
        } else {
            // Open bus: the cartridge returns the low half of the address twice.
            let low = offset & 0xFFFF;
            (low << 16) | low
        }
    }

    pub fn rom_byte(&self, offset: u32) -> u8 {
        self.rom.get(offset as usize).copied().unwrap_or(0)
    }

    pub fn phys_read32(&mut self, a: u32) -> u32 {
        if (a as usize) < RDRAM_SIZE {
            return self.s.rdram[(a >> 2) as usize];
        }
        match a >> 20 {
            0x040 => {
                if a < 0x0404_0000 {
                    let o = (a & 0x1FFC) as usize;
                    let m = &self.s.sp_mem;
                    u32::from_be_bytes([m[o], m[o + 1], m[o + 2], m[o + 3]])
                } else {
                    self.sp_reg_read(a)
                }
            }
            0x041 => self.dp_reg_read((a >> 2) & 7),
            0x043 => match (a >> 2) & 3 {
                0 => self.s.mi.mode,
                1 => 0x0202_0102,
                2 => self.s.mi.intr,
                _ => self.s.mi.mask,
            },
            0x044 => self.vi_read((a >> 2) & 0xF),
            0x045 => self.ai_read((a >> 2) & 7),
            0x046 => {
                let r = ((a >> 2) & 0xF) as usize;
                if r < 13 { self.s.pi.regs[r] } else { 0 }
            }
            0x047 => self.s.ri[((a >> 2) & 7) as usize],
            0x048 => match (a >> 2) & 7 {
                0 => self.s.si.dram_addr,
                6 => self.s.si.status,
                _ => 0,
            },
            0x080..=0x0FF => {
                let o = (a & 0x7FFC) as usize;
                let m = &self.s.cart.sram;
                u32::from_be_bytes([m[o], m[o + 1], m[o + 2], m[o + 3]])
            }
            0x100..=0x1FB => self.rom_read32(a - 0x1000_0000),
            0x1FC => {
                if a >= 0x1FC0_07C0 && a < 0x1FC0_0800 {
                    let o = (a & 0x3C) as usize;
                    let m = &self.s.pif.ram;
                    u32::from_be_bytes([m[o], m[o + 1], m[o + 2], m[o + 3]])
                } else {
                    0
                }
            }
            _ => 0,
        }
    }

    /// Writes `v` to the word at `a`, keeping the bits outside `mask`.
    pub fn phys_write32(&mut self, a: u32, v: u32, mask: u32) {
        if (a as usize) < RDRAM_SIZE {
            let w = &mut self.s.rdram[(a >> 2) as usize];
            *w = (*w & !mask) | (v & mask);
            self.mark_dirty(a);
            return;
        }
        match a >> 20 {
            0x040 => {
                if a < 0x0404_0000 {
                    let o = (a & 0x1FFC) as usize;
                    let old = u32::from_be_bytes(self.s.sp_mem[o..o + 4].try_into().unwrap());
                    let new = (old & !mask) | (v & mask);
                    self.s.sp_mem[o..o + 4].copy_from_slice(&new.to_be_bytes());
                    if o >= 0x1000 {
                        self.rsp_jit.invalidate(o as u32 - 0x1000, 4);
                    }
                } else {
                    self.sp_reg_write(a, v);
                }
            }
            0x041 => self.dp_reg_write((a >> 2) & 7, v),
            0x043 => self.mi_write((a >> 2) & 3, v),
            0x044 => self.vi_write((a >> 2) & 0xF, v),
            0x045 => self.ai_write((a >> 2) & 7, v),
            0x046 => self.pi_write((a >> 2) & 0xF, v),
            0x047 => self.s.ri[((a >> 2) & 7) as usize] = v,
            0x048 => self.si_write((a >> 2) & 7, v),
            0x080..=0x0FF => {
                let o = (a & 0x7FFC) as usize;
                let old = u32::from_be_bytes(self.s.cart.sram[o..o + 4].try_into().unwrap());
                let new = (old & !mask) | (v & mask);
                self.s.cart.sram[o..o + 4].copy_from_slice(&new.to_be_bytes());
            }
            0x1FC => {
                if a >= 0x1FC0_07C0 && a < 0x1FC0_0800 {
                    let o = (a & 0x3C) as usize;
                    let old = u32::from_be_bytes(self.s.pif.ram[o..o + 4].try_into().unwrap());
                    let new = (old & !mask) | (v & mask);
                    self.s.pif.ram[o..o + 4].copy_from_slice(&new.to_be_bytes());
                }
            }
            _ => {}
        }
    }

    // ------------------------------------------------------------ MI

    fn mi_write(&mut self, reg: u32, v: u32) {
        match reg {
            0 => {
                let mut mode = (self.s.mi.mode & !0x7F) | (v & 0x7F);
                let pairs = [(7, 8, 1 << 7), (9, 10, 1 << 8), (12, 13, 1 << 9)];
                for (clear, set, bit) in pairs {
                    if v & (1 << clear) != 0 {
                        mode &= !bit;
                    }
                    if v & (1 << set) != 0 {
                        mode |= bit;
                    }
                }
                self.s.mi.mode = mode;
                if v & (1 << 11) != 0 {
                    self.clear_mi(MI_DP);
                }
            }
            3 => {
                for i in 0..6 {
                    if v & (1 << (i * 2)) != 0 {
                        self.s.mi.mask &= !(1 << i);
                    }
                    if v & (1 << (i * 2 + 1)) != 0 {
                        self.s.mi.mask |= 1 << i;
                    }
                }
                self.update_irq();
            }
            _ => {}
        }
    }

    // ------------------------------------------------------------ VI

    pub fn vi_field_cycles(&self) -> u64 {
        self.s.vi.line_cycles * self.s.vi.lines
    }

    /// Recomputes line timing from the sync registers; called at field start.
    fn vi_update_timing(&mut self) {
        let v_sync = (self.s.vi.regs[6] & 0x3FF) as u64;
        let h_sync = (self.s.vi.regs[7] & 0xFFF) as u64;
        let h_sync = if h_sync == 0 { 3093 } else { h_sync };
        self.s.vi.lines = if v_sync == 0 { 263 } else { (v_sync + 1) / 2 };
        self.s.vi.line_cycles = (h_sync + 1) * CPU_HZ / VI_HZ;
    }

    pub fn vi_reset(&mut self) {
        self.vi_update_timing();
        self.s.vi.field_start = self.s.cycles;
        self.s.vi.field = 0;
        self.schedule_at(EV_FIELD, self.s.cycles + self.vi_field_cycles());
        self.schedule_vi_intr(self.s.cycles);
    }

    fn schedule_vi_intr(&mut self, not_before: u64) {
        let line = ((self.s.vi.regs[3] & 0x3FF) >> 1) as u64;
        if line >= self.s.vi.lines {
            self.s.events[EV_VI_INTR] = u64::MAX;
            self.update_next_event();
            return;
        }
        let mut at = self.s.vi.field_start + line * self.s.vi.line_cycles;
        if at < not_before {
            at += self.vi_field_cycles();
        }
        self.schedule_at(EV_VI_INTR, at);
    }

    fn vi_field_end(&mut self) {
        self.s.vi.field_start = self.s.cycles;
        if self.s.vi.regs[0] & 0x40 != 0 {
            self.s.vi.field ^= 1;
        } else {
            self.s.vi.field = 0;
        }
        self.vi_update_timing();
        self.schedule_at(EV_FIELD, self.s.cycles + self.vi_field_cycles());
        self.schedule_vi_intr(self.s.cycles);
        self.s.frame_done = 1;
        self.break_cpu_loop();
    }

    fn vi_read(&mut self, reg: u32) -> u32 {
        if reg == 4 {
            let elapsed = self.s.cycles.saturating_sub(self.s.vi.field_start);
            let line = (elapsed / self.s.vi.line_cycles.max(1)) as u32;
            return ((line << 1) | self.s.vi.field) & 0x3FF;
        }
        self.s.vi.regs.get(reg as usize).copied().unwrap_or(0)
    }

    fn vi_write(&mut self, reg: u32, v: u32) {
        match reg {
            3 => {
                self.s.vi.regs[3] = v;
                self.schedule_vi_intr(self.s.cycles + 1);
            }
            4 => self.clear_mi(MI_VI),
            r if (r as usize) < 14 => self.s.vi.regs[r as usize] = v,
            _ => {}
        }
    }

    // ------------------------------------------------------------ AI

    fn ai_cycles_per_sample(&self) -> u64 {
        (self.s.ai.dacrate as u64 + 1) * CPU_HZ / VI_HZ
    }

    fn ai_read(&mut self, reg: u32) -> u32 {
        match reg {
            1 => {
                if self.s.ai.fifo_count == 0 || self.s.events[EV_AI] == u64::MAX {
                    return 0;
                }
                let left = self.s.events[EV_AI].saturating_sub(self.s.cycles);
                let samples = left / self.ai_cycles_per_sample().max(1);
                ((samples * 4) as u32).min(self.s.ai.fifo_len[0]) & !7
            }
            3 => {
                let count = self.s.ai.fifo_count;
                let mut status = 0x0110_0000;
                if count >= 2 {
                    status |= 1 << 31 | 1;
                }
                if count >= 1 {
                    status |= 1 << 30;
                }
                if self.s.ai.control & 1 != 0 {
                    status |= 1 << 25;
                }
                status
            }
            _ => self.s.ai.len,
        }
    }

    fn ai_write(&mut self, reg: u32, v: u32) {
        match reg {
            0 => self.s.ai.dram_addr = v & 0xFF_FFF8,
            1 => {
                self.s.ai.len = v & 0x3_FFF8;
                if self.s.ai.len == 0 || self.s.ai.fifo_count >= 2 {
                    return;
                }
                let slot = self.s.ai.fifo_count as usize;
                self.s.ai.fifo_addr[slot] = self.s.ai.dram_addr;
                self.s.ai.fifo_len[slot] = self.s.ai.len;
                self.s.ai.fifo_count += 1;
                if slot == 0 {
                    self.ai_start_dma();
                }
            }
            2 => self.s.ai.control = v & 1,
            3 => self.clear_mi(MI_AI),
            4 => self.s.ai.dacrate = v & 0x3FFF,
            5 => self.s.ai.bitrate = v & 0xF,
            _ => {}
        }
    }

    /// Appends silence for the time the AI has been idle.
    fn ai_fill_silence(&mut self) {
        if self.audio_silent_from == u64::MAX {
            return;
        }
        let samples = (self.s.cycles - self.audio_silent_from) / self.audio_cycles_per_sample;
        self.audio.resize(self.audio.len() + samples as usize * 2, 0);
        self.audio_silent_from += samples * self.audio_cycles_per_sample;
    }

    fn ai_start_dma(&mut self) {
        let addr = self.s.ai.fifo_addr[0];
        let len = self.s.ai.fifo_len[0];
        self.ai_fill_silence();
        self.audio_silent_from = u64::MAX;
        self.audio_cycles_per_sample = self.ai_cycles_per_sample().max(1);
        for i in (0..len).step_by(4) {
            let w = self.s.rdram[(((addr + i) >> 2) as usize) & (RDRAM_SIZE / 4 - 1)];
            self.audio.push((w >> 16) as i16);
            self.audio.push(w as i16);
        }
        let samples = (len / 4) as u64;
        self.schedule(EV_AI, samples * self.audio_cycles_per_sample);
    }

    fn ai_end_of_dma(&mut self) {
        self.s.ai.fifo_addr[0] = self.s.ai.fifo_addr[1];
        self.s.ai.fifo_len[0] = self.s.ai.fifo_len[1];
        self.s.ai.fifo_count = self.s.ai.fifo_count.saturating_sub(1);
        if self.s.ai.fifo_count > 0 {
            self.ai_start_dma();
        } else {
            self.audio_silent_from = self.s.cycles;
        }
        self.raise_mi(MI_AI);
    }

    // ------------------------------------------------------------ PI

    fn pi_write(&mut self, reg: u32, v: u32) {
        match reg {
            0 => self.s.pi.regs[0] = v & 0xFF_FFFE,
            1 => self.s.pi.regs[1] = v & 0xFFFF_FFFE,
            2 => {
                // RDRAM -> cartridge (only SRAM is writable).
                let len = (v & 0xFF_FFFF) + 1;
                let (dram, cart) = (self.s.pi.regs[0], self.s.pi.regs[1]);
                if (0x0800_0000..0x1000_0000).contains(&cart) {
                    for i in 0..len {
                        let b = self.rdram_read8(dram + i);
                        self.s.cart.sram[((cart + i) & 0x7FFF) as usize] = b;
                    }
                }
                self.pi_finish(len);
            }
            3 => {
                // Cartridge -> RDRAM.
                let len = (v & 0xFF_FFFF) + 1;
                let (dram, cart) = (self.s.pi.regs[0], self.s.pi.regs[1]);
                for i in 0..len {
                    let c = cart + i;
                    let b = if (0x1000_0000..0x1FC0_0000).contains(&c) {
                        self.rom_byte(c - 0x1000_0000)
                    } else if (0x0800_0000..0x1000_0000).contains(&c) {
                        self.s.cart.sram[(c & 0x7FFF) as usize]
                    } else {
                        0
                    };
                    self.rdram_write8(dram + i, b);
                }
                self.pi_finish(len);
            }
            4 => {
                if v & 2 != 0 {
                    self.clear_mi(MI_PI);
                }
                if v & 1 != 0 {
                    self.s.pi.regs[4] = 0;
                }
            }
            r if (5..13).contains(&r) => self.s.pi.regs[r as usize] = v & 0xFF,
            _ => {}
        }
    }

    fn pi_finish(&mut self, len: u32) {
        self.s.pi.regs[0] = (self.s.pi.regs[0] + len + 7) & !7 & 0xFF_FFFF;
        self.s.pi.regs[1] = (self.s.pi.regs[1] + len + 1) & !1;
        self.s.pi.regs[4] |= 1;
        self.schedule(EV_PI, len as u64 * 2 + 200);
    }

    // ------------------------------------------------------------ SI

    fn si_write(&mut self, reg: u32, v: u32) {
        match reg {
            0 => self.s.si.dram_addr = v & 0xFF_FFF8,
            1 => {
                // PIF RAM -> RDRAM.
                self.pif_process();
                let dram = self.s.si.dram_addr;
                for i in 0..64 {
                    let b = self.s.pif.ram[i as usize];
                    self.rdram_write8(dram + i, b);
                }
                self.si_start();
            }
            4 => {
                // RDRAM -> PIF RAM.
                let dram = self.s.si.dram_addr;
                for i in 0..64 {
                    self.s.pif.ram[i as usize] = self.rdram_read8(dram + i);
                }
                self.si_start();
            }
            6 => {
                self.s.si.status &= !(1 << 12);
                self.clear_mi(MI_SI);
            }
            _ => {}
        }
    }

    fn si_start(&mut self) {
        self.s.si.status |= 1;
        self.schedule(EV_SI, 6000);
    }
}
