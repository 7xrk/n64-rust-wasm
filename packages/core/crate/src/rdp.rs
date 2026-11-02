//! RDP front end. Commands are parsed here, texture loads fill TMEM on the CPU
//! side, and every drawing command is emitted as a primitive record for the
//! WebGPU rasterizer (packages/core/src/renderer.ts). Record layouts are
//! mirrored in packages/core/src/rdp.wgsl.

use crate::state::*;
use crate::sys::System;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Tile {
    pub fmt: u32,
    pub size: u32,
    pub line: u32,
    pub tmem: u32,
    pub palette: u32,
    pub ct: u32,
    pub mt: u32,
    pub mask_t: u32,
    pub shift_t: u32,
    pub cs: u32,
    pub ms: u32,
    pub mask_s: u32,
    pub shift_s: u32,
    pub sl: u32,
    pub tl: u32,
    pub sh: u32,
    pub th: u32,
}

#[repr(C)]
pub struct Rdp {
    pub start: u32,
    pub end: u32,
    pub current: u32,
    pub status: u32,
    pub cmd: [u64; 22],
    pub cmd_len: u32,
    pub other_modes: u64,
    pub combine: u64,
    pub fill_color: u32,
    pub fog_color: u32,
    pub blend_color: u32,
    pub prim_color: u32,
    pub env_color: u32,
    pub prim_z: u32,
    pub prim_lod: u32,
    pub key_r: u32,
    pub key_gb: u32,
    pub convert: u64,
    pub scissor: [u32; 4],
    pub tex_size: u32,
    pub tex_width: u32,
    pub tex_addr: u32,
    pub ci_size: u32,
    pub ci_width: u32,
    pub ci_addr: u32,
    pub z_addr: u32,
    pub tiles: [Tile; 8],
    /// TMEM as 16-bit units; byte `b` is the high half of `tmem[b / 2]` when even.
    pub tmem: [u16; 2048],
}

const DP_XBUS: u32 = 1 << 0;
const DP_FREEZE: u32 = 1 << 1;
const DP_FLUSH: u32 = 1 << 2;
pub(crate) const DP_CBUF_READY: u32 = 1 << 7;
const DP_END_VALID: u32 = 1 << 9;
const DP_START_VALID: u32 = 1 << 10;

pub const STATE_WORDS: usize = 56;
pub const PRIM_WORDS: usize = 52;
pub const BATCH_WORDS: usize = 8;

pub const PRIM_TRI: u32 = 0;
pub const PRIM_TEXRECT: u32 = 1;
pub const PRIM_FILLRECT: u32 = 2;

/// Primitive records for one frame, consumed by the GPU backend.
pub struct GpuOut {
    pub states: Vec<u32>,
    pub prims: Vec<u32>,
    pub batches: Vec<u32>,
    pub tmem: Vec<u16>,
    pub enabled: bool,
    state_dirty: bool,
    tmem_dirty: bool,
    batch_open: bool,
}

impl GpuOut {
    pub fn new() -> GpuOut {
        GpuOut {
            states: Vec::new(),
            prims: Vec::new(),
            batches: Vec::new(),
            tmem: Vec::new(),
            enabled: true,
            state_dirty: true,
            tmem_dirty: true,
            batch_open: false,
        }
    }

    pub fn clear(&mut self) {
        self.states.clear();
        self.prims.clear();
        self.batches.clear();
        self.tmem.clear();
        self.invalidate();
    }

    /// Forces the next primitive to carry fresh state, TMEM and batch records.
    pub fn invalidate(&mut self) {
        self.state_dirty = true;
        self.tmem_dirty = true;
        self.batch_open = false;
    }
}

#[inline(always)]
fn bits(w: u64, lo: u32, len: u32) -> u32 {
    ((w >> lo) & ((1u64 << len) - 1)) as u32
}

#[inline(always)]
fn sext(v: u32, len: u32) -> i32 {
    ((v << (32 - len)) as i32) >> (32 - len)
}

impl System {
    // ------------------------------------------------------------ DP registers

    pub fn dp_reg_read(&mut self, reg: u32) -> u32 {
        let r = &self.s.rdp;
        match reg {
            0 => r.start,
            1 => r.end,
            2 => r.current,
            3 => r.status | DP_CBUF_READY,
            _ => 0,
        }
    }

    pub fn dp_reg_write(&mut self, reg: u32, v: u32) {
        match reg {
            0 => {
                if self.s.rdp.status & DP_START_VALID == 0 {
                    self.s.rdp.start = v & 0xFF_FFF8;
                    self.s.rdp.status |= DP_START_VALID;
                }
            }
            1 => {
                self.s.rdp.end = v & 0xFF_FFF8;
                if self.s.rdp.status & DP_START_VALID != 0 {
                    self.s.rdp.current = self.s.rdp.start;
                    self.s.rdp.status &= !DP_START_VALID;
                }
                self.rdp_process();
            }
            3 => {
                let st = &mut self.s.rdp.status;
                for (clear, set, flag) in [(0, 1, DP_XBUS), (2, 3, DP_FREEZE), (4, 5, DP_FLUSH)] {
                    if v & (1 << clear) != 0 {
                        *st &= !flag;
                    }
                    if v & (1 << set) != 0 {
                        *st |= flag;
                    }
                }
                if v & (1 << 2) != 0 {
                    self.rdp_process();
                }
            }
            _ => {}
        }
    }

    fn rdp_fetch(&self, a: u32) -> u64 {
        if self.s.rdp.status & DP_XBUS != 0 {
            let o = (a & 0xFF8) as usize;
            u64::from_be_bytes(self.s.sp_mem[o..o + 8].try_into().unwrap())
        } else {
            let i = ((a as usize) >> 2) & (RDRAM_SIZE / 4 - 2);
            (self.s.rdram[i] as u64) << 32 | self.s.rdram[i + 1] as u64
        }
    }

    fn rdp_process(&mut self) {
        if self.s.rdp.status & DP_FREEZE != 0 {
            return;
        }
        while self.s.rdp.current < self.s.rdp.end {
            let w = self.rdp_fetch(self.s.rdp.current);
            self.s.rdp.current += 8;
            let n = self.s.rdp.cmd_len as usize;
            self.s.rdp.cmd[n] = w;
            self.s.rdp.cmd_len += 1;
            let op = (self.s.rdp.cmd[0] >> 56) as u32 & 0x3F;
            if n + 1 >= command_length(op) {
                self.s.rdp.cmd_len = 0;
                self.rdp_execute(op);
            }
        }
        self.s.rdp.status &= !DP_END_VALID;
    }

    fn rdp_execute(&mut self, op: u32) {
        let w = self.s.rdp.cmd[0];
        let r = &mut self.s.rdp;
        match op {
            0x08..=0x0F => self.rdp_triangle(op),
            0x24 | 0x25 => self.rdp_texrect(op == 0x25),
            0x29 => self.schedule(crate::state::EV_DP, 100),
            0x2A => {
                r.key_gb = w as u32;
                self.gpu.state_dirty = true;
            }
            0x2B => {
                r.key_r = w as u32;
                self.gpu.state_dirty = true;
            }
            0x2C => {
                r.convert = w & 0x3F_FFFF_FFFF_FFFF;
                self.gpu.state_dirty = true;
            }
            0x2D => {
                r.scissor = [bits(w, 44, 12), bits(w, 32, 12), bits(w, 12, 12), bits(w, 0, 12)];
                self.gpu.state_dirty = true;
            }
            0x2E => {
                r.prim_z = w as u32;
                self.gpu.state_dirty = true;
            }
            0x2F => {
                r.other_modes = w & 0x00FF_FFFF_FFFF_FFFF;
                self.gpu.state_dirty = true;
            }
            0x30 => self.rdp_load_tlut(),
            0x32 => {
                let t = &mut r.tiles[bits(w, 24, 3) as usize];
                t.sl = bits(w, 44, 12);
                t.tl = bits(w, 32, 12);
                t.sh = bits(w, 12, 12);
                t.th = bits(w, 0, 12);
                self.gpu.state_dirty = true;
            }
            0x33 => self.rdp_load_block(),
            0x34 => self.rdp_load_tile(),
            0x35 => {
                let t = &mut r.tiles[bits(w, 24, 3) as usize];
                t.fmt = bits(w, 53, 3);
                t.size = bits(w, 51, 2);
                t.line = bits(w, 41, 9);
                t.tmem = bits(w, 32, 9);
                t.palette = bits(w, 20, 4);
                t.ct = bits(w, 19, 1);
                t.mt = bits(w, 18, 1);
                t.mask_t = bits(w, 14, 4);
                t.shift_t = bits(w, 10, 4);
                t.cs = bits(w, 9, 1);
                t.ms = bits(w, 8, 1);
                t.mask_s = bits(w, 4, 4);
                t.shift_s = bits(w, 0, 4);
                self.gpu.state_dirty = true;
            }
            0x36 => self.rdp_fillrect(),
            0x37 => {
                r.fill_color = w as u32;
                self.gpu.state_dirty = true;
            }
            0x38 => {
                r.fog_color = w as u32;
                self.gpu.state_dirty = true;
            }
            0x39 => {
                r.blend_color = w as u32;
                self.gpu.state_dirty = true;
            }
            0x3A => {
                r.prim_color = w as u32;
                r.prim_lod = bits(w, 32, 16);
                self.gpu.state_dirty = true;
            }
            0x3B => {
                r.env_color = w as u32;
                self.gpu.state_dirty = true;
            }
            0x3C => {
                r.combine = w & 0x00FF_FFFF_FFFF_FFFF;
                self.gpu.state_dirty = true;
            }
            0x3D => {
                r.tex_size = bits(w, 51, 2);
                r.tex_width = bits(w, 32, 10) + 1;
                r.tex_addr = bits(w, 0, 26);
            }
            0x3E => {
                r.z_addr = bits(w, 0, 26);
                self.gpu.batch_open = false;
            }
            0x3F => {
                r.ci_size = bits(w, 51, 2);
                r.ci_width = bits(w, 32, 10) + 1;
                r.ci_addr = bits(w, 0, 26);
                self.gpu.batch_open = false;
            }
            _ => {}
        }
    }

    // ------------------------------------------------------------ TMEM

    #[inline(always)]
    fn tmem_write8(&mut self, b: u32, v: u8) {
        let h = &mut self.s.rdp.tmem[((b & 0xFFF) >> 1) as usize];
        if b & 1 == 0 {
            *h = (*h & 0x00FF) | (v as u16) << 8;
        } else {
            *h = (*h & 0xFF00) | v as u16;
        }
    }

    /// Stores a 32-bit texel split across the low (RG) and high (BA) banks.
    #[inline(always)]
    fn tmem_write32(&mut self, b: u32, texel: u32) {
        let i = ((b & 0x7FF) >> 1) as usize;
        self.s.rdp.tmem[i] = (texel >> 16) as u16;
        self.s.rdp.tmem[i + 0x400] = texel as u16;
    }

    fn rdp_load_block(&mut self) {
        let w = self.s.rdp.cmd[0];
        let tile = self.s.rdp.tiles[bits(w, 24, 3) as usize];
        let (sl, tl, sh, dxt) = (bits(w, 44, 12), bits(w, 32, 12), bits(w, 12, 12), bits(w, 0, 12));
        let r = &self.s.rdp;
        let size = r.tex_size;
        let base = r.tex_addr + (((tl * r.tex_width + sl) << size) >> 1);
        let texels = (sh.saturating_sub(sl) + 1).min(2048);
        let dst = tile.tmem * 8;
        if size == 3 {
            for k in 0..texels {
                let word = k / 2;
                let odd = ((word * dxt) >> 11) & 1;
                let texel = self.s.rdram[(((base + k * 4) >> 2) as usize) & (RDRAM_SIZE / 4 - 1)];
                self.tmem_write32((dst + k * 2) ^ (odd << 2), texel);
            }
        } else {
            let bytes = ((texels << size) + 1) >> 1;
            for b in 0..bytes {
                let odd = (((b / 8) * dxt) >> 11) & 1;
                let v = self.rdram_read8(base + b);
                self.tmem_write8((dst + b) ^ (odd << 2), v);
            }
        }
        self.set_tile_size(bits(w, 24, 3) as usize, sl << 2, tl << 2, sh << 2, tl << 2);
        self.gpu.tmem_dirty = true;
    }

    fn rdp_load_tile(&mut self) {
        let w = self.s.rdp.cmd[0];
        let idx = bits(w, 24, 3) as usize;
        let tile = self.s.rdp.tiles[idx];
        let (sl, tl, sh, th) = (bits(w, 44, 12), bits(w, 32, 12), bits(w, 12, 12), bits(w, 0, 12));
        let r = &self.s.rdp;
        let size = r.tex_size;
        let width = r.tex_width;
        let addr = r.tex_addr;
        let (s0, t0, s1, t1) = (sl >> 2, tl >> 2, sh >> 2, th >> 2);
        let line = tile.line * 8;
        for t in t0..=t1.max(t0) {
            let row = t - t0;
            let odd = (row & 1) << 2;
            let dst_row = tile.tmem * 8 + row * line;
            if size == 3 {
                for s in s0..=s1.max(s0) {
                    let texel = self.s.rdram[(((addr + (t * width + s) * 4) >> 2) as usize) & (RDRAM_SIZE / 4 - 1)];
                    self.tmem_write32((dst_row + (s - s0) * 2) ^ odd, texel);
                }
            } else {
                let src = addr + (((t * width + s0) << size) >> 1);
                let bytes = (((s1.max(s0) - s0 + 1) << size) + 1) >> 1;
                for b in 0..bytes {
                    let v = self.rdram_read8(src + b);
                    self.tmem_write8((dst_row + b) ^ odd, v);
                }
            }
        }
        self.set_tile_size(idx, sl, tl, sh, th);
        self.gpu.tmem_dirty = true;
    }

    fn rdp_load_tlut(&mut self) {
        let w = self.s.rdp.cmd[0];
        let idx = bits(w, 24, 3) as usize;
        let tile = self.s.rdp.tiles[idx];
        let (sl, tl, sh, th) = (bits(w, 44, 12), bits(w, 32, 12), bits(w, 12, 12), bits(w, 0, 12));
        let count = ((sh >> 2).saturating_sub(sl >> 2) + 1).min(256);
        let src = self.s.rdp.tex_addr + ((tl >> 2) * self.s.rdp.tex_width + (sl >> 2)) * 2;
        let dst = tile.tmem * 4;
        for i in 0..count {
            let c = self.rdram_read16(src + i * 2);
            for k in 0..4 {
                self.s.rdp.tmem[((dst + i * 4 + k) & 0x7FF) as usize] = c;
            }
        }
        self.set_tile_size(idx, sl, tl, sh, th);
        self.gpu.tmem_dirty = true;
    }

    fn set_tile_size(&mut self, idx: usize, sl: u32, tl: u32, sh: u32, th: u32) {
        let t = &mut self.s.rdp.tiles[idx];
        t.sl = sl;
        t.tl = tl;
        t.sh = sh;
        t.th = th;
        self.gpu.state_dirty = true;
    }

    // ------------------------------------------------------------ primitives

    fn scissor_px(&self) -> (i32, i32, i32, i32) {
        let s = self.s.rdp.scissor;
        (s[0] as i32 >> 2, s[1] as i32 >> 2, (s[2] as i32 + 3) >> 2, (s[3] as i32 + 3) >> 2)
    }

    fn rdp_triangle(&mut self, op: u32) {
        let c = self.s.rdp.cmd;
        let yl = sext(bits(c[0], 32, 14), 14);
        let ym = sext(bits(c[0], 16, 14), 14);
        let yh = sext(bits(c[0], 0, 14), 14);
        let xl = (c[1] >> 32) as i32 as i64;
        let dxl = c[1] as u32 as i32 as i64;
        let xh = (c[2] >> 32) as i32 as i64;
        let dxh = c[2] as u32 as i32 as i64;
        let xm = (c[3] >> 32) as i32 as i64;
        let dxm = c[3] as u32 as i32 as i64;
        let top = (yh & !3) as i64;
        let xs = [
            xh,
            xh + dxh * (yl as i64 - top) / 4,
            xm,
            xm + dxm * (ym as i64 - top) / 4,
            xl,
            xl + dxl * (yl as i64 - ym as i64) / 4,
        ];
        let xmin = (xs.iter().min().unwrap() >> 16) as i32 - 1;
        let xmax = (xs.iter().max().unwrap() >> 16) as i32 + 2;
        let words = command_length(op);
        let mut raw = [0u32; 44];
        for k in 0..words {
            raw[k * 2] = (c[k] >> 32) as u32;
            raw[k * 2 + 1] = c[k] as u32;
        }
        let flags = op & 7;
        let header = PRIM_TRI | flags << 4 | bits(c[0], 48, 3) << 8 | bits(c[0], 55, 1) << 12 | bits(c[0], 51, 3) << 13;
        self.emit(header, (xmin, yh >> 2, xmax, ((yl + 3) >> 2) + 1), &raw[..words * 2]);
    }

    fn rdp_texrect(&mut self, flip: bool) {
        let c = self.s.rdp.cmd;
        let (xl, yl, xh, yh) = (bits(c[0], 44, 12), bits(c[0], 32, 12), bits(c[0], 12, 12), bits(c[0], 0, 12));
        let tile = bits(c[0], 24, 3);
        let raw = [
            xl,
            yl,
            xh,
            yh,
            bits(c[1], 48, 16) as u16 as i16 as i32 as u32,
            bits(c[1], 32, 16) as u16 as i16 as i32 as u32,
            bits(c[1], 16, 16) as u16 as i16 as i32 as u32,
            bits(c[1], 0, 16) as u16 as i16 as i32 as u32,
        ];
        let header = PRIM_TEXRECT | 2 << 4 | (flip as u32) << 7 | tile << 8;
        let bbox = (xh as i32 >> 2, yh as i32 >> 2, (xl as i32 >> 2) + 1, (yl as i32 >> 2) + 1);
        self.emit(header, bbox, &raw);
    }

    fn rdp_fillrect(&mut self) {
        let w = self.s.rdp.cmd[0];
        let (xl, yl, xh, yh) = (bits(w, 44, 12), bits(w, 32, 12), bits(w, 12, 12), bits(w, 0, 12));
        let bbox = (xh as i32 >> 2, yh as i32 >> 2, (xl as i32 >> 2) + 1, (yl as i32 >> 2) + 1);
        self.emit(PRIM_FILLRECT, bbox, &[xl, yl, xh, yh]);
    }

    fn state_block(&self) -> [u32; STATE_WORDS] {
        let r = &self.s.rdp;
        let mut b = [0u32; STATE_WORDS];
        b[0] = (r.other_modes >> 32) as u32;
        b[1] = r.other_modes as u32;
        b[2] = (r.combine >> 32) as u32;
        b[3] = r.combine as u32;
        b[4] = r.fill_color;
        b[5] = r.fog_color;
        b[6] = r.blend_color;
        b[7] = r.prim_color;
        b[8] = r.env_color;
        b[9] = r.prim_z;
        b[10] = r.prim_lod;
        b[11] = r.key_r;
        b[12] = r.key_gb;
        b[13] = (r.convert >> 32) as u32;
        b[14] = r.convert as u32;
        b[19..23].copy_from_slice(&r.scissor);
        for (k, t) in r.tiles.iter().enumerate() {
            let o = 24 + k * 4;
            b[o] = t.fmt
                | t.size << 3
                | t.palette << 5
                | t.ct << 9
                | t.mt << 10
                | t.cs << 11
                | t.ms << 12
                | t.mask_t << 13
                | t.shift_t << 17
                | t.mask_s << 21
                | t.shift_s << 25;
            b[o + 1] = t.line | t.tmem << 16;
            b[o + 2] = t.sl | t.tl << 16;
            b[o + 3] = t.sh | t.th << 16;
        }
        b
    }

    fn emit(&mut self, header: u32, bbox: (i32, i32, i32, i32), raw: &[u32]) {
        if !self.gpu.enabled {
            return;
        }
        let (sx0, sy0, sx1, sy1) = self.scissor_px();
        let x0 = bbox.0.max(sx0).max(0);
        let y0 = bbox.1.max(sy0).max(0);
        let x1 = bbox.2.min(sx1).min(self.s.rdp.ci_width as i32);
        let y1 = bbox.3.min(sy1).min(1024);
        if x0 >= x1 || y0 >= y1 {
            return;
        }
        let r = &self.s.rdp;
        if !self.gpu.batch_open {
            self.gpu.batches.extend_from_slice(&[
                r.ci_addr,
                r.ci_width,
                r.ci_size,
                r.z_addr,
                (self.gpu.prims.len() / PRIM_WORDS) as u32,
                0,
                u32::MAX,
                0,
            ]);
            self.gpu.batch_open = true;
        }
        if self.gpu.state_dirty {
            let block = self.state_block();
            self.gpu.states.extend_from_slice(&block);
            self.gpu.state_dirty = false;
        }
        if self.gpu.tmem_dirty {
            self.gpu.tmem.extend_from_slice(&self.s.rdp.tmem);
            self.gpu.tmem_dirty = false;
        }
        let state_index = (self.gpu.states.len() / STATE_WORDS - 1) as u32;
        let tmem_index = (self.gpu.tmem.len() / 2048 - 1) as u32;
        let mut rec = [0u32; PRIM_WORDS];
        rec[0] = header;
        rec[1] = state_index;
        rec[2] = tmem_index;
        rec[3] = x0 as u32;
        rec[4] = y0 as u32;
        rec[5] = x1 as u32;
        rec[6] = y1 as u32;
        rec[8..8 + raw.len()].copy_from_slice(raw);
        self.gpu.prims.extend_from_slice(&rec);
        let b = self.gpu.batches.len() - BATCH_WORDS;
        let batch = &mut self.gpu.batches[b..];
        batch[5] += 1;
        let (bx0, by0) = (batch[6] & 0xFFFF, batch[6] >> 16);
        let (bx1, by1) = (batch[7] & 0xFFFF, batch[7] >> 16);
        let nx0 = bx0.min(x0 as u32);
        let ny0 = by0.min(y0 as u32);
        let nx1 = bx1.max(x1 as u32);
        let ny1 = by1.max(y1 as u32);
        batch[6] = nx0 | ny0 << 16;
        batch[7] = nx1 | ny1 << 16;
    }
}

/// Length in 64-bit words of the command with opcode `op`.
fn command_length(op: u32) -> usize {
    match op {
        0x08..=0x0F => 4 + if op & 4 != 0 { 8 } else { 0 } + if op & 2 != 0 { 8 } else { 0 } + if op & 1 != 0 { 2 } else { 0 },
        0x24 | 0x25 => 2,
        _ => 1,
    }
}
