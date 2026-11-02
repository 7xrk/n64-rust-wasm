//! wasm code for one RSP block. Every translated instruction has the exact
//! semantics of its interpreter counterpart in `rsp.rs`; anything without a
//! translation calls the interpreter through the imported `op`.

use core::mem::offset_of;

use super::{kind, Kind};
use crate::rdp::{Rdp, DP_CBUF_READY};
use crate::rsp::{Rsp, ELEMENTS};

/// Linear memory addresses of the state the generated code touches.
#[derive(Clone, Copy)]
pub struct Addrs {
    pub rsp: u32,
    pub rdp: u32,
    pub dmem: u32,
}

impl Addrs {
    fn rsp(&self, field: usize) -> u32 {
        self.rsp + field as u32
    }

    fn gpr(&self, r: u32) -> u32 {
        self.rsp(offset_of!(Rsp, gpr)) + 4 * r
    }

    fn vr(&self, v: u32) -> u32 {
        self.rsp(offset_of!(Rsp, vr)) + 16 * v
    }

    /// Accumulator lanes `2k` and `2k + 1`.
    fn acc(&self, k: u32) -> u32 {
        self.rsp(offset_of!(Rsp, acc)) + 16 * k
    }
}

// Opcodes.
const LOOP: u8 = 0x03;
const IF: u8 = 0x04;
const ELSE: u8 = 0x05;
const END: u8 = 0x0B;
const BR_IF: u8 = 0x0D;
const CALL: u8 = 0x10;
const DROP: u8 = 0x1A;
const SELECT: u8 = 0x1B;
const LOCAL_GET: u8 = 0x20;
const LOCAL_SET: u8 = 0x21;
const LOCAL_TEE: u8 = 0x22;
const I32_LOAD: u8 = 0x28;
const I32_LOAD8_S: u8 = 0x2C;
const I32_LOAD8_U: u8 = 0x2D;
const I32_LOAD16_U: u8 = 0x2F;
const I32_STORE: u8 = 0x36;
const I32_STORE8: u8 = 0x3A;
const I32_STORE16: u8 = 0x3B;
const I32_CONST: u8 = 0x41;
const I32_EQZ: u8 = 0x45;
const I32_EQ: u8 = 0x46;
const I32_NE: u8 = 0x47;
const I32_LT_S: u8 = 0x48;
const I32_LT_U: u8 = 0x49;
const I32_GT_S: u8 = 0x4A;
const I32_LE_S: u8 = 0x4C;
const I32_LE_U: u8 = 0x4D;
const I32_GE_S: u8 = 0x4E;
const I32_ADD: u8 = 0x6A;
const I32_SUB: u8 = 0x6B;
const I32_AND: u8 = 0x71;
const I32_OR: u8 = 0x72;
const I32_XOR: u8 = 0x73;
const I32_SHL: u8 = 0x74;
const I32_SHR_S: u8 = 0x75;
const I32_SHR_U: u8 = 0x76;
const I32_EXTEND16_S: u8 = 0xC1;

// SIMD opcodes, after the 0xFD prefix.
const V128_LOAD: u32 = 0x00;
const V128_STORE: u32 = 0x0B;
const V128_CONST: u32 = 0x0C;
const I8X16_SHUFFLE: u32 = 0x0D;
const I16X8_SPLAT: u32 = 0x10;
const I16X8_EQ: u32 = 0x2D;
const I16X8_NE: u32 = 0x2E;
const I16X8_LT_S: u32 = 0x2F;
const I16X8_LT_U: u32 = 0x30;
const I16X8_GT_S: u32 = 0x31;
const I32X4_EQ: u32 = 0x37;
const V128_NOT: u32 = 0x4D;
const V128_AND: u32 = 0x4E;
const V128_ANDNOT: u32 = 0x4F;
const V128_OR: u32 = 0x50;
const V128_XOR: u32 = 0x51;
const V128_BITSELECT: u32 = 0x52;
const I16X8_BITMASK: u32 = 0x84;
const I16X8_NARROW_I32X4_S: u32 = 0x85;
const I16X8_SHR_S: u32 = 0x8C;
const I16X8_ADD: u32 = 0x8E;
const I16X8_SUB: u32 = 0x91;
const I32X4_EXTEND_LOW_I16X8_S: u32 = 0xA7;
const I32X4_EXTEND_HIGH_I16X8_S: u32 = 0xA8;
const I32X4_EXTEND_LOW_I16X8_U: u32 = 0xA9;
const I32X4_EXTEND_HIGH_I16X8_U: u32 = 0xAA;
const I32X4_ADD: u32 = 0xAE;
const I32X4_SUB: u32 = 0xB1;
const I32X4_MUL: u32 = 0xB5;
const I32X4_EXTMUL_LOW_I16X8_S: u32 = 0xBC;
const I32X4_EXTMUL_HIGH_I16X8_S: u32 = 0xBD;
const I32X4_EXTMUL_LOW_I16X8_U: u32 = 0xBE;
const I32X4_EXTMUL_HIGH_I16X8_U: u32 = 0xBF;
const I64X2_EXTEND_LOW_I32X4_S: u32 = 0xC7;
const I64X2_EXTEND_HIGH_I32X4_S: u32 = 0xC8;
const I64X2_EXTEND_LOW_I32X4_U: u32 = 0xC9;
const I64X2_EXTEND_HIGH_I32X4_U: u32 = 0xCA;
const I64X2_SHL: u32 = 0xCB;
const I64X2_SHR_S: u32 = 0xCC;
const I64X2_SHR_U: u32 = 0xCD;
const I64X2_ADD: u32 = 0xCE;

// Local 0 is the step budget parameter. Locals 1..31 cache the scalar
// registers of the same number.
const BUDGET: u32 = 0;
const ADDR: u32 = 32;
const COND: u32 = 33;
const TARGET: u32 = 34;
const STEPS: u32 = 35;
const I32_LOCALS: u32 = 36;
// v128 locals: the two vector operands, the accumulator lanes of a multiply
// and temporaries.
const VA: u32 = I32_LOCALS;
const VB: u32 = VA + 1;
const W: u32 = VB + 1;
const T: u32 = W + 4;
const V128_LOCALS: u32 = 14;

const ACC_MASK: u64 = (1 << 48) - 1;
/// Lane n holds bit n, to spread a flag byte across lanes.
const LANE_BITS: [u16; 8] = [1, 2, 4, 8, 16, 32, 64, 128];
/// Low 32 bits of each 64-bit lane of two vectors, as four 32-bit lanes.
const LOW32_OF_I64: [u8; 16] = [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27];
/// Low 16 bits of each 32-bit lane of two vectors, as eight 16-bit lanes.
const LOW16_OF_I32: [u8; 16] = [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29];
/// Swaps the bytes of each 16-bit lane: big-endian DMEM to host lanes.
const SWAP16: [u8; 16] = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14];

/// Where a block goes next.
enum Exit {
    To(u32),
    /// The target is in the TARGET local.
    Register,
    /// Taken when the COND local is set; otherwise falls through.
    If(u32),
}

fn uleb(out: &mut Vec<u8>, mut v: u32) {
    loop {
        let b = (v & 0x7F) as u8;
        v >>= 7;
        if v == 0 {
            out.push(b);
            return;
        }
        out.push(b | 0x80);
    }
}

fn sleb(out: &mut Vec<u8>, mut v: i32) {
    loop {
        let b = (v & 0x7F) as u8;
        v >>= 7;
        if (v == 0 && b & 0x40 == 0) || (v == -1 && b & 0x40 != 0) {
            out.push(b);
            return;
        }
        out.push(b | 0x80);
    }
}

/// Builds the module for the block at `pc`. It imports the core's memory and
/// the interpreter callback `op(instr)`, and exports `run(budget) -> steps`.
pub fn module(a: Addrs, pc: u32, words: &[u32]) -> Vec<u8> {
    fn section(m: &mut Vec<u8>, id: u8, content: &[u8]) {
        m.push(id);
        uleb(m, content.len() as u32);
        m.extend_from_slice(content);
    }
    let mut m = b"\0asm\x01\0\0\0".to_vec();
    // Types: 0 = (i32) -> (), 1 = (i32) -> (i32).
    section(&mut m, 1, &[2, 0x60, 1, 0x7F, 0, 0x60, 1, 0x7F, 1, 0x7F]);
    let mut imports = vec![2];
    imports.extend_from_slice(b"\x03env\x06memory\x02\x00\x00");
    imports.extend_from_slice(b"\x03env\x02op\x00\x00");
    section(&mut m, 2, &imports);
    section(&mut m, 3, &[1, 1]);
    section(&mut m, 7, b"\x01\x03run\x00\x01");
    let mut func = vec![2];
    uleb(&mut func, I32_LOCALS - 1);
    func.push(0x7F);
    uleb(&mut func, V128_LOCALS);
    func.push(0x7B);
    func.extend(Emitter::block(a, pc, words));
    func.push(END);
    let mut code = vec![1];
    uleb(&mut code, func.len() as u32);
    code.extend(func);
    section(&mut m, 10, &code);
    m
}

/// Where the branch `i` at `pc` goes once its delay slot has run.
fn branch_exit(i: u32, pc: u32) -> Exit {
    match i >> 26 {
        0x00 => Exit::Register,
        0x02 | 0x03 => Exit::To((i & 0x3FF) << 2),
        _ => Exit::If(pc.wrapping_add(4).wrapping_add(((i as u16 as i16 as i32) << 2) as u32) & 0xFFC),
    }
}

struct Emitter {
    code: Vec<u8>,
    a: Addrs,
    /// Registers whose local holds the current value.
    loaded: u32,
    /// Registers whose local is newer than memory.
    dirty: u32,
}

impl Emitter {
    /// A block that branches back to its own start repeats inside wasm while
    /// the budget allows another pass.
    fn block(a: Addrs, pc: u32, words: &[u32]) -> Vec<u8> {
        let mut e = Emitter { code: Vec::new(), a, loaded: 0, dirty: 0 };
        let n = words.len() as u32;
        let last = pc + 4 * (n - 1);
        let (exit, fall) = match words.iter().position(|&i| kind(i) == Kind::Branch) {
            Some(k) => (branch_exit(words[k], pc + 4 * k as u32), (last + 4) & 0xFFC),
            None => (Exit::To((last + 4) & 0xFFC), 0),
        };
        let looping = match exit {
            Exit::To(t) | Exit::If(t) => t == pc,
            Exit::Register => false,
        };
        if looping {
            e.code.extend_from_slice(&[LOOP, 0x40]);
        }
        for (k, &i) in words.iter().enumerate() {
            if kind(i) == Kind::Branch {
                e.branch(i, pc + 4 * k as u32);
            } else {
                e.instr(i);
            }
        }
        e.flush();
        if !looping {
            e.exit(exit, fall);
            e.konst(n);
            return e.code;
        }
        // Another pass needs the branch taken and `n` more steps of budget.
        e.local(LOCAL_GET, STEPS);
        e.konst(n);
        e.op(I32_ADD);
        e.local(LOCAL_TEE, STEPS);
        e.konst(n);
        e.op(I32_ADD);
        e.local(LOCAL_GET, BUDGET);
        e.op(I32_LE_U);
        if matches!(exit, Exit::If(_)) {
            e.local(LOCAL_GET, COND);
            e.op(I32_AND);
        }
        e.code.extend_from_slice(&[BR_IF, 0, END]);
        // Locals loaded in the last pass were loaded in code that may not run.
        e.loaded = 0;
        e.exit(exit, fall);
        e.local(LOCAL_GET, STEPS);
        e.code
    }

    fn op(&mut self, op: u8) {
        self.code.push(op);
    }

    fn local(&mut self, op: u8, l: u32) {
        self.code.push(op);
        uleb(&mut self.code, l);
    }

    fn konst(&mut self, v: u32) {
        self.code.push(I32_CONST);
        sleb(&mut self.code, v as i32);
    }

    fn mem(&mut self, op: u8, align: u32, offset: u32) {
        self.code.push(op);
        uleb(&mut self.code, align);
        uleb(&mut self.code, offset);
    }

    /// Loads the i32 at a fixed address.
    fn load(&mut self, addr: u32) {
        self.konst(0);
        self.mem(I32_LOAD, 2, addr);
    }

    /// Stores the i32 constant `v` at a fixed address.
    fn store_const(&mut self, addr: u32, v: u32) {
        self.konst(0);
        self.konst(v);
        self.mem(I32_STORE, 2, addr);
    }

    /// Pushes register `r`, loading it into its local on first use.
    fn get(&mut self, r: u32) {
        if r == 0 {
            self.konst(0);
        } else if self.loaded & (1 << r) != 0 {
            self.local(LOCAL_GET, r);
        } else {
            self.load(self.a.gpr(r));
            self.local(LOCAL_TEE, r);
            self.loaded |= 1 << r;
        }
    }

    /// Pops into register `r`; writes to r0 are dropped.
    fn set(&mut self, r: u32) {
        if r == 0 {
            self.op(DROP);
            return;
        }
        self.local(LOCAL_SET, r);
        self.loaded |= 1 << r;
        self.dirty |= 1 << r;
    }

    fn flush_reg(&mut self, r: u32) {
        if self.dirty & (1 << r) == 0 {
            return;
        }
        self.konst(0);
        self.local(LOCAL_GET, r);
        self.mem(I32_STORE, 2, self.a.gpr(r));
        self.dirty &= !(1 << r);
    }

    fn flush(&mut self) {
        for r in 1..32 {
            self.flush_reg(r);
        }
    }

    fn binary(&mut self, x: u32, y: u32, op: u8, d: u32) {
        self.get(x);
        self.get(y);
        self.op(op);
        self.set(d);
    }

    fn with_imm(&mut self, x: u32, imm: u32, op: u8, d: u32) {
        self.get(x);
        self.konst(imm);
        self.op(op);
        self.set(d);
    }

    /// Runs `i` in the interpreter, which reads and writes registers in memory.
    fn interpret(&mut self, i: u32) {
        let rt = (i >> 16) & 31;
        let cop2_move = i >> 26 == 0x12 && i & (1 << 25) == 0;
        match i >> 26 {
            0x32 | 0x3A => self.flush_reg((i >> 21) & 31),
            _ if cop2_move && matches!((i >> 21) & 31, 4 | 6) => self.flush_reg(rt),
            _ => {}
        }
        self.konst(i);
        self.code.push(CALL);
        uleb(&mut self.code, 0);
        // MFC2 and CFC2 write rt.
        if !(cop2_move && matches!((i >> 21) & 31, 0 | 2)) {
            return;
        }
        self.loaded &= !(1 << rt);
        self.dirty &= !(1 << rt);
        if rt == 0 {
            // The interpreter clears r0 after every step.
            self.store_const(self.a.gpr(0), 0);
        }
    }

    /// Pushes the DMEM byte at ADDR + `k`.
    fn dmem_byte(&mut self, k: u32) {
        self.local(LOCAL_GET, ADDR);
        if k != 0 {
            self.konst(k);
            self.op(I32_ADD);
        }
        self.konst(0xFFF);
        self.op(I32_AND);
        self.mem(I32_LOAD8_U, 0, self.a.dmem);
    }

    /// Pushes the big-endian value of DMEM bytes ADDR + `first` onwards.
    fn dmem_value(&mut self, first: u32, bytes: u32) {
        for k in 0..bytes {
            self.dmem_byte(first + k);
            let shift = 8 * (bytes - 1 - k);
            if shift != 0 {
                self.konst(shift);
                self.op(I32_SHL);
            }
            if k != 0 {
                self.op(I32_OR);
            }
        }
    }

    /// Stores the low byte of what `value` pushes at DMEM ADDR + `k`.
    fn dmem_store_byte(&mut self, k: u32, value: impl FnOnce(&mut Emitter)) {
        self.local(LOCAL_GET, ADDR);
        if k != 0 {
            self.konst(k);
            self.op(I32_ADD);
        }
        self.konst(0xFFF);
        self.op(I32_AND);
        value(self);
        self.mem(I32_STORE8, 0, self.a.dmem);
    }

    /// Stores `bytes` bytes of register `t` big-endian at ADDR.
    fn dmem_store(&mut self, t: u32, bytes: u32) {
        for k in 0..bytes {
            let shift = 8 * (bytes - 1 - k);
            self.dmem_store_byte(k, |e| {
                e.get(t);
                if shift != 0 {
                    e.konst(shift);
                    e.op(I32_SHR_U);
                }
            });
        }
    }

    fn address(&mut self, s: u32, imm: u32) {
        self.get(s);
        self.konst(imm);
        self.op(I32_ADD);
        self.local(LOCAL_SET, ADDR);
    }

    /// Emits a non-branch instruction.
    fn instr(&mut self, i: u32) {
        let s = (i >> 21) & 31;
        let t = (i >> 16) & 31;
        let d = (i >> 11) & 31;
        let sa = (i >> 6) & 31;
        let imm = i as u16 as i16 as i32 as u32;
        let uimm = i & 0xFFFF;
        match i >> 26 {
            0x00 => match i & 0x3F {
                0x00 => self.with_imm(t, sa, I32_SHL, d),
                0x02 => self.with_imm(t, sa, I32_SHR_U, d),
                0x03 => self.with_imm(t, sa, I32_SHR_S, d),
                // wasm masks shift counts to 5 bits, as `rs & 31` does.
                0x04 => self.binary(t, s, I32_SHL, d),
                0x06 => self.binary(t, s, I32_SHR_U, d),
                0x07 => self.binary(t, s, I32_SHR_S, d),
                0x20 | 0x21 => self.binary(s, t, I32_ADD, d),
                0x22 | 0x23 => self.binary(s, t, I32_SUB, d),
                0x24 => self.binary(s, t, I32_AND, d),
                0x25 => self.binary(s, t, I32_OR, d),
                0x26 => self.binary(s, t, I32_XOR, d),
                0x27 => {
                    self.get(s);
                    self.get(t);
                    self.op(I32_OR);
                    self.konst(u32::MAX);
                    self.op(I32_XOR);
                    self.set(d);
                }
                0x2A => self.binary(s, t, I32_LT_S, d),
                0x2B => self.binary(s, t, I32_LT_U, d),
                _ => {}
            },
            0x08 | 0x09 => self.with_imm(s, imm, I32_ADD, t),
            0x0A => self.with_imm(s, imm, I32_LT_S, t),
            0x0B => self.with_imm(s, imm, I32_LT_U, t),
            0x0C => self.with_imm(s, uimm, I32_AND, t),
            0x0D => self.with_imm(s, uimm, I32_OR, t),
            0x0E => self.with_imm(s, uimm, I32_XOR, t),
            0x0F => {
                self.konst(i << 16);
                self.set(t);
            }
            0x10 if s == 0 => self.mfc0(t, d),
            0x12 => {
                if i & (1 << 25) == 0 || !self.vector(i) {
                    self.interpret(i);
                }
            }
            0x20 | 0x24 => {
                self.address(s, imm);
                self.local(LOCAL_GET, ADDR);
                self.konst(0xFFF);
                self.op(I32_AND);
                self.mem(if i >> 26 == 0x20 { I32_LOAD8_S } else { I32_LOAD8_U }, 0, self.a.dmem);
                self.set(t);
            }
            0x21 | 0x25 => {
                self.address(s, imm);
                self.dmem_value(0, 2);
                if i >> 26 == 0x21 {
                    self.op(I32_EXTEND16_S);
                }
                self.set(t);
            }
            0x23 | 0x27 => {
                self.address(s, imm);
                self.dmem_value(0, 4);
                self.set(t);
            }
            0x28 => {
                self.address(s, imm);
                self.dmem_store(t, 1);
            }
            0x29 => {
                self.address(s, imm);
                self.dmem_store(t, 2);
            }
            0x2B => {
                self.address(s, imm);
                self.dmem_store(t, 4);
            }
            0x32 => self.lwc2(i),
            0x3A => self.swc2(i),
            _ => {}
        }
    }

    /// MFC0: SP registers 0-7 and DP registers 8-15, as `sp_reg_read` and
    /// `dp_reg_read` return them.
    fn mfc0(&mut self, t: u32, d: u32) {
        let a = self.a;
        let rsp = |f: usize| Some(a.rsp(f));
        let rdp = |f: usize| Some(a.rdp + f as u32);
        let field = match d {
            0 => rsp(offset_of!(Rsp, mem_addr)),
            1 => rsp(offset_of!(Rsp, dram_addr)),
            2 => rsp(offset_of!(Rsp, rd_len)),
            3 => rsp(offset_of!(Rsp, wr_len)),
            4 => rsp(offset_of!(Rsp, status)),
            7 => rsp(offset_of!(Rsp, semaphore)),
            8 => rdp(offset_of!(Rdp, start)),
            9 => rdp(offset_of!(Rdp, end)),
            10 => rdp(offset_of!(Rdp, current)),
            11 => rdp(offset_of!(Rdp, status)),
            _ => None,
        };
        let Some(addr) = field else {
            self.konst(0);
            self.set(t);
            return;
        };
        self.load(addr);
        if d == 11 {
            self.konst(DP_CBUF_READY);
            self.op(I32_OR);
        }
        self.set(t);
        if d == 7 {
            // Reading the semaphore takes it.
            self.store_const(addr, 1);
        }
    }

    /// Emits the part of a branch that runs before its delay slot: the
    /// condition, the register target and the link, all from values read
    /// before the delay slot can change them.
    fn branch(&mut self, i: u32, pc: u32) {
        let s = (i >> 21) & 31;
        let t = (i >> 16) & 31;
        let link = (pc + 8) & 0xFFC;
        let (op, y) = match i >> 26 {
            0x00 => {
                self.get(s);
                self.konst(0xFFC);
                self.op(I32_AND);
                self.local(LOCAL_SET, TARGET);
                if i & 0x3F == 0x09 {
                    self.konst(link);
                    self.set((i >> 11) & 31);
                }
                return;
            }
            0x02 => return,
            0x03 => {
                self.konst(link);
                self.set(31);
                return;
            }
            0x01 => (if t & 1 == 0 { I32_LT_S } else { I32_GE_S }, None),
            0x04 => (I32_EQ, Some(t)),
            0x05 => (I32_NE, Some(t)),
            0x06 => (I32_LE_S, None),
            _ => (I32_GT_S, None),
        };
        self.get(s);
        match y {
            Some(y) => self.get(y),
            None => self.konst(0),
        }
        self.op(op);
        self.local(LOCAL_SET, COND);
        if i >> 26 == 0x01 && t & 0x10 != 0 {
            self.konst(link);
            self.set(31);
        }
    }

    /// Writes PC and next PC for the exit; `fall` is the not-taken address.
    fn exit(&mut self, exit: Exit, fall: u32) {
        match exit {
            Exit::To(pc) => {
                self.konst(pc);
                self.local(LOCAL_SET, TARGET);
            }
            Exit::Register => {}
            Exit::If(target) => {
                self.konst(target);
                self.konst(fall);
                self.local(LOCAL_GET, COND);
                self.op(SELECT);
                self.local(LOCAL_SET, TARGET);
            }
        }
        self.konst(0);
        self.local(LOCAL_GET, TARGET);
        self.mem(I32_STORE, 2, self.a.rsp(offset_of!(Rsp, pc)));
        self.konst(0);
        self.local(LOCAL_GET, TARGET);
        self.konst(4);
        self.op(I32_ADD);
        self.konst(0xFFC);
        self.op(I32_AND);
        self.mem(I32_STORE, 2, self.a.rsp(offset_of!(Rsp, next_pc)));
    }

    // ------------------------------------------------------------ SIMD

    fn simd(&mut self, op: u32) {
        self.code.push(0xFD);
        uleb(&mut self.code, op);
    }

    fn vload(&mut self, addr: u32) {
        self.konst(0);
        self.simd(V128_LOAD);
        uleb(&mut self.code, 0);
        uleb(&mut self.code, addr);
    }

    /// Stores what `value` pushes at a fixed address.
    fn vstore(&mut self, addr: u32, value: impl FnOnce(&mut Emitter)) {
        self.konst(0);
        value(self);
        self.simd(V128_STORE);
        uleb(&mut self.code, 0);
        uleb(&mut self.code, addr);
    }

    fn vconst(&mut self, bytes: [u8; 16]) {
        self.simd(V128_CONST);
        self.code.extend_from_slice(&bytes);
    }

    fn vconst16(&mut self, lanes: [u16; 8]) {
        let mut b = [0u8; 16];
        for (n, l) in lanes.iter().enumerate() {
            b[2 * n..2 * n + 2].copy_from_slice(&l.to_le_bytes());
        }
        self.vconst(b);
    }

    fn vconst64(&mut self, lane: u64) {
        let mut b = [0u8; 16];
        b[..8].copy_from_slice(&lane.to_le_bytes());
        b[8..].copy_from_slice(&lane.to_le_bytes());
        self.vconst(b);
    }

    fn shuffle(&mut self, lanes: [u8; 16]) {
        self.simd(I8X16_SHUFFLE);
        self.code.extend_from_slice(&lanes);
    }

    /// Pushes local `x` with its bytes permuted by `lanes`.
    fn permute(&mut self, x: u32, lanes: [u8; 16]) {
        self.local(LOCAL_GET, x);
        self.local(LOCAL_GET, x);
        self.shuffle(lanes);
    }

    /// Pushes a lane mask with lane n set when bit `shift + n` of the u16 at
    /// `addr` is set.
    fn flag_lanes(&mut self, addr: u32, shift: u32) {
        self.konst(0);
        self.mem(I32_LOAD16_U, 1, addr);
        if shift != 0 {
            self.konst(shift);
            self.op(I32_SHR_U);
        }
        self.simd(I16X8_SPLAT);
        self.vconst16(LANE_BITS);
        self.simd(V128_AND);
        self.vconst16(LANE_BITS);
        self.simd(I16X8_EQ);
    }

    /// Stores the lane sign bits of what `mask` pushes into the u16 at `addr`.
    fn store_bitmask(&mut self, addr: u32, mask: impl FnOnce(&mut Emitter)) {
        self.konst(0);
        mask(self);
        self.simd(I16X8_BITMASK);
        self.mem(I32_STORE16, 1, addr);
    }

    fn clear_flags(&mut self, field: usize) {
        self.konst(0);
        self.konst(0);
        self.mem(I32_STORE16, 1, self.a.rsp(field));
    }

    /// Loads the operands of vector instruction `i` into VA and VB, with the
    /// element selector applied to VB.
    fn operands(&mut self, i: u32) {
        let e = ((i >> 21) & 15) as usize;
        self.vload(self.a.vr((i >> 11) & 31));
        self.local(LOCAL_SET, VA);
        self.vload(self.a.vr((i >> 16) & 31));
        self.local(LOCAL_SET, VB);
        if e >= 2 {
            let mut lanes = [0u8; 16];
            for n in 0..8 {
                let l = ELEMENTS[e][n] as u8;
                lanes[2 * n] = 2 * l;
                lanes[2 * n + 1] = 2 * l + 1;
            }
            self.permute(VB, lanes);
            self.local(LOCAL_SET, VB);
        }
    }

    /// Replaces the low 16 bits of each accumulator lane with local `x`.
    fn set_acc_low(&mut self, x: u32) {
        for k in 0..4 {
            self.vstore(self.a.acc(k), |e| {
                e.vload(e.a.acc(k));
                e.vconst64(!0xFFFF);
                e.simd(V128_AND);
                e.local(LOCAL_GET, x);
                e.simd(if k < 2 { I32X4_EXTEND_LOW_I16X8_U } else { I32X4_EXTEND_HIGH_I16X8_U });
                e.simd(if k % 2 == 0 { I64X2_EXTEND_LOW_I32X4_U } else { I64X2_EXTEND_HIGH_I32X4_U });
                e.simd(V128_OR);
            });
        }
    }

    /// Writes local `x` to vd and to the low accumulator slice.
    fn result_and_acc(&mut self, x: u32, vd: u32) {
        self.set_acc_low(x);
        self.vstore(self.a.vr(vd), |e| e.local(LOCAL_GET, x));
    }

    /// Pushes the low 16 bits of the 64-bit lanes in locals `x..x + 4`.
    fn low16_of_i64(&mut self, x: u32) {
        for h in 0..2 {
            self.local(LOCAL_GET, x + 2 * h);
            self.local(LOCAL_GET, x + 2 * h + 1);
            self.shuffle(LOW32_OF_I64);
        }
        self.shuffle(LOW16_OF_I32);
    }

    /// Translates a vector computational instruction. Returns false for the
    /// ones left to the interpreter.
    fn vector(&mut self, i: u32) -> bool {
        let funct = i & 0x3F;
        let vd = (i >> 6) & 31;
        let vco = offset_of!(Rsp, vco);
        let vcc = offset_of!(Rsp, vcc);
        match funct {
            0x00 | 0x04..=0x08 | 0x0C..=0x0F => {
                self.operands(i);
                self.multiply(funct, vd);
            }
            0x10 | 0x11 => {
                self.operands(i);
                let (add32, add16) = if funct == 0x10 { (I32X4_ADD, I16X8_ADD) } else { (I32X4_SUB, I16X8_SUB) };
                // Carry in as 0 or 1 per lane.
                self.flag_lanes(self.a.rsp(vco), 0);
                self.vconst16([1; 8]);
                self.simd(V128_AND);
                self.local(LOCAL_SET, T);
                for (h, (ext_s, ext_u)) in [(I32X4_EXTEND_LOW_I16X8_S, I32X4_EXTEND_LOW_I16X8_U), (I32X4_EXTEND_HIGH_I16X8_S, I32X4_EXTEND_HIGH_I16X8_U)].into_iter().enumerate() {
                    self.local(LOCAL_GET, VA);
                    self.simd(ext_s);
                    self.local(LOCAL_GET, VB);
                    self.simd(ext_s);
                    self.simd(add32);
                    self.local(LOCAL_GET, T);
                    self.simd(ext_u);
                    self.simd(add32);
                    self.local(LOCAL_SET, T + 1 + h as u32);
                }
                self.local(LOCAL_GET, VA);
                self.local(LOCAL_GET, VB);
                self.simd(add16);
                self.local(LOCAL_GET, T);
                self.simd(add16);
                self.local(LOCAL_SET, T + 3);
                self.set_acc_low(T + 3);
                self.vstore(self.a.vr(vd), |e| {
                    e.local(LOCAL_GET, T + 1);
                    e.local(LOCAL_GET, T + 2);
                    e.simd(I16X8_NARROW_I32X4_S);
                });
                self.clear_flags(vco);
            }
            0x14 | 0x15 => {
                self.operands(i);
                self.local(LOCAL_GET, VA);
                self.local(LOCAL_GET, VB);
                self.simd(if funct == 0x14 { I16X8_ADD } else { I16X8_SUB });
                self.local(LOCAL_SET, T);
                self.result_and_acc(T, vd);
                let vco = self.a.rsp(vco);
                if funct == 0x14 {
                    // A lane carries out when its 16-bit sum wrapped below VA.
                    self.store_bitmask(vco, |e| {
                        e.local(LOCAL_GET, T);
                        e.local(LOCAL_GET, VA);
                        e.simd(I16X8_LT_U);
                    });
                } else {
                    // Borrow in the low byte, not-equal in the high byte.
                    self.konst(0);
                    self.local(LOCAL_GET, VA);
                    self.local(LOCAL_GET, VB);
                    self.simd(I16X8_LT_U);
                    self.simd(I16X8_BITMASK);
                    self.local(LOCAL_GET, VA);
                    self.local(LOCAL_GET, VB);
                    self.simd(I16X8_NE);
                    self.simd(I16X8_BITMASK);
                    self.konst(8);
                    self.op(I32_SHL);
                    self.op(I32_OR);
                    self.mem(I32_STORE16, 1, vco);
                }
            }
            0x1D => {
                let shift = match (i >> 21) & 15 {
                    8 => 32,
                    9 => 16,
                    10 => 0,
                    _ => {
                        self.vstore(self.a.vr(vd), |e| e.vconst([0; 16]));
                        return true;
                    }
                };
                for k in 0..4 {
                    self.vload(self.a.acc(k));
                    if shift != 0 {
                        self.konst(shift);
                        self.simd(I64X2_SHR_U);
                    }
                    self.local(LOCAL_SET, T + k);
                }
                self.vstore(self.a.vr(vd), |e| e.low16_of_i64(T));
            }
            0x20..=0x23 => {
                self.operands(i);
                self.flag_lanes(self.a.rsp(vco), 0);
                self.flag_lanes(self.a.rsp(vco), 8);
                // T: carry and not-equal; T + 1: not-equal; T + 2: equal lanes.
                self.local(LOCAL_TEE, T + 1);
                self.simd(V128_AND);
                self.local(LOCAL_SET, T);
                self.local(LOCAL_GET, VA);
                self.local(LOCAL_GET, VB);
                self.simd(I16X8_EQ);
                self.local(LOCAL_SET, T + 2);
                match funct {
                    0x20 | 0x23 => {
                        self.local(LOCAL_GET, VA);
                        self.local(LOCAL_GET, VB);
                        self.simd(if funct == 0x20 { I16X8_LT_S } else { I16X8_GT_S });
                        self.local(LOCAL_GET, T + 2);
                        self.local(LOCAL_GET, T);
                        self.simd(if funct == 0x20 { V128_AND } else { V128_ANDNOT });
                        self.simd(V128_OR);
                    }
                    0x21 => {
                        self.local(LOCAL_GET, T + 2);
                        self.local(LOCAL_GET, T + 1);
                        self.simd(V128_ANDNOT);
                    }
                    _ => {
                        self.local(LOCAL_GET, T + 2);
                        self.simd(V128_NOT);
                        self.local(LOCAL_GET, T + 1);
                        self.simd(V128_OR);
                    }
                }
                self.local(LOCAL_SET, T + 3);
                self.store_bitmask(self.a.rsp(vcc), |e| e.local(LOCAL_GET, T + 3));
                self.select(T + 3, vd);
            }
            0x27 => {
                self.operands(i);
                self.flag_lanes(self.a.rsp(vcc), 0);
                self.local(LOCAL_SET, T + 3);
                self.select(T + 3, vd);
            }
            0x28..=0x2D => {
                self.operands(i);
                self.local(LOCAL_GET, VA);
                self.local(LOCAL_GET, VB);
                self.simd(match funct & !1 {
                    0x28 => V128_AND,
                    0x2A => V128_OR,
                    _ => V128_XOR,
                });
                if funct & 1 != 0 {
                    self.simd(V128_NOT);
                }
                self.local(LOCAL_SET, T);
                self.result_and_acc(T, vd);
            }
            _ => return false,
        }
        true
    }

    /// Picks VA where mask local `m` is set and VB elsewhere, writes it to vd
    /// and the low accumulator slice, and clears VCO.
    fn select(&mut self, m: u32, vd: u32) {
        self.local(LOCAL_GET, VA);
        self.local(LOCAL_GET, VB);
        self.local(LOCAL_GET, m);
        self.simd(V128_BITSELECT);
        self.local(LOCAL_SET, T);
        self.result_and_acc(T, vd);
        self.clear_flags(offset_of!(Rsp, vco));
    }

    /// The multiply group, following `multiply` in rsp.rs: a 32-bit lane
    /// product, widened to 64 bits, shifted, added to the sign-extended
    /// accumulator for the accumulating forms, and kept to 48 bits.
    fn multiply(&mut self, funct: u32, vd: u32) {
        let unsigned = funct & 7 == 4;
        for (h, ext) in [(0, I32X4_EXTEND_LOW_I16X8_S), (1, I32X4_EXTEND_HIGH_I16X8_S)] {
            self.local(LOCAL_GET, VA);
            match funct & 7 {
                0 | 7 => {
                    self.local(LOCAL_GET, VB);
                    self.simd(if h == 0 { I32X4_EXTMUL_LOW_I16X8_S } else { I32X4_EXTMUL_HIGH_I16X8_S });
                }
                4 => {
                    self.local(LOCAL_GET, VB);
                    self.simd(if h == 0 { I32X4_EXTMUL_LOW_I16X8_U } else { I32X4_EXTMUL_HIGH_I16X8_U });
                }
                // Signed by unsigned products fit in 32 bits.
                5 | 6 => {
                    let ext_u = ext + 2;
                    self.simd(if funct & 7 == 5 { ext } else { ext_u });
                    self.local(LOCAL_GET, VB);
                    self.simd(if funct & 7 == 5 { ext_u } else { ext });
                    self.simd(I32X4_MUL);
                }
                _ => unreachable!(),
            }
            self.local(LOCAL_SET, T + h);
        }
        for k in 0..4 {
            self.vstore(self.a.acc(k), |e| {
                e.local(LOCAL_GET, T + k / 2);
                e.simd(match (k % 2 == 0, unsigned) {
                    (true, false) => I64X2_EXTEND_LOW_I32X4_S,
                    (false, false) => I64X2_EXTEND_HIGH_I32X4_S,
                    (true, true) => I64X2_EXTEND_LOW_I32X4_U,
                    (false, true) => I64X2_EXTEND_HIGH_I32X4_U,
                });
                match funct & 7 {
                    0 => {
                        e.konst(1);
                        e.simd(I64X2_SHL);
                        if funct == 0x00 {
                            e.vconst64(0x8000);
                            e.simd(I64X2_ADD);
                        }
                    }
                    4 => {
                        e.konst(16);
                        e.simd(I64X2_SHR_U);
                    }
                    7 => {
                        e.konst(16);
                        e.simd(I64X2_SHL);
                    }
                    _ => {}
                }
                if funct & 8 != 0 {
                    e.vload(e.a.acc(k));
                    e.sext48();
                    e.simd(I64X2_ADD);
                }
                e.sext48();
                e.local(LOCAL_TEE, W + k);
                e.vconst64(ACC_MASK);
                e.simd(V128_AND);
            });
        }
        self.vstore(self.a.vr(vd), |e| match funct {
            0x04 => e.low16_of_i64(W),
            0x06 | 0x0C | 0x0E => e.sat_low(),
            _ => e.sat_mid(),
        });
    }

    fn sext48(&mut self) {
        self.konst(16);
        self.simd(I64X2_SHL);
        self.konst(16);
        self.simd(I64X2_SHR_S);
    }

    /// Leaves the middle accumulator slice, acc >> 16 as 32 bits, in T + 2
    /// (lanes 0-3) and T + 3 (lanes 4-7), and pushes it clamped to 16 bits.
    fn sat_mid(&mut self) {
        for h in 0..2 {
            for k in 0..2 {
                self.local(LOCAL_GET, W + 2 * h + k);
                self.konst(16);
                self.simd(I64X2_SHR_S);
            }
            self.shuffle(LOW32_OF_I64);
            self.local(LOCAL_SET, T + 2 + h);
        }
        self.local(LOCAL_GET, T + 2);
        self.local(LOCAL_GET, T + 3);
        self.simd(I16X8_NARROW_I32X4_S);
    }

    /// Pushes the low accumulator slice, or 0 / 0xFFFF where the accumulator
    /// is below / above the signed 32-bit range.
    fn sat_low(&mut self) {
        self.sat_mid();
        self.local(LOCAL_SET, T + 4);
        self.low16_of_i64(W);
        self.local(LOCAL_GET, T + 4);
        self.konst(15);
        self.simd(I16X8_SHR_S);
        self.simd(V128_NOT);
        // In range exactly where the middle slice survives clamping.
        for (h, ext) in [(0, I32X4_EXTEND_LOW_I16X8_S), (1, I32X4_EXTEND_HIGH_I16X8_S)] {
            self.local(LOCAL_GET, T + 2 + h);
            self.local(LOCAL_GET, T + 4);
            self.simd(ext);
            self.simd(I32X4_EQ);
        }
        self.simd(I16X8_NARROW_I32X4_S);
        self.simd(V128_BITSELECT);
    }

    // ------------------------------------------------------------ vector memory

    /// Vector loads: the common aligned forms inline, the rest interpreted.
    fn lwc2(&mut self, i: u32) {
        let (vt, op, e) = ((i >> 16) & 31, (i >> 11) & 31, (i >> 7) & 15);
        let bytes = match op {
            1 => 2,
            2 => 4,
            3 => 8,
            4 => 16,
            _ => return self.interpret(i),
        };
        // Below 16 bytes, an even element keeps every byte inside the register.
        if (op < 4 && (e & 1 != 0 || e + bytes > 16)) || (op == 4 && e != 0) {
            return self.interpret(i);
        }
        self.vector_address(i, bytes);
        if op < 4 {
            for p in 0..bytes / 2 {
                self.konst(0);
                self.dmem_value(2 * p, 2);
                self.mem(I32_STORE16, 1, self.a.vr(vt) + e + 2 * p);
            }
            return;
        }
        // LQV loads all 16 bytes only from an aligned address.
        self.aligned_or_interpret(i, |e| {
            e.vstore(e.a.vr(vt), |e| {
                e.local(LOCAL_GET, ADDR);
                e.konst(0xFFF);
                e.op(I32_AND);
                e.simd(V128_LOAD);
                uleb(&mut e.code, 0);
                uleb(&mut e.code, e.a.dmem);
                e.local(LOCAL_TEE, T);
                e.local(LOCAL_GET, T);
                e.shuffle(SWAP16);
            });
        });
    }

    /// Vector stores: the common aligned forms inline, the rest interpreted.
    fn swc2(&mut self, i: u32) {
        let (vt, op, e) = ((i >> 16) & 31, (i >> 11) & 31, (i >> 7) & 15);
        let bytes = match op {
            1 => 2,
            2 => 4,
            3 => 8,
            4 => 16,
            _ => return self.interpret(i),
        };
        if (op < 4 && (e & 1 != 0 || e + bytes > 16)) || (op == 4 && e != 0) {
            return self.interpret(i);
        }
        self.vector_address(i, bytes);
        if op < 4 {
            for k in 0..bytes {
                let lane = self.a.vr(vt) + e + (k & !1);
                self.dmem_store_byte(k, |em| {
                    em.konst(0);
                    em.mem(I32_LOAD16_U, 1, lane);
                    if k & 1 == 0 {
                        em.konst(8);
                        em.op(I32_SHR_U);
                    }
                });
            }
            return;
        }
        self.aligned_or_interpret(i, |e| {
            e.local(LOCAL_GET, ADDR);
            e.konst(0xFFF);
            e.op(I32_AND);
            e.vload(e.a.vr(vt));
            e.local(LOCAL_TEE, T);
            e.local(LOCAL_GET, T);
            e.shuffle(SWAP16);
            e.simd(V128_STORE);
            uleb(&mut e.code, 0);
            uleb(&mut e.code, e.a.dmem);
        });
    }

    /// ADDR = base + the signed 7-bit offset scaled by the access size.
    fn vector_address(&mut self, i: u32, bytes: u32) {
        let off = ((i & 0x7F) as i32) << 25 >> 25;
        self.address((i >> 21) & 31, (off * bytes as i32) as u32);
    }

    /// Runs `fast` when ADDR is 16-byte aligned and the interpreter otherwise.
    fn aligned_or_interpret(&mut self, i: u32, fast: impl FnOnce(&mut Emitter)) {
        self.flush_reg((i >> 21) & 31);
        self.local(LOCAL_GET, ADDR);
        self.konst(15);
        self.op(I32_AND);
        self.op(I32_EQZ);
        self.code.extend_from_slice(&[IF, 0x40]);
        fast(self);
        self.op(ELSE);
        self.interpret(i);
        self.op(END);
    }
}
