//! RSP recompiler. A straight run of IMEM up to and including the first
//! branch and its delay slot becomes one small wasm module. The host compiles
//! it and adds its function to the core's function table. Scalar, COP0 read
//! and the common vector instructions are translated; the rest call back into
//! the interpreter. After a block the machine is exactly what
//! the interpreter leaves after the same number of steps, so the JIT never
//! changes emulation results.

mod emit;

use crate::rsp::ST_SSTEP;
use crate::sys::System;
use emit::Addrs;

const IMEM_WORDS: usize = 1024;
/// Longest block, so a block seldom has more steps than a slice has left.
const MAX_BLOCK: usize = 64;
const UNKNOWN: u32 = 0;
const INTERPRET: u32 = 1;
/// Entries above this hold a block index plus `FIRST_BLOCK`.
const FIRST_BLOCK: u32 = 2;

struct Block {
    /// Slot in the core's function table.
    func: u32,
    words: Vec<u32>,
}

pub struct RspJit {
    /// Compiles a module and returns its function's table slot, or 0. Only the
    /// wasm build has a host that can compile.
    compiler: Option<fn(&[u8]) -> u32>,
    pub on: bool,
    /// What runs at each IMEM word: UNKNOWN, INTERPRET or a block.
    entry: [u32; IMEM_WORDS],
    /// IMEM byte offset just past the code each entry was built from.
    entry_end: [u32; IMEM_WORDS],
    blocks: Vec<Block>,
    /// Blocks ever compiled at each IMEM word, so microcode that comes back
    /// after an overlay is not compiled again.
    by_pc: Vec<Vec<u32>>,
}

#[cfg(target_arch = "wasm32")]
fn host_compile(module: &[u8]) -> u32 {
    #[link(wasm_import_module = "env")]
    extern "C" {
        fn rsp_jit_compile(ptr: *const u8, len: u32) -> u32;
    }
    unsafe { rsp_jit_compile(module.as_ptr(), module.len() as u32) }
}

/// Runs a block for at most `budget` steps and returns the steps it ran.
/// Kept out of line so the state the block writes is read back from memory.
#[inline(never)]
fn call(func: u32, budget: u32) -> u32 {
    let f: extern "C" fn(u32) -> u32 = unsafe { core::mem::transmute(func as usize) };
    f(budget)
}

impl RspJit {
    pub fn new() -> RspJit {
        RspJit {
            #[cfg(target_arch = "wasm32")]
            compiler: Some(host_compile),
            #[cfg(not(target_arch = "wasm32"))]
            compiler: None,
            on: true,
            entry: [UNKNOWN; IMEM_WORDS],
            entry_end: [0; IMEM_WORDS],
            blocks: Vec::new(),
            by_pc: vec![Vec::new(); IMEM_WORDS],
        }
    }

    /// Forgets the entries built from the `len` IMEM bytes at offset `at`,
    /// which wrap at the end of IMEM like DMA does.
    pub fn invalidate(&mut self, at: u32, len: u32) {
        let end = at + len;
        for w in 0..IMEM_WORDS {
            let start = w as u32 * 4;
            let hit = (start < end && self.entry_end[w] > at) || (end > 0x1000 && start < end - 0x1000);
            if hit {
                self.entry[w] = UNKNOWN;
            }
        }
    }

    pub fn invalidate_all(&mut self) {
        self.entry.fill(UNKNOWN);
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Plain,
    Branch,
    /// Can halt the RSP, move its PC or rewrite IMEM, so it is interpreted.
    Barrier,
}

fn kind(i: u32) -> Kind {
    match i >> 26 {
        0x00 => match i & 0x3F {
            0x08 | 0x09 => Kind::Branch,
            0x0D => Kind::Barrier,
            _ => Kind::Plain,
        },
        0x01..=0x07 => Kind::Branch,
        0x10 if (i >> 21) & 31 == 4 => Kind::Barrier,
        _ => Kind::Plain,
    }
}

impl System {
    /// Runs one block at the RSP PC if it fits in `budget` steps. Returns the
    /// steps run, or None when the next step must be interpreted.
    #[inline(always)]
    pub fn rsp_jit_run(&mut self, budget: u64) -> Option<u64> {
        if !self.rsp_jit.on || self.rsp_jit.compiler.is_none() || self.s.rsp.branch_pending != 0 || self.s.rsp.status & ST_SSTEP != 0 {
            return None;
        }
        let w = (self.s.rsp.pc >> 2) as usize;
        if self.rsp_jit.entry[w] == UNKNOWN {
            self.rsp_jit_build(w);
        }
        let e = self.rsp_jit.entry[w];
        if e == INTERPRET {
            return None;
        }
        let b = &self.rsp_jit.blocks[(e - FIRST_BLOCK) as usize];
        if b.words.len() as u64 > budget {
            return None;
        }
        Some(call(b.func, budget.min(u32::MAX as u64) as u32) as u64)
    }

    fn imem_word(&self, a: u32) -> u32 {
        let o = 0x1000 | a as usize;
        u32::from_be_bytes(self.s.sp_mem[o..o + 4].try_into().unwrap())
    }

    /// Instruction words of the block starting at `pc`; empty when the first
    /// step has to be interpreted.
    fn rsp_block_words(&self, pc: u32) -> Vec<u32> {
        let mut words = Vec::new();
        let mut a = pc;
        while words.len() < MAX_BLOCK && a < 0x1000 {
            let i = self.imem_word(a);
            match kind(i) {
                Kind::Barrier => break,
                Kind::Plain => {
                    words.push(i);
                    a += 4;
                }
                Kind::Branch => {
                    // The delay slot must be in IMEM and translatable too.
                    if a + 4 < 0x1000 && kind(self.imem_word(a + 4)) == Kind::Plain {
                        words.push(i);
                        words.push(self.imem_word(a + 4));
                    }
                    break;
                }
            }
        }
        words
    }

    #[inline(never)]
    fn rsp_jit_build(&mut self, w: usize) {
        let pc = (w as u32) << 2;
        let j = &self.rsp_jit;
        let reuse = j.by_pc[w].iter().copied().find(|&b| {
            let words = &j.blocks[b as usize].words;
            words.iter().enumerate().all(|(k, &i)| self.imem_word(pc + 4 * k as u32) == i)
        });
        if let Some(b) = reuse {
            self.rsp_jit.entry_end[w] = pc + 4 * self.rsp_jit.blocks[b as usize].words.len() as u32;
            self.rsp_jit.entry[w] = b + FIRST_BLOCK;
            return;
        }
        let words = self.rsp_block_words(pc);
        self.rsp_jit.entry_end[w] = pc + 4 * words.len().max(1) as u32;
        if words.is_empty() {
            self.rsp_jit.entry[w] = INTERPRET;
            return;
        }
        let compiler = self.rsp_jit.compiler.unwrap();
        let addrs = Addrs {
            rsp: &self.s.rsp as *const _ as usize as u32,
            rdp: &self.s.rdp as *const _ as usize as u32,
            dmem: self.s.sp_mem.as_ptr() as usize as u32,
        };
        let func = compiler(&emit::module(addrs, pc, &words));
        if func == 0 {
            self.rsp_jit.entry[w] = INTERPRET;
            return;
        }
        let b = self.rsp_jit.blocks.len() as u32;
        self.rsp_jit.blocks.push(Block { func, words });
        self.rsp_jit.by_pc[w].push(b);
        self.rsp_jit.entry[w] = b + FIRST_BLOCK;
    }
}
