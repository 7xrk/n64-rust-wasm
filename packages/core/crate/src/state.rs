//! The complete machine state. Everything in `State` is plain data so a
//! savestate is a byte copy of the struct.

use crate::cpu::Cpu;
use crate::rdp::Rdp;
use crate::rsp::Rsp;

pub const RDRAM_SIZE: usize = 0x80_0000;
pub const PAGE_SHIFT: u32 = 12;
pub const PAGE_COUNT: usize = RDRAM_SIZE >> PAGE_SHIFT;

pub const STATE_MAGIC: u32 = 0x3436_5354; // "TS64"
pub const STATE_VERSION: u32 = 1;

// Scheduler event slots.
pub const EV_VI_INTR: usize = 0;
pub const EV_FIELD: usize = 1;
pub const EV_COMPARE: usize = 2;
pub const EV_AI: usize = 3;
pub const EV_PI: usize = 4;
pub const EV_SI: usize = 5;
pub const EV_SP: usize = 6;
pub const EV_DP: usize = 7;
pub const EV_COUNT: usize = 8;

// MI interrupt bits.
pub const MI_SP: u32 = 1 << 0;
pub const MI_SI: u32 = 1 << 1;
pub const MI_AI: u32 = 1 << 2;
pub const MI_VI: u32 = 1 << 3;
pub const MI_PI: u32 = 1 << 4;
pub const MI_DP: u32 = 1 << 5;

#[repr(C)]
pub struct Mi {
    pub mode: u32,
    pub intr: u32,
    pub mask: u32,
}

#[repr(C)]
pub struct Vi {
    pub regs: [u32; 14],
    pub field: u32,
    pub field_start: u64,
    pub line_cycles: u64,
    pub lines: u64,
}

#[repr(C)]
pub struct Ai {
    pub dram_addr: u32,
    pub len: u32,
    pub control: u32,
    pub dacrate: u32,
    pub bitrate: u32,
    pub fifo_addr: [u32; 2],
    pub fifo_len: [u32; 2],
    pub fifo_count: u32,
}

#[repr(C)]
pub struct Pi {
    pub regs: [u32; 13],
}

#[repr(C)]
pub struct Si {
    pub dram_addr: u32,
    pub status: u32,
}

#[repr(C)]
pub struct Pif {
    pub ram: [u8; 64],
}

#[repr(C)]
pub struct Cart {
    pub eeprom: [u8; 2048],
    pub sram: [u8; 0x8000],
}

#[repr(C)]
pub struct Input {
    /// Joybus pad response per port: byte0 | byte1 << 8 | x << 16 | y << 24.
    pub pads: [u32; 4],
    pub connected: u32,
    /// Set when the game reads port 0 during the current frame.
    pub polled: u32,
}

#[repr(C)]
pub struct State {
    pub magic: u32,
    pub version: u32,
    pub cycles: u64,
    pub events: [u64; EV_COUNT],
    pub frame: u64,
    pub frame_done: u32,
    pub boot_done: u32,
    pub entry_point: u32,
    pub cpu: Cpu,
    pub rsp: Rsp,
    pub rdp: Rdp,
    pub mi: Mi,
    pub vi: Vi,
    pub ai: Ai,
    pub pi: Pi,
    pub si: Si,
    pub ri: [u32; 8],
    pub pif: Pif,
    pub cart: Cart,
    pub input: Input,
    /// SP DMEM (0x000-0xFFF) and IMEM (0x1000-0x1FFF) as big-endian bytes.
    pub sp_mem: [u8; 0x2000],
    /// RDRAM as 32-bit words in host order; byte `a` lives at `a ^ 3`.
    pub rdram: [u32; RDRAM_SIZE / 4],
}

impl State {
    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            core::slice::from_raw_parts(self as *const State as *const u8, core::mem::size_of::<State>())
        }
    }
}
