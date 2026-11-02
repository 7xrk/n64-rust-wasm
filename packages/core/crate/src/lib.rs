//! N64 core exported to JavaScript as a flat C ABI. One machine instance
//! lives in linear memory; JS reads buffers through the returned pointers.

pub mod cpu;
pub mod pif;
pub mod rdp;
pub mod rsp;
pub mod rsp_jit;
pub mod state;
pub mod sys;

use state::*;
use sys::System;

static mut SYS: *mut System = core::ptr::null_mut();

fn sys() -> &'static mut System {
    unsafe {
        if SYS.is_null() {
            SYS = Box::into_raw(System::new());
        }
        &mut *SYS
    }
}

impl System {
    pub fn reset(&mut self) {
        let pads = self.s.input.pads;
        let connected = self.s.input.connected;
        let eeprom = self.s.cart.eeprom;
        let sram = self.s.cart.sram;
        unsafe {
            core::ptr::write_bytes(&mut *self.s as *mut State, 0, 1);
        }
        self.s.magic = STATE_MAGIC;
        self.s.version = STATE_VERSION;
        self.s.input.pads = pads;
        self.s.input.connected = connected;
        self.s.cart.eeprom = eeprom;
        self.s.cart.sram = sram;
        self.s.events = [u64::MAX; EV_COUNT];
        self.hle_boot();
        self.vi_reset();
        self.reschedule_compare();
        self.after_state_change();
    }

    /// Powers on with erased cartridge saves, so a run depends only on its inputs.
    pub fn power_on(&mut self) {
        self.s.cart.eeprom.fill(0xFF);
        self.s.cart.sram.fill(0);
        self.reset();
    }

    /// Rebuilds everything derived from `State` that is kept outside it.
    pub fn after_state_change(&mut self) {
        self.rsp_jit.invalidate_all();
        self.tlb_rebuild();
        self.update_next_event();
        self.update_irq();
        self.gpu.clear();
        self.dirty.fill(u32::MAX);
        self.snap_dirty.fill(u32::MAX);
        self.audio.clear();
        self.audio_silent_from = if self.s.events[EV_AI] == u64::MAX { self.s.cycles } else { u64::MAX };
    }
}

#[no_mangle]
pub extern "C" fn n64_rom_alloc(len: u32) -> *mut u8 {
    let s = sys();
    s.rom = vec![0; len as usize];
    s.rom.as_mut_ptr()
}

/// Normalizes the ROM written into the buffer from `n64_rom_alloc` and powers on.
#[no_mangle]
pub extern "C" fn n64_load_rom() {
    let s = sys();
    System::normalize_rom(&mut s.rom);
    s.s.input.connected = 1;
    s.power_on();
}

/// Soft reset: the cartridge saves survive, as on the console's reset button.
#[no_mangle]
pub extern "C" fn n64_reset() {
    sys().reset();
}

#[no_mangle]
pub extern "C" fn n64_power_on() {
    sys().power_on();
}

/// Cartridge saves (EEPROM then SRAM), written by JS before a soft reset to
/// power on from given save data.
#[no_mangle]
pub extern "C" fn n64_save_ptr() -> *mut u8 {
    &mut sys().s.cart as *mut Cart as *mut u8
}

#[no_mangle]
pub extern "C" fn n64_save_size() -> u32 {
    core::mem::size_of::<Cart>() as u32
}

/// Runs until the next VI field. Returns 1 when the game polled controller 1.
#[no_mangle]
pub extern "C" fn n64_run_frame() -> u32 {
    let s = sys();
    s.run_frame();
    s.s.input.polled
}

#[no_mangle]
pub extern "C" fn n64_set_pad(port: u32, value: u32) {
    sys().s.input.pads[(port & 3) as usize] = value;
}

#[no_mangle]
pub extern "C" fn n64_set_connected(mask: u32) {
    sys().s.input.connected = mask & 0xF;
}

#[no_mangle]
pub extern "C" fn n64_frame() -> f64 {
    sys().s.frame as f64
}

/// Emulated seconds since power-on.
#[no_mangle]
pub extern "C" fn n64_time() -> f64 {
    sys().s.cycles as f64 / sys::CPU_HZ as f64
}

#[no_mangle]
pub extern "C" fn n64_pc() -> u32 {
    sys().s.cpu.pc
}

/// Interpreter callback for RSP blocks compiled by `rsp_jit`.
#[no_mangle]
pub extern "C" fn n64_rsp_jit_op(instr: u32) {
    sys().rsp_execute(instr, 0, false);
}

/// Switches the RSP recompiler, to check it against the interpreter.
#[no_mangle]
pub extern "C" fn n64_rsp_jit(on: u32) {
    sys().rsp_jit.on = on != 0;
}

// ---------------------------------------------------------------- savestates

#[no_mangle]
pub extern "C" fn n64_state_ptr() -> *mut u8 {
    &mut *sys().s as *mut State as *mut u8
}

#[no_mangle]
pub extern "C" fn n64_state_size() -> u32 {
    core::mem::size_of::<State>() as u32
}

/// Validates a state copied in through `n64_state_ptr`. Returns 0 on success.
#[no_mangle]
pub extern "C" fn n64_state_loaded() -> u32 {
    let s = sys();
    if s.s.magic != STATE_MAGIC || s.s.version != STATE_VERSION {
        s.reset();
        return 1;
    }
    s.after_state_change();
    0
}

// ---------------------------------------------------------------- memory

#[no_mangle]
pub extern "C" fn n64_rdram_ptr() -> *const u32 {
    sys().s.rdram.as_ptr()
}

#[no_mangle]
pub extern "C" fn n64_rdram_size() -> u32 {
    RDRAM_SIZE as u32
}

#[no_mangle]
pub extern "C" fn n64_poke8(paddr: u32, v: u32) {
    sys().rdram_write8(paddr, v as u8);
}

#[no_mangle]
pub extern "C" fn n64_poke16(paddr: u32, v: u32) {
    sys().rdram_write16(paddr & !1, v as u16);
}

#[no_mangle]
pub extern "C" fn n64_poke32(paddr: u32, v: u32) {
    sys().rdram_write32(paddr & !3, v);
}

/// Translates a CPU virtual address to physical, or returns u32::MAX.
#[no_mangle]
pub extern "C" fn n64_virt_to_phys(vaddr: u32) -> u32 {
    sys().translate_quiet(vaddr).unwrap_or(u32::MAX)
}

#[no_mangle]
pub extern "C" fn n64_dirty_ptr() -> *const u32 {
    sys().dirty.as_ptr()
}

#[no_mangle]
pub extern "C" fn n64_dirty_clear() {
    sys().dirty.fill(0);
}

#[no_mangle]
pub extern "C" fn n64_snap_dirty_ptr() -> *const u32 {
    sys().snap_dirty.as_ptr()
}

#[no_mangle]
pub extern "C" fn n64_snap_dirty_clear() {
    sys().snap_dirty.fill(0);
}

/// Byte offset of RDRAM inside the savestate; everything after it is RDRAM.
#[no_mangle]
pub extern "C" fn n64_state_rdram_offset() -> u32 {
    core::mem::offset_of!(State, rdram) as u32
}

// ---------------------------------------------------------------- video

#[no_mangle]
pub extern "C" fn n64_vi_regs_ptr() -> *const u32 {
    sys().s.vi.regs.as_ptr()
}

#[no_mangle]
pub extern "C" fn n64_vi_field() -> u32 {
    sys().s.vi.field
}

#[no_mangle]
pub extern "C" fn n64_gpu_enable(on: u32) {
    sys().gpu.enabled = on != 0;
}

#[no_mangle]
pub extern "C" fn n64_gpu_clear() {
    sys().gpu.clear();
}

#[no_mangle]
pub extern "C" fn n64_gpu_states_ptr() -> *const u32 {
    sys().gpu.states.as_ptr()
}

#[no_mangle]
pub extern "C" fn n64_gpu_states_len() -> u32 {
    sys().gpu.states.len() as u32
}

#[no_mangle]
pub extern "C" fn n64_gpu_prims_ptr() -> *const u32 {
    sys().gpu.prims.as_ptr()
}

#[no_mangle]
pub extern "C" fn n64_gpu_prims_len() -> u32 {
    sys().gpu.prims.len() as u32
}

#[no_mangle]
pub extern "C" fn n64_gpu_batches_ptr() -> *const u32 {
    sys().gpu.batches.as_ptr()
}

#[no_mangle]
pub extern "C" fn n64_gpu_batches_len() -> u32 {
    sys().gpu.batches.len() as u32
}

#[no_mangle]
pub extern "C" fn n64_gpu_tmem_ptr() -> *const u16 {
    sys().gpu.tmem.as_ptr()
}

#[no_mangle]
pub extern "C" fn n64_gpu_tmem_len() -> u32 {
    sys().gpu.tmem.len() as u32
}

// ---------------------------------------------------------------- audio

#[no_mangle]
pub extern "C" fn n64_audio_ptr() -> *const i16 {
    sys().audio.as_ptr()
}

#[no_mangle]
pub extern "C" fn n64_audio_len() -> u32 {
    sys().audio.len() as u32
}

#[no_mangle]
pub extern "C" fn n64_audio_rate() -> f64 {
    sys::CPU_HZ as f64 / sys().audio_cycles_per_sample as f64
}

#[no_mangle]
pub extern "C" fn n64_audio_clear() {
    sys().audio.clear();
}

// ---------------------------------------------------------------- native tests

#[cfg(test)]
mod tests {
    use super::*;

    /// Boots the ROM given in N64_ROM and reports progress. Run with
    /// `N64_ROM=path cargo test --release boot -- --nocapture`.
    #[test]
    fn boot() {
        let Ok(path) = std::env::var("N64_ROM") else { return };
        let rom = std::fs::read(path).unwrap();
        let ptr = n64_rom_alloc(rom.len() as u32);
        unsafe { core::ptr::copy_nonoverlapping(rom.as_ptr(), ptr, rom.len()) };
        n64_load_rom();
        let frames: u32 = std::env::var("N64_FRAMES").ok().and_then(|v| v.parse().ok()).unwrap_or(120);
        let start = std::time::Instant::now();
        for f in 0..frames {
            n64_run_frame();
            let s = sys();
            if f % 10 == 0 || f + 1 == frames {
                println!(
                    "frame {f:4} pc={:08x} boot={} vi_origin={:06x} vi_ctrl={:x} prims={} batches={} audio={} mi={:02x}/{:02x}",
                    s.s.cpu.pc,
                    s.s.boot_done,
                    s.s.vi.regs[1],
                    s.s.vi.regs[0],
                    s.gpu.prims.len() / rdp::PRIM_WORDS,
                    s.gpu.batches.len() / rdp::BATCH_WORDS,
                    s.audio.len(),
                    s.s.mi.intr,
                    s.s.mi.mask,
                );
            }
            s.gpu.clear();
            s.audio.clear();
        }
        println!("{} frames in {:?}", frames, start.elapsed());
    }
}
