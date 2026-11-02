//! PIF: high-level boot, joybus controllers and EEPROM.

use crate::state::*;
use crate::sys::System;

/// N64 memory size variable read by libultra.
const OS_MEM_SIZE: u32 = 0x318;

impl System {
    /// Converts .v64 / .n64 images to big-endian byte order.
    pub fn normalize_rom(rom: &mut [u8]) {
        match rom.get(0..4) {
            Some([0x37, 0x80, 0x40, 0x12]) => {
                for c in rom.chunks_exact_mut(2) {
                    c.swap(0, 1);
                }
            }
            Some([0x40, 0x12, 0x37, 0x80]) => {
                for c in rom.chunks_exact_mut(4) {
                    c.reverse();
                }
            }
            _ => {}
        }
    }

    /// CIC seed derived from the IPL3 in the ROM header area.
    fn cic_seed(&self) -> u32 {
        let crc = crc32(&self.rom[0x40..0x1000.min(self.rom.len())]);
        match crc {
            0x6170_A4A1 => 0x3F, // 6101
            0x90BB_6CB5 => 0x3F, // 6102
            0x0B05_0EE0 => 0x78, // 6103
            0x98BC_2C86 => 0x91, // 6105
            0xACC8_580A => 0x85, // 6106
            _ => 0x3F,
        }
    }

    fn game_code(&self) -> [u8; 3] {
        [self.rom_byte(0x3C), self.rom_byte(0x3D), self.rom_byte(0x3E)]
    }

    /// Simulates the PIF boot ROM and starts the cartridge IPL3 in DMEM.
    pub fn hle_boot(&mut self) {
        let seed = self.cic_seed();
        let code = self.game_code();
        self.eeprom_16k = EEPROM_16K_GAMES.iter().any(|g| g[..] == code[..2]);

        self.s.sp_mem.fill(0);
        let n = 0x1000.min(self.rom.len());
        self.s.sp_mem[0x40..n].copy_from_slice(&self.rom[0x40..n]);

        let cpu = &mut self.s.cpu;
        cpu.gpr[19] = 0; // cartridge
        cpu.gpr[20] = if code[2] == b'P' || code[2] == b'D' || code[2] == b'F' { 0 } else { 1 };
        cpu.gpr[21] = 0; // cold reset
        cpu.gpr[22] = seed as u64;
        cpu.gpr[23] = 0;
        cpu.gpr[11] = 0xFFFF_FFFF_A400_0040;
        cpu.gpr[29] = 0xFFFF_FFFF_A400_1FF0;
        cpu.gpr[31] = 0xFFFF_FFFF_A400_1550;
        cpu.cop0[12] = 0x3400_0000;
        cpu.cop0[16] = 0x0006_E463;
        cpu.cop0[15] = 0x0B22;
        cpu.cop0[1] = 31;
        cpu.pc = 0xA400_0040;
        cpu.next_pc = 0xA400_0044;

        // Skip RDRAM init in IPL3 and tell the OS how much memory exists.
        self.s.ri[3] = 0x14;
        self.s.ri[0] = 0x0E;
        self.s.ri[1] = 0x40;
        self.s.ri[4] = 0x0006_3634;
        self.rdram_write32(OS_MEM_SIZE, RDRAM_SIZE as u32);
        self.s.entry_point = u32::from_be_bytes([
            self.rom_byte(8),
            self.rom_byte(9),
            self.rom_byte(10),
            self.rom_byte(11),
        ]);
        self.s.boot_done = 0;
        self.s.rsp.status = 1; // halted
    }

    /// Called when IPL3 jumps into the game.
    pub fn boot_finished(&mut self) {
        self.s.boot_done = 1;
        self.rdram_write32(OS_MEM_SIZE, RDRAM_SIZE as u32);
    }

    /// Executes the joybus command list in PIF RAM. Runs on every SI read:
    /// libultra repeats a controller read without rewriting the list.
    pub fn pif_process(&mut self) {
        let mut channel = 0usize;
        let mut i = 0usize;
        while i < 63 {
            let tx = self.s.pif.ram[i];
            match tx {
                0x00 => {
                    channel += 1;
                    i += 1;
                    continue;
                }
                0xFE => break,
                0xFD | 0xFF => {
                    i += 1;
                    continue;
                }
                _ => {}
            }
            if i + 1 >= 63 {
                break;
            }
            let tx_len = (tx & 0x3F) as usize;
            let rx_len = (self.s.pif.ram[i + 1] & 0x3F) as usize;
            let cmd_at = i + 2;
            let rx_at = cmd_at + tx_len;
            if rx_at + rx_len > 64 {
                break;
            }
            let ok = self.joybus_command(channel, cmd_at, tx_len, rx_at, rx_len);
            if !ok {
                self.s.pif.ram[i + 1] |= 0x80;
            }
            i = rx_at + rx_len;
            channel += 1;
        }
        self.s.pif.ram[63] &= !1;
    }

    fn joybus_command(&mut self, channel: usize, cmd_at: usize, tx_len: usize, rx_at: usize, rx_len: usize) -> bool {
        if tx_len == 0 {
            return true;
        }
        let cmd = self.s.pif.ram[cmd_at];
        if channel < 4 {
            if self.s.input.connected & (1 << channel) == 0 {
                return false;
            }
            match cmd {
                0x00 | 0xFF => {
                    // Standard controller, no pak.
                    self.write_rx(rx_at, rx_len, &[0x05, 0x00, 0x02]);
                }
                0x01 => {
                    if channel == 0 {
                        self.s.input.polled = 1;
                    }
                    let pad = self.s.input.pads[channel].to_le_bytes();
                    self.write_rx(rx_at, rx_len, &pad);
                }
                _ => return false,
            }
            return true;
        }
        if channel == 4 {
            match cmd {
                0x00 | 0xFF => {
                    let kind = if self.eeprom_16k { 0xC0 } else { 0x80 };
                    self.write_rx(rx_at, rx_len, &[0x00, kind, 0x00]);
                }
                0x04 => {
                    let block = self.s.pif.ram[cmd_at + 1] as usize * 8;
                    let mut data = [0u8; 8];
                    data.copy_from_slice(&self.s.cart.eeprom[block % 2048..block % 2048 + 8]);
                    self.write_rx(rx_at, rx_len, &data);
                }
                0x05 => {
                    let block = (self.s.pif.ram[cmd_at + 1] as usize * 8) % 2048;
                    for k in 0..8 {
                        self.s.cart.eeprom[block + k] = self.s.pif.ram[cmd_at + 2 + k];
                    }
                    self.write_rx(rx_at, rx_len, &[0x00]);
                }
                _ => return false,
            }
            return true;
        }
        false
    }

    fn write_rx(&mut self, at: usize, len: usize, data: &[u8]) {
        for k in 0..len.min(data.len()) {
            self.s.pif.ram[at + k] = data[k];
        }
    }
}

/// Games whose save chip is a 16 Kbit EEPROM, matched on the 2-letter game ID.
const EEPROM_16K_GAMES: &[[u8; 2]] = &[
    *b"YS", // Yoshi's Story
    *b"B7", // Banjo-Tooie
];

pub fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 { (crc >> 1) ^ 0xEDB8_8320 } else { crc >> 1 };
        }
    }
    !crc
}
