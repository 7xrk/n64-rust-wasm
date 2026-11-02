# N64 TAS

A Nintendo 64 emulator for making tool-assisted speedruns, running in the browser.

The emulator is written from scratch in Rust and compiled to WebAssembly. No outside emulator code is used. Rendering runs on WebGPU.

## Features

- **Deterministic emulation.** The CPU, RSP, RDP command processing, PIF and cartridge saves all run on the CPU side. GPU output never flows back into emulated memory. The same initial state and inputs always give the same result.
- **Movies.** A movie records four controller ports per frame. It powers on from the cartridge saves embedded in it.
- **Piano roll.** Inputs can be edited, inserted, deleted, copied and pasted at any frame. Editing a past frame recomputes up to the current one. Each edit is one undo step.
- **Snapshots and seeking.** Full states are taken at a fixed frame interval. Pages that did not change are shared with the previous snapshot. A seek replays from the nearest snapshot.
- **Savestates.** Nine slots, plus export and import as files.
- **Scene history.** A screenshot and state are kept every 120 frames, to jump back to a past moment from the controller.
- **Memory tools.** RAM search, watches, PAR / GameShark codes, and an in-app agent that searches memory with the Anthropic API.
- **Video recording.** Recordings are MP4 files with H.264 video and AAC audio. Their timestamps come from emulated time, so the file plays smoothly however fast the host ran.
- **Internal resolution.** 1× to 4×, chosen in the settings.
- **Input.** Keyboard, Gamepad API, and the Nintendo Switch Online N64 controller through WebHID.

## Requirements

- A browser with WebGPU. Desktop Chrome or Edge is recommended, and is required for WebHID.
- Node.js and pnpm.
- Rust with the `wasm32-unknown-unknown` target.

## Getting started

```bash
pnpm install
```

```bash
pnpm dev
```

`pnpm dev` builds the WebAssembly core and starts the development server. Open the page and choose a ROM file.

## Commands

| Command                       | Purpose                                                |
| ----------------------------- | ------------------------------------------------------ |
| `pnpm dev`                    | Build the core and start the development server        |
| `pnpm -F n64core build:wasm`  | Rebuild only the core. Run it after changing Rust code |
| `pnpm build`                  | Build everything                                       |
| `npx vp check --fix packages` | Lint and format                                        |
| `npx tsc -p packages/front`   | Type-check the app                                     |
| `npx tsc -p packages/core`    | Type-check the core package                            |

Run cargo inside `packages/core/crate`. The SIMD setting and the exported function table in `crate/.cargo/config.toml` only apply there.

A native boot test runs a ROM without a browser. It is slow, so start with few frames.

```bash
N64_ROM=path/to/rom.z64 N64_FRAMES=300 cargo test --release --manifest-path packages/core/crate/Cargo.toml boot -- --nocapture
```

## Hotkeys

| Key       | Action                  |
| --------- | ----------------------- |
| P         | Pause / resume          |
| .         | Advance one frame       |
| ,         | Step back one frame     |
| \`        | Toggle fast-forward     |
| 1–9       | Load slot               |
| Shift+1–9 | Save to slot            |
| R         | Toggle record / play    |
| ⌘Z / ⌘⇧Z  | Undo / redo movie edits |

Controller buttons can be remapped in the settings.

## Movie files

A `.n64tas` file holds these parts in order:

1. A one-line JSON header.
2. Little-endian input words, four ports per frame.
3. The cartridge saves at power-on.
4. Gzipped snapshots, at least 200 frames apart.

The snapshots are only used when the WebAssembly build hash matches. Otherwise the movie is recomputed from its inputs.

## Layout

| Path                  | Contents                                                                                        |
| --------------------- | ----------------------------------------------------------------------------------------------- |
| `packages/core/crate` | The Rust emulator core. It exposes a flat C ABI to WebAssembly                                  |
| `packages/core/src`   | The TypeScript wrapper `Core`, the WebGPU renderer and its WGSL shaders. Published as `n64core` |
| `packages/front`      | The app. A Preact UI on the main thread drives the emulator in a Web Worker                     |

The RSP runs through a JIT that translates IMEM into WebAssembly block by block. Its results match the interpreter bit for bit. The RDP does not draw pixels on the CPU. It records primitives, state and TMEM, and a compute shader draws them into frame memory in N64 format.

## Scripting

The page exposes `window.n64` for scripts and automation. It loads ROMs and movies by URL, calls every worker API, builds pad values, waits on emulator state and takes screenshots. Run `n64.help` in the console for the full reference.
