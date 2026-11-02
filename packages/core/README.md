# n64core

Nintendo 64 emulator core compiled to WebAssembly, with a WebGPU renderer for its RDP output.

## Usage

The WebAssembly module ships as `n64core/n64core.wasm`. Pass its URL to `Core.load`.

```ts
import { Core, Renderer, packPad } from 'n64core';
import wasmUrl from 'n64core/n64core.wasm?url'; // Vite; use your bundler's asset URL import otherwise

const core = await Core.load(wasmUrl);
core.loadRom(rom);

const adapter = await navigator.gpu.requestAdapter();
const device = await adapter!.requestDevice({
  requiredLimits: { maxStorageBufferBindingSize: adapter!.limits.maxStorageBufferBindingSize },
});
const context = canvas.getContext('webgpu')!;
const format = navigator.gpu.getPreferredCanvasFormat();
context.configure({ device, format, alphaMode: 'opaque' });
const renderer = new Renderer(device, context, format);

core.setRendering(true);
function tick() {
  core.setPad(0, packPad(['A']));
  core.runFrame();
  renderer.frame(core);
  requestAnimationFrame(tick);
}
tick();
```

`Renderer` takes WebGPU types from the DOM lib. With a TypeScript version whose DOM lib lacks them, add `@webgpu/types`.

## Building

Requires Rust with the `wasm32-unknown-unknown` target.

```bash
pnpm build
```
