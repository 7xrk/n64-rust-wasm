/** WebGPU backend: replays the core's RDP primitive records into frame memory, a
 * GPU copy of RDRAM at an adjustable resolution, and scans the VI framebuffer out
 * to a canvas. */

import { type Core, PAGE_SIZE, RDRAM_SIZE } from './core';
import fbSource from './fb.wgsl?raw';
import rdpSource from './rdp.wgsl?raw';
import viSource from './vi.wgsl?raw';

const BATCH_WORDS = 8;
const CAPTURE_WIDTH = 640;
const CAPTURE_HEIGHT = 480;
const BATCH_STRIDE = 256;
const MAX_BATCHES = 256;
const PAGE_COUNT = RDRAM_SIZE / PAGE_SIZE;
/** Internal resolution multipliers; frame memory at 4x is the largest buffer WebGPU guarantees. */
export const SCALES = [1, 2, 3, 4];

/** Bytes of frame memory: one word per RDRAM halfword and sample (see rdp.wgsl). */
function frameMemorySize(scale: number): number {
  return (RDRAM_SIZE / 2) * scale * scale * 4;
}

class GrowableBuffer {
  buffer: GPUBuffer;

  constructor(
    private readonly device: GPUDevice,
    private readonly label: string,
  ) {
    this.buffer = this.create(64 * 1024);
  }

  private create(size: number): GPUBuffer {
    return this.device.createBuffer({
      label: this.label,
      size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
  }

  /** Uploads `data`; returns true when the buffer had to be reallocated. */
  upload(data: ArrayBufferView<ArrayBuffer>): boolean {
    let grown = false;
    if (data.byteLength > this.buffer.size) {
      this.buffer.destroy();
      this.buffer = this.create(2 ** Math.ceil(Math.log2(data.byteLength)));
      grown = true;
    }
    if (data.byteLength > 0) {
      this.device.queue.writeBuffer(this.buffer, 0, data.buffer, data.byteOffset, data.byteLength);
    }
    return grown;
  }
}

export class Renderer {
  private scale = 1;
  private fb: GPUBuffer;
  /** RDRAM pages the CPU wrote, staged for `expand` in fb.wgsl. */
  private readonly upload: GPUBuffer;
  private readonly pageList: GPUBuffer;
  private readonly pageStaging = new Uint32Array(PAGE_COUNT);
  private readonly states: GrowableBuffer;
  private readonly prims: GrowableBuffer;
  private readonly tmem: GrowableBuffer;
  private readonly batchUniform: GPUBuffer;
  private readonly viUniform: GPUBuffer;
  private readonly rdpLayout: GPUBindGroupLayout;
  private readonly rdpModule: GPUShaderModule;
  private readonly viModule: GPUShaderModule;
  private readonly fbModule: GPUShaderModule;
  private readonly presentPipeline: GPURenderPipeline;
  private readonly capturePipeline: GPURenderPipeline;
  private readonly viSamplers: GPUSampler[];
  // Pipelines and bind groups that depend on the scale or on frame memory.
  private rdpPipeline!: GPUComputePipeline;
  private rdpBindGroup!: GPUBindGroup;
  private expandPipeline!: GPUComputePipeline;
  private expandBindGroup!: GPUBindGroup;
  private filterPipeline!: GPURenderPipeline;
  private filterBindGroup!: GPUBindGroup;
  /** The VI-filtered framebuffer at frame memory resolution, resampled by the present passes. */
  private filtered!: GPUTexture;
  private presentBindGroup!: GPUBindGroup;
  private captureBindGroup!: GPUBindGroup;
  private readonly batchStaging = new Uint32Array((MAX_BATCHES * BATCH_STRIDE) / 4);
  /** Frame memory copies per savestate slot, holding the rendered framebuffers. */
  private readonly slotCopies = new Map<number, GPUBuffer>();

  constructor(
    private readonly device: GPUDevice,
    private readonly context: GPUCanvasContext,
    format: GPUTextureFormat,
  ) {
    this.fb = this.createFrameMemory();
    this.upload = device.createBuffer({ label: 'rdram upload', size: RDRAM_SIZE, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.pageList = device.createBuffer({
      label: 'dirty pages',
      size: PAGE_COUNT * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.states = new GrowableBuffer(device, 'rdp states');
    this.prims = new GrowableBuffer(device, 'rdp prims');
    this.tmem = new GrowableBuffer(device, 'rdp tmem');
    this.batchUniform = device.createBuffer({
      size: MAX_BATCHES * BATCH_STRIDE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.viUniform = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    const storage = (binding: number, type: GPUBufferBindingType): GPUBindGroupLayoutEntry => ({
      binding,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type },
    });
    this.rdpLayout = device.createBindGroupLayout({
      entries: [
        storage(0, 'storage'),
        storage(1, 'read-only-storage'),
        storage(2, 'read-only-storage'),
        storage(3, 'read-only-storage'),
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true } },
      ],
    });
    this.rdpModule = device.createShaderModule({ label: 'rdp', code: rdpSource });
    this.viModule = device.createShaderModule({ label: 'vi', code: viSource });
    this.fbModule = device.createShaderModule({ label: 'frame memory', code: fbSource });
    this.presentPipeline = this.viPipeline('fs_present', format);
    this.capturePipeline = this.viPipeline('fs_present', 'rgba8unorm');
    this.viSamplers = [device.createSampler({ magFilter: 'linear', minFilter: 'linear' }), device.createSampler()];
    this.createScaledPipelines();
    this.resizeFiltered(320, 240);
  }

  private createFrameMemory(): GPUBuffer {
    return this.device.createBuffer({
      label: 'frame memory',
      size: frameMemorySize(this.scale),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
  }

  private viPipeline(entryPoint: string, target: GPUTextureFormat, constants?: Record<string, number>): GPURenderPipeline {
    return this.device.createRenderPipeline({
      layout: 'auto',
      vertex: { module: this.viModule, entryPoint: 'vs' },
      fragment: { module: this.viModule, entryPoint, targets: [{ format: target }], constants },
    });
  }

  private bindBuffers(pipeline: GPUComputePipeline | GPURenderPipeline, buffers: Record<number, GPUBuffer>): GPUBindGroup {
    return this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: Object.entries(buffers).map(([binding, buffer]) => ({ binding: Number(binding), resource: { buffer } })),
    });
  }

  private createScaledPipelines(): void {
    const constants = { SCALE: this.scale };
    this.rdpPipeline = this.device.createComputePipeline({
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.rdpLayout] }),
      compute: { module: this.rdpModule, entryPoint: 'main', constants },
    });
    this.rdpBindGroup = this.createRdpBindGroup();
    this.expandPipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: this.fbModule, entryPoint: 'expand', constants },
    });
    this.expandBindGroup = this.bindBuffers(this.expandPipeline, { 0: this.fb, 1: this.upload, 2: this.pageList });
    this.filterPipeline = this.viPipeline('fs_filter', 'rgba8unorm', constants);
    this.filterBindGroup = this.bindBuffers(this.filterPipeline, { 0: this.fb, 1: this.viUniform });
  }

  /** Changes the internal resolution, carrying the rendered frame memory over. Savestate slots lose their
   * framebuffers, since those were kept at the old resolution. */
  setScale(scale: number): void {
    if (scale === this.scale) return;
    const old = this.fb;
    const rescale = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: this.fbModule, entryPoint: 'rescale', constants: { SCALE: scale, OLD_SCALE: this.scale } },
    });
    this.scale = scale;
    this.fb = this.createFrameMemory();
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(rescale);
    pass.setBindGroup(0, this.bindBuffers(rescale, { 0: this.fb, 3: old }));
    // One invocation per halfword in 64-wide workgroups, on a 256-wide grid.
    pass.dispatchWorkgroups(256, RDRAM_SIZE / 2 / 64 / 256);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    old.destroy();
    this.clearSlots();
    this.createScaledPipelines();
  }

  private resizeFiltered(width: number, height: number): void {
    this.filtered?.destroy();
    this.filtered = this.device.createTexture({
      size: [width, height],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    const bindPresent = (pipeline: GPURenderPipeline) =>
      this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 1, resource: { buffer: this.viUniform } },
          { binding: 2, resource: this.filtered.createView() },
          { binding: 3, resource: this.viSamplers[0] },
          { binding: 4, resource: this.viSamplers[1] },
        ],
      });
    this.presentBindGroup = bindPresent(this.presentPipeline);
    this.captureBindGroup = bindPresent(this.capturePipeline);
  }

  /** Keeps a copy of frame memory, which alone holds the rendered framebuffers. */
  saveSlot(slot: number): void {
    const size = frameMemorySize(this.scale);
    let copy = this.slotCopies.get(slot);
    if (!copy) {
      copy = this.device.createBuffer({ size, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
      this.slotCopies.set(slot, copy);
    }
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.fb, 0, copy, 0, size);
    this.device.queue.submit([encoder.finish()]);
  }

  /** Restores the copy made by saveSlot and shows it. Returns false when absent. */
  loadSlot(slot: number, core: Core): boolean {
    const copy = this.slotCopies.get(slot);
    if (!copy) return false;
    core.clearDirty();
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(copy, 0, this.fb, 0, frameMemorySize(this.scale));
    this.device.queue.submit([encoder.finish()]);
    this.frame(core);
    return true;
  }

  clearSlots(): void {
    for (const b of this.slotCopies.values()) b.destroy();
    this.slotCopies.clear();
  }

  /** Reads back the current VI output as 640x480 RGBA. */
  async capture(): Promise<{ width: number; height: number; data: Uint8ClampedArray<ArrayBuffer> }> {
    const texture = this.device.createTexture({
      size: [CAPTURE_WIDTH, CAPTURE_HEIGHT],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    const readback = this.device.createBuffer({
      size: CAPTURE_WIDTH * CAPTURE_HEIGHT * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.device.createCommandEncoder();
    this.drawPass(encoder, texture.createView(), this.capturePipeline, this.captureBindGroup);
    encoder.copyTextureToBuffer({ texture }, { buffer: readback, bytesPerRow: CAPTURE_WIDTH * 4 }, [CAPTURE_WIDTH, CAPTURE_HEIGHT]);
    this.device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const data = new Uint8ClampedArray(readback.getMappedRange().slice(0));
    readback.destroy();
    texture.destroy();
    return { width: CAPTURE_WIDTH, height: CAPTURE_HEIGHT, data };
  }

  private createRdpBindGroup(): GPUBindGroup {
    return this.device.createBindGroup({
      layout: this.rdpLayout,
      entries: [
        { binding: 0, resource: { buffer: this.fb } },
        { binding: 1, resource: { buffer: this.states.buffer } },
        { binding: 2, resource: { buffer: this.prims.buffer } },
        { binding: 3, resource: { buffer: this.tmem.buffer } },
        { binding: 4, resource: { buffer: this.batchUniform, size: BATCH_WORDS * 4 } },
      ],
    });
  }

  /** Uploads the RDRAM pages the CPU side changed since the last frame and expands them into frame memory. */
  private uploadDirtyPages(core: Core, encoder: GPUCommandEncoder): void {
    const dirty = core.dirtyPages();
    const rdram = core.rdram();
    let count = 0;
    let page = 0;
    while (page < PAGE_COUNT) {
      if (!((dirty[page >>> 5] >>> (page & 31)) & 1)) {
        page++;
        continue;
      }
      const start = page;
      while (page < PAGE_COUNT && (dirty[page >>> 5] >>> (page & 31)) & 1) this.pageStaging[count++] = page++;
      const offset = start * PAGE_SIZE;
      this.device.queue.writeBuffer(this.upload, offset, rdram.buffer, rdram.byteOffset + offset, (page - start) * PAGE_SIZE);
    }
    core.clearDirty();
    if (count === 0) return;
    this.device.queue.writeBuffer(this.pageList, 0, this.pageStaging.buffer, 0, count * 4);
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.expandPipeline);
    pass.setBindGroup(0, this.expandBindGroup);
    pass.dispatchWorkgroups(count, PAGE_SIZE / 4 / 64);
    pass.end();
  }

  /** Draws everything the core emitted this frame and presents the VI output. */
  frame(core: Core): void {
    const encoder = this.device.createCommandEncoder();
    this.uploadDirtyPages(core, encoder);
    const batches = core.gpuBatches();
    const count = Math.min(batches.length / BATCH_WORDS, MAX_BATCHES);
    const grown = [this.states.upload(core.gpuStates()), this.prims.upload(core.gpuPrims()), this.tmem.upload(core.gpuTmem())];
    if (grown.some(Boolean)) {
      this.rdpBindGroup = this.createRdpBindGroup();
    }
    for (let i = 0; i < count; i++) {
      this.batchStaging.set(batches.subarray(i * BATCH_WORDS, (i + 1) * BATCH_WORDS), (i * BATCH_STRIDE) / 4);
    }
    if (count > 0) {
      this.device.queue.writeBuffer(this.batchUniform, 0, this.batchStaging.buffer, 0, count * BATCH_STRIDE);
    }

    if (count > 0) {
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.rdpPipeline);
      for (let i = 0; i < count; i++) {
        const b = i * BATCH_WORDS;
        const x0 = batches[b + 6] & 0xffff;
        const y0 = batches[b + 6] >>> 16;
        const x1 = batches[b + 7] & 0xffff;
        const y1 = batches[b + 7] >>> 16;
        if (x1 <= x0 || y1 <= y0) continue;
        pass.setBindGroup(0, this.rdpBindGroup, [i * BATCH_STRIDE]);
        // Each invocation draws one sample; a workgroup covers 8x8.
        pass.dispatchWorkgroups(Math.ceil(((x1 - x0) * this.scale) / 8), Math.ceil(((y1 - y0) * this.scale) / 8));
      }
      pass.end();
    }
    core.gpuClear();

    const [width, height] = this.writeViUniform(core.viRegs());
    if (width !== this.filtered.width || height !== this.filtered.height) {
      this.resizeFiltered(width, height);
    }
    this.drawPass(encoder, this.filtered.createView(), this.filterPipeline, this.filterBindGroup);
    this.drawPass(encoder, this.context.getCurrentTexture().createView(), this.presentPipeline, this.presentBindGroup);
    this.device.queue.submit([encoder.finish()]);
  }

  private drawPass(encoder: GPUCommandEncoder, view: GPUTextureView, pipeline: GPURenderPipeline, bindGroup: GPUBindGroup): void {
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();
  }

  /** Writes the VI uniform and returns the framebuffer area it scans out, in frame memory samples. */
  private writeViUniform(regs: Uint32Array): [number, number] {
    const ctrl = regs[0];
    const width = regs[2] & 0xfff;
    const hStart = (regs[9] >>> 16) & 0x3ff;
    const hEnd = regs[9] & 0x3ff;
    const vStart = (regs[10] >>> 16) & 0x3ff;
    const vEnd = regs[10] & 0x3ff;
    const xScale = (regs[12] & 0xfff) / 1024;
    const yScale = (regs[13] & 0xfff) / 1024;
    let srcW = (hEnd - hStart) * xScale;
    let srcH = ((vEnd - vStart) / 2) * yScale;
    if (!(srcW > 0) || !(srcH > 0)) {
      srcW = width;
      srcH = (width * 3) / 4;
    }
    srcW *= this.scale;
    srcH *= this.scale;
    const data = new ArrayBuffer(32);
    const u = new Uint32Array(data);
    const f = new Float32Array(data);
    u[0] = regs[1] & 0xffffff;
    u[1] = width;
    u[2] = ctrl;
    f[4] = srcW;
    f[5] = srcH;
    this.device.queue.writeBuffer(this.viUniform, 0, data);
    return [Math.max(1, Math.ceil(srcW)), Math.max(1, Math.ceil(srcH))];
  }
}
