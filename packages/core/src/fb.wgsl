// Frame memory upkeep; the layout is described in rdp.wgsl. `expand` copies
// RDRAM pages the CPU wrote into every sample of their halfwords, keeping the
// hidden bits. `rescale` carries frame memory over to another SCALE, taking
// each new sample from the old sample that covers it.

override SCALE: u32 = 1u;
override OLD_SCALE: u32 = 1u;

const PAGE_WORDS: u32 = 1024u;

@group(0) @binding(0) var<storage, read_write> fb: array<u32>;
// RDRAM as host-order words.
@group(0) @binding(1) var<storage, read> upload: array<u32>;
// Indices of the 4 KiB pages to expand.
@group(0) @binding(2) var<storage, read> pages: array<u32>;
@group(0) @binding(3) var<storage, read> old: array<u32>;

// One workgroup per 64 words: x picks the page, y the 64-word run in it.
@compute @workgroup_size(64)
fn expand(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let word = pages[wg.x] * PAGE_WORDS + wg.y * 64u + li;
  let w = upload[word];
  for (var h = 0u; h < 2u; h++) {
    let base = (word * 2u + h) * SCALE * SCALE;
    let v = select(w & 0xFFFFu, w >> 16u, h == 0u);
    for (var s = 0u; s < SCALE * SCALE; s++) {
      fb[base + s] = (fb[base + s] & 0x30000u) | v;
    }
  }
}

// One invocation per halfword, in a 256-wide grid of 64-halfword workgroups.
@compute @workgroup_size(64)
fn rescale(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let half = (wg.y * 256u + wg.x) * 64u + li;
  for (var j = 0u; j < SCALE; j++) {
    for (var i = 0u; i < SCALE; i++) {
      let o = (j * OLD_SCALE / SCALE) * OLD_SCALE + i * OLD_SCALE / SCALE;
      fb[half * SCALE * SCALE + j * SCALE + i] = old[half * OLD_SCALE * OLD_SCALE + o];
    }
  }
}
