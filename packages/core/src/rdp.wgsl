// RDP rasterizer. Each invocation owns two horizontally adjacent pixels so
// 16-bit color and Z words are never shared between invocations. A workgroup
// covers a 16x8 tile and walks the batch's primitives in submission order.
// Record layouts are produced by crate/src/rdp.rs.

const STATE_WORDS: u32 = 56u;
const PRIM_WORDS: u32 = 52u;
const RDRAM_MASK: u32 = 0x1FFFFFu;
// RDRAM's 9th bits follow RDRAM in the buffer, one word per RDRAM word. Each
// 16-bit pixel of that word owns two bits, the first pixel the higher pair.
const HIDDEN: u32 = 0x200000u;

const PRIM_TRI: u32 = 0u;
const PRIM_TEXRECT: u32 = 1u;

const CYCLE_2: u32 = 1u;
const CYCLE_COPY: u32 = 2u;
const CYCLE_FILL: u32 = 3u;

struct Batch {
  ci_addr: u32,
  ci_width: u32,
  ci_size: u32,
  z_addr: u32,
  first: u32,
  count: u32,
  xy0: u32,
  xy1: u32,
};

@group(0) @binding(0) var<storage, read_write> rdram: array<u32>;
@group(0) @binding(1) var<storage, read> states: array<u32>;
@group(0) @binding(2) var<storage, read> prims: array<u32>;
@group(0) @binding(3) var<storage, read> tmem: array<u32>;
@group(0) @binding(4) var<uniform> batch: Batch;

var<workgroup> hit: array<u32, 64>;

var<private> color: array<u32, 2>;
// The Z word holding both pixels of this invocation.
var<private> zpair: u32;
var<private> color_hidden: u32;
var<private> z_hidden: u32;
var<private> color_dirty: bool;
var<private> z_dirty: bool;

// Per-primitive context shared by the pixel functions.
var<private> st: u32;
var<private> tm: u32;
var<private> om_hi: u32;
var<private> om_lo: u32;

fn unpack_rgba(c: u32) -> vec4<i32> {
  return vec4<i32>(i32(c >> 24u), i32((c >> 16u) & 255u), i32((c >> 8u) & 255u), i32(c & 255u));
}

fn sext(v: u32, bits: u32) -> i32 {
  return i32(v << (32u - bits)) >> (32u - bits);
}

// ------------------------------------------------------------------ TMEM

fn tmem16(i: u32) -> u32 {
  let w = tmem[tm + ((i & 0x7FFu) >> 1u)];
  return select(w & 0xFFFFu, w >> 16u, (i & 1u) == 1u);
}

fn tmem8(b: u32) -> u32 {
  let h = tmem16((b & 0xFFFu) >> 1u);
  return select(h >> 8u, h & 255u, (b & 1u) == 1u);
}

fn decode5551(c: u32) -> vec4<i32> {
  let r = (c >> 11u) & 31u;
  let g = (c >> 6u) & 31u;
  let b = (c >> 1u) & 31u;
  return vec4<i32>(i32((r << 3u) | (r >> 2u)), i32((g << 3u) | (g >> 2u)), i32((b << 3u) | (b >> 2u)), i32((c & 1u) * 255u));
}

fn decode_tlut(index: u32) -> vec4<i32> {
  let c = tmem16(0x400u + index * 4u);
  if (((om_hi >> 14u) & 1u) == 1u) {
    return vec4<i32>(i32(c >> 8u), i32(c >> 8u), i32(c >> 8u), i32(c & 255u));
  }
  return decode5551(c);
}

struct Tile {
  fmt: u32,
  size: u32,
  palette: u32,
  ct: u32,
  mt: u32,
  cs: u32,
  ms: u32,
  mask_t: u32,
  shift_t: u32,
  mask_s: u32,
  shift_s: u32,
  line: u32,
  tbase: u32,
  sl: i32,
  tl: i32,
  sh: i32,
  th: i32,
};

fn load_tile(index: u32) -> Tile {
  let o = st + 24u + (index & 7u) * 4u;
  let w0 = states[o];
  let w1 = states[o + 1u];
  let w2 = states[o + 2u];
  let w3 = states[o + 3u];
  var t: Tile;
  t.fmt = w0 & 7u;
  t.size = (w0 >> 3u) & 3u;
  t.palette = (w0 >> 5u) & 15u;
  t.ct = (w0 >> 9u) & 1u;
  t.mt = (w0 >> 10u) & 1u;
  t.cs = (w0 >> 11u) & 1u;
  t.ms = (w0 >> 12u) & 1u;
  t.mask_t = (w0 >> 13u) & 15u;
  t.shift_t = (w0 >> 17u) & 15u;
  t.mask_s = (w0 >> 21u) & 15u;
  t.shift_s = (w0 >> 25u) & 15u;
  t.line = (w1 & 0x1FFu) * 8u;
  t.tbase = ((w1 >> 16u) & 0x1FFu) * 8u;
  t.sl = i32(w2 & 0xFFFu);
  t.tl = i32((w2 >> 16u) & 0xFFFu);
  t.sh = i32(w3 & 0xFFFu);
  t.th = i32((w3 >> 16u) & 0xFFFu);
  return t;
}

fn texel(t: Tile, s: u32, tt: u32) -> vec4<i32> {
  let row = t.tbase + tt * t.line;
  let swz = (tt & 1u) << 2u;
  let tlut = ((om_hi >> 15u) & 1u) == 1u;
  if (t.size == 3u) {
    let a = ((row + s * 2u) ^ swz) & 0x7FFu;
    let rg = tmem16(a >> 1u);
    let ba = tmem16((a >> 1u) + 0x400u);
    return vec4<i32>(i32(rg >> 8u), i32(rg & 255u), i32(ba >> 8u), i32(ba & 255u));
  }
  if (t.size == 2u) {
    let c = tmem16(((row + s * 2u) ^ swz) >> 1u);
    if (t.fmt == 3u) {
      return vec4<i32>(i32(c >> 8u), i32(c >> 8u), i32(c >> 8u), i32(c & 255u));
    }
    if (t.fmt == 2u || tlut) {
      return decode_tlut(c >> 8u);
    }
    return decode5551(c);
  }
  if (t.size == 1u) {
    let b = tmem8((row + s) ^ swz);
    if (tlut) {
      return decode_tlut(b);
    }
    if (t.fmt == 3u) {
      let i = (b >> 4u) * 17u;
      return vec4<i32>(i32(i), i32(i), i32(i), i32((b & 15u) * 17u));
    }
    return vec4<i32>(i32(b), i32(b), i32(b), i32(b));
  }
  let b = tmem8((row + (s >> 1u)) ^ swz);
  let n = select(b >> 4u, b & 15u, (s & 1u) == 1u);
  if (tlut) {
    return decode_tlut((t.palette << 4u) | n);
  }
  if (t.fmt == 3u) {
    let i3 = n >> 1u;
    let i = (i3 << 5u) | (i3 << 2u) | (i3 >> 1u);
    return vec4<i32>(i32(i), i32(i), i32(i), i32((n & 1u) * 255u));
  }
  return vec4<i32>(i32(n * 17u), i32(n * 17u), i32(n * 17u), i32(n * 17u));
}

fn shift_coord(c: i32, shift: u32) -> i32 {
  if (shift == 0u) {
    return c;
  }
  if (shift <= 10u) {
    return c >> shift;
  }
  return c << (16u - shift);
}

// Applies clamp / mirror / mask to an integer texel coordinate.
fn wrap_coord(c: i32, max_c: i32, clamp_bit: u32, mirror: u32, mask: u32) -> u32 {
  var v = c;
  if (clamp_bit == 1u || mask == 0u) {
    v = clamp(v, 0, max(max_c, 0));
  }
  if (mask != 0u) {
    let m = min(mask, 10u);
    if (mirror == 1u && ((v >> m) & 1) == 1) {
      v = ~v;
    }
    v = v & ((1 << m) - 1);
  }
  return u32(v) & 0x3FFu;
}

fn sample_tile(index: u32, s10_5: f32, t10_5: f32) -> vec4<i32> {
  let t = load_tile(index);
  let s = shift_coord(i32(floor(s10_5)), t.shift_s) - (t.sl << 3u);
  let tt = shift_coord(i32(floor(t10_5)), t.shift_t) - (t.tl << 3u);
  let max_s = (t.sh - t.sl) >> 2u;
  let max_t = (t.th - t.tl) >> 2u;
  let s0 = s >> 5u;
  let t0 = tt >> 5u;
  let a = texel(t, wrap_coord(s0, max_s, t.cs, t.ms, t.mask_s), wrap_coord(t0, max_t, t.ct, t.mt, t.mask_t));
  if (((om_hi >> 13u) & 1u) == 0u) {
    return a;
  }
  // Three-point filter as performed by the RDP.
  let fs = s & 31;
  let ft = tt & 31;
  let s1 = wrap_coord(s0 + 1, max_s, t.cs, t.ms, t.mask_s);
  let t1 = wrap_coord(t0 + 1, max_t, t.ct, t.mt, t.mask_t);
  let sw = wrap_coord(s0, max_s, t.cs, t.ms, t.mask_s);
  let tw = wrap_coord(t0, max_t, t.ct, t.mt, t.mask_t);
  let b = texel(t, s1, tw);
  let c = texel(t, sw, t1);
  if (fs + ft < 32) {
    return a + ((b - a) * fs + (c - a) * ft) / 32;
  }
  let d = texel(t, s1, t1);
  return d + ((c - d) * (32 - fs) + (b - d) * (32 - ft)) / 32;
}

// ------------------------------------------------------------------ combiner

struct Inputs {
  combined: vec4<i32>,
  tex0: vec4<i32>,
  tex1: vec4<i32>,
  prim: vec4<i32>,
  shade: vec4<i32>,
  env: vec4<i32>,
  noise: i32,
  lod_frac: i32,
  prim_lod: i32,
};

fn sel_rgb(s: u32, v: Inputs) -> vec3<i32> {
  switch (s) {
    case 0u: { return v.combined.rgb; }
    case 1u: { return v.tex0.rgb; }
    case 2u: { return v.tex1.rgb; }
    case 3u: { return v.prim.rgb; }
    case 4u: { return v.shade.rgb; }
    case 5u: { return v.env.rgb; }
    default: { return vec3<i32>(0); }
  }
}

fn cc_a(s: u32, v: Inputs) -> vec3<i32> {
  if (s == 6u) { return vec3<i32>(256); }
  if (s == 7u) { return vec3<i32>(v.noise); }
  return sel_rgb(s, v);
}

fn cc_b(s: u32, v: Inputs) -> vec3<i32> {
  if (s == 6u) {
    let kr = states[st + 11u];
    let kgb = states[st + 12u];
    return vec3<i32>(i32((kr >> 8u) & 255u), i32(kgb >> 24u), i32((kgb >> 8u) & 255u));
  }
  if (s == 7u) { return vec3<i32>(sext((states[st + 14u] >> 9u) & 0x1FFu, 9u)); }
  return sel_rgb(s, v);
}

fn cc_c(s: u32, v: Inputs) -> vec3<i32> {
  switch (s) {
    case 6u: {
      let kr = states[st + 11u];
      let kgb = states[st + 12u];
      return vec3<i32>(i32(kr & 255u), i32((kgb >> 16u) & 255u), i32(kgb & 255u));
    }
    case 7u: { return vec3<i32>(v.combined.a); }
    case 8u: { return vec3<i32>(v.tex0.a); }
    case 9u: { return vec3<i32>(v.tex1.a); }
    case 10u: { return vec3<i32>(v.prim.a); }
    case 11u: { return vec3<i32>(v.shade.a); }
    case 12u: { return vec3<i32>(v.env.a); }
    case 13u: { return vec3<i32>(v.lod_frac); }
    case 14u: { return vec3<i32>(v.prim_lod); }
    case 15u: { return vec3<i32>(sext(states[st + 14u] & 0x1FFu, 9u)); }
    default: { return sel_rgb(s, v); }
  }
}

fn cc_d(s: u32, v: Inputs) -> vec3<i32> {
  if (s == 6u) { return vec3<i32>(256); }
  return sel_rgb(s, v);
}

fn sel_alpha(s: u32, v: Inputs) -> i32 {
  switch (s) {
    case 0u: { return v.combined.a; }
    case 1u: { return v.tex0.a; }
    case 2u: { return v.tex1.a; }
    case 3u: { return v.prim.a; }
    case 4u: { return v.shade.a; }
    case 5u: { return v.env.a; }
    case 6u: { return 256; }
    default: { return 0; }
  }
}

fn ca_c(s: u32, v: Inputs) -> i32 {
  switch (s) {
    case 0u: { return v.lod_frac; }
    case 6u: { return v.prim_lod; }
    case 7u: { return 0; }
    default: { return sel_alpha(s, v); }
  }
}

// The combiner's 9-bit result wraps: 0x100-0x17F saturates, 0x180-0x1FF is negative.
fn cc_clamp(v: i32) -> i32 {
  let r = v & 0x1FF;
  if ((r & 0x180) == 0x100) { return 255; }
  if ((r & 0x180) == 0x180) { return 0; }
  return r & 255;
}

fn combine(cycle: u32, v: Inputs) -> vec4<i32> {
  let hi = states[st + 2u];
  let lo = states[st + 3u];
  var ra: u32; var rc: u32; var aa: u32; var ac: u32;
  var rb: u32; var rd: u32; var ab: u32; var ad: u32;
  if (cycle == 0u) {
    ra = (hi >> 20u) & 15u; rc = (hi >> 15u) & 31u; aa = (hi >> 12u) & 7u; ac = (hi >> 9u) & 7u;
    rb = (lo >> 28u) & 15u; rd = (lo >> 15u) & 7u; ab = (lo >> 12u) & 7u; ad = (lo >> 9u) & 7u;
  } else {
    ra = (hi >> 5u) & 15u; rc = hi & 31u; aa = (lo >> 21u) & 7u; ac = (lo >> 18u) & 7u;
    rb = (lo >> 24u) & 15u; rd = (lo >> 6u) & 7u; ab = (lo >> 3u) & 7u; ad = lo & 7u;
  }
  let rgb = (cc_a(ra, v) - cc_b(rb, v)) * cc_c(rc, v) + cc_d(rd, v) * 256 + vec3<i32>(128);
  let al = (sel_alpha(aa, v) - sel_alpha(ab, v)) * ca_c(ac, v) + sel_alpha(ad, v) * 256 + 128;
  return vec4<i32>(cc_clamp(rgb.r >> 8u), cc_clamp(rgb.g >> 8u), cc_clamp(rgb.b >> 8u), cc_clamp(al >> 8u));
}

// ------------------------------------------------------------------ blender

fn blend_rgb(s: u32, pix: vec3<i32>, mem: vec3<i32>) -> vec3<i32> {
  switch (s) {
    case 0u: { return pix; }
    case 1u: { return mem; }
    case 2u: { return unpack_rgba(states[st + 6u]).rgb; }
    default: { return unpack_rgba(states[st + 5u]).rgb; }
  }
}

// Blender mux selectors of `cycle`: 1a, 1b, 2a, 2b.
fn blend_sel(cycle: u32) -> vec4<u32> {
  let sh = (1u - cycle) * 2u;
  return (vec4<u32>(om_lo) >> (vec4<u32>(28u, 24u, 20u, 16u) + sh)) & vec4<u32>(3u);
}

// Without force_blend the sum is normalized by the weights, which turns the
// usual pixel/memory coverage weights into the antialiasing edge average.
fn blend(cycle: u32, pix: vec3<i32>, pix_a: i32, shade_a: i32, mem: vec4<i32>, normalize: bool) -> vec3<i32> {
  let sel = blend_sel(cycle);
  let m1b = sel.y;
  let m2b = sel.w;
  let p = blend_rgb(sel.x, pix, mem.rgb);
  let m = blend_rgb(sel.z, pix, mem.rgb);
  var a: i32;
  switch (m1b) {
    case 0u: { a = pix_a; }
    case 1u: { a = unpack_rgba(states[st + 5u]).a; }
    case 2u: { a = shade_a; }
    default: { a = 0; }
  }
  var b: i32;
  switch (m2b) {
    case 0u: { b = 255 - a; }
    case 1u: { b = mem.a; }
    case 2u: { b = 255; }
    default: { b = 0; }
  }
  var a5 = a >> 3u;
  var b5 = b >> 3u;
  if (m2b == 1u) {
    a5 &= 0x3C;
    b5 |= 3;
  }
  let sum = p * a5 + m * (b5 + 1);
  if (!normalize) {
    return clamp(sum >> vec3<u32>(5u), vec3<i32>(0), vec3<i32>(255));
  }
  return clamp(sum / ((a5 & ~3) + (b5 & ~3) + 4), vec3<i32>(0), vec3<i32>(255));
}

// ------------------------------------------------------------------ depth

fn z_decompress(c: u32) -> u32 {
  let e = (c >> 11u) & 7u;
  let m = c & 0x7FFu;
  switch (e) {
    case 0u: { return m << 6u; }
    case 1u: { return 0x20000u + (m << 5u); }
    case 2u: { return 0x30000u + (m << 4u); }
    case 3u: { return 0x38000u + (m << 3u); }
    case 4u: { return 0x3C000u + (m << 2u); }
    case 5u: { return 0x3E000u + (m << 1u); }
    case 6u: { return 0x3F000u + m; }
    default: { return 0x3F800u + m; }
  }
}

fn z_compress(z: u32) -> u32 {
  if (z < 0x20000u) { return (z >> 6u); }
  if (z < 0x30000u) { return (1u << 11u) | ((z - 0x20000u) >> 5u); }
  if (z < 0x38000u) { return (2u << 11u) | ((z - 0x30000u) >> 4u); }
  if (z < 0x3C000u) { return (3u << 11u) | ((z - 0x38000u) >> 3u); }
  if (z < 0x3E000u) { return (4u << 11u) | ((z - 0x3C000u) >> 2u); }
  if (z < 0x3F000u) { return (5u << 11u) | ((z - 0x3E000u) >> 1u); }
  if (z < 0x3F800u) { return (6u << 11u) | (z - 0x3F000u); }
  return (7u << 11u) | ((z - 0x3F800u) & 0x7FFu);
}

// ------------------------------------------------------------------ framebuffer

fn hidden_bits(h: u32, j: u32) -> u32 {
  return (h >> (2u - j * 2u)) & 3u;
}

fn with_hidden_bits(h: u32, j: u32, v: u32) -> u32 {
  let sh = 2u - j * 2u;
  return (h & ~(3u << sh)) | ((v & 3u) << sh);
}

// The memory color as the blender reads it; alpha carries the stored coverage.
fn mem_color(j: u32, cvg: u32) -> vec4<i32> {
  let c = color[j];
  if (batch.ci_size == 3u) {
    return vec4<i32>(unpack_rgba(c).rgb, i32(cvg << 5u));
  }
  return vec4<i32>(i32((c >> 8u) & 0xF8u), i32((c >> 3u) & 0xF8u), i32((c << 2u) & 0xF8u), i32(cvg << 5u));
}

// Stored coverage minus one (0-7). Without image read the RDP assumes full.
fn mem_cvg(j: u32) -> u32 {
  if (((om_lo >> 6u) & 1u) == 0u) {
    return 7u;
  }
  if (batch.ci_size == 3u) {
    return (color[j] >> 5u) & 7u;
  }
  return ((color[j] & 1u) << 2u) | hidden_bits(color_hidden, j);
}

const MAGIC_SQUARE = array<i32, 16>(0, 6, 1, 7, 4, 2, 5, 3, 3, 5, 2, 4, 7, 1, 6, 0);
const BAYER = array<i32, 16>(0, 4, 1, 5, 4, 0, 5, 1, 3, 7, 2, 6, 7, 3, 6, 2);

// Color dither threshold (0-7); 7 never rounds up.
fn color_dither(x: u32, y: u32, noise_value: i32) -> i32 {
  let i = (y & 3u) * 4u + (x & 3u);
  switch ((om_hi >> 6u) & 3u) {
    case 0u: { return MAGIC_SQUARE[i]; }
    case 1u: { return BAYER[i]; }
    case 2u: { return noise_value & 7; }
    default: { return 7; }
  }
}

// Alpha dither offset (0-7): the color pattern, its inverse, noise or none.
fn alpha_dither(cdith: i32, noise_value: i32) -> i32 {
  switch ((om_hi >> 4u) & 3u) {
    case 0u: { return cdith; }
    case 1u: { return ~cdith & 7; }
    case 2u: { return (noise_value >> 3u) & 7; }
    default: { return 0; }
  }
}

// Rounds 8-bit color up to the next 5-bit step when its dropped bits exceed the
// dither threshold, so the VI dither filter can recover the lost precision.
fn dither(c: vec3<i32>, d: i32) -> vec3<i32> {
  if (batch.ci_size == 3u) {
    return c;
  }
  let up = (c & vec3<i32>(7)) > vec3<i32>(d);
  return min(select(c, (c & vec3<i32>(0xF8)) + 8, up), vec3<i32>(255));
}

// Writes a pixel with its coverage (0-7). 16-bit pixels keep coverage in the
// alpha bit and the hidden bits; 32-bit pixels keep the alpha the caller gives.
fn store_color(j: u32, c: vec4<i32>, cvg: u32) {
  color_dirty = true;
  let u = vec4<u32>(clamp(c, vec4<i32>(0), vec4<i32>(255)));
  if (batch.ci_size == 3u) {
    color[j] = (u.r << 24u) | (u.g << 16u) | (u.b << 8u) | u.a;
    return;
  }
  color[j] = ((u.r >> 3u) << 11u) | ((u.g >> 3u) << 6u) | ((u.b >> 3u) << 1u) | (cvg >> 2u);
  color_hidden = with_hidden_bits(color_hidden, j, cvg);
}

// ------------------------------------------------------------------ primitives

// 16.16 attribute `comp` of the coefficient block starting at prims[b].
fn attr(b: u32, comp: u32, row: u32) -> f32 {
  let wi = prims[b + row * 2u + (comp >> 1u)];
  let wf = prims[b + (row + 2u) * 2u + (comp >> 1u)];
  let sh = select(16u, 0u, (comp & 1u) == 1u);
  return f32(i32((((wi >> sh) & 0xFFFFu) << 16u) | ((wf >> sh) & 0xFFFFu)));
}

struct Raster {
  covered: bool,
  shade: vec4<i32>,
  s: f32,
  t: f32,
  z: u32,
  // Integer depth slope per pixel, used as the comparison tolerance.
  dz: u32,
  // Number of the 8 subsamples covered.
  cvg: u32,
  // Whether the first subsample is covered; it decides writes without AA.
  cvbit: bool,
};

// The RDP rounds the triangle's depth slope up to a power of two.
fn normalize_dzpix(sum: u32) -> u32 {
  if ((sum & 0xC000u) != 0u) {
    return 0x8000u;
  }
  if (sum == 0u) {
    return 1u;
  }
  return 2u << (31u - countLeadingZeros(sum));
}

// 4-bit depth slope code stored beside each Z value.
fn dz_compress(v: u32) -> u32 {
  return select(0u, 8u, (v & 0xFF00u) != 0u) | select(0u, 4u, (v & 0xF0F0u) != 0u) |
    select(0u, 2u, (v & 0xCCCCu) != 0u) | select(0u, 1u, (v & 0xAAAAu) != 0u);
}

fn noise(x: u32, y: u32) -> i32 {
  var h = (x * 0x27d4eb2du) ^ (y * 0x165667b1u);
  h = (h ^ (h >> 15u)) * 0x85ebca6bu;
  return i32((h >> 24u) & 255u);
}

fn rasterize_triangle(base: u32, x: u32, y: u32) -> Raster {
  var r: Raster;
  r.covered = false;
  let raw = base + 8u;
  let w0hi = prims[raw];
  let w0lo = prims[raw + 1u];
  let yl = sext(w0hi & 0x3FFFu, 14u);
  let ym = sext((w0lo >> 16u) & 0x3FFFu, 14u);
  let yh = sext(w0lo & 0x3FFFu, 14u);
  let lft = ((w0hi >> 23u) & 1u) == 1u;
  let xl = f32(i32(prims[raw + 2u]));
  let dxl = f32(i32(prims[raw + 3u]));
  let xh = f32(i32(prims[raw + 4u]));
  let dxh = f32(i32(prims[raw + 5u]));
  let xm = f32(i32(prims[raw + 6u]));
  let dxm = f32(i32(prims[raw + 7u]));
  let ytop = yh & ~3;
  var cvg = 0u;
  for (var sub = 0; sub < 4; sub++) {
    let k = i32(y) * 4 + sub;
    if (k < yh || k >= yl) {
      continue;
    }
    let fk = f32(k - ytop) * 0.25;
    let xhk = (xh + dxh * fk) / 65536.0;
    var xmk: f32;
    if (k < ym) {
      xmk = (xm + dxm * fk) / 65536.0;
    } else {
      xmk = (xl + dxl * f32(k - ym) * 0.25) / 65536.0;
    }
    let left = select(xmk, xhk, lft);
    let right = select(xhk, xmk, lft);
    let off = select(0.0, 0.25, (sub & 1) == 1);
    let sx = f32(x) + off;
    let first = sx >= left && sx < right;
    r.cvbit = r.cvbit || (sub == 0 && first);
    cvg += select(0u, 1u, first);
    cvg += select(0u, 1u, sx + 0.5 >= left && sx + 0.5 < right);
  }
  if (cvg == 0u) {
    return r;
  }
  r.covered = true;
  r.cvg = cvg;

  let flags = (prims[base] >> 4u) & 7u;
  let dl = f32(i32(y) - (ytop >> 2u));
  let xh_line = (xh + dxh * f32(i32(y) * 4 - ytop) * 0.25) / 65536.0;
  let dx = f32(x) - xh_line;
  var block = raw + 8u;
  r.shade = vec4<i32>(0);
  if ((flags & 4u) != 0u) {
    var c: array<i32, 4>;
    for (var i = 0u; i < 4u; i++) {
      let v = attr(block, i, 0u) + attr(block, i, 4u) * dl + attr(block, i, 1u) * dx;
      c[i] = clamp(i32(floor(v / 65536.0)), 0, 255);
    }
    r.shade = vec4<i32>(c[0], c[1], c[2], c[3]);
    block += 16u;
  }
  if ((flags & 2u) != 0u) {
    let s = attr(block, 0u, 0u) + attr(block, 0u, 4u) * dl + attr(block, 0u, 1u) * dx;
    let t = attr(block, 1u, 0u) + attr(block, 1u, 4u) * dl + attr(block, 1u, 1u) * dx;
    if (((om_hi >> 19u) & 1u) == 1u) {
      let w = attr(block, 2u, 0u) + attr(block, 2u, 4u) * dl + attr(block, 2u, 1u) * dx;
      // S and T are pre-multiplied by W, which is normalized to 0x7FFF.
      let inv = 32768.0 / select(w, 1.0, w == 0.0);
      r.s = s * inv;
      r.t = t * inv;
    } else {
      r.s = s / 65536.0;
      r.t = t / 65536.0;
    }
    block += 16u;
  }
  if ((flags & 1u) != 0u) {
    let z0 = f32(i32(prims[block]));
    let dzdx = f32(i32(prims[block + 1u]));
    let dzde = f32(i32(prims[block + 2u]));
    let dzdy = f32(i32(prims[block + 3u]));
    let z = z0 + dzde * dl + dzdx * dx;
    r.dz = normalize_dzpix(u32((abs(dzdx) + abs(dzdy)) / 65536.0) & 0xFFFFu);
    // Z is s15.16; the 18-bit depth keeps the integer part and 3 fraction bits.
    r.z = u32(clamp(i32(floor(z / 8192.0)), 0, 0x3FFFF));
  }
  return r;
}

fn rasterize_rect(base: u32, x: u32, y: u32, cycle: u32) -> Raster {
  var r: Raster;
  let raw = base + 8u;
  let xl = i32(prims[raw]);
  let yl = i32(prims[raw + 1u]);
  let xh = i32(prims[raw + 2u]);
  let yh = i32(prims[raw + 3u]);
  let px = i32(x);
  let py = i32(y);
  if (cycle >= CYCLE_COPY) {
    r.covered = px >= (xh >> 2u) && px <= (xl >> 2u) && py >= (yh >> 2u) && py <= (yl >> 2u);
  } else {
    r.covered = px * 4 >= xh && px * 4 < xl && py * 4 >= yh && py * 4 < yl;
  }
  r.cvg = 8u;
  r.cvbit = r.covered;
  r.shade = vec4<i32>(0);
  r.z = (states[st + 9u] >> 16u) << 3u;
  r.dz = states[st + 9u] & 0xFFFFu;
  if ((prims[base] & 15u) == PRIM_TEXRECT) {
    let dx = f32(px - (xh >> 2u));
    let dy = f32(py - (yh >> 2u));
    let s = f32(i32(prims[raw + 4u]));
    let t = f32(i32(prims[raw + 5u]));
    var dsdx = f32(i32(prims[raw + 6u]));
    let dtdy = f32(i32(prims[raw + 7u]));
    if (cycle == CYCLE_COPY) {
      dsdx = dsdx / 4.0;
    }
    if (((prims[base] >> 7u) & 1u) == 1u) {
      r.s = s + dsdx * dy / 32.0;
      r.t = t + dtdy * dx / 32.0;
    } else {
      r.s = s + dsdx * dx / 32.0;
      r.t = t + dtdy * dy / 32.0;
    }
  }
  return r;
}

struct Depth {
  ok: bool,
  // Within the slope tolerance of memory or behind it, i.e. the same surface.
  farther: bool,
  cvg: u32,
};

// Hardware depth compare. The tolerance is the larger of the pixel's and the
// stored slope, and interior pixels require being strictly in front.
fn depth_test(j: u32, sz: u32, dzpix: u32, overflow: bool, cvg: u32) -> Depth {
  var d: Depth;
  d.cvg = cvg;
  let zword = select(zpair & 0xFFFFu, zpair >> 16u, j == 0u);
  let oz = z_decompress(zword >> 2u);
  var dzmem = 1u << (((zword & 3u) << 2u) | hidden_bits(z_hidden, j));
  var coplanar = false;
  let exponent = (zword >> 13u) & 15u;
  if (exponent < 3u) {
    if (dzmem != 0x8000u) {
      dzmem = max(dzmem << 1u, 16u >> exponent);
    } else {
      coplanar = true;
      dzmem = 0xFFFFu;
    }
  }
  let dznew = 1u << (31u - countLeadingZeros(dzpix | dzmem));
  let dz = i32(dznew << 3u);
  let z = i32(sz);
  let o = i32(oz);
  d.farther = coplanar || z + dz >= o;
  let nearer = coplanar || z - dz <= o;
  let infront = z < o;
  let max_z = oz == 0x3FFFFu;
  switch ((om_lo >> 10u) & 3u) {
    // Interpenetrating: an edge crossing memory gets coverage by depth overlap.
    case 1u: {
      if (infront && d.farther && overflow) {
        let enc = dz_compress(dznew & 0xFFFFu);
        d.cvg = (((((oz >> enc) - (sz >> enc)) & 15u) * cvg) >> 3u) & 15u;
        d.ok = true;
      } else {
        d.ok = max_z || select(nearer, infront, overflow);
      }
    }
    // Transparent
    case 2u: { d.ok = max_z || infront; }
    // Decal
    case 3u: { d.ok = d.farther && nearer && !max_z; }
    // Opaque
    default: { d.ok = max_z || select(nearer, infront, overflow); }
  }
  return d;
}

fn draw(index: u32, x: u32, y: u32, j: u32) {
  let base = index * PRIM_WORDS;
  if (x < prims[base + 3u] || x >= prims[base + 5u] || y < prims[base + 4u] || y >= prims[base + 6u]) {
    return;
  }
  let header = prims[base];
  let kind = header & 15u;
  st = prims[base + 1u] * STATE_WORDS;
  tm = prims[base + 2u] * 1024u;
  om_hi = states[st];
  om_lo = states[st + 1u];
  let cycle = (om_hi >> 20u) & 3u;

  if (cycle == CYCLE_FILL) {
    if (kind == PRIM_TRI) {
      return;
    }
    let fill = states[st + 4u];
    if (batch.ci_size == 3u) {
      color[j] = fill;
    } else {
      color[j] = select(fill & 0xFFFFu, fill >> 16u, j == 0u);
      color_hidden = with_hidden_bits(color_hidden, j, select(0u, 3u, (color[j] & 1u) == 1u));
    }
    color_dirty = true;
    return;
  }

  var r: Raster;
  if (kind == PRIM_TRI) {
    r = rasterize_triangle(base, x, y);
  } else {
    r = rasterize_rect(base, x, y, cycle);
  }
  if (!r.covered) {
    return;
  }
  let tile = (header >> 8u) & 7u;
  let alpha_compare = (om_lo & 1u) == 1u;
  let blend_color = unpack_rgba(states[st + 6u]);

  if (cycle == CYCLE_COPY) {
    let c = sample_tile(tile, r.s, r.t);
    if (alpha_compare && c.a == 0) {
      return;
    }
    store_color(j, c, select(0u, 7u, c.a >= 128));
    return;
  }

  var v: Inputs;
  v.prim = unpack_rgba(states[st + 7u]);
  v.env = unpack_rgba(states[st + 8u]);
  v.shade = r.shade;
  v.noise = noise(x, y);
  v.lod_frac = 0;
  v.prim_lod = i32(states[st + 10u] & 255u);
  v.combined = vec4<i32>(0);
  if (((header >> 4u) & 2u) != 0u) {
    v.tex0 = sample_tile(tile, r.s, r.t);
    if (cycle == CYCLE_2) {
      v.tex1 = sample_tile(tile + 1u, r.s, r.t);
    } else {
      v.tex1 = v.tex0;
    }
  }
  var c = combine(0u, v);
  if (cycle == CYCLE_2) {
    v.combined = c;
    c = combine(1u, v);
  }

  // Coverage can be scaled by alpha, and alpha can be replaced by coverage.
  var cvg = r.cvg;
  let cvg_x_alpha = ((om_lo >> 12u) & 1u) == 1u;
  let scaled = (c.a * i32(cvg) + 4) >> 3u;
  if (cvg_x_alpha) {
    cvg = u32(scaled >> 5u) & 15u;
  }
  let cdith = color_dither(x, y, v.noise);
  let adith = alpha_dither(cdith, v.noise);
  if (((om_lo >> 13u) & 1u) == 1u) {
    c.a = min(select(i32(cvg) << 5u, scaled, cvg_x_alpha), 255);
  } else {
    c.a = min(c.a + adith, 255);
  }
  let shade_a = min(r.shade.a + adith, 255);

  if (alpha_compare) {
    let threshold = select(blend_color.a, v.noise, ((om_lo >> 1u) & 1u) == 1u);
    if (c.a < threshold) {
      return;
    }
  }

  let aa = ((om_lo >> 3u) & 1u) == 1u;
  let force_blend = ((om_lo >> 14u) & 1u) == 1u;
  let memcvg = mem_cvg(j);
  // Coverage overflowing a full pixel marks an interior pixel, which replaces
  // memory; partial coverage over the same surface blends as an edge.
  let overflow = ((memcvg + cvg) & 8u) != 0u;
  let prim_depth = ((om_lo >> 2u) & 1u) == 1u;
  let zsrc = select(r.z, (states[st + 9u] >> 16u) << 3u, prim_depth);
  let dzpix = select(r.dz, states[st + 9u] & 0xFFFFu, prim_depth);
  var farther = true;
  if (((om_lo >> 4u) & 1u) == 1u) {
    let d = depth_test(j, zsrc, dzpix, overflow, cvg);
    if (!d.ok) {
      return;
    }
    farther = d.farther;
    cvg = d.cvg;
  }
  if (!select(r.cvbit, cvg != 0u, aa)) {
    return;
  }
  let blend_en = force_blend || (!overflow && aa && farther);

  let mem = mem_color(j, memcvg);
  let final_cycle = select(0u, 1u, cycle == CYCLE_2);
  var out = c.rgb;
  if (cycle == CYCLE_2) {
    out = blend(0u, out, c.a, shade_a, mem, false);
  }
  let sel = blend_sel(final_cycle);
  if (((om_lo >> 7u) & 1u) == 1u && !overflow) {
    // Color on coverage: only the coverage changes.
    out = blend_rgb(sel.z, out, mem.rgb);
  } else if (!blend_en || (sel.y == 0u && sel.w == 0u && c.a >= 255)) {
    out = blend_rgb(sel.x, out, mem.rgb);
  } else {
    out = blend(final_cycle, out, c.a, shade_a, mem, !force_blend);
  }

  var final_cvg: u32;
  switch ((om_lo >> 8u) & 3u) {
    // Clamp
    case 0u: { final_cvg = min(select(cvg - 1u, cvg + memcvg, blend_en), 7u); }
    // Wrap
    case 1u: { final_cvg = (cvg + memcvg) & 7u; }
    // Zap
    case 2u: { final_cvg = 7u; }
    // Save
    default: { final_cvg = memcvg; }
  }
  store_color(j, vec4<i32>(dither(out, cdith), i32(final_cvg << 5u)), final_cvg);

  if (((om_lo >> 5u) & 1u) == 1u) {
    let enc = dz_compress(dzpix);
    let nz = ((z_compress(zsrc) << 2u) | (enc >> 2u)) & 0xFFFFu;
    if (j == 0u) {
      zpair = (zpair & 0xFFFFu) | (nz << 16u);
    } else {
      zpair = (zpair & 0xFFFF0000u) | nz;
    }
    z_hidden = with_hidden_bits(z_hidden, j, enc);
    z_dirty = true;
  }
}

// ------------------------------------------------------------------ entry

@compute @workgroup_size(8, 8)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let bx0 = (batch.xy0 & 0xFFFFu) & ~1u;
  let by0 = batch.xy0 >> 16u;
  let bx1 = batch.xy1 & 0xFFFFu;
  let by1 = batch.xy1 >> 16u;
  let tx0 = bx0 + wg.x * 16u;
  let ty0 = by0 + wg.y * 8u;
  let px = tx0 + lid.x * 2u;
  let py = ty0 + lid.y;
  let width = batch.ci_width;
  let in_bounds = px < width && px < bx1 && py < by1;

  let pix = py * width + px;
  let wide = batch.ci_size == 3u;
  let caddr = select((batch.ci_addr + pix * 2u) >> 2u, (batch.ci_addr + pix * 4u) >> 2u, wide) & RDRAM_MASK;
  let zaddr = ((batch.z_addr + pix * 2u) >> 2u) & RDRAM_MASK;
  if (in_bounds) {
    if (wide) {
      color[0] = rdram[caddr];
      color[1] = rdram[(caddr + 1u) & RDRAM_MASK];
    } else {
      let w = rdram[caddr];
      color[0] = w >> 16u;
      color[1] = w & 0xFFFFu;
      color_hidden = rdram[HIDDEN + caddr];
    }
    zpair = rdram[zaddr];
    z_hidden = rdram[HIDDEN + zaddr];
  }

  var chunk = 0u;
  loop {
    if (chunk >= batch.count) {
      break;
    }
    let pi = chunk + li;
    var h = 0u;
    if (pi < batch.count) {
      let base = (batch.first + pi) * PRIM_WORDS;
      let overlap = prims[base + 3u] < tx0 + 16u && prims[base + 5u] > tx0 && prims[base + 4u] < ty0 + 8u && prims[base + 6u] > ty0;
      h = select(0u, 1u, overlap);
    }
    hit[li] = h;
    workgroupBarrier();
    let n = min(64u, batch.count - chunk);
    for (var k = 0u; k < n; k++) {
      if (hit[k] != 0u && in_bounds) {
        draw(batch.first + chunk + k, px, py, 0u);
        draw(batch.first + chunk + k, px + 1u, py, 1u);
      }
    }
    workgroupBarrier();
    chunk += 64u;
  }

  if (in_bounds && color_dirty) {
    if (wide) {
      rdram[caddr] = color[0];
      if (px + 1u < width) {
        rdram[(caddr + 1u) & RDRAM_MASK] = color[1];
      }
    } else {
      rdram[caddr] = (color[0] << 16u) | (color[1] & 0xFFFFu);
      rdram[HIDDEN + caddr] = color_hidden;
    }
  }
  if (in_bounds && z_dirty) {
    rdram[zaddr] = zpair;
    rdram[HIDDEN + zaddr] = z_hidden;
  }
}
