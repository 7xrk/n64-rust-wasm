// VI scanout in two passes: `fs_filter` applies the VI's antialiasing, dither and
// divot filters at framebuffer resolution, `fs_present` resamples it to the output.

struct Vi {
  origin: u32,
  width: u32,
  ctrl: u32,
  pad: u32,
  src_size: vec2<f32>,
};

const RDRAM_MASK: u32 = 0x1FFFFFu;
// Hidden bits follow RDRAM; see rdp.wgsl.
const HIDDEN: u32 = 0x200000u;

@group(0) @binding(0) var<storage, read> rdram: array<u32>;
@group(0) @binding(1) var<uniform> vi: Vi;
@group(0) @binding(2) var filtered: texture_2d<f32>;
@group(0) @binding(3) var linear_sampler: sampler;
@group(0) @binding(4) var nearest_sampler: sampler;

struct VsOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VsOut {
  let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
  var o: VsOut;
  o.pos = vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
  o.uv = vec2<f32>(p.x, 1.0 - p.y);
  return o;
}

struct Pixel {
  rgb: vec3<i32>,
  // Coverage minus one; 7 is a fully covered pixel.
  cvg: u32,
};

fn fetch(p: vec2<i32>) -> Pixel {
  let q = clamp(p, vec2<i32>(0), vec2<i32>(i32(vi.width), i32(ceil(vi.src_size.y))) - 1);
  let i = u32(q.y) * vi.width + u32(q.x);
  if ((vi.ctrl & 3u) == 3u) {
    let w = rdram[((vi.origin + i * 4u) >> 2u) & RDRAM_MASK];
    return Pixel(vec3<i32>(vec3<u32>(w >> 24u, w >> 16u, w >> 8u) & vec3<u32>(255u)), (w >> 5u) & 7u);
  }
  let a = vi.origin + i * 2u;
  let word = (a >> 2u) & RDRAM_MASK;
  let second = (a & 2u) == 2u;
  let c = select(rdram[word] >> 16u, rdram[word] & 0xFFFFu, second);
  let hidden = (rdram[HIDDEN + word] >> select(2u, 0u, second)) & 3u;
  return Pixel(vec3<i32>(vec3<u32>(c >> 8u, c >> 3u, c << 2u) & vec3<u32>(0xF8u)), ((c & 1u) << 2u) | hidden);
}

// Dither filter: each neighbor brighter or darker than the center moves it by
// one 8-bit step, recovering the precision the RDP dithered away.
fn restore(p: vec2<i32>, center: vec3<i32>) -> vec3<i32> {
  var sum = center;
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      if (dx != 0 || dy != 0) {
        sum += sign((fetch(p + vec2<i32>(dx, dy)).rgb >> vec3<u32>(3u)) - (center >> vec3<u32>(3u)));
      }
    }
  }
  return sum & vec3<i32>(255);
}

const AA_NEIGHBORS = array<vec2<i32>, 6>(
  vec2<i32>(-1, -1), vec2<i32>(1, -1), vec2<i32>(-2, 0), vec2<i32>(2, 0), vec2<i32>(-1, 1), vec2<i32>(1, 1)
);

// Edge antialiasing: the background behind a partially covered pixel is
// estimated from fully covered neighbors (penultimate max + min - center) and
// mixed in by the missing coverage.
fn antialias(p: vec2<i32>, center: Pixel) -> vec3<i32> {
  var max1 = center.rgb;
  var max2 = center.rgb;
  var min1 = center.rgb;
  var min2 = center.rgb;
  var count = 1;
  for (var k = 0; k < 6; k++) {
    let n = fetch(p + AA_NEIGHBORS[k]);
    if (n.cvg != 7u) {
      continue;
    }
    let v = n.rgb;
    if (count == 1) {
      max2 = min(max1, v);
      max1 = max(max1, v);
      min2 = max(min1, v);
      min1 = min(min1, v);
    } else {
      max2 = select(max(max2, v), max1, v > max1);
      max1 = max(max1, v);
      min2 = select(min(min2, v), min1, v < min1);
      min1 = min(min1, v);
    }
    count++;
  }
  let delta = max2 + min2 - center.rgb * 2;
  return (((delta * i32(7u - center.cvg) + 4) >> vec3<u32>(3u)) + center.rgb) & vec3<i32>(255);
}

fn filter_pixel(p: vec2<i32>) -> Pixel {
  var px = fetch(p);
  // AA modes 2 and 3 ignore coverage.
  if (((vi.ctrl >> 9u) & 1u) == 1u) {
    px.cvg = 7u;
  }
  if (px.cvg != 7u) {
    px.rgb = antialias(p, px);
  } else if ((vi.ctrl & 0x10000u) != 0u) {
    px.rgb = restore(p, px.rgb);
  }
  return px;
}

fn median3(a: vec3<i32>, b: vec3<i32>, c: vec3<i32>) -> vec3<i32> {
  return max(min(a, b), min(max(a, b), c));
}

@fragment
fn fs_filter(in: VsOut) -> @location(0) vec4<f32> {
  if ((vi.ctrl & 3u) < 2u || vi.width == 0u) {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
  }
  let p = vec2<i32>(in.pos.xy);
  let center = filter_pixel(p);
  var rgb = center.rgb;
  // Divot filter: a horizontal median removes the one-pixel notches the
  // antialiasing filter leaves along edges.
  if ((vi.ctrl & 0x10u) != 0u) {
    let left = filter_pixel(p - vec2<i32>(1, 0));
    let right = filter_pixel(p + vec2<i32>(1, 0));
    if ((center.cvg & left.cvg & right.cvg) != 7u) {
      rgb = median3(left.rgb, rgb, right.rgb);
    }
  }
  return vec4<f32>(vec3<f32>(rgb) / 255.0, 1.0);
}

// Gamma correction is a square root curve; gamma dither adds 6 bits of noise
// below the 8-bit input before taking the root.
fn gamma(rgb: vec3<f32>, pos: vec2<f32>) -> vec3<f32> {
  let c = vec3<u32>(round(rgb * 255.0));
  var h = (u32(pos.x) * 0x27d4eb2du) ^ (u32(pos.y) * 0x165667b1u);
  h = (h ^ (h >> 15u)) * 0x85ebca6bu;
  let noise = select(0u, h >> 26u, (vi.ctrl & 4u) != 0u);
  let g = floor(sqrt(vec3<f32>((c << vec3<u32>(6u)) | vec3<u32>(noise)))) * 2.0;
  return min(g, vec3<f32>(255.0)) / 255.0;
}

@fragment
fn fs_present(in: VsOut) -> @location(0) vec4<f32> {
  if ((vi.ctrl & 3u) < 2u || vi.width == 0u) {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
  }
  let uv = in.uv * vi.src_size / vec2<f32>(textureDimensions(filtered));
  // AA mode 3 replicates pixels; the other modes resample bilinearly.
  var rgb: vec3<f32>;
  if (((vi.ctrl >> 8u) & 3u) == 3u) {
    rgb = textureSampleLevel(filtered, nearest_sampler, uv, 0.0).rgb;
  } else {
    rgb = textureSampleLevel(filtered, linear_sampler, uv, 0.0).rgb;
  }
  if ((vi.ctrl & 8u) != 0u) {
    rgb = gamma(rgb, in.pos.xy);
  }
  return vec4<f32>(rgb, 1.0);
}
