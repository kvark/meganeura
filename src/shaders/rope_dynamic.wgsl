// RoPE with dynamic position offset read from a storage buffer.
// Same as rope.wgsl but pos_offset comes from a u32 buffer instead of params.

struct Params {
    seq: u32,
    dim: u32,
    theta_bits: u32,
    // Static offset added to the dynamic one (zero from the builders).
    pos_offset: u32,
    head_dim: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

var<storage> src: array<f32>;
var<storage, read_write> dst: array<f32>;
var<storage> pos_offset_buf: array<u32>;
var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let half_dim = params.dim / 2u;
    let total = params.seq * half_dim;
    if i >= total { return; }

    let row = i / half_dim;
    let pos = row + params.pos_offset + pos_offset_buf[0];
    let pair_in_row = i % half_dim;
    let theta = bitcast<f32>(params.theta_bits);

    // Apply RoPE per-head
    let half_head = params.head_dim / 2u;
    let head = pair_in_row / half_head;
    let pair_in_head = pair_in_row % half_head;

    let exponent = -2.0 * f32(pair_in_head) / f32(params.head_dim);
    let inv_freq = pow(theta, exponent);
    let angle = f32(pos) * inv_freq;
    let cos_val = cos(angle);
    let sin_val = sin(angle);

    // HuggingFace "half-split" convention within each head
    let base = row * params.dim + head * params.head_dim;
    let idx0 = base + pair_in_head;
    let idx1 = base + pair_in_head + half_head;
    let v0 = src[idx0];
    let v1 = src[idx1];

    dst[idx0] = v0 * cos_val - v1 * sin_val;
    dst[idx1] = v0 * sin_val + v1 * cos_val;
}

// The storage binding is shared with the dynamic-offset entry point; this
// entry interprets it as one absolute position per row.
@compute @workgroup_size(256)
fn with_positions(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let half_dim = params.dim / 2u;
    let total = params.seq * half_dim;
    if i >= total { return; }

    let row = i / half_dim;
    let pos = pos_offset_buf[row];
    let pair_in_row = i % half_dim;
    let theta = bitcast<f32>(params.theta_bits);

    let half_head = params.head_dim / 2u;
    let head = pair_in_row / half_head;
    let pair_in_head = pair_in_row % half_head;
    let exponent = -2.0 * f32(pair_in_head) / f32(params.head_dim);
    let angle = f32(pos) * pow(theta, exponent);
    let cos_val = cos(angle);
    let sin_val = sin(angle);

    let base = row * params.dim + head * params.head_dim;
    let idx0 = base + pair_in_head;
    let idx1 = idx0 + half_head;
    let v0 = src[idx0];
    let v1 = src[idx1];
    dst[idx0] = v0 * cos_val - v1 * sin_val;
    dst[idx1] = v0 * sin_val + v1 * cos_val;
}


// The freq-factor form: a per-pair divisor on top of the uniform base, as
// GGML's rope consumes the `rope_freqs` tensor — pair i's angle runs at
// `pos * inv_freq_i / factors[i]`, so a factor of 1.0 keeps the standard
// rotation and a huge one freezes the pair. One factor per pair, in the
// pair layout the base-derived rotation already uses.
var<storage> factors: array<f32>;

@compute @workgroup_size(256)
fn with_factors(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let half_dim = params.dim / 2u;
    let total = params.seq * half_dim;
    if i >= total { return; }

    let row = i / half_dim;
    let pos = row + params.pos_offset + pos_offset_buf[0];
    let pair_in_row = i % half_dim;
    let theta = bitcast<f32>(params.theta_bits);

    let half_head = params.head_dim / 2u;
    let head = pair_in_row / half_head;
    let pair_in_head = pair_in_row % half_head;
    let exponent = -2.0 * f32(pair_in_head) / f32(params.head_dim);
    let inv_freq = pow(theta, exponent) / factors[pair_in_head];
    let angle = f32(pos) * inv_freq;
    let cos_val = cos(angle);
    let sin_val = sin(angle);

    let base = row * params.dim + head * params.head_dim;
    let idx0 = base + pair_in_head;
    let idx1 = idx0 + half_head;
    let v0 = src[idx0];
    let v1 = src[idx1];
    dst[idx0] = v0 * cos_val - v1 * sin_val;
    dst[idx1] = v0 * sin_val + v1 * cos_val;
}
