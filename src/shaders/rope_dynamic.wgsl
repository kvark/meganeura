// RoPE with dynamic position offset read from a storage buffer.
// Same as rope.wgsl but pos_offset comes from a u32 buffer instead of params.

struct Params {
    seq: u32,
    dim: u32,
    theta_bits: u32,
    _pad: u32,
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
    let pos = row + pos_offset_buf[0];
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
