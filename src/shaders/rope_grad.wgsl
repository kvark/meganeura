struct Params {
    seq: u32,
    dim: u32,
    theta_bits: u32,
    pos_offset: u32,
    head_dim: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

var<storage> src: array<f32>;       // grad_output
var<storage, read_write> dst: array<f32>;  // grad_input
var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let half_dim = params.dim / 2u;
    let total = params.seq * half_dim;
    if i >= total { return; }

    let row = i / half_dim;
    let pos = row + params.pos_offset;
    let pair_in_row = i % half_dim;
    let theta = bitcast<f32>(params.theta_bits);

    let half_head = params.head_dim / 2u;
    let head = pair_in_row / half_head;
    let pair_in_head = pair_in_row % half_head;

    let exponent = -2.0 * f32(pair_in_head) / f32(params.head_dim);
    let inv_freq = pow(theta, exponent);
    let angle = f32(pos) * inv_freq;
    let cos_val = cos(angle);
    let sin_val = sin(angle);

    // Same index layout as forward
    let base = row * params.dim + head * params.head_dim;
    let idx0 = base + pair_in_head;
    let idx1 = base + pair_in_head + half_head;
    let g0 = src[idx0];
    let g1 = src[idx1];

    // Inverse rotation (transpose of rotation matrix):
    // grad_x0 = grad_y0 * cos + grad_y1 * sin
    // grad_x1 = -grad_y0 * sin + grad_y1 * cos
    dst[idx0] = g0 * cos_val + g1 * sin_val;
    dst[idx1] = -g0 * sin_val + g1 * cos_val;
}
