// SwiGLUConcat: input[M, 2*N] → output[M, N]
// gate = input[:, :N], up = input[:, N:]
// output = silu(gate) * up
//
// Forward bindings: src (input), dst (output), params
// Params: len = M*N (output elements), half_n = N

struct Params {
    len: u32,
    half_n: u32,
    _pad1: u32,
    _pad2: u32,
}

var<storage> src_a: array<f32>;
var<storage> src_b: array<f32>;
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;

// Forward: src_a = input[M, 2*N], dst = output[M, N]
@compute @workgroup_size(256)
fn swiglu_concat(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.len { return; }
    let row = i / params.half_n;
    let col = i % params.half_n;
    let wide = params.half_n * 2u;
    let gate = src_a[row * wide + col];
    let up = src_a[row * wide + params.half_n + col];
    let sig = 1.0 / (1.0 + exp(-gate));
    dst[i] = gate * sig * up;
}

// Backward: src_a = input[M, 2*N], src_b = grad_out[M, N], dst = grad_input[M, 2*N]
// d_gate = grad_out * up * dsilu(gate)
// d_up   = grad_out * silu(gate)
@compute @workgroup_size(256)
fn swiglu_concat_grad(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.len { return; }
    let row = i / params.half_n;
    let col = i % params.half_n;
    let wide = params.half_n * 2u;
    let gate = src_a[row * wide + col];
    let up = src_a[row * wide + params.half_n + col];
    let grad_out = src_b[i];
    let sig = 1.0 / (1.0 + exp(-gate));
    let silu_g = gate * sig;
    let dsilu_g = sig + silu_g * (1.0 - sig);
    dst[row * wide + col] = grad_out * up * dsilu_g;
    dst[row * wide + params.half_n + col] = grad_out * silu_g;
}
