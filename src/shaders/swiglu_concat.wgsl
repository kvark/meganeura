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

// GeGLUConcat: same layout as SwiGLUConcat, gelu(gate) * up.
fn gelu_tanh(x: f32) -> f32 {
    let x3 = x * x * x;
    let inner = 0.7978845608 * (x + 0.044715 * x3);
    return 0.5 * x * (1.0 + tanh(inner));
}

fn dgelu_tanh(x: f32) -> f32 {
    let x2 = x * x;
    let x3 = x2 * x;
    let inner = 0.7978845608 * (x + 0.044715 * x3);
    let t = tanh(inner);
    let sech2 = 1.0 - t * t;
    let dinner = 0.7978845608 * (1.0 + 0.134145 * x2);
    return 0.5 * (1.0 + t) + 0.5 * x * sech2 * dinner;
}

@compute @workgroup_size(256)
fn geglu_concat(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.len { return; }
    let row = i / params.half_n;
    let col = i % params.half_n;
    let wide = params.half_n * 2u;
    let gate = src_a[row * wide + col];
    let up = src_a[row * wide + params.half_n + col];
    dst[i] = gelu_tanh(gate) * up;
}

@compute @workgroup_size(256)
fn geglu_concat_grad(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.len { return; }
    let row = i / params.half_n;
    let col = i % params.half_n;
    let wide = params.half_n * 2u;
    let gate = src_a[row * wide + col];
    let up = src_a[row * wide + params.half_n + col];
    let grad_out = src_b[i];
    dst[row * wide + col] = grad_out * up * dgelu_tanh(gate);
    dst[row * wide + params.half_n + col] = grad_out * gelu_tanh(gate);
}
