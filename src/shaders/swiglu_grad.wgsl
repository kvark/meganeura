struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

var<storage> src_a: array<f32>;
var<storage> src_b: array<f32>;
var<storage> src_c: array<f32>;
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;

// dsilu(g) = sig(g) + silu(g) * (1 - sig(g))
// swiglu_grad_gate: dst[i] = grad_out[i] * up[i] * dsilu(gate[i])
// src_a = grad_out, src_b = gate, src_c = up
@compute @workgroup_size(256)
fn swiglu_grad_gate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.len { return; }
    let grad_out = src_a[i];
    let gate = src_b[i];
    let up = src_c[i];
    let sig = 1.0 / (1.0 + exp(-gate));
    let silu_g = gate * sig;
    let dsilu_g = sig + silu_g * (1.0 - sig);
    dst[i] = grad_out * up * dsilu_g;
}

// swiglu_grad_up: dst[i] = grad_out[i] * silu(gate[i])
// src_a = grad_out, src_b = gate
@compute @workgroup_size(256)
fn swiglu_grad_up(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.len { return; }
    let grad_out = src_a[i];
    let gate = src_b[i];
    let sig = 1.0 / (1.0 + exp(-gate));
    dst[i] = grad_out * gate * sig;
}

// silu_grad: dst[i] = grad_out[i] * dsilu(x[i])
// dsilu(x) = sig(x) + x * sig(x) * (1 - sig(x))
// src_a = grad_out, src_b = x
@compute @workgroup_size(256)
fn silu_grad(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.len { return; }
    let grad_out = src_a[i];
    let x = src_b[i];
    let sig = 1.0 / (1.0 + exp(-x));
    let dsilu = sig + x * sig * (1.0 - sig);
    dst[i] = grad_out * dsilu;
}
