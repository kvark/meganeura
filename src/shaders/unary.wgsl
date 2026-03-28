struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

var<storage> src: array<f32>;
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;

@compute @workgroup_size(256)
fn relu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.len { return; }
    dst[i] = max(src[i], 0.0);
}

@compute @workgroup_size(256)
fn sigmoid(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.len { return; }
    let val = src[i];
    dst[i] = 1.0 / (1.0 + exp(-val));
}

@compute @workgroup_size(256)
fn neg(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.len { return; }
    dst[i] = -src[i];
}

// silu: x * sigmoid(x) = x / (1 + exp(-x))
@compute @workgroup_size(256)
fn silu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.len { return; }
    let val = src[i];
    dst[i] = val / (1.0 + exp(-val));
}

// gelu approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
@compute @workgroup_size(256)
fn gelu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.len { return; }
    let x = src[i];
    let x3 = x * x * x;
    let inner = 0.7978845608 * (x + 0.044715 * x3);
    dst[i] = 0.5 * x * (1.0 + tanh(inner));
}
