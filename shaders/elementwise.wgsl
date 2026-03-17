// Element-wise operations on flat buffers.
// Each entry point operates on arrays of f32.

struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> input_a: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

// Secondary input for binary ops
@group(1) @binding(0) var<storage, read> input_b: array<f32>;

@compute @workgroup_size(256)
fn relu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.len { return; }
    output[i] = max(input_a[i], 0.0);
}

@compute @workgroup_size(256)
fn sigmoid(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.len { return; }
    output[i] = 1.0 / (1.0 + exp(-input_a[i]));
}

@compute @workgroup_size(256)
fn neg(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.len { return; }
    output[i] = -input_a[i];
}

@compute @workgroup_size(256)
fn add(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.len { return; }
    output[i] = input_a[i] + input_b[i];
}

@compute @workgroup_size(256)
fn mul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.len { return; }
    output[i] = input_a[i] * input_b[i];
}

@compute @workgroup_size(256)
fn greater(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.len { return; }
    output[i] = select(0.0, 1.0, input_a[i] > input_b[i]);
}

// Bias add: out[i] = a[i] + bias[i % bias_len]
// Params.len = total elements, bias uses input_b
struct BiasParams {
    len: u32,
    bias_len: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(3) var<uniform> bias_params: BiasParams;

@compute @workgroup_size(256)
fn bias_add(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= bias_params.len { return; }
    output[i] = input_a[i] + input_b[i % bias_params.bias_len];
}

// SGD update: param -= lr * grad
struct SgdParams {
    len: u32,
    lr: f32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(4) var<uniform> sgd_params: SgdParams;

@compute @workgroup_size(256)
fn sgd_update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= sgd_params.len { return; }
    // input_a = param (read), input_b = grad (read), output = param (write)
    output[i] = input_a[i] - sgd_params.lr * input_b[i];
}
