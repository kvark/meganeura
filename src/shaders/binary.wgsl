struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

var<storage> src_a: array<f32>;
var<storage> src_b: array<f32>;
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;

@compute @workgroup_size(256)
fn add(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.len { return; }
    dst[i] = src_a[i] + src_b[i];
}

@compute @workgroup_size(256)
fn mul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.len { return; }
    dst[i] = src_a[i] * src_b[i];
}

@compute @workgroup_size(256)
fn greater(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.len { return; }
    dst[i] = select(0.0, 1.0, src_a[i] > src_b[i]);
}

// swiglu: silu(gate) * up = (gate / (1 + exp(-gate))) * up
@compute @workgroup_size(256)
fn swiglu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.len { return; }
    let gate = src_a[i];
    let up = src_b[i];
    let sig = 1.0 / (1.0 + exp(-gate));
    dst[i] = gate * sig * up;
}
