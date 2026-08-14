struct Params {
    m: u32,
    n: u32,
    _pad0: u32,
    _pad1: u32,
}

var<storage> src: array<f32>;
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    if row >= params.m || col >= params.n { return; }
    // dst[col * m + row] = src[row * n + col]
    dst[col * params.m + row] = src[row * params.n + col];
}
