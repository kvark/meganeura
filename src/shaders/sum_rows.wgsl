struct Params {
    m: u32,
    n: u32,
    _pad0: u32,
    _pad1: u32,
}

var<storage> src: array<f32>;
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;

// Sum across rows: dst[col] = sum over row of src[row * n + col]
@compute @workgroup_size(256)
fn sum_rows(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    if col >= params.n { return; }
    var acc = 0.0;
    for (var row = 0u; row < params.m; row++) {
        acc += src[row * params.n + col];
    }
    dst[col] = acc;
}
