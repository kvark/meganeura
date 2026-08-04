struct Params {
    total: u32,
    inner: u32,
    pairs: u32,
    _pad: u32,
}

var<storage> src_a: array<f32>;
var<storage> src_b: array<f32>;
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pair_index = gid.x;
    if (pair_index >= params.total) {
        return;
    }

    let row = pair_index / params.pairs;
    let vector_begin = pair_index * params.inner;
    let direction_begin = row * params.inner;
    var component = 0.0;
    for (var column = 0u; column < params.inner; column += 1u) {
        component += src_a[vector_begin + column] * src_b[direction_begin + column];
    }
    for (var column = 0u; column < params.inner; column += 1u) {
        let projection = src_b[direction_begin + column] * component;
        dst[vector_begin + column] = src_a[vector_begin + column] + (-projection);
    }
}
