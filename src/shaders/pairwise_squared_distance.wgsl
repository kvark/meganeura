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
    let output_index = gid.x;
    if (output_index >= params.total) {
        return;
    }

    let row = output_index / params.pairs;
    let left_begin = row * params.inner;
    let right_begin = output_index * params.inner;
    var sum = 0.0;
    for (var column = 0u; column < params.inner; column += 1u) {
        let delta = src_a[left_begin + column] - src_b[right_begin + column];
        sum += delta * delta;
    }
    dst[output_index] = sum;
}
