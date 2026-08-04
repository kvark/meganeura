struct Params {
    total: u32,
    inner: u32,
    pairs: u32,
    _pad: u32,
}

var<storage> src_a: array<f32>;
var<storage> src_b: array<f32>;
var<storage> src_c: array<f32>;
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;

@compute @workgroup_size(256)
fn grad_left(@builtin(global_invocation_id) gid: vec3<u32>) {
    let left_index = gid.x;
    if (left_index >= params.total) {
        return;
    }

    let row = left_index / params.inner;
    let column = left_index % params.inner;
    var sum = 0.0;
    for (var pair = 0u; pair < params.pairs; pair += 1u) {
        let pair_index = row * params.pairs + pair;
        let right_index = pair_index * params.inner + column;
        let delta = src_b[left_index] - src_c[right_index];
        let term = src_a[pair_index] * delta;
        sum += term + term;
    }
    dst[left_index] = sum;
}

@compute @workgroup_size(256)
fn grad_right(@builtin(global_invocation_id) gid: vec3<u32>) {
    let right_index = gid.x;
    if (right_index >= params.total) {
        return;
    }

    let pair_index = right_index / params.inner;
    let row = pair_index / params.pairs;
    let column = right_index % params.inner;
    let left_index = row * params.inner + column;
    let delta = src_b[left_index] - src_c[right_index];
    let term = src_a[pair_index] * delta;
    dst[right_index] = -(term + term);
}
