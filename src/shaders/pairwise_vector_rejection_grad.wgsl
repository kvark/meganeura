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
fn grad_vectors(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pair_index = gid.x;
    if (pair_index >= params.total) {
        return;
    }

    let row = pair_index / params.pairs;
    let vector_begin = pair_index * params.inner;
    let direction_begin = row * params.inner;
    var grad_component = 0.0;
    for (var column = 0u; column < params.inner; column += 1u) {
        grad_component += src_a[vector_begin + column] * src_c[direction_begin + column];
    }
    grad_component = -grad_component;
    for (var column = 0u; column < params.inner; column += 1u) {
        let indirect = src_c[direction_begin + column] * grad_component;
        dst[vector_begin + column] = src_a[vector_begin + column] + indirect;
    }
}

@compute @workgroup_size(256)
fn grad_directions(@builtin(global_invocation_id) gid: vec3<u32>) {
    let direction_index = gid.x;
    if (direction_index >= params.total) {
        return;
    }

    let row = direction_index / params.inner;
    let output_column = direction_index % params.inner;
    let direction_begin = row * params.inner;
    var sum = 0.0;
    for (var pair = 0u; pair < params.pairs; pair += 1u) {
        let vector_begin = (row * params.pairs + pair) * params.inner;
        var component = 0.0;
        var grad_component = 0.0;
        for (var column = 0u; column < params.inner; column += 1u) {
            component += src_b[vector_begin + column] * src_c[direction_begin + column];
            grad_component += src_a[vector_begin + column] * src_c[direction_begin + column];
        }
        let direct = (-src_a[vector_begin + output_column]) * component;
        let indirect = src_b[vector_begin + output_column] * (-grad_component);
        sum += direct + indirect;
    }
    dst[direction_index] = sum;
}
