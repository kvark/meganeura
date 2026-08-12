struct Params {
    total: u32,
    inner: u32,
    pairs: u32,
    mode: u32,
}

var<storage> src_a: array<f32>;
var<storage> src_b: array<f32>;
var<storage> src_c: array<f32>;
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let output_index = gid.x;
    if (output_index >= params.total) {
        return;
    }

    if params.mode == 1u {
        let pair_index = output_index / params.inner;
        let row = pair_index / params.pairs;
        let column = output_index % params.inner;
        let left_index = row * params.inner + column;
        let delta = src_b[left_index] - src_c[output_index];
        let term = src_a[pair_index] * delta;
        dst[output_index] = -(term + term);
        return;
    }

    let row = output_index / params.inner;
    let output_column = output_index % params.inner;
    var sum = 0.0;
    if params.mode == 0u {
        for (var pair = 0u; pair < params.pairs; pair += 1u) {
            let pair_index = row * params.pairs + pair;
            let right_index = pair_index * params.inner + output_column;
            let delta = src_b[output_index] - src_c[right_index];
            let term = src_a[pair_index] * delta;
            sum += term + term;
        }
    } else {
        let direction_begin = row * params.inner;
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
    }
    dst[output_index] = sum;
}
