struct Params {
    rows: u32,
    inner: u32,
    floor_bits: u32,
    round_one_bits: u32,
}

var<storage> src_a: array<f32>;
var<storage> src_b: array<f32>;
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= params.rows) {
        return;
    }

    let begin = row * params.inner;
    let round_one = bitcast<f32>(params.round_one_bits);
    var sum = 0.0;
    for (var column = 0u; column < params.inner; column += 1u) {
        sum += src_b[begin + column] * round_one;
    }
    let floor = bitcast<f32>(params.floor_bits);
    let above_floor = sum - floor;
    let denominator = max(above_floor, 0.0) + floor;
    let inverse = 1.0 / denominator;
    let negative_inverse_squared = -(inverse * inverse);
    var denominator_gradient = 0.0;
    if (above_floor > 0.0) {
        for (var column = 0u; column < params.inner; column += 1u) {
            let index = begin + column;
            let reciprocal_gradient = src_a[index] * src_b[index];
            let contribution = reciprocal_gradient * negative_inverse_squared;
            denominator_gradient += contribution * round_one;
        }
    }
    for (var column = 0u; column < params.inner; column += 1u) {
        let index = begin + column;
        dst[index] = src_a[index] * inverse + denominator_gradient;
    }
}
