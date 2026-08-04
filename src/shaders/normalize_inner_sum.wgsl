struct Params {
    rows: u32,
    inner: u32,
    floor_bits: u32,
    round_one_bits: u32,
}

var<storage> src: array<f32>;
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
        sum += src[begin + column] * round_one;
    }
    let floor = bitcast<f32>(params.floor_bits);
    let denominator = max(sum - floor, 0.0) + floor;
    let inverse = 1.0 / denominator;
    for (var column = 0u; column < params.inner; column += 1u) {
        let index = begin + column;
        dst[index] = src[index] * inverse;
    }
}
