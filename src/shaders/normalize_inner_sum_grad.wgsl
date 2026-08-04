struct Params {
    rows: u32,
    inner: u32,
    floor_bits: u32,
    _pad: u32,
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
    var sum = 0.0;
    var weighted_gradient = 0.0;
    for (var column = 0u; column < params.inner; column += 1u) {
        let index = begin + column;
        sum += src_b[index];
        weighted_gradient += src_a[index] * src_b[index];
    }
    let floor = bitcast<f32>(params.floor_bits);
    let above_floor = sum - floor;
    let denominator = max(above_floor, 0.0) + floor;
    let inverse = 1.0 / denominator;
    let denominator_gradient = select(
        0.0,
        -(weighted_gradient * (inverse * inverse)),
        above_floor > 0.0,
    );
    for (var column = 0u; column < params.inner; column += 1u) {
        let index = begin + column;
        dst[index] = src_a[index] * inverse + denominator_gradient;
    }
}
