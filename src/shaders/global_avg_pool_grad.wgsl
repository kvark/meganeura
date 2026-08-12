// Row-wise data movement shared by average-pool gradients, broadcasts, and shifts.
// Dispatch: [ceil(total/256), 1, 1].

struct Params {
    total: u32, // output element count
    spatial: u32, // row width
    mode: u32, // 0 = normalized broadcast, 1 = broadcast, 2 = shift, 3 = cumsum
    argument: u32,
}

var<storage> src: array<f32>;
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.total { return; }
    if params.mode == 3u {
        let base = i * params.spatial;
        var acc = 0.0;
        for (var offset = 0u; offset < params.spatial; offset++) {
            let col = select(offset, params.spatial - 1u - offset, params.argument != 0u);
            let index = base + col;
            let value = src[index];
            dst[index] = acc;
            acc += value;
        }
    } else if params.mode == 2u {
        let col = i32(i % params.spatial);
        let source_col = col - bitcast<i32>(params.argument);
        if source_col >= 0 && source_col < i32(params.spatial) {
            dst[i] = src[i - u32(col) + u32(source_col)];
        } else {
            dst[i] = 0.0;
        }
    } else {
        let value = src[i / params.spatial];
        dst[i] = select(value / f32(params.spatial), value, params.mode == 1u);
    }
}
