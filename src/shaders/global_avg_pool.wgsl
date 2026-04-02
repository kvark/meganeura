struct Params {
    channels: u32,
    spatial: u32,
    total_out: u32,
    _pad: u32,
}

var<storage> src: array<f32>;
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;

@compute @workgroup_size(256)
fn global_avg_pool(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.total_out { return; }

    let base = idx * params.spatial;
    var sum = 0.0;
    for (var s = 0u; s < params.spatial; s++) {
        sum += src[base + s];
    }
    dst[idx] = sum / f32(params.spatial);
}
