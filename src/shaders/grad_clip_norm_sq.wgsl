// Global gradient norm, in two kinds of dispatch sharing one module.
//
// `square != 0`: every workgroup reduces the squares of a grid-strided slice
// of one gradient and writes its sum to `acc[slot + workgroup]`. Slots are
// disjoint across all gradients, so the per-parameter dispatches need no
// barriers between them and a large gradient gets many workgroups instead of
// one.
//
// `square == 0`: a single workgroup sums `len` partials and writes the total
// to `acc[slot]`. The order is fixed, so the norm is deterministic.
//
// Pair with `grad_clip_scale`, which reads the total.

struct Params {
    len: u32,
    slot: u32,
    square: u32,
    _pad0: u32,
}

var<storage> grad: array<f32>;
var<storage, read_write> acc: array<f32>;
var<uniform> params: Params;

var<workgroup> wg_data: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let tid = lid.x;
    let stride = groups.x * 256u;
    var s: f32 = 0.0;
    var idx = wgid.x * 256u + tid;
    loop {
        if idx >= params.len { break; }
        let g = grad[idx];
        s = s + select(g, g * g, params.square != 0u);
        idx = idx + stride;
    }
    wg_data[tid] = s;
    workgroupBarrier();

    var half = 128u;
    loop {
        if half == 0u { break; }
        if tid < half {
            wg_data[tid] = wg_data[tid] + wg_data[tid + half];
        }
        workgroupBarrier();
        half = half >> 1u;
    }

    if tid == 0u {
        acc[params.slot + wgid.x] = wg_data[0];
    }
}
