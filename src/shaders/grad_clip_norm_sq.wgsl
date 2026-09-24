// Global gradient norm, in two kinds of dispatch sharing one module.
//
// `square != 0`: every workgroup reduces the squares of its tile of one
// gradient and writes the sum to `acc[slot + workgroup]`. Slots are disjoint
// across all chunks, so those dispatches need no barriers between them.
//
// `square == 0`: a single workgroup sums the `groups` partials in `grad` and
// writes the total to `acc[slot]`. The order is fixed, so the norm is
// deterministic.
//
// Pair with `grad_clip_scale`, which reads the total.

struct Params {
    count: u32,
    groups: u32,
    slot: u32,
    square: u32,
}

var<storage> grad: array<f32>;
var<storage, read_write> acc: array<f32>;
var<uniform> params: Params;

var<workgroup> wg_data: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(num_workgroups) grid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let group = group_index(wgid, grid);
    var sum = 0.0;
    if params.square == 0u {
        for (var i = tid; i < params.groups; i += WORKGROUP) {
            sum = sum + grad[i];
        }
    } else {
        if group >= params.groups { return; }
        let s = find_segment(params.count, group);
        let start = (group - s.first_group) * TILE;
        for (var k = tid; k < TILE; k += WORKGROUP) {
            let i = start + k;
            if i < s.len {
                let g = grad[s.offset + i];
                sum = sum + g * g;
            }
        }
    }
    wg_data[tid] = sum;
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
        acc[params.slot + select(0u, group, params.square != 0u)] = wg_data[0];
    }
}
