// Temporal gradient accumulation: acc[i] += grad[i] * scale.
//
// backward overwrites each param's grad buffer every step(); this pass
// adds that fresh grad into a persistent accumulator so gradients sum
// ACROSS step() calls (the PyTorch `.grad +=` semantics meganeura's
// static-graph backward lacks). `scale` is 1/micro_batches so the
// accumulator holds the mean gradient. Cleared by `zero_grad()`.

struct Params {
    count: u32,
    groups: u32,
    scale: f32,
    _pad0: u32,
}

var<storage> grad: array<f32>;
var<storage, read_write> acc: array<f32>;
var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(num_workgroups) grid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let group = group_index(wgid, grid);
    if group >= params.groups { return; }
    let s = find_segment(params.count, group);
    let start = (group - s.first_group) * TILE;
    for (var k = lid.x; k < TILE; k += WORKGROUP) {
        let i = start + k;
        if i < s.len {
            let e = s.offset + i;
            acc[e] = acc[e] + grad[e] * params.scale;
        }
    }
}
