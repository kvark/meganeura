// In-place scale every gradient by `min(1, max_norm / sqrt(acc))`,
// reading `acc` (the global gradient-norm-squared accumulator filled
// by `grad_clip_norm_sq`).

struct Params {
    count: u32,
    groups: u32,
    max_norm: f32,
    _pad0: u32,
}

var<storage, read_write> grad: array<f32>;
var<storage> acc: array<f32>;
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
    // norm_sq could be negative due to f32 rounding edge cases or
    // partial NaN if a prior backward overflowed; guard with max(0).
    let norm = sqrt(max(acc[0], 0.0));
    var scale = 1.0;
    if norm > params.max_norm && params.max_norm > 0.0 {
        scale = params.max_norm / norm;
    }
    let start = (group - s.first_group) * TILE;
    for (var k = lid.x; k < TILE; k += WORKGROUP) {
        let i = start + k;
        if i < s.len {
            let e = s.offset + i;
            grad[e] = grad[e] * scale;
        }
    }
}
