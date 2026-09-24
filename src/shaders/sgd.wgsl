struct Params {
    count: u32,
    groups: u32,
    lr: f32,
    _pad0: u32,
}

var<storage, read_write> param: array<f32>;
var<storage> grad: array<f32>;
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
    let lr = params.lr * s.lr_scale;
    let start = (group - s.first_group) * TILE;
    for (var k = lid.x; k < TILE; k += WORKGROUP) {
        let i = start + k;
        if i < s.len {
            let e = s.offset + i;
            param[e] = param[e] - lr * grad[e];
        }
    }
}
