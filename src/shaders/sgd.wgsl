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
    let i = start + lid.x * 4u;
    if i < s.len {
        if s.len - i >= 4u {
            let e = s.offset + i;
            let p = vec4<f32>(param[e], param[e + 1u], param[e + 2u], param[e + 3u]);
            let g = vec4<f32>(grad[e], grad[e + 1u], grad[e + 2u], grad[e + 3u]);
            let updated = p - vec4<f32>(lr) * g;
            param[e] = updated.x;
            param[e + 1u] = updated.y;
            param[e + 2u] = updated.z;
            param[e + 3u] = updated.w;
        } else {
            for (var component = 0u; component < 4u; component++) {
                if i + component < s.len {
                    let e = s.offset + i + component;
                    param[e] = param[e] - lr * grad[e];
                }
            }
        }
    }
}
