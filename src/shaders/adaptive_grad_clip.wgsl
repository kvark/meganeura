// Per-parameter adaptive gradient clipping: scale each gradient in place by
// min(1, clip * max(pmin, ||parameter||) / ||gradient||).
//
// Three modes of one entry point, each dispatched once per chunk in its own
// pass:
//   0 norms:  every workgroup reduces (param², grad²) over its tile and
//             writes the pair to partials[slot + workgroup]
//   1 factor: one workgroup per segment sums its parameter's pairs in a
//             fixed order and writes its scale to scales[index]
//   2 apply:  grad[i] *= scales[index] over every parameter
// A large parameter therefore spreads across the device instead of
// streaming through one workgroup, and the result stays deterministic.

struct Params {
    count: u32,
    groups: u32,
    slot: u32,
    mode: u32,
    clip: f32,
    pmin: f32,
    _pad0: u32,
    _pad1: u32,
}

var<storage> param: array<f32>;
var<storage, read_write> grad: array<f32>;
var<storage, read_write> partials: array<vec2<f32>>;
var<storage, read_write> scales: array<f32>;
var<uniform> params: Params;

var<workgroup> param_squares: array<f32, WORKGROUP>;
var<workgroup> grad_squares: array<f32, WORKGROUP>;

fn reduce_pair(tid: u32, p: f32, g: f32) -> vec2<f32> {
    param_squares[tid] = p;
    grad_squares[tid] = g;
    workgroupBarrier();
    var stride = WORKGROUP / 2u;
    loop {
        if stride == 0u { break; }
        if tid < stride {
            param_squares[tid] = param_squares[tid] + param_squares[tid + stride];
            grad_squares[tid] = grad_squares[tid] + grad_squares[tid + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    return vec2<f32>(param_squares[0], grad_squares[0]);
}

@compute @workgroup_size(WORKGROUP)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(num_workgroups) grid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let group = group_index(wgid, grid);
    if params.mode == 1u {
        if group >= params.count { return; }
        let s = segments[group];
        let tiles = (s.len + TILE - 1u) / TILE;
        var sums = vec2<f32>(0.0);
        for (var t = tid; t < tiles; t += WORKGROUP) {
            sums = sums + partials[params.slot + s.first_group + t];
        }
        let total = reduce_pair(tid, sums.x, sums.y);
        if tid == 0u {
            let param_norm = sqrt(max(total.x, 0.0));
            let grad_norm = sqrt(max(total.y, 0.0));
            let upper = params.clip * max(params.pmin, param_norm);
            var scale = 1.0;
            if grad_norm > upper && upper > 0.0 {
                scale = upper / grad_norm;
            }
            scales[s.index] = scale;
        }
        return;
    }
    if group >= params.groups { return; }
    let s = find_segment(params.count, group);
    let start = (group - s.first_group) * TILE;
    if params.mode == 0u {
        var param_sum = 0.0;
        var grad_sum = 0.0;
        for (var k = tid; k < TILE; k += WORKGROUP) {
            let i = start + k;
            if i < s.len {
                let p = param[s.offset + i];
                let g = grad[s.offset + i];
                param_sum = param_sum + p * p;
                grad_sum = grad_sum + g * g;
            }
        }
        let total = reduce_pair(tid, param_sum, grad_sum);
        if tid == 0u {
            partials[params.slot + group] = total;
        }
    } else {
        let scale = scales[s.index];
        for (var k = tid; k < TILE; k += WORKGROUP) {
            let i = start + k;
            if i < s.len {
                let e = s.offset + i;
                grad[e] = grad[e] * scale;
            }
        }
    }
}
