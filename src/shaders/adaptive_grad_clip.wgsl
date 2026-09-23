// Per-parameter adaptive gradient clipping: scale each gradient in place by
// min(1, clip * max(pmin, ||parameter||) / ||gradient||).
//
// Three modes of one entry point, each dispatched once per parameter in its
// own pass:
//   0 norms:  every workgroup reduces (param², grad²) over a grid-strided slice
//           and writes the pair to partials[slot + workgroup]
//   1 factor: one workgroup sums the parameter's `slots` pairs in a fixed
//           order and writes its scale to scales[index]
//   2 apply:  grad[i] *= scales[index] over the whole parameter
// A large parameter therefore spreads across the device instead of
// streaming through one workgroup, and the result stays deterministic.

struct Params {
    len: u32,
    clip: f32,
    pmin: f32,
    slot: u32,
    slots: u32,
    index: u32,
    mode: u32,
    _pad0: u32,
}

var<storage> param: array<f32>;
var<storage, read_write> grad: array<f32>;
var<storage, read_write> partials: array<vec2<f32>>;
var<storage, read_write> scales: array<f32>;
var<uniform> params: Params;

const WORKGROUP_SIZE: u32 = 256u;

var<workgroup> param_squares: array<f32, WORKGROUP_SIZE>;
var<workgroup> grad_squares: array<f32, WORKGROUP_SIZE>;

fn reduce_pair(tid: u32, p: f32, g: f32) -> vec2<f32> {
    param_squares[tid] = p;
    grad_squares[tid] = g;
    workgroupBarrier();
    var stride = WORKGROUP_SIZE / 2u;
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

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let tid = lid.x;
    if params.mode == 0u {
        let stride = groups.x * WORKGROUP_SIZE;
        var param_sum = 0.0;
        var grad_sum = 0.0;
        var index = wgid.x * WORKGROUP_SIZE + tid;
        loop {
            if index >= params.len { break; }
            let p = param[index];
            let g = grad[index];
            param_sum = param_sum + p * p;
            grad_sum = grad_sum + g * g;
            index = index + stride;
        }
        let total = reduce_pair(tid, param_sum, grad_sum);
        if tid == 0u {
            partials[params.slot + wgid.x] = total;
        }
    } else if params.mode == 1u {
        var sums = vec2<f32>(0.0);
        var slot = tid;
        loop {
            if slot >= params.slots { break; }
            sums = sums + partials[params.slot + slot];
            slot = slot + WORKGROUP_SIZE;
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
            scales[params.index] = scale;
        }
    } else {
        let i = gid.x;
        if i < params.len {
            grad[i] = grad[i] * scales[params.index];
        }
    }
}
