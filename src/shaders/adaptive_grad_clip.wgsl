// Per-parameter adaptive gradient clipping in one workgroup. Threads stride
// over the complete parameter leaf, reduce both L2 norms in shared memory,
// and then scale the gradient in place by
// min(1, clip * max(pmin, ||parameter||) / ||gradient||).

struct Params {
    len: u32,
    clip: f32,
    pmin: f32,
    _pad0: u32,
}

var<storage> param: array<f32>;
var<storage, read_write> grad: array<f32>;
var<uniform> params: Params;

const WORKGROUP_SIZE: u32 = 256u;
const REDUCTION_START: u32 = WORKGROUP_SIZE / 2u;

var<workgroup> param_squares: array<f32, WORKGROUP_SIZE>;
var<workgroup> grad_squares: array<f32, WORKGROUP_SIZE>;
var<workgroup> gradient_scale: f32;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    var param_sum = 0.0;
    var grad_sum = 0.0;
    var index = tid;
    loop {
        if index >= params.len { break; }
        let p = param[index];
        let g = grad[index];
        param_sum = param_sum + p * p;
        grad_sum = grad_sum + g * g;
        index = index + WORKGROUP_SIZE;
    }
    param_squares[tid] = param_sum;
    grad_squares[tid] = grad_sum;
    workgroupBarrier();

    var stride = REDUCTION_START;
    loop {
        if stride == 0u { break; }
        if tid < stride {
            param_squares[tid] = param_squares[tid] + param_squares[tid + stride];
            grad_squares[tid] = grad_squares[tid] + grad_squares[tid + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    if tid == 0u {
        let param_norm = sqrt(max(param_squares[0], 0.0));
        let grad_norm = sqrt(max(grad_squares[0], 0.0));
        let upper = params.clip * max(params.pmin, param_norm);
        gradient_scale = 1.0;
        if grad_norm > upper && upper > 0.0 {
            gradient_scale = upper / grad_norm;
        }
    }
    workgroupBarrier();

    index = tid;
    loop {
        if index >= params.len { break; }
        grad[index] = grad[index] * gradient_scale;
        index = index + WORKGROUP_SIZE;
    }
}
