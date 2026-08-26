// Adaptive gradient clipping in one workgroup per norm unit. Flattening a
// parameter to [reduction_count, unit_count] expresses both supported modes:
// leaf-wise clipping uses one unit, while Optax-compatible clipping chooses
// the rank-dependent trailing output units on the CPU.

struct Params {
    len: u32,
    unit_count: u32,
    reduction_count: u32,
    workgroups_x: u32,
    clip: f32,
    pmin: f32,
    optax_semantics: u32,
    _pad1: u32,
}

var<storage> param: array<f32>;
var<storage, read_write> grad: array<f32>;
var<uniform> params: Params;

var<workgroup> param_squares: array<f32, 256>;
var<workgroup> grad_squares: array<f32, 256>;
var<workgroup> gradient_scale: f32;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let unit = wid.x + wid.y * params.workgroups_x;
    if unit >= params.unit_count { return; }

    let tid = lid.x;
    var param_sum = 0.0;
    var grad_sum = 0.0;
    var reduction_index = tid;
    loop {
        if reduction_index >= params.reduction_count { break; }
        let index = reduction_index * params.unit_count + unit;
        if index >= params.len { break; }
        let p = param[index];
        let g = grad[index];
        param_sum = param_sum + p * p;
        grad_sum = grad_sum + g * g;
        reduction_index = reduction_index + 256u;
    }
    param_squares[tid] = param_sum;
    grad_squares[tid] = grad_sum;
    workgroupBarrier();

    var stride = 128u;
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
        // Optax uses max(gradient norm, 1e-6) for the division and applies the
        // clipped branch when the norm equals the bound. The legacy leaf-wise
        // API keeps its historical strict comparison and zero-bound no-op.
        let should_clip = select(
            grad_norm > upper && upper > 0.0,
            grad_norm >= upper,
            params.optax_semantics != 0u,
        );
        if should_clip {
            gradient_scale = upper / max(grad_norm, 1e-6);
        }
    }
    workgroupBarrier();

    reduction_index = tid;
    loop {
        if reduction_index >= params.reduction_count { break; }
        let index = reduction_index * params.unit_count + unit;
        if index >= params.len { break; }
        grad[index] = grad[index] * gradient_scale;
        reduction_index = reduction_index + 256u;
    }
}
