struct Params {
    len: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    step: f32,
    wd: f32,
    grad_group_size: u32,
    algorithm: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

var<storage, read_write> param: array<f32>;
var<storage> grad: array<f32>;
var<storage, read_write> m: array<f32>;
var<storage, read_write> v: array<f32>;
var<storage, read_write> grouped_grad_norm: array<f32>;
var<uniform> params: Params;

const ADAM_ALGORITHM_LAPROP: u32 = 1u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.len { return; }

    let g = grad[i];

    if params.grad_group_size != 0u && i % params.grad_group_size == 0u {
        var sum_squared = 0.0;
        var component = 0u;
        loop {
            if component >= params.grad_group_size { break; }
            let value = grad[i + component];
            sum_squared = sum_squared + value * value;
            component = component + 1u;
        }
        let group = i / params.grad_group_size;
        grouped_grad_norm[group] = grouped_grad_norm[group] + sqrt(sum_squared);
    }

    // Update biased second moment
    let v_new = params.beta2 * v[i] + (1.0 - params.beta2) * g * g;
    v[i] = v_new;
    let v_hat = v_new / (1.0 - pow(params.beta2, params.step));

    // Adam accumulates momentum on raw gradients and normalizes the result.
    // LaProp normalizes each gradient first and then accumulates
    // momentum, matching DreamerV3's RMS -> momentum optimizer chain.
    var moment_input = g;
    if params.algorithm == ADAM_ALGORITHM_LAPROP {
        moment_input = g / (sqrt(v_hat) + params.eps);
    }
    let m_new = params.beta1 * m[i] + (1.0 - params.beta1) * moment_input;
    m[i] = m_new;
    let m_hat = m_new / (1.0 - pow(params.beta1, params.step));

    var update = m_hat / (sqrt(v_hat) + params.eps);
    if params.algorithm == ADAM_ALGORITHM_LAPROP {
        update = m_hat;
    }

    // Update parameter. Decoupled weight decay (AdamW): the wd*param term is
    // applied directly to the weight, NOT routed through the Adam moments, so
    // it is independent of the gradient's adaptive scaling.
    param[i] = param[i] - params.lr * (update + params.wd * param[i]);
}
