struct Params {
    len: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    step: f32,
    _pad0: u32,
    _pad1: u32,
}

var<storage, read_write> param: array<f32>;
var<storage> grad: array<f32>;
var<storage, read_write> m: array<f32>;
var<storage, read_write> v: array<f32>;
var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.len { return; }

    let g = grad[i];

    // Update biased first moment
    let m_new = params.beta1 * m[i] + (1.0 - params.beta1) * g;
    // Update biased second moment
    let v_new = params.beta2 * v[i] + (1.0 - params.beta2) * g * g;

    m[i] = m_new;
    v[i] = v_new;

    // Bias-corrected estimates
    let m_hat = m_new / (1.0 - pow(params.beta1, params.step));
    let v_hat = v_new / (1.0 - pow(params.beta2, params.step));

    // Update parameter
    param[i] = param[i] - params.lr * m_hat / (sqrt(v_hat) + params.eps);
}
