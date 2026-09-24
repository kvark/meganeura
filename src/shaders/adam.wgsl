struct Params {
    count: u32,
    groups: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    step: f32,
    wd: f32,
    grad_group_size: u32,
    // Pair index whose grouped gradient norms are collected.
    grouped_index: u32,
    algorithm: u32,
    _pad0: u32,
}

var<storage, read_write> param: array<f32>;
var<storage> grad: array<f32>;
var<storage, read_write> m: array<f32>;
var<storage, read_write> v: array<f32>;
var<storage, read_write> grouped_grad_norm: array<f32>;
var<uniform> params: Params;

const ADAM_ALGORITHM_LAPROP: u32 = 1u;

fn update(s: Segment, i: u32) {
    let e = s.offset + i;
    let g = grad[e];

    let group_size = params.grad_group_size;
    if group_size != 0u && s.index == params.grouped_index && i % group_size == 0u {
        var sum_squared = 0.0;
        for (var component = 0u; component < group_size; component++) {
            let value = grad[e + component];
            sum_squared = sum_squared + value * value;
        }
        let group = i / group_size;
        grouped_grad_norm[group] = grouped_grad_norm[group] + sqrt(sum_squared);
    }

    // Update biased second moment
    let v_new = params.beta2 * v[e] + (1.0 - params.beta2) * g * g;
    v[e] = v_new;
    let v_hat = v_new / (1.0 - pow(params.beta2, params.step));

    // Adam accumulates momentum on raw gradients and normalizes the result.
    // LaProp normalizes each gradient first and then accumulates
    // momentum, matching DreamerV3's RMS -> momentum optimizer chain.
    var moment_input = g;
    if params.algorithm == ADAM_ALGORITHM_LAPROP {
        moment_input = g / (sqrt(v_hat) + params.eps);
    }
    let m_new = params.beta1 * m[e] + (1.0 - params.beta1) * moment_input;
    m[e] = m_new;
    let m_hat = m_new / (1.0 - pow(params.beta1, params.step));

    var step = m_hat / (sqrt(v_hat) + params.eps);
    if params.algorithm == ADAM_ALGORITHM_LAPROP {
        step = m_hat;
    }

    // Update parameter. Decoupled weight decay (AdamW): the wd*param term is
    // applied directly to the weight, NOT routed through the Adam moments, so
    // it is independent of the gradient's adaptive scaling.
    let lr = params.lr * s.lr_scale;
    param[e] = param[e] - lr * (step + params.wd * param[e]);
}

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
            update(s, i);
        }
    }
}
