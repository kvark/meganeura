struct Params {
    count: u32,
    groups: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    // 1 / (1 - beta1^step) and 1 / (1 - beta2^step), computed on the host.
    // The exponent is uniform across the dispatch and constant for the
    // step, so raising it per element would repeat one `pow` for every
    // parameter element. The host also keeps one rounding for the whole
    // step instead of one per thread. Must match `AdamParams` in
    // `runtime/optimizer.rs`.
    bias_correction1: f32,
    bias_correction2: f32,
    wd: f32,
    grad_group_size: u32,
    // Pair index whose grouped gradient norms are collected.
    grouped_index: u32,
    algorithm: u32,
    one_minus_beta1: f32,
    one_minus_beta2: f32,
}

var<storage, read_write> param: array<f32>;
var<storage> grad: array<f32>;
var<storage, read_write> m: array<f32>;
var<storage, read_write> v: array<f32>;
var<storage, read_write> grouped_grad_norm: array<f32>;
var<uniform> params: Params;

const ADAM_ALGORITHM_LAPROP: u32 = 1u;

// Keep tiny variance arithmetic normal even when the device flushes F32 denormals.
fn variance_scaled(value: f32) -> f32 {
    let bits = bitcast<u32>(value);
    if bits < 0x00800000u {
        return f32(bits) * 0x1p-85f;
    }
    return value * 0x1p64f;
}

fn variance_unscaled(value: f32) -> f32 {
    if value < 0x1p-62f {
        let mantissa = value * 0x1p85f;
        let lower = u32(mantissa);
        let fraction = mantissa - f32(lower);
        let up = fraction > 0.5 || (fraction == 0.5 && (lower & 1u) != 0u);
        return bitcast<f32>(lower + select(0u, 1u, up));
    }
    return value * 0x1p-64f;
}

fn variance_and_rms(old: f32, gradient: f32) -> vec2<f32> {
    let finite_history = (bitcast<u32>(old) & 0x7fffffffu) < 0x7f800000u;
    let history = select(old, 0.0, params.beta2 == 0.0 && finite_history);
    if abs(gradient) < 0x1p-32f && bitcast<u32>(history) < 0x1f800000u {
        let scaled_gradient = gradient * 0x1p32f;
        let scaled = params.beta2 * variance_scaled(history)
            + params.one_minus_beta2 * scaled_gradient * scaled_gradient;
        let stored = variance_unscaled(scaled);
        let rms = sqrt(variance_scaled(stored) * params.bias_correction2) * 0x1p-32f;
        return vec2<f32>(stored, rms);
    }
    let stored = params.beta2 * old + params.one_minus_beta2 * gradient * gradient;
    return vec2<f32>(stored, sqrt(stored * params.bias_correction2));
}

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
    let variance_rms = variance_and_rms(v[e], g);
    v[e] = variance_rms.x;
    let rms = variance_rms.y;

    // Adam accumulates momentum on raw gradients and normalizes the result.
    // LaProp normalizes each gradient first and then accumulates
    // momentum, matching DreamerV3's RMS -> momentum optimizer chain.
    var moment_input = g;
    if params.algorithm == ADAM_ALGORITHM_LAPROP {
        moment_input = g / (rms + params.eps);
    }
    let m_new = params.beta1 * m[e] + params.one_minus_beta1 * moment_input;
    m[e] = m_new;
    let m_hat = m_new * params.bias_correction1;

    var step = m_hat / (rms + params.eps);
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
