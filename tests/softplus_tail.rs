//! Relative accuracy matters when tiny positive values are later normalized.
use meganeura::{CompileOptions, Graph, Mode, SessionConfig};

fn accurate(actual: f32, expected: f64, description: &str) {
    assert!(actual.is_finite(), "{description}: nonfinite {actual}");
    assert!(
        (actual as f64 - expected).abs() <= expected.abs() * 1e-5 + 1e-38,
        "{description}: {actual} != {expected}"
    );
}

#[test]
fn softplus_tail_forward_and_backward_match_f64() {
    let scaled = [-80.0f32, -40.0, -30.0, -20.0, -18.5, -18.0, -17.749283,
        -17.187275, -17.0, -16.617382, -16.0, -10.0, -6.0, -1.0, 0.0,
        1.0, 6.0, 16.0, 30.0, 80.0];
    for beta in [0.25f32, 1.0, 10.0] {
        for schedule in [false, true] {
            let input: Vec<_> = scaled.iter().map(|x| x / beta).collect();
            let upstream: Vec<_> = (0..input.len()).map(|i| if i % 2 == 0 { 1.25 } else { -0.75 }).collect();
            let mut g = Graph::new();
            let x = g.parameter("x", &[input.len()]);
            let y = g.softplus(x, beta);
            let weight = g.input("upstream", &[input.len()]);
            let weighted = g.mul(y, weight);
            let loss = g.sum_all(weighted);
            g.set_outputs(vec![loss, y]);
            let mut options = CompileOptions::default();
            options.use_schedule_pointwise = schedule;
            let mut s = meganeura::build(&g, SessionConfig { mode: Mode::Training, options, ..SessionConfig::from_env() }).0;
            s.set_parameter("x", &input);
            s.set_input("upstream", &upstream);
            s.set_adam(0.0, 0.9, 0.999, 1e-8);
            s.step(); s.wait();
            let mut values = vec![0.0; input.len()];
            let mut gradients = values.clone();
            s.read_output_by_index(1, &mut values);
            s.read_param_grad("x", &mut gradients);
            for i in 0..input.len() {
                let v = input[i] as f64 * beta as f64;
                let t = (-v.abs()).exp();
                let expected = input[i].max(0.0) as f64 + t.ln_1p() / beta as f64;
                let derivative = if v > 0.0 { 1.0 } else { t } / (1.0 + t);
                accurate(values[i], expected, "softplus forward");
                accurate(gradients[i], upstream[i] as f64 * derivative, "softplus gradient");
            }
        }
    }
}

#[test]
fn softplus_tail_normalized_mixture_matches_f64() {
    // Two actual saturated Ommatidia selector pixels. The old implementation
    // changed one normalized weight by 0.295 despite preserving total mass.
    let logits = [-28.584324f32, -55.08445, -57.773396, -17.187275, -43.64951, 42.930416,
        -53.557167, -71.394745, -42.14466, -105.05025, -16.617382, 0.0];
    let priors = [0.02f32, 0.06, 0.12, 0.30, 0.5, 0.0,
        0.28302041, 0.28315285, 0.18885687, 0.15745437, 0.08751558, 0.0];
    let mut g = Graph::new();
    let x = g.input("x", &[2, 6]);
    let p = g.input("prior", &[2, 6]);
    let y = g.softplus(x, 1.0);
    let floor = g.constant(vec![1e-8; 12], &[2, 6]);
    let negative_floor = g.neg(floor);
    let offset = g.add(y, negative_floor);
    let clamped = g.relu(offset);
    let positive = g.add(clamped, floor);
    let raw = g.mul(positive, p);
    let total = g.sum_inner(raw);
    let total = g.broadcast_inner(total, 6);
    let normalized = g.div(raw, total);
    g.set_outputs(vec![normalized]);
    let mut s = meganeura::build(&g, SessionConfig { mode: Mode::Inference, ..SessionConfig::from_env() }).0;
    s.set_input("x", &logits); s.set_input("prior", &priors);
    s.step(); s.wait();
    let output = s.read_output(12);
    for row in 0..2 {
        let raw: Vec<_> = (0..6).map(|c| {
            let i = row * 6 + c;
            let v = logits[i] as f64;
            priors[i] as f64 * (v.max(0.0) + (-v.abs()).exp().ln_1p()).max(1e-8)
        }).collect();
        let total: f64 = raw.iter().sum();
        for c in 0..6 { accurate(output[row * 6 + c], raw[c] / total, "normalized weight"); }
    }
}
