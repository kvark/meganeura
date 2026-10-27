//! Regression coverage for the runtime-only RmsNorm + cooperative-matmul
//! prologue fusion. Ineligible matmuls must remain scalar, while eligible
//! matmuls must preserve the rsqrt dependency and observe parameter updates.

use meganeura::{CoopPolicy, Graph, Session};

fn build(cooperative: bool, expose_normalized: bool, [rows, inner, cols]: [usize; 3]) -> Session {
    let mut graph = Graph::new();
    let x = graph.input("x", &[rows, inner]);
    let norm_weight = graph.parameter("norm.weight", &[inner]);
    let normalized = graph.rms_norm(x, norm_weight, 1e-5);
    let projection = graph.parameter("projection", &[inner, cols]);
    let output = graph.matmul(normalized, projection);
    if expose_normalized {
        // Making the RmsNorm result externally visible prevents prologue
        // fusion while leaving the following matmul eligible for coop.
        graph.set_outputs(vec![normalized, output]);
    } else {
        graph.set_outputs(vec![output]);
    }
    let mut config = crate::support::gpu::inference_config();
    config.runtime.coop = if cooperative {
        CoopPolicy::AllowF16
    } else {
        CoopPolicy::Disabled
    };
    meganeura::build(&graph, config).0
}

fn run(
    session: &mut Session,
    output_index: usize,
    x: &[f32],
    norm_weight: &[f32],
    projection: &[f32],
    output_len: usize,
) -> Vec<f32> {
    session.set_parameter("norm.weight", norm_weight);
    session.set_parameter("projection", projection);
    session.set_input("x", x);
    session.step();
    session.wait();
    let mut output = vec![0.0; output_len];
    session.read_output_by_index(output_index, &mut output);
    output
}

fn relative_l2(got: &[f32], reference: &[f32]) -> f64 {
    let error_sq = got
        .iter()
        .zip(reference)
        .map(|(&a, &b)| f64::from(a - b).powi(2))
        .sum::<f64>();
    let reference_sq = reference
        .iter()
        .map(|&value| f64::from(value).powi(2))
        .sum::<f64>();
    (error_sq / reference_sq.max(f64::EPSILON)).sqrt()
}

#[test]
fn coop_rmsnorm_matmul_prologue_matches_scalar_after_updates() {
    check_after_updates([64, 64, 64], false);
}

#[test]
fn coop_rmsnorm_matmul_prologue_padded_rows_after_updates() {
    let gpu = crate::support::gpu::gpu();
    if meganeura::runtime::auto_tune(&gpu, 0).coop_caps.f16_tile == 0 {
        return;
    }
    for shape in [[17, 719, 4096], [32, 16, 4096]] {
        check_after_updates(shape, true);
    }
}

fn check_after_updates(shape @ [rows, inner, cols]: [usize; 3], require_prologue: bool) {
    let mut scalar = build(false, false, shape);
    let mut unfused_cooperative = build(true, true, shape);
    let mut cooperative = build(true, false, shape);
    let prologue_enabled = cooperative
        .plan()
        .dispatches
        .iter()
        .any(|dispatch| dispatch.use_coop() && dispatch.matmul_prologue.is_some());
    assert!(!require_prologue || prologue_enabled, "shape={shape:?}");

    let projection = (0..inner * cols)
        .map(|index| ((index * 17 % 101) as f32 - 50.0) * 0.001)
        .collect::<Vec<_>>();
    for update in 0..2 {
        let x = (0..rows * inner)
            .map(|index| ((index * 29 + update * 7) % 113) as f32 * 0.002 - 0.1)
            .collect::<Vec<_>>();
        let norm_weight = (0..inner)
            .map(|index| 0.5 + ((index * 13 + update * 11) % 37) as f32 * 0.01)
            .collect::<Vec<_>>();
        let scalar_reference = run(&mut scalar, 0, &x, &norm_weight, &projection, rows * cols);
        let cooperative_reference = run(
            &mut unfused_cooperative,
            1,
            &x,
            &norm_weight,
            &projection,
            rows * cols,
        );
        let got = run(
            &mut cooperative,
            0,
            &x,
            &norm_weight,
            &projection,
            rows * cols,
        );
        let prologue_error = relative_l2(&got, &cooperative_reference);
        let scalar_error = relative_l2(&got, &scalar_reference);
        eprintln!(
            "RmsNorm coop prologue update {update}: enabled={prologue_enabled}, prologue_l2={prologue_error:.6e}, scalar_l2={scalar_error:.6e}"
        );
        assert!(
            prologue_error < 1e-5,
            "cooperative RmsNorm matmul prologue diverged from unfused coop after update {update}: {prologue_error:.6e}"
        );
        assert!(
            scalar_error < 5e-3,
            "cooperative RmsNorm matmul diverged from scalar after update {update}: {scalar_error:.6e}"
        );
    }
}
