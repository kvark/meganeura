//! Regression coverage for the runtime-only RmsNorm + cooperative-matmul
//! prologue fusion. Ineligible matmuls must remain scalar, while eligible
//! matmuls must preserve the rsqrt dependency and observe parameter updates.

use meganeura::{Graph, Session, build_inference_session};
use std::sync::Mutex;

static GPU_TEST_LOCK: Mutex<()> = Mutex::new(());

const ROWS: usize = 64;
const INNER: usize = 64;
const COLS: usize = 64;

fn build(cooperative: bool, expose_normalized: bool) -> Session {
    // SAFETY: the test serializes its process-global feature switches.
    unsafe {
        std::env::remove_var("MEGANEURA_COOP_F16");
        if cooperative {
            std::env::remove_var("MEGANEURA_DISABLE_COOP");
        } else {
            std::env::set_var("MEGANEURA_DISABLE_COOP", "1");
        }
    }

    let mut graph = Graph::new();
    let x = graph.input("x", &[ROWS, INNER]);
    let norm_weight = graph.parameter("norm.weight", &[INNER]);
    let normalized = graph.rms_norm(x, norm_weight, 1e-5);
    let projection = graph.parameter("projection", &[INNER, COLS]);
    let output = graph.matmul(normalized, projection);
    if expose_normalized {
        // Making the RmsNorm result externally visible prevents prologue
        // fusion while leaving the following matmul eligible for coop.
        graph.set_outputs(vec![normalized, output]);
    } else {
        graph.set_outputs(vec![output]);
    }
    build_inference_session(&graph)
}

fn run(
    session: &mut Session,
    output_index: usize,
    x: &[f32],
    norm_weight: &[f32],
    projection: &[f32],
) -> Vec<f32> {
    session.set_parameter("norm.weight", norm_weight);
    session.set_parameter("projection", projection);
    session.set_input("x", x);
    session.step();
    session.wait();
    let mut output = vec![0.0; ROWS * COLS];
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
    let _guard = GPU_TEST_LOCK.lock().expect("GPU test lock poisoned");
    let mut scalar = build(false, false);
    let mut unfused_cooperative = build(true, true);
    let mut cooperative = build(true, false);
    let prologue_enabled = cooperative
        .plan()
        .dispatches
        .iter()
        .any(|dispatch| dispatch.use_coop && dispatch.matmul_prologue.is_some());

    let projection = (0..INNER * COLS)
        .map(|index| ((index * 17 % 101) as f32 - 50.0) * 0.001)
        .collect::<Vec<_>>();
    for update in 0..2 {
        let x = (0..ROWS * INNER)
            .map(|index| ((index * 29 + update * 7) % 113) as f32 * 0.002 - 0.1)
            .collect::<Vec<_>>();
        let norm_weight = (0..INNER)
            .map(|index| 0.5 + ((index * 13 + update * 11) % 37) as f32 * 0.01)
            .collect::<Vec<_>>();
        let scalar_reference = run(&mut scalar, 0, &x, &norm_weight, &projection);
        let cooperative_reference = run(&mut unfused_cooperative, 1, &x, &norm_weight, &projection);
        let got = run(&mut cooperative, 0, &x, &norm_weight, &projection);
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

    unsafe { std::env::remove_var("MEGANEURA_DISABLE_COOP") };
}
