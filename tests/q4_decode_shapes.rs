//! Q4 matmul at decode shapes (m = 1).
//!
//! `q4_matmul_correctness` in `gpu_smoke` uses m = 6, so it exercises the
//! general tiled matmul only. Autoregressive decode multiplies one row at a
//! time, and `compile.rs` specializes `m == 1 && n % 4 == 0` to a K-split
//! GEMV. `generate_module_weighted` has no Q4 variant of the GEMV shaders
//! and falls back to the f32 ones, which read the packed 4-bit blocks as
//! floats — the outputs came back around 1e37 rather than failing.
//!
//! Nothing caught it: the codegen unit test that checks Q4 shaders emit
//! `dequant_q4` lists the six tiled matmul groups and none of the three
//! GEMV ones. So Q4 was unusable for exactly the shape a decode graph is
//! made of, which is what the roadmap needs it for.
//!
//! These cases use SmolLM2-135M's real projection widths and compare
//! against the CPU quantize/dequantize reference — the same instrument
//! `q4_matmul_correctness` uses, which measures whether the GPU does what
//! Q4 should do rather than how lossy Q4 is.

use meganeura::runtime::{dequantize_q4_0, quantize_q4_0};
use meganeura::{Graph, SessionConfig};

fn gpu_q4(m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut g = Graph::new();
    let a_in = g.input("a", &[m, k]);
    let b_q4 = g.parameter_q4("b", &[k, n]);
    let c = g.matmul(a_in, b_q4);
    g.set_outputs(vec![c]);
    let mut session = meganeura::build(&g, SessionConfig::inference_from_env()).0;
    session.set_input("a", a);
    session.set_parameter("b", b);
    session.step();
    session.wait();
    session.read_output(m * n)
}

/// What Q4 should produce: quantize and dequantize on the CPU, then
/// multiply in f32.
fn cpu_q4(m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    let b_deq = dequantize_q4_0(&quantize_q4_0(b, k, n), k, n);
    let mut out = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for i in 0..k {
                acc += a[row * k + i] * b_deq[i * n + col];
            }
            out[row * n + col] = acc;
        }
    }
    out
}

fn check(label: &str, m: usize, k: usize, n: usize) {
    // A sawtooth is deliberately hostile to 32-element block quantization,
    // so any block-indexing error shows up rather than averaging out.
    let a: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect();
    let b: Vec<f32> = (0..k * n)
        .map(|i| ((i % 97) as f32 - 48.0) * 0.01)
        .collect();

    let gpu = gpu_q4(m, k, n, &a, &b);
    let cpu = cpu_q4(m, k, n, &a, &b);

    assert!(
        gpu.iter().all(|v| v.is_finite()),
        "{label}: Q4 matmul produced non-finite values"
    );
    let scale = cpu.iter().cloned().fold(0.0f32, |acc, v| acc.max(v.abs()));
    assert!(scale > 1e-3, "{label}: degenerate reference output");
    let err = gpu
        .iter()
        .zip(&cpu)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    // Both sides dequantize the same blocks and sum over the same k, so
    // this is f32 reassociation only. Observed 0.0 on lavapipe.
    assert!(
        err / scale < 1e-3,
        "{label}: GPU Q4 diverged from the CPU Q4 reference: \
         max_abs_err={err} (output scale {scale})"
    );
}

/// n % 4 == 0 makes these eligible for the K-split GEMV specialization.
#[test]
fn q4_single_row_matches_reference_on_gemv_eligible_widths() {
    // SmolLM2-135M: hidden 576, ffn 1536, vocab 49152.
    check("ffn gate/up 576x1536", 1, 576, 1536);
    check("ffn down    1536x576", 1, 1536, 576);
    check("attn qkv     576x576", 1, 576, 576);
    check("lm_head    576x49152", 1, 576, 49152);
}

/// n % 4 != 0 keeps these on the general tiled path — the control.
#[test]
fn q4_single_row_matches_reference_on_tiled_widths() {
    check("576x1534", 1, 576, 1534);
    check("576x1533", 1, 576, 1533);
}

/// Batched shapes were already covered; keep them adjacent so a future
/// GEMV change that regresses only one of the two is obvious.
#[test]
fn q4_batched_rows_match_reference() {
    check("m=2 576x1536", 2, 576, 1536);
    check("m=6 576x1536", 6, 576, 1536);
}
