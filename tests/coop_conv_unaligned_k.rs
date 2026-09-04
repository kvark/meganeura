//! Regression: Conv2dGemm cooperative-matrix kernel must be correct when
//! K = Ci*kH*kW is not a multiple of 4. The vec4 weight staging assumed
//! each [Co, K] row begins on a vec4 boundary; for Ci=14 (K=126) the rows
//! are unaligned and the load read across row boundaries, corrupting the
//! tile (max abs error ~1.9, 2M/4M elements wrong) and NaN-ing the deep
//! van-world SR model on NVIDIA. Fixed by a per-element fallback when
//! k_total % 4 != 0 (codegen::emit_forward_weight_stage).
//!
//! The coop path runs only on GPUs that advertise cooperative matrices;
//! on f16-only GPUs it is opt-in (overflow-unsafe for large activations).
//! The test selects that policy explicitly. With small bounded inputs f16
//! precision is fine, isolating the K%4 *alignment* correctness. On no-coop
//! hardware (e.g. lavapipe CI) both paths are scalar and the test trivially
//! passes.

use meganeura::{CoopPolicy, Graph, Mode, SessionConfig, SessionOptions};

fn config(mode: Mode, coop: bool) -> SessionConfig<'static> {
    SessionConfig {
        mode,
        runtime: SessionOptions {
            coop: if coop {
                CoopPolicy::AllowF16
            } else {
                CoopPolicy::Disabled
            },
            ..SessionOptions::default()
        },
        ..SessionConfig::default()
    }
}

fn conv_out(in_c: u32, coop: bool) -> Vec<f32> {
    let (batch, hw, out_c) = (32u32, 16u32, 64u32);
    let in_size = (batch * in_c * hw * hw) as usize;
    let k_size = (out_c * in_c * 9) as usize;
    let mut g = Graph::new();
    let x = g.input("x", &[in_size]);
    let k = g.parameter("k", &[k_size]);
    let y = g.conv2d(x, k, batch, in_c, hw, hw, out_c, 3, 3, 1, 1);
    g.set_outputs(vec![y]);
    let mut s = meganeura::build(&g, config(Mode::Inference, coop)).0;
    // Small bounded values so f16 coop precision is not the issue.
    let xd: Vec<f32> = (0..in_size)
        .map(|i| ((i * 31 % 17) as f32 / 16.0 - 0.5) * 0.2)
        .collect();
    let kd: Vec<f32> = (0..k_size)
        .map(|i| ((i * 13 % 19) as f32 / 18.0 - 0.5) * 0.2)
        .collect();
    s.set_parameter("k", &kd);
    s.set_input("x", &xd);
    s.step();
    s.wait();
    s.read_output((batch * out_c * hw * hw) as usize)
}

fn conv_input_grad(coop: bool) -> Vec<f32> {
    // H*W = 196 is not a multiple of the 16-wide f32 cooperative output
    // tile on Apple Silicon. The grad-input kernel must bounds-check the
    // partial right edge instead of letting coopStoreT cross NCHW rows.
    let (batch, channels, hw) = (4u32, 64u32, 14u32);
    let x_size = (batch * channels * hw * hw) as usize;
    let k_size = (channels * channels) as usize;
    let mut g = Graph::new();
    let x = g.parameter("x", &[x_size]);
    let k = g.parameter("k", &[k_size]);
    let y = g.conv2d(x, k, batch, channels, hw, hw, channels, 1, 1, 1, 0);
    let loss = g.mean_all(y);
    g.set_outputs(vec![loss]);

    let mut s = meganeura::build(&g, config(Mode::Training, coop)).0;
    let xd: Vec<f32> = (0..x_size)
        .map(|i| ((i * 31 % 17) as f32 - 8.0) * 0.01)
        .collect();
    let kd: Vec<f32> = (0..k_size)
        .map(|i| ((i * 13 % 19) as f32 - 9.0) * 0.01)
        .collect();
    s.set_parameter("x", &xd);
    s.set_parameter("k", &kd);
    s.step();
    s.wait();

    let mut grad = vec![0.0; x_size];
    s.read_param_grad("x", &mut grad);
    grad
}

fn conv_partial_output_batch(coop: bool) -> Vec<f32> {
    // Co=1 needs a partial cooperative output tile. With N>1, padding those
    // rows changes the batch stride unless this dispatch stays scalar.
    let (batch, in_c, out_c, hw) = (4u32, 6u32, 1u32, 64u32);
    let input_size = (batch * in_c * hw * hw) as usize;
    let kernel_size = (out_c * in_c) as usize;
    let mut g = Graph::new();
    let x = g.input("x", &[input_size]);
    let k = g.parameter("k", &[kernel_size]);
    let y = g.conv2d(x, k, batch, in_c, hw, hw, out_c, 1, 1, 1, 0);
    g.set_outputs(vec![y]);

    let mut s = meganeura::build(&g, config(Mode::Inference, coop)).0;
    let xd: Vec<f32> = (0..input_size)
        .map(|i| ((i * 31 % 37) as f32 - 18.0) * 0.01)
        .collect();
    let kd: Vec<f32> = (0..kernel_size).map(|i| (i as f32 + 1.0) * 0.05).collect();
    s.set_parameter("k", &kd);
    s.set_input("x", &xd);
    s.step();
    s.wait();
    s.read_output((batch * out_c * hw * hw) as usize)
}

#[test]
fn coop_conv_unaligned_k_matches_scalar() {
    // K = 14*9 = 126, not a multiple of 4 — the failing case.
    let scalar = conv_out(14, false);
    let coop = conv_out(14, true);
    let max_abs = scalar
        .iter()
        .zip(&coop)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let nan = coop.iter().filter(|v| v.is_nan()).count();
    println!("K=126 coop-vs-scalar: max_abs_diff={max_abs:.4}, coop NaNs={nan}");
    // The bug gave max_abs ~1.9; f16 coop rounding on these bounded inputs
    // is well under 0.05. (0 if the GPU has no coop — both scalar.)
    assert_eq!(nan, 0, "coop conv produced NaNs at K=126");
    assert!(
        max_abs < 0.05,
        "coop conv wrong at K=126 (not multiple of 4): max_abs_diff={max_abs}"
    );
}

#[test]
fn coop_conv_aligned_k_matches_scalar() {
    // K = 16*9 = 144, a multiple of 4 — the already-working case (guard).
    let scalar = conv_out(16, false);
    let coop = conv_out(16, true);
    let max_abs = scalar
        .iter()
        .zip(&coop)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs < 0.05,
        "coop conv wrong at K=144: max_abs_diff={max_abs}"
    );
}

#[test]
fn coop_conv_grad_input_partial_right_edge_matches_scalar() {
    let scalar = conv_input_grad(false);
    let coop = conv_input_grad(true);
    let max_abs = scalar
        .iter()
        .zip(&coop)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs < 1e-5,
        "coop grad-input wrong at partial right edge: max_abs_diff={max_abs}"
    );
}

#[test]
fn coop_conv_batched_partial_output_tile_matches_scalar() {
    let scalar = conv_partial_output_batch(false);
    let coop = conv_partial_output_batch(true);
    let max_abs = scalar
        .iter()
        .zip(&coop)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs < 1e-5,
        "batched conv with a partial output tile changed under cooperative selection: {max_abs}"
    );
}
