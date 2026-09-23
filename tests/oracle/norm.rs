//! Norms, RoPE and pairwise row ops against the reference: every lowering
//! the compiler can select, the plan-level RmsNorm fusions, and the
//! autodiff rules.
//!
//! Lowerings covered (see `compile.rs`):
//! * RmsNorm / LayerNorm: schedule reduction (one row per workgroup, and
//!   `256 / next_pow2(cols)` packed rows for `cols` in 2..=32) and the
//!   hand-written `rms_norm.wgsl` / `layer_norm.wgsl`.
//! * RmsNormGradW: single-pass column kernel (`rows < 4`), row-parallel
//!   kernel with row blocks of 1, 2, 4 and 32 plus SumRows.
//! * RmsNormGradX: packed rows (`cols` in 2..=32) and 256-lane rows.
//! * LayerNormGradWB: one block written directly (`rows = 1`), blocks of 1,
//!   2, 4 and 32 rows plus SumRows. LayerNormGradX: 256-lane rows.
//! * RoPE: static offset, dynamic offset, dynamic offset with frequency
//!   factors, explicit positions; RoPEGrad.
//! * Pairwise ops: the generated reductions and `pairwise_grad.wgsl` modes.
//! * Plan fusions: `fuse_rmsnorm_into_gemv` (M = 1, MatMulGemv and
//!   MatMulGemvBT), `fuse_rmsnorm_into_add` (both RmsNorm lowerings), and
//!   the unfused M > 1 matmul. `fuse_rmsnorm_prologues` needs cooperative
//!   matrices, which lavapipe lacks; `rms_norm_fusions` reaches it only on a
//!   device that selects them for the M > 1 matmul.
//! * One-workgroup-per-row kernels with more than 65535 rows.

use meganeura::compile::ShaderEntry;
use meganeura::graph::{Op, PairwiseGradKind, TensorType};
use meganeura::reference::{Feeds, gpu, gradients};
use meganeura::{Graph, Mode, NodeId, SessionConfig};

const EPS: f32 = 1e-5;

/// Collects failures so a sweep reports every failing variant at once.
#[derive(Default)]
struct Sweep {
    failures: Vec<String>,
}

impl Sweep {
    fn inference(&mut self, label: &str, g: &Graph, feeds: &Feeds, options: &gpu::Options) {
        let report = gpu::check_inference(g, feeds, options).unwrap();
        println!("{label}\n{report}");
        if !report.passed() {
            self.failures.push(format!("{label}:\n{report}"));
        }
    }

    fn report(&mut self, label: &str, report: meganeura::reference::Report) {
        println!("{label}\n{report}");
        if !report.passed() {
            self.failures.push(format!("{label}:\n{report}"));
        }
    }

    #[track_caller]
    fn finish(self) {
        assert!(
            self.failures.is_empty(),
            "{} failing case(s):\n{}",
            self.failures.len(),
            self.failures.join("\n")
        );
    }
}

fn options(schedule_reduction: bool) -> gpu::Options {
    let mut options = gpu::Options::default();
    options.compile.use_schedule_reduction = schedule_reduction;
    options
}

fn random_feeds(g: &Graph, seed: u64) -> Feeds {
    let mut feeds = Feeds::new();
    feeds.fill_random(g, seed, 1.0);
    feeds
}

/// The dispatches of the inference plan `options` builds for `g`.
fn plan(g: &Graph, options: &gpu::Options) -> Vec<(ShaderEntry, bool)> {
    let mut config = SessionConfig::from_env();
    config.mode = Mode::Inference;
    config.options = options.compile.clone();
    config.optimize = options.optimize;
    config.runtime.coop = options.coop;
    let (session, _) = meganeura::build(g, config);
    session
        .plan()
        .dispatches
        .iter()
        .map(|d| (d.shader.clone(), d.gemv_rmsnorm.is_some()))
        .collect()
}

fn norm_inputs(g: &mut Graph, rows: usize, cols: usize) -> (NodeId, NodeId, NodeId) {
    let x = g.input("x", &[rows, cols]);
    let w = g.parameter("w", &[cols]);
    let b = g.parameter("b", &[cols]);
    (x, w, b)
}

const FORWARD_SHAPES: &[(usize, usize)] = &[
    (1, 1),
    (1, 7),
    (3, 2),
    (300, 3),
    (5, 17),
    (9, 32),
    (4, 33),
    (3, 256),
    (2, 300),
    (1, 1000),
    (70, 64),
];

#[test]
fn rms_norm_forward() {
    let mut sweep = Sweep::default();
    for schedule in [true, false] {
        for (i, &(rows, cols)) in FORWARD_SHAPES.iter().enumerate() {
            let mut g = Graph::new();
            let (x, w, _) = norm_inputs(&mut g, rows, cols);
            let y = g.rms_norm(x, w, EPS);
            g.set_outputs(vec![y]);
            let feeds = random_feeds(&g, 10 + i as u64);
            let label = format!("rms_norm [{rows}, {cols}] schedule={schedule}");
            sweep.inference(&label, &g, &feeds, &options(schedule));
        }
    }
    sweep.finish();
}

#[test]
fn layer_norm_forward() {
    let mut sweep = Sweep::default();
    for schedule in [true, false] {
        for (i, &(rows, cols)) in FORWARD_SHAPES.iter().enumerate() {
            let mut g = Graph::new();
            let (x, w, b) = norm_inputs(&mut g, rows, cols);
            let y = g.layer_norm(x, w, b, EPS);
            g.set_outputs(vec![y]);
            let feeds = random_feeds(&g, 20 + i as u64);
            let label = format!("layer_norm [{rows}, {cols}] schedule={schedule}");
            sweep.inference(&label, &g, &feeds, &options(schedule));
        }
    }
    sweep.finish();
}

/// Rows whose mean dwarfs their spread. A two-pass variance is accurate
/// here; `E[x²] − mean²` in `f32` cancels catastrophically, and when the
/// difference rounds to zero or below, `rsqrt(var + eps)` is NaN. A
/// constant row of 300.0 is the minimal case: every intermediate is exact,
/// yet the scheduled kernel returns NaN on lavapipe, where the compiled
/// `(E[x²] − mean²) + eps` evidently no longer adds `eps` to an exact zero.
#[test]
fn layer_norm_offset_rows() {
    let mut sweep = Sweep::default();
    let (rows, cols) = (4, 64);
    for schedule in [true, false] {
        for (offset, spread) in [
            (100.0f32, 1.0f32),
            (300.0, 0.01),
            (300.0, 0.0),
            (1000.0, 0.0),
        ] {
            let mut g = Graph::new();
            let (x, w, b) = norm_inputs(&mut g, rows, cols);
            let y = g.layer_norm(x, w, b, EPS);
            g.set_outputs(vec![y]);
            let mut feeds = Feeds::new();
            let mut rng = meganeura::reference::Rng::new(7);
            let data: Vec<f32> = (0..rows * cols)
                .map(|_| offset + spread * rng.uniform(-1.0, 1.0))
                .collect();
            feeds.set("x", &data);
            feeds.fill_random(&g, 8, 1.0);
            let label = format!("layer_norm offset={offset} spread={spread} schedule={schedule}");
            sweep.inference(&label, &g, &feeds, &options(schedule));
        }
    }
    sweep.finish();
}

/// Build `op(dy, x, w)` with `dy` and `x` inputs of `[rows, cols]`.
fn norm_grad_graph(
    rows: usize,
    cols: usize,
    op: impl Fn(&mut Graph, NodeId, NodeId, NodeId) -> NodeId,
) -> Graph {
    let mut g = Graph::new();
    let dy = g.input("dy", &[rows, cols]);
    let x = g.input("x", &[rows, cols]);
    let w = g.parameter("w", &[cols]);
    let out = op(&mut g, dy, x, w);
    g.set_outputs(vec![out]);
    g
}

/// Row counts reaching every weight-gradient block size:
/// `clamp(rows / 1024, 1, 32)` rounded down to a power of two.
const WEIGHT_GRAD_ROWS: &[usize] = &[1, 2, 3, 4, 37, 2050, 4100, 33_000];

#[test]
fn rms_norm_grad_w() {
    let mut sweep = Sweep::default();
    for &rows in WEIGHT_GRAD_ROWS {
        for cols in [1, 5, 64, 300] {
            if rows * cols > 2_000_000 {
                continue;
            }
            let g = norm_grad_graph(rows, cols, |g, dy, x, w| g.rms_norm_grad_w(dy, x, w, EPS));
            let feeds = random_feeds(&g, (rows * 31 + cols) as u64);
            sweep.inference(
                &format!("rms_norm_grad_w [{rows}, {cols}]"),
                &g,
                &feeds,
                &options(true),
            );
        }
    }
    sweep.finish();
}

#[test]
fn rms_norm_grad_x() {
    let mut sweep = Sweep::default();
    for rows in [1, 5, 100] {
        for cols in [1, 2, 3, 8, 17, 32, 33, 256, 300, 700] {
            let g = norm_grad_graph(rows, cols, |g, dy, x, w| g.rms_norm_grad_x(dy, x, w, EPS));
            let feeds = random_feeds(&g, (rows * 37 + cols) as u64);
            sweep.inference(
                &format!("rms_norm_grad_x [{rows}, {cols}]"),
                &g,
                &feeds,
                &options(true),
            );
        }
    }
    sweep.finish();
}

#[test]
fn layer_norm_grad_wb() {
    let mut sweep = Sweep::default();
    for &rows in WEIGHT_GRAD_ROWS {
        for cols in [1, 5, 64, 300] {
            if rows * cols > 2_000_000 {
                continue;
            }
            let g = norm_grad_graph(rows, cols, |g, dy, x, w| {
                g.layer_norm_grad_wb(dy, x, w, EPS)
            });
            let feeds = random_feeds(&g, (rows * 41 + cols) as u64);
            sweep.inference(
                &format!("layer_norm_grad_wb [{rows}, {cols}]"),
                &g,
                &feeds,
                &options(true),
            );
        }
    }
    sweep.finish();
}

#[test]
fn layer_norm_grad_x() {
    let mut sweep = Sweep::default();
    for rows in [1, 5, 100] {
        for cols in [1, 3, 32, 256, 300, 700] {
            let g = norm_grad_graph(rows, cols, |g, dy, x, w| g.layer_norm_grad_x(dy, x, w, EPS));
            let feeds = random_feeds(&g, (rows * 43 + cols) as u64);
            sweep.inference(
                &format!("layer_norm_grad_x [{rows}, {cols}]"),
                &g,
                &feeds,
                &options(true),
            );
        }
    }
    sweep.finish();
}

/// `(rows, dim, head_dim)`: one head, several heads, a two-element head,
/// and enough pairs for several workgroups.
const ROPE_SHAPES: &[(usize, usize, u32)] = &[
    (1, 8, 8),
    (5, 8, 8),
    (3, 12, 2),
    (7, 128, 64),
    (300, 64, 16),
];

#[test]
fn rope_static() {
    let mut sweep = Sweep::default();
    for (i, &(rows, dim, head_dim)) in ROPE_SHAPES.iter().enumerate() {
        for (theta, offset) in [(10_000.0, 0), (10_000.0, 7), (500_000.0, 1000)] {
            let mut g = Graph::new();
            let x = g.input("x", &[rows, dim]);
            let y = g.rope_with_offset(x, theta, offset, head_dim);
            g.set_outputs(vec![y]);
            let feeds = random_feeds(&g, 50 + i as u64);
            sweep.inference(
                &format!("rope [{rows}, {dim}] head_dim={head_dim} theta={theta} offset={offset}"),
                &g,
                &feeds,
                &options(true),
            );

            let mut g = Graph::new();
            let dy = g.input("dy", &[rows, dim]);
            let dx = g.rope_grad(dy, theta, offset, head_dim);
            g.set_outputs(vec![dx]);
            let feeds = random_feeds(&g, 60 + i as u64);
            sweep.inference(
                &format!(
                    "rope_grad [{rows}, {dim}] head_dim={head_dim} theta={theta} offset={offset}"
                ),
                &g,
                &feeds,
                &options(true),
            );
        }
    }
    sweep.finish();
}

#[test]
fn rope_dynamic() {
    let mut sweep = Sweep::default();
    for (i, &(rows, dim, head_dim)) in ROPE_SHAPES.iter().enumerate() {
        for offset in [0u32, 13, 1000] {
            for factors in [false, true] {
                let mut g = Graph::new();
                let x = g.input("x", &[rows, dim]);
                let pos = g.input_u32("pos", &[1]);
                let y = if factors {
                    let f = g.input("factors", &[head_dim as usize / 2]);
                    g.rope_dynamic_offset_factors(x, 10_000.0, pos, head_dim, f)
                } else {
                    g.rope_dynamic_offset(x, 10_000.0, pos, head_dim)
                };
                g.set_outputs(vec![y]);
                let mut feeds = Feeds::new();
                feeds.set_u32("pos", &[offset]);
                if factors {
                    // Divisors in [1, 8], as in long-context frequency scaling.
                    let f: Vec<f32> = (0..head_dim / 2)
                        .map(|k| 1.0 + (k * 7 % 8) as f32)
                        .collect();
                    feeds.set("factors", &f);
                }
                feeds.fill_random(&g, 70 + i as u64, 1.0);
                sweep.inference(
                    &format!(
                        "rope dynamic [{rows}, {dim}] head_dim={head_dim} offset={offset} factors={factors}"
                    ),
                    &g,
                    &feeds,
                    &options(true),
                );
            }
        }
    }
    sweep.finish();
}

#[test]
fn rope_positions() {
    let mut sweep = Sweep::default();
    for (i, &(rows, dim, head_dim)) in ROPE_SHAPES.iter().enumerate() {
        for max_pos in [16u32, 4096] {
            let mut g = Graph::new();
            let x = g.input("x", &[rows, dim]);
            let pos = g.input_u32("pos", &[rows]);
            let y = g.rope_with_positions(x, 10_000.0, pos, head_dim);
            g.set_outputs(vec![y]);
            let mut rng = meganeura::reference::Rng::new(80 + i as u64);
            let positions: Vec<u32> = (0..rows).map(|_| rng.below(max_pos)).collect();
            let mut feeds = Feeds::new();
            feeds.set_u32("pos", &positions);
            feeds.fill_random(&g, 90 + i as u64, 1.0);
            sweep.inference(
                &format!("rope positions [{rows}, {dim}] head_dim={head_dim} max_pos={max_pos}"),
                &g,
                &feeds,
                &options(true),
            );
        }
    }
    sweep.finish();
}

/// `(M, D, P)` for the pairwise ops.
const PAIRWISE_SHAPES: &[(usize, usize, usize)] =
    &[(1, 1, 1), (3, 5, 2), (17, 64, 4), (300, 3, 1), (2, 300, 3)];

/// `[M, D]` directions, unit-length when `unit`.
fn directions(m: usize, d: usize, seed: u64, unit: bool) -> Vec<f32> {
    let mut rng = meganeura::reference::Rng::new(seed);
    let mut data: Vec<f32> = (0..m * d).map(|_| rng.uniform(-1.0, 1.0)).collect();
    if unit {
        for row in data.chunks_mut(d) {
            let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt();
            row.iter_mut().for_each(|v| *v /= norm);
        }
    }
    data
}

#[test]
fn pairwise_forward() {
    let mut sweep = Sweep::default();
    for (i, &(m, d, p)) in PAIRWISE_SHAPES.iter().enumerate() {
        let mut g = Graph::new();
        let left = g.input("left", &[m, d]);
        let right = g.input("right", &[m * p, d]);
        let y = g.pairwise_squared_distance(left, right, p);
        g.set_outputs(vec![y]);
        let feeds = random_feeds(&g, 100 + i as u64);
        sweep.inference(
            &format!("pairwise_squared_distance M={m} D={d} P={p}"),
            &g,
            &feeds,
            &options(true),
        );

        for unit in [true, false] {
            let mut g = Graph::new();
            let v = g.input("v", &[m * p, d]);
            let u = g.input("u", &[m, d]);
            let y = g.pairwise_vector_rejection(v, u, p);
            g.set_outputs(vec![y]);
            let mut feeds = Feeds::new();
            feeds.set("u", &directions(m, d, 110 + i as u64, unit));
            feeds.fill_random(&g, 120 + i as u64, 1.0);
            sweep.inference(
                &format!("pairwise_vector_rejection M={m} D={d} P={p} unit={unit}"),
                &g,
                &feeds,
                &options(true),
            );
        }
    }
    sweep.finish();
}

#[test]
fn pairwise_grad() {
    let mut sweep = Sweep::default();
    for (i, &(m, d, p)) in PAIRWISE_SHAPES.iter().enumerate() {
        for kind in [
            PairwiseGradKind::DistanceLeft,
            PairwiseGradKind::DistanceRight,
            PairwiseGradKind::RejectionDirections,
        ] {
            let mut g = Graph::new();
            let (grad, first, second, out) = match kind {
                PairwiseGradKind::RejectionDirections => (
                    g.input("g", &[m * p, d]),
                    g.input("v", &[m * p, d]),
                    g.input("u", &[m, d]),
                    vec![m, d],
                ),
                _ => (
                    g.input("g", &[m, p]),
                    g.input("left", &[m, d]),
                    g.input("right", &[m * p, d]),
                    if matches!(kind, PairwiseGradKind::DistanceLeft) {
                        vec![m, d]
                    } else {
                        vec![m * p, d]
                    },
                ),
            };
            let y = g.add_raw_node(
                Op::PairwiseGrad {
                    kind,
                    inner: d as u32,
                    pairs: p as u32,
                },
                vec![grad, first, second],
                TensorType::f32(out),
            );
            g.set_outputs(vec![y]);
            let feeds = random_feeds(&g, 130 + i as u64);
            sweep.inference(
                &format!("pairwise_grad {kind:?} M={m} D={d} P={p}"),
                &g,
                &feeds,
                &options(true),
            );
        }
    }
    sweep.finish();
}

/// `rms_norm → matmul` at decode (M = 1) and prefill (M > 1) shapes, with
/// `[K, N]` and `[N, K]` weights, and `rms_norm → add`. Asserts the fusion
/// each case is meant to reach actually happens.
#[test]
fn rms_norm_fusions() {
    let mut sweep = Sweep::default();
    let (k, n) = (96, 40);
    for schedule in [true, false] {
        let opts = options(schedule);
        for m in [1, 5] {
            for transposed in [false, true] {
                let mut g = Graph::new();
                let x = g.input("x", &[m, k]);
                let w = g.parameter("w", &[k]);
                let h = g.rms_norm(x, w, EPS);
                let y = if transposed {
                    let b = g.parameter("proj", &[n, k]);
                    g.matmul_bt(h, b)
                } else {
                    let b = g.parameter("proj", &[k, n]);
                    g.matmul(h, b)
                };
                g.set_outputs(vec![y]);
                let dispatches = plan(&g, &opts);
                let folded = dispatches.iter().any(|&(_, fused)| fused);
                println!("M={m} transposed={transposed}: {dispatches:?}");
                if m == 1 && !folded {
                    sweep
                        .failures
                        .push(format!("M=1 transposed={transposed} schedule={schedule}: RmsNorm was not folded into the GEMV"));
                }
                let feeds = random_feeds(&g, 140 + m as u64);
                sweep.inference(
                    &format!(
                        "rms_norm -> matmul M={m} transposed={transposed} schedule={schedule} folded={folded}"
                    ),
                    &g,
                    &feeds,
                    &opts,
                );
            }
        }

        for rows in [1, 3, 40] {
            let mut g = Graph::new();
            let x = g.input("x", &[rows, k]);
            let r = g.input("residual", &[rows, k]);
            let w = g.parameter("w", &[k]);
            let h = g.rms_norm(x, w, EPS);
            let y = g.add(h, r);
            g.set_outputs(vec![y]);
            let dispatches = plan(&g, &opts);
            let fused = dispatches
                .iter()
                .any(|(s, _)| *s == ShaderEntry::RmsNormAdd);
            if !fused {
                sweep.failures.push(format!(
                    "rows={rows} schedule={schedule}: RmsNorm+Add was not fused: {dispatches:?}"
                ));
            }
            let feeds = random_feeds(&g, 150 + rows as u64);
            sweep.inference(
                &format!("rms_norm -> add rows={rows} schedule={schedule}"),
                &g,
                &feeds,
                &opts,
            );
        }

        // Also reading the norm keeps it materialized next to the fused
        // consumer's inputs.
        let mut g = Graph::new();
        let x = g.input("x", &[1, k]);
        let w = g.parameter("w", &[k]);
        let b = g.parameter("proj", &[k, n]);
        let h = g.rms_norm(x, w, EPS);
        let y = g.matmul(h, b);
        g.set_outputs(vec![y, h]);
        let feeds = random_feeds(&g, 160);
        sweep.inference(
            &format!("rms_norm -> matmul, norm exposed schedule={schedule}"),
            &g,
            &feeds,
            &opts,
        );
    }
    sweep.finish();
}

fn check_gradients(sweep: &mut Sweep, label: &str, g: &Graph, feeds: &Feeds) {
    let report = gradients::check(g, feeds, &gradients::Options::default()).unwrap();
    sweep.report(&format!("{label}: autodiff vs finite differences"), report);
    let report = gpu::check_training(g, feeds, &gpu::Options::default()).unwrap();
    sweep.report(&format!("{label}: training step"), report);
}

#[test]
fn norm_gradients() {
    let mut sweep = Sweep::default();
    for (rows, cols) in [(1, 6), (5, 17), (4, 300)] {
        let mut g = Graph::new();
        let x = g.parameter("x", &[rows, cols]);
        let w = g.parameter("w", &[cols]);
        let y = g.rms_norm(x, w, EPS);
        let loss = gradients::weighted_loss(&mut g, y, 1, 0.7);
        g.set_outputs(vec![loss]);
        let feeds = random_feeds(&g, 200 + cols as u64);
        check_gradients(
            &mut sweep,
            &format!("rms_norm [{rows}, {cols}]"),
            &g,
            &feeds,
        );

        let mut g = Graph::new();
        let x = g.parameter("x", &[rows, cols]);
        let w = g.parameter("w", &[cols]);
        let b = g.parameter("b", &[cols]);
        let y = g.layer_norm(x, w, b, EPS);
        let loss = gradients::weighted_loss(&mut g, y, 2, -1.3);
        g.set_outputs(vec![loss]);
        let feeds = random_feeds(&g, 210 + cols as u64);
        check_gradients(
            &mut sweep,
            &format!("layer_norm [{rows}, {cols}]"),
            &g,
            &feeds,
        );
    }
    sweep.finish();
}

#[test]
fn rope_gradients() {
    let mut sweep = Sweep::default();
    for &(rows, dim, head_dim) in &[(1usize, 8usize, 8u32), (5, 12, 4), (6, 32, 16)] {
        for offset in [0, 9] {
            let mut g = Graph::new();
            let x = g.parameter("x", &[rows, dim]);
            let y = g.rope_with_offset(x, 10_000.0, offset, head_dim);
            let loss = gradients::weighted_loss(&mut g, y, 3, 0.6);
            g.set_outputs(vec![loss]);
            let feeds = random_feeds(&g, 220 + rows as u64);
            check_gradients(
                &mut sweep,
                &format!("rope [{rows}, {dim}] head_dim={head_dim} offset={offset}"),
                &g,
                &feeds,
            );
        }
    }
    sweep.finish();
}

/// The dynamic offset and the frequency factors change the rotation, and
/// `RoPEGrad` has neither, so autodiff refuses these decode-time forms
/// rather than return the gradient of a different rotation.
#[test]
fn rope_dynamic_forms_are_not_differentiated() {
    let (rows, dim, head_dim) = (3, 8, 8u32);
    for factors in [false, true] {
        let mut g = Graph::new();
        let x = g.parameter("x", &[rows, dim]);
        let pos = g.input_u32("pos", &[1]);
        let y = if factors {
            let f = g.input("factors", &[head_dim as usize / 2]);
            g.rope_dynamic_offset_factors(x, 10_000.0, pos, head_dim, f)
        } else {
            g.rope_dynamic_offset(x, 10_000.0, pos, head_dim)
        };
        let loss = gradients::weighted_loss(&mut g, y, 4, 0.6);
        g.set_outputs(vec![loss]);
        let refused = std::panic::catch_unwind(|| meganeura::autodiff::differentiate(&g));
        assert!(refused.is_err(), "factors={factors}");
    }
}

#[test]
fn pairwise_gradients() {
    let mut sweep = Sweep::default();
    for &(m, d, p) in &[(1usize, 1usize, 1usize), (3, 5, 2), (4, 7, 3)] {
        let mut g = Graph::new();
        let left = g.parameter("left", &[m, d]);
        let right = g.parameter("right", &[m * p, d]);
        let y = g.pairwise_squared_distance(left, right, p);
        let loss = gradients::weighted_loss(&mut g, y, 5, 0.8);
        g.set_outputs(vec![loss]);
        let feeds = random_feeds(&g, 240 + d as u64);
        check_gradients(
            &mut sweep,
            &format!("pairwise_squared_distance M={m} D={d} P={p}"),
            &g,
            &feeds,
        );

        for unit in [true, false] {
            let mut g = Graph::new();
            let v = g.parameter("v", &[m * p, d]);
            let u = g.parameter("u", &[m, d]);
            let y = g.pairwise_vector_rejection(v, u, p);
            let loss = gradients::weighted_loss(&mut g, y, 6, -0.9);
            g.set_outputs(vec![loss]);
            let mut feeds = Feeds::new();
            feeds.set("u", &directions(m, d, 250 + d as u64, unit));
            feeds.fill_random(&g, 260 + d as u64, 1.0);
            check_gradients(
                &mut sweep,
                &format!("pairwise_vector_rejection M={m} D={d} P={p} unit={unit}"),
                &g,
                &feeds,
            );
        }
    }
    sweep.finish();
}

/// A RoPE node with both a static `pos_offset` and a dynamic offset input:
/// the op documents both as added to the row index.
#[test]
fn rope_static_and_dynamic_offset() {
    let mut sweep = Sweep::default();
    let (rows, dim) = (4, 8);
    let mut g = Graph::new();
    let x = g.input("x", &[rows, dim]);
    let pos = g.input_u32("pos", &[1]);
    let y = g.add_raw_node(
        Op::RoPE {
            theta: 10_000.0,
            pos_offset: 3,
            head_dim: dim as u32,
            freq_factors: false,
        },
        vec![x, pos],
        TensorType::f32(vec![rows, dim]),
    );
    g.set_outputs(vec![y]);
    let mut feeds = Feeds::new();
    feeds.set_u32("pos", &[5]);
    feeds.fill_random(&g, 300, 1.0);
    sweep.inference("rope pos_offset=3 + dynamic 5", &g, &feeds, &options(true));
    sweep.finish();
}

/// More rows than one dispatch dimension holds (65535 workgroups) for the
/// kernels that run one workgroup per row.
#[test]
fn norm_many_rows() {
    let mut sweep = Sweep::default();
    let (rows, cols) = (66_000, 40);
    for schedule in [true, false] {
        let mut g = Graph::new();
        let (x, w, b) = norm_inputs(&mut g, rows, cols);
        let y = g.rms_norm(x, w, EPS);
        let z = g.layer_norm(x, w, b, EPS);
        g.set_outputs(vec![y, z]);
        let feeds = random_feeds(&g, 310);
        sweep.inference(
            &format!("rms_norm + layer_norm [{rows}, {cols}] schedule={schedule}"),
            &g,
            &feeds,
            &options(schedule),
        );
    }
    let mut g = Graph::new();
    let dy = g.input("dy", &[rows, cols]);
    let x = g.input("x", &[rows, cols]);
    let w = g.parameter("w", &[cols]);
    let a = g.rms_norm_grad_x(dy, x, w, EPS);
    let b = g.layer_norm_grad_x(dy, x, w, EPS);
    g.set_outputs(vec![a, b]);
    let feeds = random_feeds(&g, 320);
    sweep.inference(
        &format!("rms_norm_grad_x + layer_norm_grad_x [{rows}, {cols}]"),
        &g,
        &feeds,
        &options(true),
    );
    sweep.finish();
}
