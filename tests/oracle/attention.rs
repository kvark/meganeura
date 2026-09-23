//! Attention family: every forward and backward kernel variant against the
//! reference, and every attention autodiff rule against finite differences.
//!
//! Kernel selection (`Compiler::attention_dispatch*` in `src/compile.rs`,
//! default knobs): with `ept = min(head_dim, 32)` and
//! `bq = 256 / (head_dim / ept)`, a forward or dQ dispatch uses the flash
//! kernel when `q_seq >= bq` and the one-query scalar kernel otherwise; dK/dV
//! makes the same choice on the KV length (`q_seq` when causal). So flash
//! starts at 256 rows for `head_dim <= 32` (one lane per query, the tiled
//! `tpq == 1` backward path), 128 for 64, 64 for 128, 32 for 256 and 16 for
//! 512. Forward kernels stage keys in tiles of 8 with a one-key tail.

use meganeura::compile::compile_with;
use meganeura::graph::Op;
use meganeura::reference::{Feeds, gpu, gradients};
use meganeura::{Graph, NodeId, TensorType};
use std::collections::BTreeSet;
use std::panic::{AssertUnwindSafe, catch_unwind};

#[derive(Clone, Copy, Debug)]
enum Kind {
    Causal,
    Window(u32),
    Full,
    Cross,
    Mha { cross: bool },
    Rope(f32),
}

impl Kind {
    fn is_cross(self) -> bool {
        matches!(self, Kind::Mha { cross: true })
    }
}

/// Query rows, key rows, heads, KV heads, head width.
#[derive(Clone, Copy, Debug)]
struct Shape {
    q: usize,
    kv: usize,
    heads: u32,
    kv_heads: u32,
    dim: u32,
}

const fn shape(q: usize, kv: usize, heads: u32, kv_heads: u32, dim: u32) -> Shape {
    Shape {
        q,
        kv,
        heads,
        kv_heads,
        dim,
    }
}

impl Shape {
    fn q_width(self) -> usize {
        (self.heads * self.dim) as usize
    }

    fn kv_width(self) -> usize {
        (self.kv_heads * self.dim) as usize
    }
}

fn forward(g: &mut Graph, kind: Kind, s: Shape, q: NodeId, k: NodeId, v: NodeId) -> NodeId {
    let (h, kvh, hd) = (s.heads, s.kv_heads, s.dim);
    match kind {
        Kind::Causal => g.causal_attention(q, k, v, h, kvh, hd),
        Kind::Window(w) => g.sliding_window_attention(q, k, v, h, kvh, hd, w),
        Kind::Full => g.full_attention(q, k, v, h, kvh, hd),
        Kind::Cross => g.cross_attention(q, k, v, h, kvh, hd),
        Kind::Mha { cross } => g.multi_head_attn(q, k, v, h, kvh, hd, cross),
        Kind::Rope(rope_theta) => g.add_raw_node(
            Op::CausalAttentionRoPE {
                num_heads: h,
                num_kv_heads: kvh,
                head_dim: hd,
                rope_theta,
            },
            vec![q, k, v],
            TensorType::f32(vec![s.q, s.q_width()]),
        ),
    }
}

fn forward_graph(kind: Kind, s: Shape) -> Graph {
    let mut g = Graph::new();
    let q = g.input("q", &[s.q, s.q_width()]);
    let k = g.input("k", &[s.kv, s.kv_width()]);
    let v = g.input("v", &[s.kv, s.kv_width()]);
    let o = forward(&mut g, kind, s, q, k, v);
    g.set_outputs(vec![o]);
    g
}

/// The forward op and its three gradient ops, fed as autodiff feeds them
/// (rotated Q/K for RoPE), all as outputs.
fn backward_graph(kind: Kind, s: Shape) -> Graph {
    let mut g = Graph::new();
    let q = g.input("q", &[s.q, s.q_width()]);
    let k = g.input("k", &[s.kv, s.kv_width()]);
    let v = g.input("v", &[s.kv, s.kv_width()]);
    let d_out = g.input("d_out", &[s.q, s.q_width()]);
    let fwd_node = forward(&mut g, kind, s, q, k, v);
    let (gq, gk) = match kind {
        Kind::Rope(theta) => (g.rope(q, theta, s.dim), g.rope(k, theta, s.dim)),
        _ => (q, k),
    };
    let (num_heads, num_kv_heads, head_dim, is_cross) =
        (s.heads, s.kv_heads, s.dim, kind.is_cross());
    let inputs = vec![d_out, gq, gk, v];
    let q_ty = g.node(q).ty.clone();
    let k_ty = g.node(k).ty.clone();
    let dq = g.add_raw_node(
        Op::MultiHeadAttnGradQ {
            fwd_node,
            num_heads,
            num_kv_heads,
            head_dim,
            is_cross,
        },
        inputs.clone(),
        q_ty,
    );
    let dk = g.add_raw_node(
        Op::MultiHeadAttnGradK {
            fwd_node,
            num_heads,
            num_kv_heads,
            head_dim,
            is_cross,
        },
        inputs.clone(),
        k_ty.clone(),
    );
    let dv = g.add_raw_node(
        Op::MultiHeadAttnGradV {
            fwd_node,
            num_heads,
            num_kv_heads,
            head_dim,
            is_cross,
        },
        inputs,
        k_ty,
    );
    g.set_outputs(vec![fwd_node, dq, dk, dv]);
    g
}

/// `0.7 · Σ w ⊙ attention(q, k, v)` over parameters q, k, v.
fn training_graph(kind: Kind, s: Shape) -> Graph {
    let mut g = Graph::new();
    let q = g.parameter("q", &[s.q, s.q_width()]);
    let k = g.parameter("k", &[s.kv, s.kv_width()]);
    let v = g.parameter("v", &[s.kv, s.kv_width()]);
    let o = forward(&mut g, kind, s, q, k, v);
    let loss = gradients::weighted_loss(&mut g, o, 11, 0.7);
    g.set_outputs(vec![loss]);
    g
}

fn random(g: &Graph, seed: u64) -> Feeds {
    let mut feeds = Feeds::new();
    feeds.fill_random(g, seed, 1.5);
    feeds
}

/// Runs cases, keeps going past failures, and reports every kernel used.
#[derive(Default)]
struct Sweep {
    failures: Vec<String>,
    kernels: BTreeSet<String>,
}

impl Sweep {
    fn kernels(&mut self, g: &Graph, options: &gpu::Options) -> String {
        let plan = compile_with(g, &options.compile);
        let names: BTreeSet<String> = plan
            .dispatches
            .iter()
            .map(|d| format!("{:?}", d.shader))
            .collect();
        self.kernels.extend(names.iter().cloned());
        names.into_iter().collect::<Vec<_>>().join(" ")
    }

    fn run(&mut self, label: &str, check: impl FnOnce() -> Result<String, String>) {
        let outcome = catch_unwind(AssertUnwindSafe(check));
        match outcome {
            Ok(Ok(report)) => println!("ok   {label}\n{report}"),
            Ok(Err(report)) => {
                println!("FAIL {label}\n{report}");
                self.failures.push(format!("{label}\n{report}"));
            }
            Err(panic) => {
                let message = panic
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_default();
                println!("FAIL {label}: panicked: {message}");
                self.failures.push(format!("{label}: panicked: {message}"));
            }
        }
    }

    fn inference(&mut self, label: &str, g: &Graph, feeds: &Feeds, options: &gpu::Options) {
        let kernels = catch_unwind(AssertUnwindSafe(|| self.kernels(g, options)))
            .unwrap_or_else(|_| "compile panicked".to_string());
        self.run(&format!("{label} [{kernels}]"), || {
            let report = gpu::check_inference(g, feeds, options).map_err(|e| e.to_string())?;
            let text = report.to_string();
            if report.passed() { Ok(text) } else { Err(text) }
        });
    }

    fn autodiff(&mut self, label: &str, g: &Graph, feeds: &Feeds) {
        self.run(&format!("{label} autodiff"), || {
            let options = gradients::Options::default();
            let report = gradients::check(g, feeds, &options).map_err(|e| e.to_string())?;
            let text = report.to_string();
            if report.passed() { Ok(text) } else { Err(text) }
        });
    }

    fn training(&mut self, label: &str, g: &Graph, feeds: &Feeds) {
        self.run(&format!("{label} training"), || {
            let options = gpu::Options::default();
            let report = gpu::check_training(g, feeds, &options).map_err(|e| e.to_string())?;
            let text = report.to_string();
            if report.passed() { Ok(text) } else { Err(text) }
        });
    }

    #[track_caller]
    fn finish(self) {
        println!("kernels: {:?}", self.kernels);
        assert!(
            self.failures.is_empty(),
            "{} failing case(s):\n{}",
            self.failures.len(),
            self.failures.join("\n")
        );
    }
}

/// Self-attention shapes crossing every forward/dQ/dK·dV selection
/// threshold, with ragged tiles, GQA ratios 1, 2 and 4, and one-row input.
const SELF_SHAPES: &[Shape] = &[
    shape(1, 1, 2, 1, 64),     // one row: scalar everywhere
    shape(13, 13, 4, 2, 64),   // scalar, key tile of 8 plus tail
    shape(40, 40, 2, 2, 8),    // scalar, tiny heads
    shape(31, 31, 2, 1, 256),  // scalar, just below the hd=256 threshold
    shape(33, 33, 2, 1, 256),  // flash hd=256 (bq=32), ragged last block
    shape(64, 64, 4, 1, 128),  // flash hd=128 (bq=64), exact block, GQA 4
    shape(63, 63, 2, 2, 128),  // scalar, just below
    shape(130, 130, 2, 1, 64), // flash hd=64 (bq=128), ragged
    shape(260, 260, 2, 2, 32), // flash hd=32 (bq=256): one lane per query
    shape(257, 257, 1, 1, 16), // flash hd=16, tpq=1
];

fn forward_sweep(kinds: &[Kind], shapes: &[Shape]) {
    let mut sweep = Sweep::default();
    let options = gpu::Options::default();
    for &kind in kinds {
        for (n, &s) in shapes.iter().enumerate() {
            let g = forward_graph(kind, s);
            let feeds = random(&g, 100 + n as u64);
            sweep.inference(&format!("{kind:?} {s:?}"), &g, &feeds, &options);
        }
    }
    sweep.finish();
}

fn backward_sweep(kinds: &[Kind], shapes: &[Shape]) {
    let mut sweep = Sweep::default();
    let options = gpu::Options::default();
    for &kind in kinds {
        for (n, &s) in shapes.iter().enumerate() {
            let g = backward_graph(kind, s);
            let feeds = random(&g, 200 + n as u64);
            sweep.inference(&format!("{kind:?} backward {s:?}"), &g, &feeds, &options);
        }
    }
    sweep.finish();
}

#[test]
fn causal_forward() {
    forward_sweep(&[Kind::Causal], SELF_SHAPES);
}

#[test]
fn causal_forward_wide_heads() {
    // 512-wide heads: flash from 16 rows (tpq=16), scalar below.
    forward_sweep(
        &[Kind::Causal],
        &[shape(5, 5, 1, 1, 512), shape(17, 17, 2, 1, 512)],
    );
}

/// Attention kernels need power-of-two heads. Other widths (80 is common)
/// are refused when the session is built, never computed wrongly.
#[test]
fn causal_head_dim_80_is_refused() {
    for s in [shape(9, 9, 2, 1, 80), shape(70, 70, 2, 1, 80)] {
        let g = forward_graph(Kind::Causal, s);
        let refused = std::panic::catch_unwind(|| {
            let mut config = meganeura::SessionConfig::from_env();
            config.mode = meganeura::Mode::Inference;
            meganeura::build(&g, config)
        });
        assert!(refused.is_err(), "{s:?} built");
    }
}

#[test]
fn sliding_window_forward() {
    let shapes = [
        shape(1, 1, 2, 1, 64),
        shape(13, 13, 4, 2, 64),
        shape(33, 33, 2, 1, 256),
        shape(130, 130, 2, 1, 64),
        shape(260, 260, 2, 2, 32),
    ];
    forward_sweep(
        &[
            Kind::Window(1),
            Kind::Window(3),
            Kind::Window(8),
            Kind::Window(1000),
        ],
        &shapes,
    );
}

#[test]
fn full_forward() {
    forward_sweep(&[Kind::Full, Kind::Mha { cross: false }], SELF_SHAPES);
}

#[test]
fn cross_forward() {
    let shapes = [
        shape(1, 7, 2, 1, 64),
        shape(5, 1, 2, 2, 64),
        shape(7, 300, 2, 1, 64),  // scalar forward, long KV
        shape(40, 19, 2, 1, 256), // flash forward, KV tail of 3
        shape(130, 9, 4, 2, 64),  // flash forward, one KV tile plus tail
        shape(257, 20, 2, 1, 32), // flash tpq=1
    ];
    forward_sweep(&[Kind::Cross, Kind::Mha { cross: true }], &shapes);
}

#[test]
fn rope_forward() {
    forward_sweep(
        &[Kind::Rope(10_000.0)],
        &[
            shape(1, 1, 2, 1, 64),
            shape(13, 13, 4, 2, 64),
            shape(33, 33, 2, 1, 256),
        ],
    );
}

#[test]
fn causal_backward() {
    backward_sweep(&[Kind::Causal], SELF_SHAPES);
}

#[test]
fn sliding_window_backward() {
    backward_sweep(
        &[Kind::Window(1), Kind::Window(5)],
        &[
            shape(13, 13, 4, 2, 64),
            shape(33, 33, 2, 1, 256),
            shape(130, 130, 2, 1, 64),
            shape(260, 260, 2, 2, 32),
        ],
    );
}

#[test]
fn full_backward() {
    backward_sweep(&[Kind::Full, Kind::Mha { cross: false }], SELF_SHAPES);
}

#[test]
fn cross_backward() {
    let shapes = [
        shape(1, 7, 2, 1, 64),
        shape(5, 1, 2, 2, 64),
        shape(40, 9, 2, 1, 256),  // flash dQ, scalar dK/dV
        shape(5, 40, 2, 1, 256),  // scalar dQ, flash dK/dV
        shape(7, 300, 2, 2, 32),  // scalar dQ, flash dK/dV with tpq=1
        shape(257, 20, 4, 1, 32), // flash dQ tpq=1, scalar dK/dV
    ];
    backward_sweep(&[Kind::Cross, Kind::Mha { cross: true }], &shapes);
}

#[test]
fn rope_backward() {
    backward_sweep(
        &[Kind::Rope(10_000.0)],
        &[shape(13, 13, 4, 2, 64), shape(33, 33, 2, 1, 256)],
    );
}

/// The differentiable attention ops, small enough for finite differences.
const TRAINABLE: &[(Kind, Shape)] = &[
    (Kind::Causal, shape(5, 5, 2, 1, 4)),
    (Kind::Window(2), shape(6, 6, 2, 2, 4)),
    (Kind::Full, shape(5, 5, 4, 2, 2)),
    (Kind::Mha { cross: false }, shape(4, 4, 2, 1, 4)),
    (Kind::Mha { cross: true }, shape(3, 6, 2, 1, 4)),
    (Kind::Rope(100.0), shape(5, 5, 2, 1, 4)),
];

#[test]
fn attention_autodiff() {
    let mut sweep = Sweep::default();
    for (n, &(kind, s)) in TRAINABLE.iter().enumerate() {
        let g = training_graph(kind, s);
        let feeds = random(&g, 300 + n as u64);
        sweep.autodiff(&format!("{kind:?} {s:?}"), &g, &feeds);
    }
    sweep.finish();
}

#[test]
fn attention_training() {
    let mut sweep = Sweep::default();
    // The same small graphs, then shapes on the flash backward kernels.
    let flash = [
        (Kind::Causal, shape(33, 33, 2, 1, 256)),
        (Kind::Window(4), shape(33, 33, 2, 1, 256)),
        (Kind::Mha { cross: true }, shape(40, 35, 2, 1, 256)),
        (Kind::Rope(10_000.0), shape(33, 33, 2, 1, 256)),
    ];
    for (n, &(kind, s)) in TRAINABLE.iter().chain(&flash).enumerate() {
        let g = training_graph(kind, s);
        let feeds = random(&g, 400 + n as u64);
        sweep.training(&format!("{kind:?} {s:?}"), &g, &feeds);
    }
    sweep.finish();
}

fn cached_graph(s: Shape, window: Option<u32>) -> Graph {
    let mut g = Graph::new();
    let q = g.input("q", &[s.q, s.q_width()]);
    let k = g.input("k_cache", &[s.kv, s.kv_width()]);
    let v = g.input("v_cache", &[s.kv, s.kv_width()]);
    let pos = g.input_u32("kv_pos", &[1]);
    let o = match window {
        None => g.cached_attention(q, k, v, pos, s.heads, s.kv_heads, s.dim),
        Some(window) => {
            let valid = g.input_u32("valid_len", &[1]);
            g.cached_block_attention(q, k, v, pos, valid, s.heads, s.kv_heads, s.dim, window)
        }
    };
    g.set_outputs(vec![o]);
    g
}

#[test]
fn cached_attention() {
    // One query: `cached_attention.wgsl`; several: the cached flash kernel
    // (32 queries per workgroup). Positions cover a partial first key tile,
    // whole tiles plus a tail, and the last cache row.
    let mut sweep = Sweep::default();
    let options = gpu::Options::default();
    for (n, &(queries, max_seq, kv_pos)) in [
        (1, 40, 0),
        (1, 40, 20),
        (1, 40, 39),
        (5, 40, 3),
        (33, 40, 26),
    ]
    .iter()
    .enumerate()
    {
        for (heads, kv_heads) in [(2, 1), (2, 2)] {
            let s = shape(queries, max_seq, heads, kv_heads, 64);
            let g = cached_graph(s, None);
            let mut feeds = Feeds::new();
            feeds.set_u32("kv_pos", &[kv_pos]);
            feeds.fill_random(&g, 500 + n as u64, 1.5);
            sweep.inference(&format!("{s:?} kv_pos={kv_pos}"), &g, &feeds, &options);
        }
    }
    sweep.finish();
}

#[test]
fn cached_block_attention() {
    // Unsplit when block > 1 or max_seq <= 64; split-K + combine with
    // ceil(max_seq / 32) splits for single-row decode past 64 rows; forced
    // splits for blocks. Head widths cover 1..=8 values per lane. Rows past
    // `valid_len` are unspecified and must stay unwritten (NaN-poisoned).
    let mut sweep = Sweep::default();
    type Case = (usize, usize, u32, u32, u32, u32, u32, Option<u32>);
    let cases: &[Case] = &[
        // block, max_seq, head_dim, window, kv_pos, valid_len, heads/kv, splits
        (1, 40, 64, 0, 17, 1, 2, None),
        (1, 100, 64, 0, 90, 1, 2, None), // 4 splits
        (1, 100, 80, 7, 60, 1, 2, None), // split with window
        (1, 520, 4, 0, 500, 1, 1, None), // 16 splits
        (1, 70, 200, 0, 3, 1, 2, None),  // most splits empty
        (4, 30, 64, 0, 0, 4, 2, None),
        (4, 30, 64, 0, 9, 2, 1, None), // two unspecified rows
        (4, 30, 64, 0, 9, 9, 2, None), // valid_len clamps to the block
        (5, 50, 100, 3, 20, 5, 2, None),
        (5, 50, 512, 0, 12, 4, 2, None),
        (3, 96, 64, 37, 50, 3, 2, Some(4)), // forced split on a block
        (3, 96, 80, 0, 7, 2, 2, Some(16)),
    ];
    for (n, &(block, max_seq, dim, window, kv_pos, valid_len, ratio, splits)) in
        cases.iter().enumerate()
    {
        let s = shape(block, max_seq, 2, 2 / ratio, dim);
        let g = cached_graph(s, Some(window));
        let mut feeds = Feeds::new();
        feeds.set_u32("kv_pos", &[kv_pos]);
        feeds.set_u32("valid_len", &[valid_len]);
        feeds.fill_random(&g, 600 + n as u64, 1.5);
        let mut options = gpu::Options::default();
        options.compile.cached_attention_splits = splits;
        sweep.inference(
            &format!(
                "{s:?} window={window} kv_pos={kv_pos} valid_len={valid_len} splits={splits:?}"
            ),
            &g,
            &feeds,
            &options,
        );
    }
    sweep.finish();
}

#[test]
fn chunked_relative_attention() {
    let mut sweep = Sweep::default();
    let options = gpu::Options::default();
    for (n, &(seq, heads, dim, left, softcap)) in [
        (1, 2, 4, 2, 5.0f32),
        (9, 2, 4, 2, 5.0), // only the current key
        (9, 2, 4, 3, 5.0),
        (12, 1, 64, 5, 2.0), // tanh saturates
        (7, 2, 100, 4, 50.0),
        (6, 1, 512, 8, 10.0), // window longer than the sequence
    ]
    .iter()
    .enumerate()
    {
        let mut g = Graph::new();
        let width = (heads * dim) as usize;
        let q = g.input("q", &[seq, width]);
        let k = g.input("k", &[seq, width]);
        let v = g.input("v", &[seq, width]);
        let rel = g.input("relative_k", &[left as usize, width]);
        let o = g.chunked_relative_attention(q, k, v, rel, heads, dim, left, softcap);
        g.set_outputs(vec![o]);
        let feeds = random(&g, 700 + n as u64);
        sweep.inference(
            &format!("seq={seq} heads={heads} head_dim={dim} left={left} softcap={softcap}"),
            &g,
            &feeds,
            &options,
        );
    }
    sweep.finish();
}
