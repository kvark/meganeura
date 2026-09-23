//! Differential tests for the vision family: convolutions (direct,
//! depthwise, Winograd, both derivatives), max pooling and group
//! normalization, on every kernel variant the compiler selects.
//!
//! Each case names the shader it is meant to reach, and the test checks the
//! compiled plan really dispatches it, so a change to the selection
//! heuristics cannot silently leave a variant untested.

use meganeura::compile::{ExecutionPlan, Kernel, ShaderEntry, group_norm_chunks};
use meganeura::graph::Op;
use meganeura::reference::{Feeds, Report, Rng, gpu, gradients};
use meganeura::{CoopPolicy, Graph, Mode, NodeId, Session, SessionConfig, SessionOptions};
use std::sync::Arc;

/// Shaders the session compiled for `graph` under `options` dispatches.
fn dispatched(graph: &Graph, mode: Mode, options: &gpu::Options) -> Vec<ShaderEntry> {
    let mut config = SessionConfig::from_env();
    config.mode = mode;
    config.options = options.compile.clone();
    config.optimize = options.optimize;
    config.runtime.coop = options.coop;
    let (session, _) = meganeura::build(graph, config);
    session
        .plan()
        .dispatches
        .iter()
        .map(|d| d.shader.clone())
        .collect()
}

/// Collects failures so one broken case does not hide the others.
#[derive(Default)]
struct Failures(Vec<String>);

impl Failures {
    fn report(&mut self, label: &str, report: &Report) {
        println!("{label}\n{report}");
        if !report.passed() {
            self.0.push(format!("{label}:\n{report}"));
        }
    }

    /// Record that `label` did not compile to `expected`.
    fn expect_shader(&mut self, label: &str, shaders: &[ShaderEntry], expected: &ShaderEntry) {
        if !shaders.contains(expected) {
            self.0.push(format!(
                "{label}: expected a {expected:?} dispatch, plan has {shaders:?}"
            ));
        }
    }

    #[track_caller]
    fn assert_none(&self) {
        assert!(self.0.is_empty(), "{}", self.0.join("\n"));
    }
}

fn inference(
    failures: &mut Failures,
    label: &str,
    graph: &Graph,
    feeds: &Feeds,
    options: &gpu::Options,
    expected: &[ShaderEntry],
) {
    let shaders = dispatched(graph, Mode::Inference, options);
    for e in expected {
        failures.expect_shader(label, &shaders, e);
    }
    let report = gpu::check_inference(graph, feeds, options).unwrap();
    failures.report(&format!("{label} {expected:?}"), &report);
}

fn random(n: usize, seed: u64, lo: f32, hi: f32) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| rng.uniform(lo, hi)).collect()
}

/// A 2D convolution configuration.
#[derive(Clone, Copy, Debug)]
struct Conv {
    n: u32,
    ci: u32,
    h: u32,
    w: u32,
    co: u32,
    kh: u32,
    kw: u32,
    stride: u32,
    ph: u32,
    pw: u32,
}

impl Conv {
    #[allow(clippy::too_many_arguments)]
    const fn new(
        n: u32,
        ci: u32,
        (h, w): (u32, u32),
        co: u32,
        (kh, kw): (u32, u32),
        stride: u32,
        (ph, pw): (u32, u32),
    ) -> Self {
        Self {
            n,
            ci,
            h,
            w,
            co,
            kh,
            kw,
            stride,
            ph,
            pw,
        }
    }

    fn out(self) -> (u32, u32) {
        (
            (self.h + 2 * self.ph - self.kh) / self.stride + 1,
            (self.w + 2 * self.pw - self.kw) / self.stride + 1,
        )
    }

    fn x_len(self) -> usize {
        (self.n * self.ci * self.h * self.w) as usize
    }

    fn w_len(self) -> usize {
        (self.co * self.ci * self.kh * self.kw) as usize
    }

    fn y_len(self) -> usize {
        let (oh, ow) = self.out();
        (self.n * self.co * oh * ow) as usize
    }

    fn conv(self, g: &mut Graph, x: NodeId, w: NodeId) -> NodeId {
        g.conv2d_hw(
            x,
            w,
            self.n,
            self.ci,
            self.h,
            self.w,
            self.co,
            self.kh,
            self.kw,
            self.stride,
            self.ph,
            self.pw,
        )
    }

    fn forward_graph(self) -> Graph {
        let mut g = Graph::new();
        let x = g.input("x", &[self.x_len()]);
        let w = g.parameter("w", &[self.w_len()]);
        let y = self.conv(&mut g, x, w);
        g.set_outputs(vec![y]);
        g
    }

    fn grad_input_graph(self) -> Graph {
        let mut g = Graph::new();
        let dy = g.input("dy", &[self.y_len()]);
        let w = g.input("w", &[self.w_len()]);
        let dx = g.conv2d_grad_input(
            dy,
            w,
            self.n,
            self.ci,
            self.h,
            self.w,
            self.co,
            self.kh,
            self.kw,
            self.stride,
            self.ph,
            self.pw,
        );
        g.set_outputs(vec![dx]);
        g
    }

    fn grad_weight_graph(self) -> Graph {
        let mut g = Graph::new();
        let dy = g.input("dy", &[self.y_len()]);
        let x = g.input("x", &[self.x_len()]);
        let dw = g.conv2d_grad_weight(
            dy,
            x,
            self.ci,
            self.h,
            self.w,
            self.co,
            self.kh,
            self.kw,
            self.stride,
            self.ph,
            self.pw,
        );
        g.set_outputs(vec![dw]);
        g
    }
}

/// Tile selection (`compile::conv_register_tile`): the widest of 64/32/16
/// whose grid `ceil(M/t) · ceil(N/t) · batch` has at least 64 workgroups,
/// else 16. Forward: M = Co, N = oH·oW, batch = N.
#[test]
fn conv2d_forward() {
    use ShaderEntry::{Conv2dGemm, Conv2dGemm16, Conv2dGemmSmall};
    let cases = [
        // 64-wide tile: 2·16·2 = 64 workgroups; Co, oH·oW, K = 45 unaligned.
        (Conv::new(2, 5, (31, 33), 65, (3, 3), 1, (1, 1)), Conv2dGemm),
        // 32-wide tile: 2·8·4 = 64 at 32, 16 at 64.
        (
            Conv::new(4, 3, (15, 17), 33, (3, 3), 1, (1, 1)),
            Conv2dGemmSmall,
        ),
        // 16-wide tail tile; a single row of output channels.
        (Conv::new(1, 3, (5, 6), 7, (3, 3), 1, (1, 1)), Conv2dGemm16),
        // 1×1 kernel: degenerate im2col (the removed MatMul shortcut read
        // NCHW transposed; this pins the layout).
        (
            Conv::new(2, 19, (7, 9), 13, (1, 1), 1, (0, 0)),
            Conv2dGemm16,
        ),
        (
            Conv::new(4, 24, (16, 16), 40, (1, 1), 1, (0, 0)),
            Conv2dGemmSmall,
        ),
        // Stride 2 with H, W not multiples of the stride, padding 1.
        (
            Conv::new(2, 6, (13, 11), 10, (3, 3), 2, (1, 1)),
            Conv2dGemm16,
        ),
        // Stride 2, 7×7 stem-like kernel, padding 3, odd extents.
        (
            Conv::new(1, 3, (23, 21), 17, (7, 7), 2, (3, 3)),
            Conv2dGemm16,
        ),
        // Stride 2 1×1 projection shortcut: drops the last row/column.
        (Conv::new(2, 9, (9, 7), 11, (1, 1), 2, (0, 0)), Conv2dGemm16),
        // Stride 3 larger than the kernel: skipped input pixels.
        (
            Conv::new(1, 4, (10, 11), 5, (2, 2), 3, (0, 0)),
            Conv2dGemm16,
        ),
        // Separate padding: Conv1d emulation (H = 1, kW = 5, pW = 2).
        (
            Conv::new(3, 8, (1, 37), 12, (1, 5), 1, (0, 2)),
            Conv2dGemm16,
        ),
        // Non-square kernel, padding only in H.
        (Conv::new(2, 4, (9, 8), 6, (3, 2), 1, (2, 0)), Conv2dGemm16),
        // Padding larger than half the kernel: all-padding border outputs.
        (Conv::new(1, 2, (4, 5), 3, (3, 3), 1, (3, 2)), Conv2dGemm16),
    ];
    let options = gpu::Options::default();
    let mut failures = Failures::default();
    for (i, (c, shader)) in cases.into_iter().enumerate() {
        let g = c.forward_graph();
        let mut feeds = Feeds::new();
        feeds.fill_random(&g, 100 + i as u64, 1.0);
        inference(
            &mut failures,
            &format!("{c:?}"),
            &g,
            &feeds,
            &options,
            &[shader],
        );
    }
    failures.assert_none();
}

/// `optimize::apply_winograd_conv_fusions` rewrites inference-mode 3×3
/// stride-1 convolutions with symmetric padding, a parameter weight and
/// `Ci · Co >= 4096` into F(2,3) Winograd (weight, input and output
/// transforms around a batched matmul). The same shapes with
/// `no_winograd` (`MEGANEURA_NO_WINOGRAD`) take the direct GEMM.
#[test]
fn conv2d_winograd() {
    let cases = [
        // Aligned channels, even output (whole 2×2 tiles).
        Conv::new(2, 64, (8, 8), 64, (3, 3), 1, (1, 1)),
        // Odd output extents (partial tiles), no padding, batch 3,
        // channels off every tile width; 67 · 63 = 4221.
        Conv::new(3, 67, (7, 9), 63, (3, 3), 1, (0, 0)),
        // Padding 2: output larger than the input.
        Conv::new(1, 32, (5, 6), 130, (3, 3), 1, (2, 2)),
    ];
    let mut failures = Failures::default();
    for (i, &c) in cases.iter().enumerate() {
        let g = c.forward_graph();
        let mut feeds = Feeds::new();
        feeds.fill_random(&g, 200 + i as u64, 1.0);
        let mut options = gpu::Options::default();
        options.optimize.no_winograd = false;
        inference(
            &mut failures,
            &format!("winograd {c:?}"),
            &g,
            &feeds,
            &options,
            &[
                ShaderEntry::WinogradWeightTransform,
                ShaderEntry::WinogradInputTransform,
                ShaderEntry::WinogradBatchedMatMul,
                ShaderEntry::WinogradOutputTransform,
            ],
        );
        options.optimize.no_winograd = true;
        let shaders = dispatched(&g, Mode::Inference, &options);
        if shaders.contains(&ShaderEntry::WinogradBatchedMatMul) {
            failures
                .0
                .push(format!("no_winograd still used Winograd for {c:?}"));
        }
        let report = gpu::check_inference(&g, &feeds, &options).unwrap();
        failures.report(&format!("direct {c:?} {shaders:?}"), &report);
    }
    // Just below the channel threshold, and asymmetric padding, stay direct.
    for c in [
        Conv::new(1, 63, (6, 6), 65, (3, 3), 1, (1, 1)),
        Conv::new(1, 64, (6, 6), 64, (3, 3), 1, (1, 0)),
    ] {
        let g = c.forward_graph();
        let mut options = gpu::Options::default();
        options.optimize.no_winograd = false;
        let shaders = dispatched(&g, Mode::Inference, &options);
        if shaders.contains(&ShaderEntry::WinogradBatchedMatMul) {
            failures.0.push(format!("{c:?} should not use Winograd"));
        }
    }
    failures.assert_none();
}

/// The reference defines `WinogradConv2d` as the 3×3 convolution with its
/// original weight; evaluate the rewritten graph against the source graph.
#[test]
fn winograd_reference_matches_conv2d() {
    let c = Conv::new(2, 64, (5, 7), 65, (3, 3), 1, (1, 1));
    let g = c.forward_graph();
    let mut rewritten = g.deep_clone();
    let mut fusions = Vec::new();
    meganeura::optimize::apply_winograd_conv_fusions(
        &mut rewritten,
        &mut fusions,
        &meganeura::OptimizeConfig::default(),
    );
    assert_eq!(fusions.len(), 1, "the conv was not rewritten");
    // The transformed weight is appended after the conv that reads it.
    let rewritten = rewritten.toposort();
    let mut feeds = Feeds::new();
    feeds.fill_random(&g, 3, 1.0);
    // The transformed weight's value is derived; any value of the right
    // length must give the same result.
    feeds.fill_random(&rewritten, 4, 1.0);
    let want = meganeura::reference::evaluate_outputs(&g, &feeds).unwrap();
    let got = meganeura::reference::evaluate_outputs(&rewritten, &feeds).unwrap();
    assert_eq!(want, got);
}

/// `Conv2dDw` has one kernel: 16×16 output tiles per (batch, channel)
/// plane, kernels up to 7×7 staged in shared memory.
#[test]
fn conv2d_depthwise() {
    // (batch, channels, (h, w), (kh, kw), stride, (ph, pw))
    let cases = [
        (2, 17, (12, 10), (3, 3), 1, (1, 1)),
        // Output wider and taller than one 16×16 tile.
        (1, 3, (37, 20), (3, 3), 1, (1, 1)),
        // Stride 2, odd extents, 5×5 with padding 2.
        (3, 5, (13, 15), (5, 5), 2, (2, 2)),
        // The largest supported kernel, asymmetric padding.
        (1, 4, (11, 9), (7, 7), 1, (3, 1)),
        // Non-square kernel, no padding, stride 3.
        (2, 6, (10, 14), (1, 4), 3, (0, 0)),
    ];
    let options = gpu::Options::default();
    let mut failures = Failures::default();
    for (i, &(n, c, (h, w), (kh, kw), stride, (ph, pw))) in cases.iter().enumerate() {
        let mut g = Graph::new();
        let x = g.input("x", &[(n * c * h * w) as usize]);
        let k = g.parameter("k", &[(c * kh * kw) as usize]);
        let y = g.conv2d_dw(x, k, n, c, h, w, kh, kw, stride, ph, pw);
        g.set_outputs(vec![y]);
        let mut feeds = Feeds::new();
        feeds.fill_random(&g, 300 + i as u64, 1.0);
        inference(
            &mut failures,
            &format!("depthwise {:?}", cases[i]),
            &g,
            &feeds,
            &options,
            &[ShaderEntry::Conv2dDw],
        );
    }
    failures.assert_none();
}

/// Grad-input GEMM: M = Ci, N = H·W, K = Co·kH·kW, batch on the grid.
#[test]
fn conv2d_grad_input() {
    use ShaderEntry::{Conv2dGradInputGemm, Conv2dGradInputGemm16, Conv2dGradInputGemmSmall};
    let cases = [
        // 64: 2·16·2 = 64 workgroups.
        (
            Conv::new(2, 65, (31, 33), 5, (3, 3), 1, (1, 1)),
            Conv2dGradInputGemm,
        ),
        // 32: 2·8·4 = 64 at 32.
        (
            Conv::new(4, 33, (15, 17), 3, (3, 3), 1, (1, 1)),
            Conv2dGradInputGemmSmall,
        ),
        (
            Conv::new(1, 7, (5, 6), 3, (3, 3), 1, (1, 1)),
            Conv2dGradInputGemm16,
        ),
        (
            Conv::new(2, 13, (7, 9), 19, (1, 1), 1, (0, 0)),
            Conv2dGradInputGemm16,
        ),
        // Stride 2: input rows/columns no output window reaches get zero.
        (
            Conv::new(2, 6, (13, 11), 10, (3, 3), 2, (1, 1)),
            Conv2dGradInputGemm16,
        ),
        (
            Conv::new(2, 9, (9, 7), 11, (1, 1), 2, (0, 0)),
            Conv2dGradInputGemm16,
        ),
        (
            Conv::new(1, 3, (23, 21), 17, (7, 7), 2, (3, 3)),
            Conv2dGradInputGemm16,
        ),
        (
            Conv::new(1, 4, (10, 11), 5, (2, 2), 3, (0, 0)),
            Conv2dGradInputGemm16,
        ),
        (
            Conv::new(3, 8, (1, 37), 12, (1, 5), 1, (0, 2)),
            Conv2dGradInputGemm16,
        ),
        (
            Conv::new(2, 4, (9, 8), 6, (3, 2), 1, (2, 0)),
            Conv2dGradInputGemm16,
        ),
    ];
    let options = gpu::Options::default();
    let mut failures = Failures::default();
    for (i, (c, shader)) in cases.into_iter().enumerate() {
        let g = c.grad_input_graph();
        let mut feeds = Feeds::new();
        feeds.fill_random(&g, 400 + i as u64, 1.0);
        inference(
            &mut failures,
            &format!("{c:?}"),
            &g,
            &feeds,
            &options,
            &[shader],
        );
    }
    failures.assert_none();
}

/// Grad-weight GEMM: M = Co, N = Ci·kH·kW, with the batch folded into K,
/// so the grid never has a batch axis.
#[test]
fn conv2d_grad_weight() {
    use ShaderEntry::{Conv2dGradWeightGemm, Conv2dGradWeightGemm16, Conv2dGradWeightGemmSmall};
    let cases = [
        // 64: ceil(520/64) · ceil(450/64) = 9 · 8 = 72 workgroups.
        (
            Conv::new(1, 50, (4, 4), 520, (3, 3), 1, (1, 1)),
            Conv2dGradWeightGemm,
        ),
        // 32: 2 · 32 = 64 at 32, 1 · 16 at 64.
        (
            Conv::new(2, 111, (3, 3), 64, (3, 3), 1, (1, 1)),
            Conv2dGradWeightGemmSmall,
        ),
        // 16, batch 3 folded into K.
        (
            Conv::new(3, 5, (9, 7), 7, (3, 3), 1, (1, 1)),
            Conv2dGradWeightGemm16,
        ),
        (
            Conv::new(2, 19, (7, 9), 13, (1, 1), 1, (0, 0)),
            Conv2dGradWeightGemm16,
        ),
        (
            Conv::new(2, 6, (13, 11), 10, (3, 3), 2, (1, 1)),
            Conv2dGradWeightGemm16,
        ),
        (
            Conv::new(2, 9, (9, 7), 11, (1, 1), 2, (0, 0)),
            Conv2dGradWeightGemm16,
        ),
        (
            Conv::new(1, 3, (23, 21), 17, (7, 7), 2, (3, 3)),
            Conv2dGradWeightGemm16,
        ),
        (
            Conv::new(1, 4, (10, 11), 5, (2, 2), 3, (0, 0)),
            Conv2dGradWeightGemm16,
        ),
        (
            Conv::new(3, 8, (1, 37), 12, (1, 5), 1, (0, 2)),
            Conv2dGradWeightGemm16,
        ),
        (
            Conv::new(2, 4, (9, 8), 6, (3, 2), 1, (2, 0)),
            Conv2dGradWeightGemm16,
        ),
    ];
    let options = gpu::Options::default();
    let mut failures = Failures::default();
    for (i, (c, shader)) in cases.into_iter().enumerate() {
        let g = c.grad_weight_graph();
        let mut feeds = Feeds::new();
        feeds.fill_random(&g, 500 + i as u64, 1.0);
        inference(
            &mut failures,
            &format!("{c:?}"),
            &g,
            &feeds,
            &options,
            &[shader],
        );
    }
    failures.assert_none();
}

/// A max-pooling configuration: (batch, channels, (h, w), (kh, kw), stride, padding).
type Pool = (u32, u32, (u32, u32), (u32, u32), u32, u32);

const POOLS: [Pool; 7] = [
    // 2×2 stride 2, even and odd extents (the odd last row is dropped).
    (2, 3, (8, 8), (2, 2), 2, 0),
    (1, 4, (9, 7), (2, 2), 2, 0),
    // ResNet stem: 3×3 stride 2 padding 1, odd extents.
    (2, 5, (13, 11), (3, 3), 2, 1),
    // Overlapping windows, stride 1: one input wins several windows.
    (1, 3, (6, 7), (3, 3), 1, 1),
    // Non-square kernel with padding, stride 2.
    (3, 2, (7, 10), (2, 3), 2, 1),
    // Stride larger than the kernel: skipped inputs get zero gradient.
    (1, 2, (11, 10), (2, 2), 3, 0),
    // Many planes: several workgroups of 256 threads.
    (4, 16, (10, 10), (3, 3), 2, 1),
];

fn pool_sizes(p: Pool) -> (usize, usize) {
    let (n, c, (h, w), (kh, kw), s, pad) = p;
    let oh = (h + 2 * pad - kh) / s + 1;
    let ow = (w + 2 * pad - kw) / s + 1;
    ((n * c * h * w) as usize, (n * c * oh * ow) as usize)
}

fn max_pool(g: &mut Graph, x: NodeId, p: Pool) -> NodeId {
    let (n, c, (h, w), (kh, kw), s, pad) = p;
    g.max_pool_2d(x, n, c, h, w, kh, kw, s, pad)
}

/// Values from `{0, 1, 2}` so that most windows hold tied maxima.
fn tied(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| rng.below(3) as f32).collect()
}

#[test]
fn max_pool_2d() {
    let options = gpu::Options::default();
    let mut failures = Failures::default();
    for (i, &p) in POOLS.iter().enumerate() {
        let (x_len, _) = pool_sizes(p);
        let mut g = Graph::new();
        let x = g.input("x", &[x_len]);
        let y = max_pool(&mut g, x, p);
        g.set_outputs(vec![y]);
        for (kind, data) in [
            ("distinct", random(x_len, 600 + i as u64, -1.0, 1.0)),
            ("all negative", random(x_len, 610 + i as u64, -3.0, -1.0)),
        ] {
            let mut feeds = Feeds::new();
            feeds.set("x", &data);
            inference(
                &mut failures,
                &format!("max pool {p:?} {kind}"),
                &g,
                &feeds,
                &options,
                &[ShaderEntry::MaxPool2d],
            );
        }
    }
    failures.assert_none();
}

/// `MaxPool2dGrad` routes each output gradient to its window's first
/// maximum in row-major order; tied inputs check the tie-break.
#[test]
fn max_pool_2d_grad() {
    let options = gpu::Options::default();
    let mut failures = Failures::default();
    for (i, &p) in POOLS.iter().enumerate() {
        let (_, c, (h, w), (kh, kw), stride, padding) = p;
        let (x_len, y_len) = pool_sizes(p);
        let mut g = Graph::new();
        let dy = g.input("dy", &[y_len]);
        let x = g.input("x", &[x_len]);
        let dx = g.add_raw_node(
            Op::MaxPool2dGrad {
                channels: c,
                in_h: h,
                in_w: w,
                kernel_h: kh,
                kernel_w: kw,
                stride,
                padding,
            },
            vec![dy, x],
            meganeura::TensorType::f32(vec![x_len]),
        );
        g.set_outputs(vec![dx]);
        for (kind, data) in [
            ("distinct", random(x_len, 700 + i as u64, -1.0, 1.0)),
            ("tied", tied(x_len, 710 + i as u64)),
            ("all negative", random(x_len, 720 + i as u64, -3.0, -1.0)),
        ] {
            let mut feeds = Feeds::new();
            feeds.set("x", &data);
            feeds.fill_random(&g, 730 + i as u64, 1.0);
            inference(
                &mut failures,
                &format!("max pool grad {p:?} {kind}"),
                &g,
                &feeds,
                &options,
                &[ShaderEntry::MaxPool2dGrad],
            );
        }
    }
    failures.assert_none();
}

/// A group-norm configuration: (batch, channels, spatial, groups).
type Norm = (u32, u32, u32, u32);

/// Single-pass kernels (`group_norm_chunks == 1`) and the two-pass
/// statistics + apply path, including a group size the first chunk count
/// does not divide (8084 = 4 · 2021: 3 chunks by work, lowered to 2).
const NORMS: [(Norm, u32); 7] = [
    ((2, 6, 35, 3), 1),
    // One group (layer-norm-like), strided loops over 256 threads.
    ((1, 5, 300, 1), 1),
    // One channel per group (instance-norm-like).
    ((3, 4, 49, 4), 1),
    ((1, 8, 64 * 64, 2), 8),
    ((2, 16, 32 * 32, 4), 2),
    ((1, 8, 43 * 47, 2), 2),
    // Group size exactly at the chunking threshold.
    ((1, 4, 1024, 1), 2),
];

fn group_norm_graph(norm: Norm, eps: f32, silu: bool) -> Graph {
    let (n, c, s, groups) = norm;
    let mut g = Graph::new();
    let x = g.input("x", &[(n * c * s) as usize]);
    let w = g.parameter("w", &[c as usize]);
    let b = g.parameter("b", &[c as usize]);
    let y = g.group_norm(x, w, b, n, c, s, groups, eps);
    let y = if silu { g.silu(y) } else { y };
    g.set_outputs(vec![y]);
    g
}

#[test]
fn group_norm() {
    let options = gpu::Options::default();
    let mut failures = Failures::default();
    for (i, &(norm, chunks)) in NORMS.iter().enumerate() {
        let (n, c, s, groups) = norm;
        assert_eq!(group_norm_chunks(n, c, s, groups), chunks, "{norm:?}");
        let (single, fused) = if chunks == 1 {
            (ShaderEntry::GroupNorm, ShaderEntry::GroupNormSilu)
        } else {
            (ShaderEntry::GroupNormApply, ShaderEntry::GroupNormApply)
        };
        for (silu, shader) in [(false, single), (true, fused)] {
            let g = group_norm_graph(norm, 1e-5, silu);
            let mut feeds = Feeds::new();
            feeds.fill_random(&g, 800 + i as u64, 1.0);
            let label = format!("group norm {norm:?} chunks {chunks} silu {silu}");
            inference(&mut failures, &label, &g, &feeds, &options, &[shader]);
        }
    }
    failures.assert_none();
}

/// Inputs away from zero mean. GroupNorm is shift-invariant, and a stable
/// `f32` evaluation errs by about `eps · (|x| + |mean|) / std`, which the
/// reference magnitude allows for. The chunked path derives the variance
/// as `E[x²] − mean²` in `f32`, which errs by `eps · mean² / std²`: at
/// offset 3000 (mean / std ≈ 5200) the variance cancels to zero and the
/// output is about 180× too large, while the single-pass kernel passes.
#[test]
fn group_norm_offset_inputs() {
    let options = gpu::Options::default();
    let mut failures = Failures::default();
    for &(norm, chunks) in &[NORMS[0], NORMS[3]] {
        for offset in [4.0f32, 30.0, 3000.0] {
            let g = group_norm_graph(norm, 1e-5, false);
            let (n, c, s, _) = norm;
            let mut feeds = Feeds::new();
            let len = (n * c * s) as usize;
            feeds.set("x", &random(len, 900, offset - 1.0, offset + 1.0));
            feeds.fill_random(&g, 901, 1.0);
            let label = format!("group norm {norm:?} chunks {chunks} offset {offset}");
            let report = gpu::check_inference(&g, &feeds, &options).unwrap();
            failures.report(&label, &report);
        }
    }
    failures.assert_none();
}

#[test]
fn group_norm_grads() {
    let options = gpu::Options::default();
    let mut failures = Failures::default();
    let norms: [Norm; 5] = [
        (2, 6, 35, 3),
        (1, 5, 300, 1),
        (3, 4, 49, 4),
        (2, 16, 32 * 32, 4),
        (1, 12, 1000, 3),
    ];
    for (i, &norm) in norms.iter().enumerate() {
        let (n, c, s, groups) = norm;
        let len = (n * c * s) as usize;
        let eps = 1e-5;

        let mut g = Graph::new();
        let dy = g.input("dy", &[len]);
        let x = g.input("x", &[len]);
        let w = g.input("w", &[c as usize]);
        let dx = g.group_norm_grad_input(dy, x, w, n, c, s, groups, eps);
        g.set_outputs(vec![dx]);
        let mut feeds = Feeds::new();
        feeds.fill_random(&g, 1000 + i as u64, 1.0);
        inference(
            &mut failures,
            &format!("group norm grad input {norm:?}"),
            &g,
            &feeds,
            &options,
            &[
                ShaderEntry::GroupNormGradStats,
                ShaderEntry::GroupNormGradInput,
            ],
        );

        let mut g = Graph::new();
        let dy = g.input("dy", &[len]);
        let x = g.input("x", &[len]);
        let dwb = g.group_norm_grad_weight_bias(dy, x, c, s, groups, eps);
        g.set_outputs(vec![dwb]);
        inference(
            &mut failures,
            &format!("group norm grad weight/bias {norm:?}"),
            &g,
            &feeds,
            &options,
            &[
                ShaderEntry::GroupNormGradStats,
                ShaderEntry::GroupNormGradWeightBias,
            ],
        );
    }
    failures.assert_none();
}

/// Check `graph`'s derivatives against finite differences and the
/// training session against the reference.
fn autodiff(failures: &mut Failures, label: &str, graph: &Graph, feeds: &Feeds) {
    let report = gradients::check(graph, feeds, &gradients::Options::default()).unwrap();
    failures.report(&format!("{label} autodiff"), &report);
    let options = gpu::Options::default();
    let shaders = dispatched(graph, Mode::Training, &options);
    let report = gpu::check_training(graph, feeds, &options).unwrap();
    failures.report(&format!("{label} training {shaders:?}"), &report);
}

#[test]
fn conv2d_autodiff() {
    let cases = [
        Conv::new(2, 3, (6, 7), 4, (3, 3), 1, (1, 1)),
        Conv::new(1, 2, (9, 8), 3, (3, 3), 2, (1, 1)),
        Conv::new(2, 3, (5, 5), 2, (1, 1), 2, (0, 0)),
        Conv::new(2, 2, (1, 11), 3, (1, 3), 1, (0, 1)),
        Conv::new(1, 2, (6, 5), 2, (2, 3), 3, (1, 2)),
    ];
    let mut failures = Failures::default();
    for (i, &c) in cases.iter().enumerate() {
        let mut g = Graph::new();
        let x = g.parameter("x", &[c.x_len()]);
        let w = g.parameter("w", &[c.w_len()]);
        let y = c.conv(&mut g, x, w);
        let loss = gradients::weighted_loss(&mut g, y, 1100 + i as u64, 0.7);
        g.set_outputs(vec![loss]);
        let mut feeds = Feeds::new();
        feeds.fill_random(&g, 1110 + i as u64, 1.0);
        autodiff(&mut failures, &format!("{c:?}"), &g, &feeds);
    }
    failures.assert_none();
}

/// Training through a convolution large enough for the 64- and 32-wide
/// derivative tiles; finite differences would be too slow here, so only
/// the device is compared with the reference derivative graph.
#[test]
fn conv2d_training_wide_tiles() {
    let mut failures = Failures::default();
    for (i, c) in [
        Conv::new(2, 65, (31, 33), 5, (3, 3), 1, (1, 1)),
        Conv::new(2, 111, (3, 3), 64, (3, 3), 1, (1, 1)),
    ]
    .into_iter()
    .enumerate()
    {
        let mut g = Graph::new();
        let x = g.parameter("x", &[c.x_len()]);
        let w = g.parameter("w", &[c.w_len()]);
        let y = c.conv(&mut g, x, w);
        let loss = gradients::weighted_loss(&mut g, y, 1200 + i as u64, 0.3);
        g.set_outputs(vec![loss]);
        let mut feeds = Feeds::new();
        feeds.fill_random(&g, 1210 + i as u64, 1.0);
        let options = gpu::Options::default();
        let shaders = dispatched(&g, Mode::Training, &options);
        let report = gpu::check_training(&g, &feeds, &options).unwrap();
        failures.report(&format!("{c:?} training {shaders:?}"), &report);
    }
    failures.assert_none();
}

#[test]
fn max_pool_2d_autodiff() {
    let mut failures = Failures::default();
    for (i, &p) in POOLS[..6].iter().enumerate() {
        let (x_len, _) = pool_sizes(p);
        let mut g = Graph::new();
        let x = g.parameter("x", &[x_len]);
        let y = max_pool(&mut g, x, p);
        let loss = gradients::weighted_loss(&mut g, y, 1300 + i as u64, -1.3);
        g.set_outputs(vec![loss]);
        let mut feeds = Feeds::new();
        // Distinct values: a finite-difference step must not change winners.
        feeds.fill_random(&g, 1310 + i as u64, 1.0);
        autodiff(&mut failures, &format!("max pool {p:?}"), &g, &feeds);
    }
    failures.assert_none();
}

#[test]
fn group_norm_autodiff() {
    let mut failures = Failures::default();
    let norms: [Norm; 4] = [(2, 6, 10, 3), (1, 4, 12, 1), (2, 3, 7, 3), (1, 4, 1024, 1)];
    for (i, &norm) in norms.iter().enumerate() {
        let (n, c, s, groups) = norm;
        let mut g = Graph::new();
        let x = g.parameter("x", &[(n * c * s) as usize]);
        let w = g.parameter("w", &[c as usize]);
        let b = g.parameter("b", &[c as usize]);
        let y = g.group_norm(x, w, b, n, c, s, groups, 1e-3);
        let loss = gradients::weighted_loss(&mut g, y, 1400 + i as u64, 0.9);
        g.set_outputs(vec![loss]);
        let mut feeds = Feeds::new();
        feeds.fill_random(&g, 1410 + i as u64, 1.0);
        autodiff(&mut failures, &format!("group norm {norm:?}"), &g, &feeds);
    }
    failures.assert_none();
}

/// Conv → GroupNorm → ReLU → MaxPool → Conv: each op's backward consumes
/// another op's gradient, with batch 2.
#[test]
fn vision_chain_autodiff() {
    let c1 = Conv::new(2, 3, (8, 7), 4, (3, 3), 1, (1, 1));
    let pool: Pool = (2, 4, (8, 7), (2, 2), 2, 0);
    let c2 = Conv::new(2, 4, (4, 3), 3, (3, 3), 1, (1, 0));
    let mut g = Graph::new();
    let x = g.input("x", &[c1.x_len()]);
    let w1 = g.parameter("w1", &[c1.w_len()]);
    let gw = g.parameter("gw", &[4]);
    let gb = g.parameter("gb", &[4]);
    let w2 = g.parameter("w2", &[c2.w_len()]);
    let y = c1.conv(&mut g, x, w1);
    let y = g.group_norm(y, gw, gb, 2, 4, 8 * 7, 2, 1e-5);
    let y = g.relu(y);
    let y = max_pool(&mut g, y, pool);
    let y = c2.conv(&mut g, y, w2);
    let loss = gradients::weighted_loss(&mut g, y, 1500, 0.5);
    g.set_outputs(vec![loss]);
    let mut feeds = Feeds::new();
    feeds.fill_random(&g, 1501, 1.0);
    let mut failures = Failures::default();
    autodiff(&mut failures, "chain", &g, &feeds);
    failures.assert_none();
}

/// Run a hand-edited plan for `graph` with
/// poisoned buffers and scalar kernels, and compare its outputs with the
/// reference.
fn check_plan(graph: &Graph, feeds: &Feeds, plan: ExecutionPlan) -> Report {
    use meganeura::reference::{Comparison, Tolerance, check, evaluate, magnitudes};
    let values = evaluate(graph, feeds).unwrap();
    let context = meganeura::init_gpu_context_with(meganeura::GpuOptions::from_env()).unwrap();
    let mut session = Session::with_context_opts(
        plan,
        Arc::new(context),
        SessionOptions {
            poison: true,
            coop: CoopPolicy::Disabled,
            ..Default::default()
        },
    );
    for node in graph.nodes() {
        match node.op {
            Op::Input { ref name } => session.set_input(name, &feeds.f32(name).unwrap()),
            Op::Parameter { ref name } => session.set_parameter(name, &feeds.f32(name).unwrap()),
            _ => {}
        }
    }
    session.step();
    session.wait();
    let mut report = Report::default();
    for (index, &id) in graph.outputs().iter().enumerate() {
        let node = graph.node(id);
        let inputs: Vec<_> = node.inputs.iter().map(|&i| &values[i as usize]).collect();
        let want = &values[id as usize];
        let magnitude = magnitudes(graph, node, &inputs, want).unwrap();
        let mut got = vec![0.0f32; want.len()];
        session.read_output_by_index(index, &mut got);
        report.comparisons.push(Comparison {
            what: format!("output {index} (%{id})"),
            result: check(&got, &want.data, &magnitude, Tolerance::default()),
        });
    }
    report
}

fn is_conv_gemm(shader: &ShaderEntry) -> bool {
    use ShaderEntry::*;
    matches!(
        shader,
        Conv2dGemm
            | Conv2dGemmSmall
            | Conv2dGemm16
            | Conv2dGradInputGemm
            | Conv2dGradInputGemmSmall
            | Conv2dGradInputGemm16
            | Conv2dGradWeightGemm
            | Conv2dGradWeightGemmSmall
            | Conv2dGradWeightGemm16
    )
}

/// The compiler emits every implicit-GEMM convolution as the exact-divisor
/// kernel with a 16-deep K stage; the tuner may replace it with a 32-deep K
/// stage or the uniform software-divisor shader. Force each on each tile
/// width and direction. Most reductions here (K = 45, 27, 54, 90, 18, 84)
/// are multiples of neither stage.
#[test]
fn conv2d_tuned_kernels() {
    let cases: [(&str, Graph); 9] = [
        (
            "forward 64",
            Conv::new(2, 5, (31, 33), 65, (3, 3), 1, (1, 1)).forward_graph(),
        ),
        (
            "forward 32",
            Conv::new(4, 3, (15, 17), 33, (3, 3), 1, (1, 1)).forward_graph(),
        ),
        (
            "forward 16",
            Conv::new(2, 6, (13, 11), 10, (3, 3), 2, (1, 1)).forward_graph(),
        ),
        (
            "grad input 64",
            Conv::new(2, 65, (31, 33), 5, (3, 3), 1, (1, 1)).grad_input_graph(),
        ),
        (
            "grad input 32",
            Conv::new(4, 33, (15, 17), 3, (3, 3), 1, (1, 1)).grad_input_graph(),
        ),
        (
            "grad input 16",
            Conv::new(2, 6, (13, 11), 10, (3, 3), 2, (1, 1)).grad_input_graph(),
        ),
        (
            "grad weight 64",
            Conv::new(1, 50, (4, 4), 520, (3, 3), 1, (1, 1)).grad_weight_graph(),
        ),
        (
            "grad weight 32",
            Conv::new(2, 111, (3, 3), 64, (3, 3), 1, (1, 1)).grad_weight_graph(),
        ),
        (
            "grad weight 16",
            Conv::new(2, 6, (13, 11), 10, (3, 3), 2, (1, 1)).grad_weight_graph(),
        ),
    ];
    let mut failures = Failures::default();
    for (i, (label, g)) in cases.iter().enumerate() {
        let mut feeds = Feeds::new();
        feeds.fill_random(g, 1600 + i as u64, 1.0);
        for kernel in [
            Kernel::SpecializedConv { k_tile: 16 },
            Kernel::SpecializedConv { k_tile: 32 },
            Kernel::Default,
        ] {
            let mut plan = meganeura::compile::compile(g);
            let conv = plan
                .dispatches
                .iter_mut()
                .find(|d| is_conv_gemm(&d.shader))
                .unwrap();
            let shader = conv.shader.clone();
            conv.kernel = kernel.clone();
            let report = check_plan(g, &feeds, plan);
            failures.report(&format!("{label} {shader:?} {kernel:?}"), &report);
        }
    }
    failures.assert_none();
}

/// Split-K weight gradients (`ExecutionPlan::split_conv_weight_gradients`,
/// chosen by the measured training search): partial products over slices
/// of the batch · oH · oW reduction, summed by `SumRows`.
#[test]
fn conv2d_grad_weight_split_k() {
    use ShaderEntry::{
        Conv2dGradWeightGemmSplit, Conv2dGradWeightGemmSplit16, Conv2dGradWeightGemmSplitSmall,
    };
    let cases = [
        // K = 2 · 6 · 6 = 72 on the 64-wide tile.
        (
            Conv::new(2, 50, (6, 6), 520, (3, 3), 1, (1, 1)),
            Conv2dGradWeightGemmSplit,
        ),
        // K = 3 · 4 · 4 = 48 on the 32-wide tile.
        (
            Conv::new(3, 111, (4, 4), 64, (3, 3), 1, (1, 1)),
            Conv2dGradWeightGemmSplitSmall,
        ),
        // K = 2 · 7 · 6 = 84, stride 2, on the 16-wide tile.
        (
            Conv::new(2, 6, (13, 11), 10, (3, 3), 2, (1, 1)),
            Conv2dGradWeightGemmSplit16,
        ),
    ];
    let mut failures = Failures::default();
    for (i, (c, split_shader)) in cases.into_iter().enumerate() {
        let g = c.grad_weight_graph();
        let mut feeds = Feeds::new();
        feeds.fill_random(&g, 1700 + i as u64, 1.0);
        for splits in [2, 3] {
            let mut plan = meganeura::compile::compile(&g);
            let index = plan
                .dispatches
                .iter()
                .position(|d| is_conv_gemm(&d.shader))
                .unwrap();
            if let Err(e) = plan.split_conv_weight_gradients(&[(index, splits)], usize::MAX) {
                failures
                    .0
                    .push(format!("{c:?} splits {splits}: refused: {e:?}"));
                continue;
            }
            let shaders: Vec<_> = plan.dispatches.iter().map(|d| d.shader.clone()).collect();
            failures.expect_shader(&format!("{c:?}"), &shaders, &split_shader);
            let report = check_plan(&g, &feeds, plan);
            failures.report(&format!("{c:?} splits {splits} {split_shader:?}"), &report);
        }
    }
    failures.assert_none();
}

/// A pooling window that covers only padding has no maximum. PyTorch
/// forbids it (`padding <= kernel / 2`); the builder does not, and the
/// kernel writes `-f32::MAX` there, so the reference rejects the graph.
#[test]
fn max_pool_padding_only_window_is_invalid() {
    let p: Pool = (1, 1, (3, 3), (2, 2), 2, 2);
    let (x_len, _) = pool_sizes(p);
    let mut g = Graph::new();
    let x = g.input("x", &[x_len]);
    let y = max_pool(&mut g, x, p);
    g.set_outputs(vec![y]);
    let mut feeds = Feeds::new();
    feeds.fill_random(&g, 1800, 1.0);
    let result = meganeura::reference::evaluate_outputs(&g, &feeds);
    assert!(
        matches!(result, Err(meganeura::reference::Error::Invalid { .. })),
        "{result:?}"
    );
}

/// Inference rewrites `silu(group_norm(x))` into `GroupNormSilu`; cover
/// both its single-dispatch and chunked lowerings, the latter with a large
/// mean so the chunk statistics must combine without cancellation.
#[test]
fn group_norm_silu_fusion() {
    let options = gpu::Options::default();
    let mut failures = Failures::default();
    for &(n, c, s, groups, offset) in &[
        (2u32, 6u32, 35u32, 3u32, 0.0f32),
        (1, 8, 4096, 2, 0.0),
        (1, 8, 4096, 2, 300.0),
    ] {
        let mut g = Graph::new();
        let len = (n * c * s) as usize;
        let x = g.input("x", &[len]);
        let w = g.parameter("w", &[c as usize]);
        let b = g.parameter("b", &[c as usize]);
        let y = g.group_norm(x, w, b, n, c, s, groups, 1e-5);
        let y = g.silu(y);
        g.set_outputs(vec![y]);
        let mut fused = g.deep_clone();
        meganeura::optimize::apply_group_norm_silu_fusions(&mut fused, &mut Vec::new());
        assert!(
            fused
                .nodes()
                .iter()
                .any(|node| matches!(node.op, Op::GroupNormSilu { .. })),
            "group_norm + silu was not fused for {:?}",
            (n, c, s, groups)
        );
        let mut feeds = Feeds::new();
        feeds.set("x", &random(len, 950, offset - 1.0, offset + 1.0));
        feeds.fill_random(&g, 951, 1.0);
        let report = gpu::check_inference(&g, &feeds, &options).unwrap();
        failures.report(
            &format!("group_norm_silu {:?} offset {offset}", (n, c, s, groups)),
            &report,
        );
    }
    failures.assert_none();
}

/// Training at 64×64 and larger reductions: kernel gradients summed over
/// 4096 positions, a 3×3 → 3×3 → residual → 1×1 chain (whose 1×1 input
/// gradient once had its workgroup axes swapped at this size), GroupNorm
/// over 4096 positions, the FiLM broadcast `e[C,1] · ones[1,4096]`, and a
/// mean over 40960 elements. Every gradient element is compared.
#[test]
fn large_spatial_training() {
    let options = gpu::Options::default();
    let mut failures = Failures::default();
    let res = 64u32;
    let mut graphs: Vec<(&str, Graph)> = Vec::new();

    let mut g = Graph::new();
    let x = g.input("x", &[(2 * res * res) as usize]);
    let k = g.parameter("k", &[2 * 2 * 9]);
    let y = g.conv2d(x, k, 1, 2, res, res, 2, 3, 3, 1, 1);
    let loss = gradients::weighted_loss(&mut g, y, 1300, 0.9);
    g.set_outputs(vec![loss]);
    graphs.push(("conv3x3 64x64", g));

    let mut g = Graph::new();
    let x = g.input("x", &[(3 * res * res) as usize]);
    let w0 = g.parameter("w0", &[4 * 3 * 9]);
    let a = g.conv2d(x, w0, 1, 3, res, res, 4, 3, 3, 1, 1);
    let w1 = g.parameter("w1", &[4 * 4 * 9]);
    let b = g.conv2d(a, w1, 1, 4, res, res, 4, 3, 3, 1, 1);
    let sum = g.add(a, b);
    let w2 = g.parameter("w2", &[2 * 4]);
    let pred = g.conv2d(sum, w2, 1, 4, res, res, 2, 1, 1, 1, 0);
    let loss = gradients::weighted_loss(&mut g, pred, 1301, 0.7);
    g.set_outputs(vec![loss]);
    graphs.push(("conv residual + 1x1 64x64", g));

    let mut g = Graph::new();
    let spatial = res * res;
    let x = g.input("x", &[(4 * spatial) as usize]);
    let w = g.parameter("w", &[(4 * spatial) as usize]);
    let xw = g.mul(x, w);
    let gw = g.parameter("gn_w", &[4]);
    let gb = g.parameter("gn_b", &[4]);
    let y = g.group_norm(xw, gw, gb, 1, 4, spatial, 2, 1e-5);
    let loss = gradients::weighted_loss(&mut g, y, 1302, 1.7);
    g.set_outputs(vec![loss]);
    graphs.push(("group_norm 4096", g));

    let mut g = Graph::new();
    let e = g.parameter("e", &[8, 1]);
    let ones = g.constant(vec![1.0; 4096], &[1, 4096]);
    let plane = g.matmul(e, ones);
    let loss = gradients::weighted_loss(&mut g, plane, 1303, 0.9);
    g.set_outputs(vec![loss]);
    graphs.push(("film e[8,1] x ones[1,4096]", g));

    let mut g = Graph::new();
    let n = 40960;
    let x = g.input("x", &[n]);
    let w = g.parameter("w", &[n]);
    let pred = g.mul(x, w);
    let target = g.input("target", &[n]);
    let loss = g.mse_loss(pred, target);
    g.set_outputs(vec![loss]);
    graphs.push(("mse 40960", g));

    for (i, (label, g)) in graphs.iter().enumerate() {
        let mut feeds = Feeds::new();
        feeds.fill_random(g, 1310 + i as u64, 1.0);
        let report = gpu::check_training(g, &feeds, &options).unwrap();
        failures.report(label, &report);
    }
    failures.assert_none();
}

/// Inference at the super-resolution model's scale: a 64→64 3×3 convolution
/// at 96×96 (Winograd), GroupNorm + SiLU over 64 channels at 128×128 with
/// 16 groups (fused, chunked), and a 3×1 convolution over a flat 129×1 image.
#[test]
fn super_resolution_scale() {
    let options = gpu::Options::default();
    let mut failures = Failures::default();
    let mut graphs: Vec<(&str, Graph)> = Vec::new();

    // 48² Winograd tiles × 64 channels still exceed one dispatch dimension.
    let (c, conv_hw, hw) = (64u32, 96u32, 128u32);
    let mut g = Graph::new();
    let x = g.input("x", &[(c * conv_hw * conv_hw) as usize]);
    let k = g.parameter("k", &[(c * c * 9) as usize]);
    let y = g.conv2d(x, k, 1, c, conv_hw, conv_hw, c, 3, 3, 1, 1);
    g.set_outputs(vec![y]);
    graphs.push(("winograd 64x64 at 96x96", g));

    let mut g = Graph::new();
    let x = g.input("x", &[(c * hw * hw) as usize]);
    let w = g.parameter("w", &[c as usize]);
    let b = g.parameter("b", &[c as usize]);
    let y = g.group_norm(x, w, b, 1, c, hw * hw, 16, 1e-5);
    let y = g.silu(y);
    g.set_outputs(vec![y]);
    graphs.push(("group_norm_silu 64ch 128x128", g));

    let mut g = Graph::new();
    let x = g.input("x", &[3 * 129]);
    let k = g.parameter("k", &[5 * 3 * 3]);
    let y = g.conv2d_hw(x, k, 1, 3, 129, 1, 5, 3, 1, 1, 1, 0);
    g.set_outputs(vec![y]);
    graphs.push(("conv3x1 over 129x1", g));

    for (i, (label, g)) in graphs.iter().enumerate() {
        let mut feeds = Feeds::new();
        feeds.fill_random(g, 1400 + i as u64, 1.0);
        let report = gpu::check_inference(g, &feeds, &options).unwrap();
        failures.report(label, &report);
    }
    failures.assert_none();
}
