//! Cooperative-matrix matmul correctness for skinny / non-aligned
//! dimensions.
//!
//! Each test builds a tiny `MatMul` / `MatMulAT` / `MatMulBT` graph with
//! a shape that historically tripped one of three coop bugs:
//!
//!   1. vec4 staging along the K axis: `(tr4 + 4u) <= k` zero-padded the
//!      entire tile whenever K < 4 (or whenever K % 4 != 0 caused the
//!      last partial tile to be skipped). Hidden gradients silently
//!      collapsed to zero in any model with a scalar output head.
//!
//!   2. vec4 staging along the N axis: analogous bug for forward passes
//!      where N is small or non-multiple-of-4.
//!
//!   3. Output store: `coopStoreT` with row stride `n` writes a 16×16
//!      sub-tile, and when `n % 16 != 0` lanes overflow into the next
//!      row's leading columns. The runtime auto-switch now refuses coop
//!      in that regime; the test checks the result is still correct
//!      because the scalar fallback runs.
//!
//! The intent is to cover the cases by running them on the GPU and
//! comparing against a straight-line CPU reference. Cases pass on f16
//! cooperative_matrix (the production path on RDNA3 / Volta+) with
//! generous tolerance because f16 inputs lose ~10 mantissa bits.

use meganeura::{Graph, Mode, SessionConfig, build};

fn pcg_inputs(n: usize, seed: u32) -> Vec<f32> {
    // Deterministic, uniform in [-1, 1].
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(747796405).wrapping_add(2891336453);
            let w = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277803737);
            let r = (w >> 22) ^ w;
            (r as f32) * (2.0 / u32::MAX as f32) - 1.0
        })
        .collect()
}

fn cpu_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    // A is [m, k] row-major, B is [k, n] row-major, C is [m, n] row-major.
    let mut c = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0_f32;
            for l in 0..k {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

fn cpu_matmul_at(a: &[f32], b: &[f32], k: usize, m: usize, n: usize) -> Vec<f32> {
    // A is [k, m] row-major, B is [k, n] row-major. C = A^T @ B, shape [m, n].
    let mut c = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0_f32;
            for l in 0..k {
                sum += a[l * m + i] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

fn cpu_matmul_bt(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    // A is [m, k] row-major, B is [n, k] row-major. C = A @ B^T, shape [m, n].
    let mut c = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0_f32;
            for l in 0..k {
                sum += a[i * k + l] * b[j * k + l];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

fn assert_close(gpu: &[f32], cpu: &[f32], label: &str) {
    assert_eq!(gpu.len(), cpu.len());
    let max_abs = gpu
        .iter()
        .zip(cpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    let max_cpu = cpu
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f32, f32::max)
        .max(1e-6);
    let rel = max_abs / max_cpu;
    // f16 inputs in the coop path lose ~10 mantissa bits; allow 2%.
    let first_bad = gpu
        .iter()
        .zip(cpu)
        .position(|(g, c)| (g - c).abs() / max_cpu > 0.02);
    assert!(
        rel < 0.02,
        "{label}: rel err {:.3}% (max|gpu-cpu|={:.3e}, max|cpu|={:.3e}), first bad idx {:?}\n  gpu[..8]={:?}\n  cpu[..8]={:?}",
        rel * 100.0,
        max_abs,
        max_cpu,
        first_bad,
        &gpu[..8.min(gpu.len())],
        &cpu[..8.min(cpu.len())],
    );
}

fn run_matmul(m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut g = Graph::new();
    let a = g.input("a", &[m, k]);
    let b = g.input("b", &[k, n]);
    let c = g.matmul(a, b);
    g.set_outputs(vec![c]);
    let (mut session, _) = build(
        &g,
        SessionConfig {
            mode: Mode::Inference,
            ..Default::default()
        },
    );
    let a_data = pcg_inputs(m * k, 1);
    let b_data = pcg_inputs(k * n, 2);
    session.set_input("a", &a_data);
    session.set_input("b", &b_data);
    session.step();
    session.wait();
    let mut gpu = vec![0.0_f32; m * n];
    session.read_output_by_index(0, &mut gpu);
    let cpu = cpu_matmul(&a_data, &b_data, m, k, n);
    assert_close(&gpu, &cpu, &format!("matmul m={m} k={k} n={n}"));
    gpu
}

fn run_matmul_at(m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut g = Graph::new();
    // MatMulAT: A is [K, M], B is [K, N], output [M, N].
    let a = g.input("a", &[k, m]);
    let b = g.input("b", &[k, n]);
    let c = g.matmul_at(a, b);
    g.set_outputs(vec![c]);
    let (mut session, _) = build(
        &g,
        SessionConfig {
            mode: Mode::Inference,
            ..Default::default()
        },
    );
    let a_data = pcg_inputs(k * m, 1);
    let b_data = pcg_inputs(k * n, 2);
    session.set_input("a", &a_data);
    session.set_input("b", &b_data);
    session.step();
    session.wait();
    let mut gpu = vec![0.0_f32; m * n];
    session.read_output_by_index(0, &mut gpu);
    let cpu = cpu_matmul_at(&a_data, &b_data, k, m, n);
    assert_close(&gpu, &cpu, &format!("matmul_at m={m} k={k} n={n}"));
    gpu
}

fn run_matmul_bt(m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut g = Graph::new();
    // MatMulBT: A is [M, K], B is [N, K], output [M, N].
    let a = g.input("a", &[m, k]);
    let b = g.input("b", &[n, k]);
    let c = g.matmul_bt(a, b);
    g.set_outputs(vec![c]);
    let (mut session, _) = build(
        &g,
        SessionConfig {
            mode: Mode::Inference,
            ..Default::default()
        },
    );
    let a_data = pcg_inputs(m * k, 1);
    let b_data = pcg_inputs(n * k, 2);
    session.set_input("a", &a_data);
    session.set_input("b", &b_data);
    session.step();
    session.wait();
    let mut gpu = vec![0.0_f32; m * n];
    session.read_output_by_index(0, &mut gpu);
    let cpu = cpu_matmul_bt(&a_data, &b_data, m, n, k);
    assert_close(&gpu, &cpu, &format!("matmul_bt m={m} k={k} n={n}"));
    gpu
}

// ---------- K direction (vec4_a / vec4_b_transposed) ----------
// Forward MatMul with K < 4 or K % 4 != 0. The output dims are
// aligned, so coop routing is enabled and the kernel must handle the
// K-axis staging via the per-lane fallback.

#[test]
fn matmul_skinny_k1() {
    run_matmul(4096, 1, 32);
}

#[test]
fn matmul_skinny_k3() {
    run_matmul(4096, 3, 32);
}

#[test]
fn matmul_non_aligned_k18() {
    // K=18 hits the "last partial K tile" case: the t=16 iteration only
    // has K=16,17 in bounds; without the fix those two are lost.
    run_matmul(4096, 18, 32);
}

#[test]
fn matmul_bt_skinny_k1() {
    // The original bug: backward through a `[hidden, 1]` output head.
    run_matmul_bt(4096, 1, 32);
}

#[test]
fn matmul_bt_skinny_k3() {
    run_matmul_bt(4096, 3, 32);
}

#[test]
fn matmul_bt_non_aligned_k18() {
    run_matmul_bt(4096, 18, 32);
}

// ---------- N direction (output store + vec4_b) ----------
// Output dim N small or non-multiple-of-16. The store stride bug means
// coop can't safely run; the runtime auto-switch should refuse coop and
// the scalar fallback should produce correct results.

#[test]
fn matmul_n1() {
    run_matmul(4096, 32, 1);
}

#[test]
fn matmul_n3() {
    run_matmul(4096, 32, 3);
}

#[test]
fn matmul_n8() {
    run_matmul(4096, 32, 8);
}

#[test]
fn matmul_n16_aligned() {
    // N=16 is the smallest N for which coop can safely store. Verifies
    // the gate doesn't accidentally exclude valid coop shapes.
    run_matmul(4096, 32, 16);
}

#[test]
fn matmul_n33_non_aligned() {
    run_matmul(4096, 32, 33);
}

#[test]
fn matmul_bt_n33_non_aligned() {
    run_matmul_bt(4096, 32, 33);
}

// ---------- M direction (vec4_a_transposed + padded output store) ----------
// A partial final row tile is safe when N is tile-aligned: the runtime pads
// the output allocation through the next complete M tile, and consumers keep
// using the logical M extent. These exercise enough workgroups to ensure the
// f16 cooperative path is selected on supported hardware.

#[test]
fn matmul_m50_n256() {
    run_matmul(50, 128, 256);
}

#[test]
fn matmul_bt_m50_n256() {
    run_matmul_bt(50, 128, 256);
}

#[test]
fn matmul_at_m50_n256() {
    run_matmul_at(50, 128, 256);
}

// Smaller MatMulAT shapes remain useful coverage for transposed staging,
// although they may stay below the cooperative workgroup threshold.

#[test]
fn matmul_at_m1() {
    run_matmul_at(1, 4096, 32);
}

#[test]
fn matmul_at_m3() {
    run_matmul_at(3, 4096, 32);
}

#[test]
fn matmul_at_m16_aligned() {
    run_matmul_at(16, 4096, 32);
}

#[test]
fn matmul_at_m33_non_aligned() {
    run_matmul_at(33, 4096, 32);
}

// ---------- Shapes seen in blade-volume-train's volumetric forward ----------
//
// The differentiable trainer needs `[P, L] @ [L, L] = [P, L]` where P is the
// number of pixels per Adam step and L is the maximum traversal-path length.
// `[1024, 16] @ [16, 16]` was reported producing NaN output despite valid
// non-NaN inputs.

#[test]
fn matmul_pl_at_ll_p1024_l16() {
    run_matmul(1024, 16, 16);
}

#[test]
fn matmul_pl_at_ll_p1024_l24() {
    run_matmul(1024, 24, 24);
}

#[test]
fn matmul_pl_at_ll_p784_l16() {
    // Boundary: P=784 is the smallest P that reproduced the NaN in
    // blade-volume-train.
    run_matmul(784, 16, 16);
}

/// Build `sigmoid(matmul(a, b))` and compare to the CPU reference.
///
/// Hits `fuse_epilogues` which folds the sigmoid into the matmul as an
/// epilogue, replacing the matmul output buffer with the sigmoid buffer.
/// The shader path is `matmul.wgsl` with `$STORE_BODY` rewritten to apply
/// the epilogue before storing.
fn run_matmul_sigmoid(m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut g = Graph::new();
    let a = g.input("a", &[m, k]);
    let b = g.input("b", &[k, n]);
    let c = g.matmul(a, b);
    let s = g.sigmoid(c);
    g.set_outputs(vec![s]);
    let (mut session, _) = build(
        &g,
        SessionConfig {
            mode: Mode::Inference,
            ..Default::default()
        },
    );
    let a_data = pcg_inputs(m * k, 1);
    let b_data = pcg_inputs(k * n, 2);
    session.set_input("a", &a_data);
    session.set_input("b", &b_data);
    session.step();
    session.wait();
    let mut gpu = vec![0.0_f32; m * n];
    session.read_output_by_index(0, &mut gpu);
    let cpu_mm = cpu_matmul(&a_data, &b_data, m, k, n);
    let cpu: Vec<f32> = cpu_mm.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect();
    assert_close(&gpu, &cpu, &format!("matmul+sigmoid m={m} k={k} n={n}"));
    gpu
}

#[test]
fn matmul_sigmoid_p1024_l16() {
    // Regression test: fuse_epilogues was producing wrong output past the
    // first 64 rows for [M=1024, K=16] @ [K=16, N=16], leaving the rest
    // of the [M, N]=[1024, 16] result as zero. Downstream `recip(sig)`
    // then surfaced inf in blade-volume-train.
    run_matmul_sigmoid(1024, 16, 16);
}

#[test]
fn matmul_sigmoid_p1024_l24() {
    run_matmul_sigmoid(1024, 24, 24);
}

#[test]
fn matmul_sigmoid_p784_l16() {
    run_matmul_sigmoid(784, 16, 16);
}

#[test]
fn matmul_sigmoid_coop_eligible_grid() {
    // 32×4 = 128 cooperative workgroups with a 32×32 output tile. This
    // crosses the f16 cooperative-selection threshold on wide discrete GPUs,
    // exercising the shared-memory accumulator-to-epilogue bridge rather
    // than merely validating the scalar fallback.
    run_matmul_sigmoid(1024, 16, 128);
}
