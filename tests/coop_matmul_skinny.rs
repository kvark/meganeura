//! Cooperative-matrix matmul correctness for skinny / non-aligned
//! dimensions.
//!
//! Each case builds a tiny `MatMul` / `MatMulAT` / `MatMulBT` graph with
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

use crate::support::data::{assert_close, matmul_reference, pcg_inputs};
use meganeura::{Graph, Mode, SessionConfig, build};

/// Build `layout`'s product (0 `MatMul`, 1 `MatMulAT`, 2 `MatMulBT`),
/// optionally with a sigmoid that `fuse_epilogues` folds into its store, and
/// compare it with the f64 reference. f16 cooperative inputs lose ~10
/// mantissa bits; allow 2% of the largest output.
fn check(layout: usize, m: usize, k: usize, n: usize, sigmoid: bool) {
    let mut g = Graph::new();
    let a = g.input("a", &if layout == 1 { [k, m] } else { [m, k] });
    let b = g.input("b", &if layout == 2 { [n, k] } else { [k, n] });
    let c = match layout {
        1 => g.matmul_at(a, b),
        2 => g.matmul_bt(a, b),
        _ => g.matmul(a, b),
    };
    let c = if sigmoid { g.sigmoid(c) } else { c };
    g.set_outputs(vec![c]);
    let (mut session, _) = build(
        &g,
        SessionConfig {
            mode: Mode::Inference,
            ..crate::support::gpu::config()
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
    let cpu: Vec<f32> = matmul_reference(layout, &a_data, &b_data, [m, n, k])
        .into_iter()
        .map(|x| if sigmoid { 1.0 / (1.0 + (-x).exp()) } else { x })
        .map(|x| x as f32)
        .collect();
    assert_close(
        &format!("layout={layout} sigmoid={sigmoid} m={m} k={k} n={n}"),
        &gpu,
        &cpu,
        0.02,
    );
}

#[test]
fn skinny_and_unaligned_products_match_cpu() {
    for (layout, m, k, n, sigmoid) in [
        // K direction (vec4_a / vec4_b_transposed): K < 4, and K = 18, whose
        // last partial K tile has only K = 16, 17 in bounds. The output dims
        // are aligned, so cooperative routing stays enabled. The transposed-B
        // K = 1 case is the original bug: backward through a `[hidden, 1]`
        // output head.
        (0, 4096, 1, 32, false),
        (0, 4096, 18, 32, false),
        (2, 4096, 1, 32, false),
        (2, 4096, 18, 32, false),
        // N direction (output store + vec4_b): N small or not a multiple of
        // 16. The store stride bug means coop cannot run safely; the runtime
        // refuses it and the scalar fallback must be correct. N = 16 is the
        // smallest safe cooperative store, which the gate must not exclude.
        (0, 4096, 32, 3, false),
        (0, 4096, 32, 16, false),
        (0, 4096, 32, 33, false),
        (2, 4096, 32, 33, false),
        // M direction (vec4_a_transposed + padded output store): a partial
        // final row tile is safe when N is tile-aligned, because the runtime
        // pads the output through the next complete M tile. These launch
        // enough workgroups to select the f16 cooperative path where present.
        (0, 50, 128, 256, false),
        (2, 50, 128, 256, false),
        (1, 50, 128, 256, false),
        // A smaller transposed-A product, possibly below the cooperative
        // workgroup threshold, still covers transposed staging.
        (1, 33, 4096, 32, false),
        // blade-volume-train's volumetric forward, `[P, L] @ [L, L]`:
        // `[1024, 16] @ [16, 16]` produced NaN from valid inputs.
        (0, 1024, 16, 16, false),
        // fuse_epilogues produced wrong output past the first 64 rows of the
        // same product, leaving the rest zero; downstream `recip(sig)` then
        // surfaced inf.
        (0, 1024, 16, 16, true),
        // 32×4 = 128 cooperative workgroups with a 32×32 output tile cross
        // the f16 selection threshold on wide discrete GPUs, exercising the
        // shared-memory accumulator-to-epilogue bridge rather than the
        // scalar fallback.
        (0, 1024, 16, 128, true),
    ] {
        check(layout, m, k, n, sigmoid);
    }
}
