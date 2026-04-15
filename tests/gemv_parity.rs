//! Parity tests for the M=1 GEMV matmul path.
//!
//! The compiler routes Op::MatMul with M=1 through `ShaderEntry::MatMulGemv`
//! (a dedicated GEMV kernel, `shaders/matmul_gemv.wgsl`). This test verifies
//! it produces numerically identical output to the tiled matmul for
//! representative decode-sized shapes.

use meganeura::{Graph, build_inference_session, compile};

/// Reference CPU matmul for [1,K] × [K,N] → [1,N].
fn cpu_gemv(a: &[f32], b: &[f32], k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; n];
    for col in 0..n {
        let mut acc = 0.0_f32;
        for kk in 0..k {
            acc += a[kk] * b[kk * n + col];
        }
        out[col] = acc;
    }
    out
}

fn run_gpu_gemv(a_data: &[f32], b_data: &[f32], k: usize, n: usize) -> Vec<f32> {
    let mut g = Graph::new();
    let a = g.input("a", &[1, k]);
    let b = g.parameter("b", &[k, n]);
    let c = g.matmul(a, b);
    g.set_outputs(vec![c]);

    let mut session = build_inference_session(&g);

    // Sanity: when N % 4 == 0 the plan should route through GEMV;
    // otherwise it falls back to the tile matmul.
    let plan = session.plan();
    let gemv_count = plan
        .dispatches
        .iter()
        .filter(|d| matches!(d.shader, compile::ShaderEntry::MatMulGemv))
        .count();
    if n.is_multiple_of(4) {
        assert_eq!(
            gemv_count, 1,
            "expected one MatMulGemv dispatch for n%4==0, found {}",
            gemv_count
        );
    } else {
        assert_eq!(
            gemv_count, 0,
            "expected tile-matmul fallback for n%4!=0, got {} GEMV",
            gemv_count
        );
    }

    session.set_input("a", a_data);
    session.set_parameter("b", b_data);
    session.step();
    session.wait();
    session.read_output(n)
}

fn assert_close(a: &[f32], b: &[f32], rel_tol: f32, abs_tol: f32) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let scale = x.abs().max(y.abs()).max(1e-6);
        let rel = (x - y).abs() / scale;
        let abs = (x - y).abs();
        assert!(
            abs <= abs_tol || rel <= rel_tol,
            "mismatch at [{i}]: gpu={x}, cpu={y}, rel={rel:.3e}, abs={abs:.3e}",
        );
    }
}

fn test_shape(k: usize, n: usize, seed: u32) {
    // Deterministic non-uniform data so column structure exercises the kernel.
    let a: Vec<f32> = (0..k)
        .map(|i| ((i as u32 ^ seed) as f32 * 0.003).sin())
        .collect();
    let b: Vec<f32> = (0..k * n)
        .map(|i| ((i as u32 ^ seed.wrapping_mul(31)) as f32 * 0.0007).cos())
        .collect();

    let gpu = run_gpu_gemv(&a, &b, k, n);
    let cpu = cpu_gemv(&a, &b, k, n);
    // fp32 matmul accumulates K products; rel tol scales with sqrt(K) roughly.
    // Allow 1e-4 rel / 1e-5 abs which covers K up to a few thousand.
    assert_close(&gpu, &cpu, 1e-4, 1e-5);
}

#[test]
fn gemv_square_small() {
    test_shape(64, 64, 1);
}

#[test]
fn gemv_smollm2_qproj() {
    // Real SmolLM2-135M decode Q-projection shape.
    test_shape(576, 576, 2);
}

#[test]
fn gemv_smollm2_mlp_up() {
    test_shape(576, 1536, 3);
}

#[test]
fn gemv_smollm2_mlp_down() {
    test_shape(1536, 576, 4);
}

#[test]
fn gemv_smolvla_shapes() {
    test_shape(720, 720, 5);
    test_shape(720, 2048, 6);
}

#[test]
fn gemv_non_multiple_of_256() {
    // N not a multiple of the workgroup size — exercises the
    // `col < n` bounds check at the tail.
    test_shape(128, 300, 7);
    test_shape(128, 257, 8);
    test_shape(128, 1, 9);
}

// ---- FusedMatMulAdd (GEMV + residual) ----

/// CPU reference: 1×K × K×N + D[1,N].
fn cpu_gemv_add(a: &[f32], b: &[f32], d: &[f32], k: usize, n: usize) -> Vec<f32> {
    let mut out = cpu_gemv(a, b, k, n);
    for (o, r) in out.iter_mut().zip(d.iter()) {
        *o += *r;
    }
    out
}

fn test_gemv_add_shape(k: usize, n: usize, seed: u32) {
    let a: Vec<f32> = (0..k)
        .map(|i| ((i as u32 ^ seed) as f32 * 0.003).sin())
        .collect();
    let b: Vec<f32> = (0..k * n)
        .map(|i| ((i as u32 ^ seed.wrapping_mul(31)) as f32 * 0.0007).cos())
        .collect();
    let d: Vec<f32> = (0..n)
        .map(|i| ((i as u32 ^ seed.wrapping_mul(7)) as f32 * 0.01).sin() * 0.1)
        .collect();

    let mut g = Graph::new();
    let a_n = g.input("a", &[1, k]);
    let b_n = g.parameter("b", &[k, n]);
    let d_n = g.parameter("d", &[1, n]);
    let mm = g.matmul(a_n, b_n);
    let out = g.add(mm, d_n);
    g.set_outputs(vec![out]);

    let mut session = build_inference_session(&g);
    // Sanity: the optimizer should fuse MatMul+Add to FusedMatMulAdd, which
    // at M=1 with N%4==0 routes through MatMulGemvAdd.
    let plan = session.plan();
    let gemv_add_count = plan
        .dispatches
        .iter()
        .filter(|disp| matches!(disp.shader, compile::ShaderEntry::MatMulGemvAdd))
        .count();
    assert_eq!(
        gemv_add_count,
        1,
        "expected one MatMulGemvAdd dispatch, got {}; plan:\n{:?}",
        gemv_add_count,
        plan.dispatches
            .iter()
            .map(|d| format!("{:?}", d.shader))
            .collect::<Vec<_>>(),
    );

    session.set_input("a", &a);
    session.set_parameter("b", &b);
    session.set_parameter("d", &d);
    session.step();
    session.wait();
    let gpu = session.read_output(n);
    let cpu = cpu_gemv_add(&a, &b, &d, k, n);
    assert_close(&gpu, &cpu, 1e-4, 1e-5);
}

#[test]
fn gemv_add_smollm2_mlp_down() {
    // MLP down + residual at decode: 1×1536 × 1536×576 + residual
    test_gemv_add_shape(1536, 576, 100);
}

#[test]
fn gemv_add_smollm2_o_proj() {
    // Attention out-proj + residual: 1×576 × 576×576 + residual
    test_gemv_add_shape(576, 576, 101);
}

#[test]
fn gemv_add_smolvla_mlp_down() {
    test_gemv_add_shape(2048, 720, 102);
}

#[test]
fn gemv_non_multiple_k() {
    // K not a multiple of 256 — exercises the shared-memory chunk tail.
    test_shape(100, 256, 10);
    test_shape(511, 256, 11);
    test_shape(513, 256, 12);
}
