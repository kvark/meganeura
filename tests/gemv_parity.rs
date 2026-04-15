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

    // Sanity: the compiled plan should use the GEMV kernel for M=1.
    let plan = session.plan();
    let gemv_count = plan
        .dispatches
        .iter()
        .filter(|d| matches!(d.shader, compile::ShaderEntry::MatMulGemv))
        .count();
    assert_eq!(
        gemv_count, 1,
        "expected exactly one MatMulGemv dispatch, found {}",
        gemv_count
    );

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

#[test]
fn gemv_non_multiple_k() {
    // K not a multiple of 256 — exercises the shared-memory chunk tail.
    test_shape(100, 256, 10);
    test_shape(511, 256, 11);
    test_shape(513, 256, 12);
}
