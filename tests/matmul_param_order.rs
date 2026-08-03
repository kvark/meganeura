//! Ground truth for the plain Op::MatMul uniform param order.
//! compile.rs pushes params [m, k, n] for ShaderEntry::MatMul/MatMulGemv
//! while matmul.wgsl declares struct Params {m, n, k} — for any k != n
//! one of them is wrong. Verify C = A @ B against CPU for non-square
//! shapes through the real session pipeline.

use meganeura::Graph;

fn check(m: usize, k: usize, n: usize) {
    let mut g = Graph::new();
    let a = g.input("a", &[m, k]);
    let b = g.input("b", &[k, n]);
    let c = g.matmul(a, b);
    g.set_outputs(vec![c]);

    let mut session = meganeura::build(&g, meganeura::SessionConfig::inference_from_env()).0;
    let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1 + 0.5).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.01 - 0.3).collect();
    session.set_input("a", &a_data);
    session.set_input("b", &b_data);
    session.step();
    session.wait();
    let got = session.read_output(m * n);

    let mut want = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0;
            for t in 0..k {
                s += a_data[i * k + t] * b_data[t * n + j];
            }
            want[i * n + j] = s;
        }
    }
    let max_diff = got
        .iter()
        .zip(&want)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-3,
        "matmul [{m}x{k}]@[{k}x{n}] wrong: max_diff={max_diff:.4} got[..4]={:?} want[..4]={:?}",
        &got[..4.min(got.len())],
        &want[..4.min(want.len())]
    );
}

#[test]
fn matmul_square() {
    check(8, 8, 8);
}

#[test]
fn matmul_k_lt_n() {
    // The M4 repro shape: grad_sum = grad_pred[256,2] @ W[2,4].
    check(256, 2, 4);
}

#[test]
fn matmul_k_gt_n() {
    check(16, 32, 8);
}

#[test]
fn matmul_gemv_m1() {
    // m=1, n % 4 == 0 routes to MatMulGemv.
    check(1, 1, 512);
}

#[test]
fn matmul_gemv_m1_k8() {
    check(1, 8, 64);
}
