//! Mixed-precision embedding parity: `embedding_f16(idx, to_f16(w))` must
//! match the f32 `embedding(idx, w)` in both forward value and the gradient
//! w.r.t. the f32 master parameter `w`, within f16 rounding tolerance.
//! This validates the f16-coefficients path (halved gather bytes) end to
//! end — forward gather + straight-through backward + scatter-add.

use meganeura::{Graph, build_session};

fn run(use_f16: bool) -> (f32, Vec<f32>) {
    let vocab = 5usize;
    let seq = 12usize;
    let hidden = 8usize;
    let mut g = Graph::new();
    let idx = g.input_u32("idx", &[seq]);
    let w = g.parameter("w", &[vocab, hidden]);
    let x = g.input("x", &[seq, hidden]);
    let gathered = if use_f16 {
        let w16 = g.to_f16(w);
        g.embedding_f16(idx, w16)
    } else {
        g.embedding(idx, w)
    };
    let prod = g.mul(gathered, x);
    let loss = g.sum_all(prod);
    g.set_outputs(vec![loss]);

    let mut s = build_session(&g);
    let w_init: Vec<f32> = (0..vocab * hidden).map(|i| (i as f32 % 7.0 - 3.0) * 0.3).collect();
    let x_data: Vec<f32> = (0..seq * hidden).map(|i| (i as f32 % 5.0 - 2.0) * 0.4).collect();
    let idx_data: Vec<u32> = (0..seq).map(|i| (i * 2 % vocab) as u32).collect();
    s.set_parameter("w", &w_init);
    s.set_input_u32("idx", &idx_data);
    s.set_input("x", &x_data);
    s.set_learning_rate(0.0);
    s.step();
    s.wait();
    let loss = s.read_loss();
    let mut grad = vec![0.0f32; vocab * hidden];
    s.read_param_grad("w", &mut grad);
    (loss, grad)
}

#[test]
fn f16_embedding_matches_f32_within_rounding() {
    let (loss_f32, grad_f32) = run(false);
    let (loss_f16, grad_f16) = run(true);

    // f16 has ~3 significant digits; allow ~1e-2 relative + small abs floor.
    let rel = |a: f32, b: f32| (a - b).abs() / a.abs().max(b.abs()).max(1e-4);
    assert!(
        rel(loss_f32, loss_f16) < 2e-2,
        "loss mismatch: f32={loss_f32}, f16={loss_f16}"
    );
    for (i, (a, b)) in grad_f32.iter().zip(grad_f16.iter()).enumerate() {
        assert!(
            rel(*a, *b) < 2e-2 || (a - b).abs() < 1e-3,
            "grad[{i}] mismatch: f32={a}, f16={b}"
        );
    }
    eprintln!("f16 embedding parity OK: loss f32={loss_f32:.5} f16={loss_f16:.5}");
}
