//! Reproducer for the bug where an intermediate MatMul that's also a user
//! output gets Nop'd by the MatMul+Add fusion (e.g. when followed by
//! `mse_loss(matmul(h, W), target)`). Toposort then silently drops the
//! dead output from `outputs()`, collapsing `num_outputs()` and making
//! `read_output_by_index(1)` panic.

use meganeura::{Graph, build_session, nn};

#[test]
fn output_matmul_survives_matmul_add_fusion() {
    // Graph: pred = fc(h); loss = mse_loss(pred, target)
    // mse_loss decomposes to Add(pred, -target) -> Mul(diff, diff) -> MeanAll.
    // The MatMul+Add fusion used to Nop `pred` even though it was a user
    // output, silently dropping it.
    let bs = 2usize;
    let mut g = Graph::new();
    let h = g.input("h", &[bs, 4]);
    let target = g.input("target", &[bs, 3]);
    let fc = nn::Linear::new(&mut g, "fc", 4, 3);
    let pred_pre_bias = g.matmul(h, fc.weight);
    // Use a raw matmul (not fc.forward) so there's no bias — the only
    // Add operation on `pred_pre_bias` is the one inside mse_loss.
    let loss = g.mse_loss(pred_pre_bias, target);
    g.set_outputs(vec![loss, pred_pre_bias]);

    let mut s = build_session(&g);
    assert_eq!(
        s.num_outputs(),
        2,
        "both loss and pred should survive optimization (got {})",
        s.num_outputs()
    );
    s.set_parameter("fc.weight", &[0.2; 12]);
    s.set_parameter("fc.bias", &[0.0; 3]);
    let input: Vec<f32> = (0..bs * 4).map(|i| (i as f32 + 1.0) * 0.1).collect();
    s.set_input("h", &input);
    s.set_input("target", &vec![0.0f32; bs * 3]);
    s.set_learning_rate(0.0);
    s.step();
    s.wait();

    let mut pred = vec![-999.0f32; bs * 3];
    s.read_output_by_index(1, &mut pred);

    // Compute expected pred = h @ W (W = 0.2 all-ones 4x3).
    // row i: sum_j h[i,j] * 0.2 = 0.2 * sum of that row
    for i in 0..bs {
        let row_sum: f32 = input[i * 4..(i + 1) * 4].iter().sum();
        let expected = 0.2 * row_sum;
        for j in 0..3 {
            let got = pred[i * 3 + j];
            let diff = (got - expected).abs();
            assert!(
                diff < 1e-5,
                "pred[{i},{j}]: got {got}, expected {expected} (diff {diff})"
            );
        }
    }
}
