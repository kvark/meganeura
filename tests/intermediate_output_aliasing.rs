//! Reproducer for the bug where an intermediate node (with both forward
//! consumers and a backward gradient path) returned to the user via
//! `set_outputs` becomes corrupted after `step()` at batch_size > 1.

use meganeura::{Graph, build_session, nn};

#[test]
fn intermediate_node_output_is_stable_across_batch_sizes() {
    for &bs in &[1usize, 2, 4] {
        let mut g = Graph::new();
        let x = g.input("x", &[bs, 4]);
        let target = g.input("target", &[bs, 3]);

        let fc1 = nn::Linear::new(&mut g, "fc1", 4, 4);
        let norm = nn::RmsNorm::new(&mut g, "norm.weight", 4, 1e-5);
        let fc_out = nn::Linear::new(&mut g, "fc_out", 4, 3);
        let fc2 = nn::Linear::new(&mut g, "fc2", 3, 3);

        let h = fc1.forward(&mut g, x);
        let h = g.relu(h);
        let h = norm.forward(&mut g, h);
        let y = fc_out.forward(&mut g, h);
        let y2 = fc2.forward(&mut g, y);
        let loss = g.mse_loss(y2, target);
        g.set_outputs(vec![loss, y]);

        let mut s = build_session(&g);
        s.set_parameter("fc1.weight", &vec![0.1; 16]);
        s.set_parameter("fc1.bias", &vec![0.1; 4]);
        s.set_parameter("norm.weight", &vec![1.0; 4]);
        s.set_parameter("fc_out.weight", &vec![0.2; 12]);
        s.set_parameter("fc_out.bias", &vec![0.0; 3]);
        s.set_parameter("fc2.weight", &vec![0.3; 9]);
        s.set_parameter("fc2.bias", &vec![0.0; 3]);

        let input: Vec<f32> = (0..bs * 4).map(|i| (i as f32 + 1.0) * 0.1).collect();
        s.set_input("x", &input);
        s.set_input("target", &vec![0.0f32; bs * 3]);
        s.set_learning_rate(0.0);
        s.step();
        s.wait();

        let mut out = vec![-999.0f32; bs * 3];
        s.read_output_by_index(1, &mut out);
        eprintln!("bs={bs}: {out:?}");

        // Each row should match fc_out(norm(relu(fc1(x_row)))) computed by
        // a pure forward-only session. No cross-row bleed, no NaN, no stale.
        //
        // Build a second session with only forward to compute expected.
        let mut fg = Graph::new();
        let fx = fg.input("x", &[bs, 4]);
        let ffc1 = nn::Linear::new(&mut fg, "fc1", 4, 4);
        let fnorm = nn::RmsNorm::new(&mut fg, "norm.weight", 4, 1e-5);
        let ffc_out = nn::Linear::new(&mut fg, "fc_out", 4, 3);
        let fh = ffc1.forward(&mut fg, fx);
        let fh = fg.relu(fh);
        let fh = fnorm.forward(&mut fg, fh);
        let fy = ffc_out.forward(&mut fg, fh);
        fg.set_outputs(vec![fy]);
        let mut fs = meganeura::train::build_inference_session(&fg);
        fs.set_parameter("fc1.weight", &vec![0.1; 16]);
        fs.set_parameter("fc1.bias", &vec![0.1; 4]);
        fs.set_parameter("norm.weight", &vec![1.0; 4]);
        fs.set_parameter("fc_out.weight", &vec![0.2; 12]);
        fs.set_parameter("fc_out.bias", &vec![0.0; 3]);
        fs.set_input("x", &input);
        fs.step();
        fs.wait();
        let mut expected = vec![0.0f32; bs * 3];
        fs.read_output_by_index(0, &mut expected);
        eprintln!("bs={bs} expected: {expected:?}");

        for (i, (a, e)) in out.iter().zip(expected.iter()).enumerate() {
            let diff = (a - e).abs();
            assert!(
                diff < 1e-4,
                "bs={bs} elem {i}: got {a}, expected {e} (diff {diff})"
            );
        }
    }
}
