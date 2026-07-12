//! Step-local intermediates default to `Memory::Device` (no CPU
//! mapping); `MEGANEURA_NO_DEVICE_LOCAL=1` forces everything back to
//! host-visible `Memory::Shared`. Training must produce the same numbers
//! in both layouts, and user-visible buffers (loss, outputs, params)
//! must remain readable regardless.
//!
//! Kept in its own test binary: the env var is process-global and the
//! sessions here are built sequentially to avoid racing other tests.

use meganeura::{Graph, build_session, nn};

fn model(bs: usize) -> Graph {
    let mut g = Graph::new();
    let x = g.input("x", &[bs, 8]);
    let target = g.input("target", &[bs, 4]);
    let fc1 = nn::Linear::new(&mut g, "fc1", 8, 16);
    let norm = nn::RmsNorm::new(&mut g, "norm.weight", 16, 1e-5);
    let fc2 = nn::Linear::new(&mut g, "fc2", 16, 4);
    let h = fc1.forward(&mut g, x);
    let h = g.relu(h);
    let h = norm.forward(&mut g, h);
    let y = fc2.forward(&mut g, h);
    let loss = g.mse_loss(y, target);
    g.set_outputs(vec![loss, y]);
    g
}

fn run(bs: usize) -> (f32, Vec<f32>) {
    let g = model(bs);
    let mut s = build_session(&g);
    s.set_parameter("fc1.weight", &vec![0.05; 8 * 16]);
    s.set_parameter("fc1.bias", &[0.1; 16]);
    s.set_parameter("norm.weight", &[1.0; 16]);
    s.set_parameter("fc2.weight", &vec![0.1; 16 * 4]);
    s.set_parameter("fc2.bias", &[0.0; 4]);
    let x: Vec<f32> = (0..bs * 8).map(|i| 0.1 * (i % 7) as f32).collect();
    s.set_input("x", &x);
    s.set_input("target", &vec![0.5; bs * 4]);
    s.set_learning_rate(0.1);
    let mut losses = Vec::new();
    for _ in 0..3 {
        s.step();
        s.wait();
        losses.push(s.read_loss());
    }
    let mut y = vec![0.0f32; bs * 4];
    s.read_output_by_index(1, &mut y);
    (losses[losses.len() - 1], y)
}

#[test]
fn device_local_intermediates_match_shared() {
    // Force the all-host-visible layout for the baseline.
    unsafe { std::env::set_var("MEGANEURA_NO_DEVICE_LOCAL", "1") };
    let (shared_loss, shared_y) = run(4);
    unsafe { std::env::remove_var("MEGANEURA_NO_DEVICE_LOCAL") };

    // Default layout: step-local intermediates are device-local.
    let (device_loss, device_y) = run(4);

    assert!(
        (shared_loss - device_loss).abs() <= 1e-6 * shared_loss.abs().max(1.0),
        "loss diverged: shared {shared_loss} vs device-local {device_loss}"
    );
    for (i, (s, d)) in shared_y.iter().zip(device_y.iter()).enumerate() {
        assert!(
            (s - d).abs() <= 1e-6 * s.abs().max(1.0),
            "output elem {i} diverged: shared {s} vs device-local {d}"
        );
    }
    // Training actually happened (loss is finite and the run converged
    // the same way in both layouts).
    assert!(device_loss.is_finite());
}
