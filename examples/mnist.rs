/// MNIST training example for Meganeura.
///
/// Demonstrates the full pipeline:
///   Graph definition → autodiff → egglog optimization → compile → GPU execution
///
/// This example uses synthetic data since we don't bundle MNIST files.
/// Replace with real MNIST loading for actual training.
use meganeura::{Graph, build_session};

fn main() {
    env_logger::init();

    let batch = 32;
    let input_dim = 784; // 28x28
    let hidden = 128;
    let classes = 10;
    let epochs = 10;
    let steps_per_epoch = 100;
    let lr = 0.01_f32;

    // --- Build the model graph ---
    let mut g = Graph::new();

    // Inputs
    let x = g.input("x", &[batch, input_dim]);
    let labels = g.input("labels", &[batch, classes]);

    // Layer 1: linear + relu
    let w1 = g.parameter("w1", &[input_dim, hidden]);
    let b1 = g.parameter("b1", &[hidden]);
    let mm1 = g.matmul(x, w1);
    let h1 = g.bias_add(mm1, b1);
    let a1 = g.relu(h1);

    // Layer 2: linear → logits
    let w2 = g.parameter("w2", &[hidden, classes]);
    let b2 = g.parameter("b2", &[classes]);
    let mm2 = g.matmul(a1, w2);
    let logits = g.bias_add(mm2, b2);

    // Loss
    let loss = g.cross_entropy_loss(logits, labels);
    g.set_outputs(vec![loss]);

    println!("forward graph:\n{}", g);

    // --- Build training session ---
    // This runs: autodiff → egglog optimize → compile → GPU init
    println!("building session (autodiff + egglog + compile)...");
    let mut session = build_session(&g);
    println!(
        "session ready: {} buffers, {} dispatches",
        session.plan().buffers.len(),
        session.plan().dispatches.len()
    );

    // --- Initialize parameters ---
    // Xavier initialization
    let w1_data = xavier_init(input_dim, hidden);
    let b1_data = vec![0.0_f32; hidden];
    let w2_data = xavier_init(hidden, classes);
    let b2_data = vec![0.0_f32; classes];

    session.set_parameter("w1", &w1_data);
    session.set_parameter("b1", &b1_data);
    session.set_parameter("w2", &w2_data);
    session.set_parameter("b2", &b2_data);

    // --- Training loop ---
    println!("training...");
    for epoch in 0..epochs {
        let mut total_loss = 0.0_f32;
        for step in 0..steps_per_epoch {
            // Generate synthetic batch
            let (images, targets) = synthetic_batch(batch, input_dim, classes);
            session.set_input("x", &images);
            session.set_input("labels", &targets);

            // Forward + backward
            session.step();
            session.wait();

            // SGD update (CPU fallback until GPU shader binding is complete)
            session.sgd_step_cpu(lr);

            let loss_val = session.read_loss();
            total_loss += loss_val;

            if step % 50 == 0 {
                println!("  epoch {} step {}: loss = {:.4}", epoch, step, loss_val);
            }
        }
        println!(
            "epoch {}: avg_loss = {:.4}",
            epoch,
            total_loss / steps_per_epoch as f32
        );
    }

    println!("done!");
}

fn xavier_init(fan_in: usize, fan_out: usize) -> Vec<f32> {
    use std::f32::consts::PI;
    let scale = (2.0 / (fan_in + fan_out) as f32).sqrt();
    let n = fan_in * fan_out;
    // Simple pseudo-random using sine (deterministic, no external deps)
    (0..n)
        .map(|i| {
            let x = (i as f32 + 1.0) * 0.618_034; // golden ratio
            (x * PI * 2.0).sin() * scale
        })
        .collect()
}

fn synthetic_batch(batch: usize, input_dim: usize, classes: usize) -> (Vec<f32>, Vec<f32>) {
    let images: Vec<f32> = (0..batch * input_dim)
        .map(|i| ((i % 256) as f32) / 255.0)
        .collect();

    // One-hot labels
    let mut labels = vec![0.0_f32; batch * classes];
    for b in 0..batch {
        labels[b * classes + (b % classes)] = 1.0;
    }

    (images, labels)
}
