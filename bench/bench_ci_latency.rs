/// CI latency benchmark: measures data upload + GPU processing + result download.
///
/// Runs a small MLP that exercises the full runtime path. Compilation
/// happens before timing — only the hot loop (upload, dispatch, readback)
/// is measured.
///
/// Usage:
///   cargo run --release --example bench_ci_latency
use std::time::Instant;

use meganeura::{Graph, build_inference_session, build_session};

fn main() {
    env_logger::init();

    let batch = 16;
    let input_dim = 128;
    let hidden = 64;
    let classes = 10;

    // ── Build & compile (not timed) ──

    // Training graph (forward + backward + SGD)
    let mut g = Graph::new();
    let x = g.input("x", &[batch, input_dim]);
    let labels = g.input("labels", &[batch, classes]);
    let w1 = g.parameter("w1", &[input_dim, hidden]);
    let b1 = g.parameter("b1", &[hidden]);
    let h1 = g.matmul(x, w1);
    let h1 = g.bias_add(h1, b1);
    let h1 = g.relu(h1);
    let w2 = g.parameter("w2", &[hidden, hidden]);
    let b2 = g.parameter("b2", &[hidden]);
    let h2 = g.matmul(h1, w2);
    let h2 = g.bias_add(h2, b2);
    let h2 = g.relu(h2);
    let w3 = g.parameter("w3", &[hidden, classes]);
    let out = g.matmul(h2, w3);
    let loss = g.cross_entropy_loss(out, labels);
    g.set_outputs(vec![loss]);
    let mut train_session = build_session(&g);

    // Inference graph (forward only, logits output)
    let mut g_inf = Graph::new();
    let xi = g_inf.input("x", &[batch, input_dim]);
    let w1i = g_inf.parameter("w1", &[input_dim, hidden]);
    let b1i = g_inf.parameter("b1", &[hidden]);
    let h1i = g_inf.matmul(xi, w1i);
    let h1i = g_inf.bias_add(h1i, b1i);
    let h1i = g_inf.relu(h1i);
    let w2i = g_inf.parameter("w2", &[hidden, hidden]);
    let b2i = g_inf.parameter("b2", &[hidden]);
    let h2i = g_inf.matmul(h1i, w2i);
    let h2i = g_inf.bias_add(h2i, b2i);
    let h2i = g_inf.relu(h2i);
    let w3i = g_inf.parameter("w3", &[hidden, classes]);
    let logits = g_inf.matmul(h2i, w3i);
    g_inf.set_outputs(vec![logits]);
    let mut inf_session = build_inference_session(&g_inf);

    // Load weights (not timed)
    let w1_data = vec![0.01_f32; input_dim * hidden];
    let b1_data = vec![0.0_f32; hidden];
    let w2_data = vec![0.01_f32; hidden * hidden];
    let b2_data = vec![0.0_f32; hidden];
    let w3_data = vec![0.01_f32; hidden * classes];
    for s in [
        &mut train_session as &mut meganeura::Session,
        &mut inf_session,
    ] {
        s.set_parameter("w1", &w1_data);
        s.set_parameter("b1", &b1_data);
        s.set_parameter("w2", &w2_data);
        s.set_parameter("b2", &b2_data);
        s.set_parameter("w3", &w3_data);
    }

    let x_data: Vec<f32> = (0..batch * input_dim)
        .map(|i| ((i * 7 + 13) % 256) as f32 / 255.0)
        .collect();
    let mut labels_data = vec![0.0_f32; batch * classes];
    for b in 0..batch {
        labels_data[b * classes + (b % classes)] = 1.0;
    }

    // ── Warmup (not timed) ──
    let warmup = 3;
    for _ in 0..warmup {
        inf_session.set_input("x", &x_data);
        inf_session.step();
        inf_session.wait();
        let _ = inf_session.read_output(batch * classes);
    }
    for _ in 0..warmup {
        train_session.set_input("x", &x_data);
        train_session.set_input("labels", &labels_data);
        train_session.step();
        train_session.wait();
        let _ = train_session.read_loss();
    }

    // ── Timed: upload + dispatch + readback ──
    let runs = 10;

    // Forward (inference)
    let mut fwd_times = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t0 = Instant::now();
        inf_session.set_input("x", &x_data);
        inf_session.step();
        inf_session.wait();
        let _ = inf_session.read_output(batch * classes);
        fwd_times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }

    // Training step (forward + backward + SGD)
    let mut train_times = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t0 = Instant::now();
        train_session.set_input("x", &x_data);
        train_session.set_input("labels", &labels_data);
        train_session.step();
        train_session.wait();
        let _ = train_session.read_loss();
        train_times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }

    // ── Statistics ──
    let fwd_avg = fwd_times.iter().sum::<f64>() / runs as f64;
    let fwd_med = median(&mut fwd_times);
    let train_avg = train_times.iter().sum::<f64>() / runs as f64;
    let train_med = median(&mut train_times);
    let loss_val = train_session.read_loss();
    assert!(loss_val.is_finite(), "loss not finite: {}", loss_val);

    // ── JSON output ──
    println!("{{");
    println!("  \"benchmark\": \"ci_latency\",");
    println!("  \"forward_avg_ms\": {:.2},", fwd_avg);
    println!("  \"forward_median_ms\": {:.2},", fwd_med);
    println!("  \"train_step_avg_ms\": {:.2},", train_avg);
    println!("  \"train_step_median_ms\": {:.2},", train_med);
    println!("  \"loss\": {:.6},", loss_val);
    println!("  \"runs\": {},", runs);
    println!("  \"warmup\": {}", warmup);
    println!("}}");
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}
