/// CI latency benchmark: measures data upload + GPU processing + result download.
///
/// Runs a small MLP that exercises the full runtime path. Compilation
/// happens before timing — only the hot loop (upload, dispatch, readback)
/// is measured.
///
/// Usage:
///   cargo run --release --example bench_ci_latency
use std::time::Instant;

use meganeura::Graph;

mod paired;

fn main() {
    env_logger::init();
    let mut paired_options = paired::Options::default();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        assert!(
            paired_options.argument(&arg, &mut args),
            "unknown argument {arg}"
        );
    }
    let paired_gpu = paired_options.enabled.then(|| paired_options.context());

    let batch = 16;
    let input_dim = 128;
    let hidden = 64;
    let classes = 10;

    // ── Build & compile (not timed) ──

    // Training graph, measured both with and without an SGD update.
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
    let (mut train_session, train_construction_ms) = if let Some(ref gpu) = paired_gpu {
        paired_options.build(&g, gpu.clone(), true, false)
    } else {
        let construction = Instant::now();
        let session = meganeura::build(&g, meganeura::SessionConfig::from_env()).0;
        (session, construction.elapsed().as_secs_f64() * 1000.0)
    };

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
    let (mut inf_session, forward_construction_ms) = if let Some(ref gpu) = paired_gpu {
        paired_options.build(&g_inf, gpu.clone(), false, false)
    } else {
        let construction = Instant::now();
        let session = meganeura::build(&g_inf, meganeura::SessionConfig::inference_from_env()).0;
        (session, construction.elapsed().as_secs_f64() * 1000.0)
    };

    // Load weights (not timed)
    let w1_data = vec![0.01_f32; input_dim * hidden];
    let b1_data = vec![0.0_f32; hidden];
    let w2_data = vec![0.01_f32; hidden * hidden];
    let b2_data = vec![0.0_f32; hidden];
    let w3_data = vec![0.01_f32; hidden * classes];
    let initialize = |s: &mut meganeura::Session| {
        s.set_parameter("w1", &w1_data);
        s.set_parameter("b1", &b1_data);
        s.set_parameter("w2", &w2_data);
        s.set_parameter("b2", &b2_data);
        s.set_parameter("w3", &w3_data);
    };
    initialize(&mut train_session);
    initialize(&mut inf_session);

    let x_data: Vec<f32> = (0..batch * input_dim)
        .map(|i| ((i * 7 + 13) % 256) as f32 / 255.0)
        .collect();
    let mut labels_data = vec![0.0_f32; batch * classes];
    for b in 0..batch {
        labels_data[b * classes + (b % classes)] = 1.0;
    }

    if let Some(gpu) = paired_gpu {
        let (mut candidate_inf, candidate_forward_ms) =
            paired_options.build(&g_inf, gpu.clone(), false, true);
        let (mut candidate_train, candidate_training_ms) =
            paired_options.build(&g, gpu, true, true);
        let forward = paired::measure(
            "inference",
            &mut inf_session,
            &mut candidate_inf,
            [forward_construction_ms, candidate_forward_ms],
            paired_options,
            initialize,
            |session| {
                session.set_input("x", &x_data);
                session.step();
                session.wait();
                let _ = session.read_output(batch * classes);
            },
            |session| session.read_output(batch * classes),
        );
        let execute_training = |session: &mut meganeura::Session| {
            session.set_input("x", &x_data);
            session.set_input("labels", &labels_data);
            session.step();
            session.wait();
            let _ = session.read_loss();
        };
        let observe_training = |session: &mut meganeura::Session, update: bool| {
            let mut values = vec![session.read_loss()];
            let mut changed = false;
            for (name, initial) in [
                ("w1", &w1_data),
                ("b1", &b1_data),
                ("w2", &w2_data),
                ("b2", &b2_data),
                ("w3", &w3_data),
            ] {
                let mut value = vec![0.0; initial.len()];
                session.read_param_grad(name, &mut value);
                values.extend_from_slice(&value);
                session.read_param(name, &mut value);
                changed |= &value != initial;
                values.extend_from_slice(&value);
            }
            assert!(
                !update || changed,
                "SGD benchmark did not update parameters"
            );
            values
        };
        let training = paired::measure(
            "forward_backward",
            &mut train_session,
            &mut candidate_train,
            [train_construction_ms, candidate_training_ms],
            paired_options,
            initialize,
            execute_training,
            |session| observe_training(session, false),
        );
        train_session.set_learning_rate(0.001);
        candidate_train.set_learning_rate(0.001);
        let update = paired::measure(
            "forward_backward_sgd",
            &mut train_session,
            &mut candidate_train,
            [train_construction_ms, candidate_training_ms],
            paired_options,
            initialize,
            execute_training,
            |session| observe_training(session, true),
        );
        println!(
            "{}",
            serde_json::json!({
                "benchmark": "ci_latency_paired", "boundary": "input upload + completed step + output/loss readback",
                "state": "identical initial parameters restored before each execution outside timing; fixed SGD learning rate 0.001",
                "qualification": "full logits or loss, every parameter gradient, and every parameter after each sample",
                "cases": [forward, training, update],
            })
        );
        return;
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

    // Forward + backward: preserve the existing CI metric and its baseline.
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

    let train_execution = train_session.execution_report();
    let loss_before_updates = train_session.read_loss();
    train_session.set_learning_rate(0.001);
    for _ in 0..warmup {
        train_session.set_input("x", &x_data);
        train_session.set_input("labels", &labels_data);
        train_session.step();
        train_session.wait();
        let _ = train_session.read_loss();
    }
    let mut update_times = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t0 = Instant::now();
        train_session.set_input("x", &x_data);
        train_session.set_input("labels", &labels_data);
        train_session.step();
        train_session.wait();
        let _ = train_session.read_loss();
        update_times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }

    // ── Statistics ──
    let fwd_avg = fwd_times.iter().sum::<f64>() / runs as f64;
    let fwd_med = median(&fwd_times);
    let train_avg = train_times.iter().sum::<f64>() / runs as f64;
    let train_med = median(&train_times);
    let loss_val = train_session.read_loss();
    assert!(loss_val.is_finite(), "loss not finite: {}", loss_val);
    let mut trained = vec![0.0; w3_data.len()];
    train_session.read_param("w3", &mut trained);
    assert!(trained.iter().all(|x| x.is_finite()));
    assert_ne!(
        trained, w3_data,
        "the training benchmark did not update parameters"
    );

    // ── JSON output ──
    println!(
        "{}",
        serde_json::json!({
            "benchmark": "ci_latency",
            "forward_avg_ms": fwd_avg,
            "forward_median_ms": fwd_med,
            "train_step_avg_ms": train_avg,
            "train_step_median_ms": train_med,
            "forward_samples_ms": fwd_times,
            "train_step_samples_ms": train_times,
            "train_update_avg_ms": update_times.iter().sum::<f64>() / runs as f64,
            "train_update_median_ms": median(&update_times),
            "train_update_samples_ms": update_times,
            "forward_construction_ms": forward_construction_ms,
            "train_construction_ms": train_construction_ms,
            "forward_resident_bytes": inf_session.memory_summary().total_allocated_bytes(),
            "train_resident_bytes": train_session.memory_summary().total_allocated_bytes(),
            "forward_execution": inf_session.execution_report(),
            "train_execution": train_execution,
            "train_update_execution": train_session.execution_report(),
            "device": inf_session.device_information().device_name,
            "driver": inf_session.device_information().driver_name,
            "loss": loss_before_updates,
            "loss_after_updates": loss_val,
            "learning_rate": 0.001,
            "runs": runs,
            "warmup": warmup,
        })
    );
}

fn median(values: &[f64]) -> f64 {
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}
