use std::time::Instant;

fn main() {
    env_logger::init();
    let config = meganeura::models::smollm2::SmolLM2Config::smollm2_135m();
    let seq_len: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);

    eprintln!("SmolLM2-135M training benchmark (seq={})", seq_len);
    let g = meganeura::models::smollm2::build_training_graph(&config, seq_len);

    let mut sess = meganeura::build_session(&g);
    eprintln!(
        "  {} dispatches, {} groups",
        sess.plan().dispatches.len(),
        sess.num_groups()
    );

    for (name, buf_ref) in sess.plan().param_buffers.clone() {
        let n = sess.plan().buffers[buf_ref.0 as usize] / 4;
        let data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin() * 0.1).collect();
        sess.set_parameter(&name, &data);
    }

    let tokens: Vec<u32> = (0..seq_len as u32)
        .map(|i| i % config.vocab_size as u32)
        .collect();
    let mut labels = vec![0.0f32; seq_len * config.vocab_size];
    for i in 0..seq_len {
        labels[i * config.vocab_size + (i + 1) % config.vocab_size] = 1.0;
    }

    sess.set_learning_rate(1e-5);
    sess.set_input_u32("token_ids", &tokens);
    sess.set_input("labels", &labels);

    // Warmup
    eprintln!("warmup (3 steps)...");
    for _ in 0..3 {
        sess.step();
        sess.wait();
    }

    // Benchmark
    let runs = 5;
    eprintln!("benchmarking ({} steps)...", runs);
    let mut times = Vec::new();
    for i in 0..runs {
        let t0 = Instant::now();
        sess.step();
        sess.wait();
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        times.push(ms);
        eprintln!("  step {}: {:.1}ms", i + 1, ms);
    }
    let avg = times.iter().sum::<f64>() / runs as f64;
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[runs / 2];
    eprintln!("  avg: {:.1}ms, median: {:.1}ms", avg, median);

    // GPU profiling if MEGANEURA_PROFILE=1
    if std::env::var("MEGANEURA_PROFILE").is_ok() {
        sess.set_profiling(true);
        sess.step();
        sess.wait();
        sess.step();
        sess.wait();
        sess.step();
        sess.wait();
        sess.dump_gpu_timings();
        sess.set_profiling(false);
    }
}
