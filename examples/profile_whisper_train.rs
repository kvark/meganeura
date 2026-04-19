//! Profile Whisper-tiny training to see what dominates after the
//! LayerNorm forward rewrite. Mirrors profile_whisper_inference but
//! builds the training graph (encoder + projection + MSE loss).

use meganeura::{
    build_session,
    models::whisper::{self, WhisperConfig},
};

fn main() {
    env_logger::init();

    let gpu = meganeura::runtime::init_gpu_context().expect("gpu");
    let result = meganeura::runtime::auto_tune(&gpu, 64);
    eprintln!("coop_matrix_available={}", result.coop_caps.is_supported());
    meganeura::runtime::install_auto_tune(result);
    drop(gpu);

    let config = WhisperConfig::whisper_tiny();
    let batch = 1u32;
    let mel_len = 3000u32;
    let g = whisper::build_training_graph(&config, batch, mel_len);

    let mut sess = build_session(&g);
    eprintln!(
        "Whisper-tiny training: {} dispatches, {} buffers",
        sess.plan().dispatches.len(),
        sess.plan().buffers.len(),
    );

    for (name, buf_ref) in sess.plan().param_buffers.clone() {
        let n = sess.plan().buffers[buf_ref.0 as usize] / 4;
        let data: Vec<f32> = (0..n)
            .map(|i| {
                let h = name.len().wrapping_mul(31).wrapping_add(i);
                ((h % 200) as f32 - 100.0) * 0.005
            })
            .collect();
        sess.set_parameter(&name, &data);
    }

    let mel: Vec<f32> = (0..config.n_mels * mel_len as usize)
        .map(|i| ((i * 17 + 5) % 1000) as f32 / 1000.0 - 0.5)
        .collect();
    sess.set_input("mel", &mel);

    sess.set_learning_rate(1e-5);

    eprintln!("warmup (3 steps)...");
    for _ in 0..3 {
        sess.step();
        sess.wait();
    }

    sess.set_profiling(true);
    sess.step();
    sess.wait();
    sess.step();
    sess.wait();
    sess.step();
    sess.wait();
    sess.dump_gpu_timings();
}
