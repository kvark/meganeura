//! Profile Whisper-tiny encoder inference to find the per-kernel
//! breakdown that explains the 3x gap vs PyTorch.
//!
//! Uses random-initialized weights (no HuggingFace download). Runs a
//! few warmup steps, then enables profiling on the next step and dumps
//! GPU pass timings via Session::dump_gpu_timings().
//!
//! Usage:
//!   MEGANEURA_DEVICE_ID=<id> cargo run --release --example profile_whisper_inference

use meganeura::{
    Graph, build_inference_session,
    models::whisper::{self, WhisperConfig},
};

fn main() {
    env_logger::init();

    // Install coop_matrix_available global so the FlashAttentionCoop
    // dispatch (gated additionally by MEGANEURA_FLASH_FWD_COOP=1) can
    // fire on capable GPUs.
    let gpu = meganeura::runtime::init_gpu_context().expect("gpu");
    let result = meganeura::runtime::auto_tune(&gpu, 64);
    eprintln!("coop_matrix_available={}", result.coop_matrix_available);
    meganeura::runtime::install_auto_tune(result);
    drop(gpu);

    let config = WhisperConfig::whisper_tiny();
    let batch = 1u32;
    let mel_len = 3000u32;
    let seq_len = ((mel_len + 2 - 3) / 2 + 1) as usize;

    let mut g = Graph::new();
    let hidden = whisper::build_encoder(&mut g, &config, batch, mel_len);
    g.set_outputs(vec![hidden]);

    let mut session = build_inference_session(&g);
    eprintln!(
        "Whisper-tiny encoder: {} dispatches, {} buffers, seq_len={}",
        session.plan().dispatches.len(),
        session.plan().buffers.len(),
        seq_len,
    );

    // Random init for every parameter buffer.
    for (name, buf_ref) in session.plan().param_buffers.clone() {
        let n = session.plan().buffers[buf_ref.0 as usize] / 4;
        let data: Vec<f32> = (0..n)
            .map(|i| {
                let h = name.len().wrapping_mul(31).wrapping_add(i);
                ((h % 200) as f32 - 100.0) * 0.005
            })
            .collect();
        session.set_parameter(&name, &data);
    }

    let mel: Vec<f32> = (0..config.n_mels * mel_len as usize)
        .map(|i| ((i * 17 + 5) % 1000) as f32 / 1000.0 - 0.5)
        .collect();

    // Warmup: 3 steps without profiling so the driver caches everything.
    session.set_input("mel", &mel);
    for _ in 0..3 {
        session.step();
    }
    session.wait();

    // Profiled step: per-pass GPU timestamps go through encoder.timings()
    // and are printed by dump_gpu_timings(). The dump happens after the
    // *next* step() finishes.
    session.set_profiling(true);
    session.step();
    session.wait();
    session.step();
    session.wait();
    session.step();
    session.wait();
    session.dump_gpu_timings();
}
