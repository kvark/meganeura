//! Profile ResNet-50 training to find the per-kernel breakdown for
//! the worst Meganeura↔PyTorch training gap (4.9x on RTX 5080).
//!
//! Usage:
//!   MEGANEURA_DEVICE_ID=<id> cargo run --release --example profile_resnet50_train

use meganeura::models::resnet;

fn main() {
    env_logger::init();

    // Install coop-matrix availability so the runtime promotes the 3x3
    // stride-1 backward dispatches to the generated conv coop kernels.
    // Without this the scalar Conv2dGradInputGemm runs (34% of training
    // time on RTX 5080).
    let gpu = meganeura::init_gpu_context_with(meganeura::GpuOptions::from_env()).expect("gpu");
    let result = meganeura::runtime::auto_tune(&gpu, 64);
    eprintln!(
        "coop_matrix_available={} f32_tile={} f16_tile={}",
        result.coop_caps.is_supported(),
        result.coop_caps.f32_tile,
        result.coop_caps.f16_tile
    );
    drop(gpu);

    let batch = 1u32;
    let g = resnet::build_resnet50_training(batch);

    let mut sess = meganeura::build(&g, meganeura::SessionConfig::from_env()).0;
    eprintln!(
        "ResNet-50 training: {} dispatches, {} buffers",
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

    // Synthetic input (224x224 RGB) and one-hot labels (1000 classes).
    let img: Vec<f32> = (0..(batch as usize * 3 * 224 * 224))
        .map(|i| ((i % 256) as f32) / 256.0)
        .collect();
    let mut labels = vec![0.0f32; batch as usize * 1000];
    labels[0] = 1.0;

    sess.set_learning_rate(1e-4);
    sess.set_input("image", &img);
    sess.set_input("labels", &labels);

    eprintln!("warmup (3 steps)...");
    for _ in 0..3 {
        sess.step();
        sess.wait();
    }

    let mut samples = Vec::new();
    for _ in 0..10 {
        let start = std::time::Instant::now();
        sess.step();
        sess.wait();
        samples.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    samples.sort_by(f64::total_cmp);
    eprintln!(
        "unprofiled step median {:.3} ms (min {:.3}, max {:.3})",
        samples[samples.len() / 2],
        samples[0],
        samples[samples.len() - 1]
    );

    sess.set_profiling(true);
    sess.step();
    sess.wait();
    let timings = sess.gpu_timings();
    let dispatches = &sess.plan().dispatches;
    let mut ranked: Vec<_> = timings
        .iter()
        .enumerate()
        .filter_map(|(i, (_, dur))| {
            dispatches
                .get(i)
                .map(|d| (dur.as_secs_f64() * 1000.0, i, d))
        })
        .collect();
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
    eprintln!(
        "profiled passes {} / dispatches {}",
        timings.len(),
        dispatches.len()
    );
    for (ms, index, dispatch) in ranked.iter().take(30) {
        eprintln!(
            "  {ms:7.3} ms  #{index:<3} {:?} wg={:?} params={:?}",
            dispatch.shader,
            dispatch.workgroups,
            &dispatch.params[..dispatch.params.len().min(12)]
        );
    }
    sess.dump_gpu_timings();
}
