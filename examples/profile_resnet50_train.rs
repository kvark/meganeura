//! Profile ResNet-50 training to find the per-kernel breakdown for
//! the worst Meganeura↔PyTorch training gap (4.9x on RTX 5080).
//!
//! Usage:
//!   MEGANEURA_DEVICE_ID=<id> cargo run --release --example profile_resnet50_train

use meganeura::{build_session, models::resnet};

fn main() {
    env_logger::init();

    // Install coop-matrix availability + fusion register costs so the
    // compile path picks Conv2dGradInputGemmCoop3x3 for the 3x3
    // stride-1 backward dispatches. Without this the scalar
    // Conv2dGradInputGemm runs (34% of training time on RTX 5080).
    let gpu = meganeura::runtime::init_gpu_context().expect("gpu");
    let result = meganeura::runtime::auto_tune(&gpu, 64);
    eprintln!("coop_matrix_available={}", result.coop_matrix_available);
    meganeura::runtime::install_auto_tune(result);
    drop(gpu);

    let batch = 1u32;
    let g = resnet::build_resnet50_training(batch);

    let mut sess = build_session(&g);
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

    sess.set_profiling(true);
    sess.step();
    sess.wait();
    sess.step();
    sess.wait();
    sess.step();
    sess.wait();
    sess.dump_gpu_timings();
}
