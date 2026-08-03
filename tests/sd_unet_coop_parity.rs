//! Regression for the Apple f32 cooperative-matrix path. Apple exposes 8x8
//! tiles; the generated forward-convolution kernel previously used a 16x16
//! vec4 staging map and wrote beyond its 64-element workgroup arrays. The
//! opening 4-to-64-channel convolution then corrupted the whole SD U-Net.

use meganeura::models::sd_unet::{self, SDUNetConfig};
use meganeura::{Graph, Session};
use std::sync::Mutex;

static GPU_TEST_LOCK: Mutex<()> = Mutex::new(());

fn name_seed(name: &str) -> f32 {
    let mut hash = 0u32;
    for byte in name.bytes() {
        hash = hash.wrapping_mul(31).wrapping_add(u32::from(byte));
    }
    (hash % 10_000) as f32
}

fn initialize(session: &mut Session) {
    for (name, buffer) in session.plan().param_buffers.clone() {
        let len = session.plan().buffers[buffer.0 as usize] / size_of::<f32>();
        let data = if name.contains(".norm") && name.ends_with(".weight") {
            vec![1.0; len]
        } else if name.contains(".norm") && name.ends_with(".bias") {
            vec![0.0; len]
        } else {
            let seed = name_seed(&name);
            (0..len)
                .map(|index| (index as f32 * 0.01 + seed).sin() * 0.02)
                .collect()
        };
        session.set_parameter(&name, &data);
    }
}

fn run(cooperative: bool) -> Vec<f32> {
    // SAFETY: the test holds GPU_TEST_LOCK while changing process-global
    // feature switches and is intended to run serially with other GPU tests.
    unsafe {
        std::env::remove_var("MEGANEURA_COOP_F16");
        if cooperative {
            std::env::remove_var("MEGANEURA_DISABLE_COOP");
        } else {
            std::env::set_var("MEGANEURA_DISABLE_COOP", "1");
        }
    }

    let config = SDUNetConfig::small();
    let output_len =
        (config.batch_size * config.in_channels * config.resolution * config.resolution) as usize;
    let mut graph = Graph::new();
    let output = sd_unet::build_unet(&mut graph, &config);
    graph.set_outputs(vec![output]);
    let mut session = meganeura::build(&graph, meganeura::SessionConfig::inference_from_env()).0;
    initialize(&mut session);

    let noisy_latent = (0..output_len)
        .map(|index| (index as f32 * 0.01).sin())
        .collect::<Vec<_>>();
    let timestep_embedding = (0..(config.batch_size * config.time_input_dim) as usize)
        .map(|index| (index as f32 * 0.005).sin())
        .collect::<Vec<_>>();
    let text_context = (0..(config.context_len * config.context_dim) as usize)
        .map(|index| (index as f32 * 0.003).cos() * 0.1)
        .collect::<Vec<_>>();
    session.set_input("noisy_latent", &noisy_latent);
    session.set_input("timestep_embedding", &timestep_embedding);
    session.set_input("text_context", &text_context);
    session.step();
    session.wait();
    session.read_output(output_len)
}

#[test]
fn sd_unet_f32_coop_matches_scalar() {
    let _guard = GPU_TEST_LOCK.lock().expect("GPU test lock poisoned");
    let scalar = run(false);
    let cooperative = run(true);

    let mut error_sq = 0.0f64;
    let mut reference_sq = 0.0f64;
    let mut max_abs = 0.0f32;
    for (&got, &reference) in cooperative.iter().zip(&scalar) {
        let error = got - reference;
        error_sq += f64::from(error) * f64::from(error);
        reference_sq += f64::from(reference) * f64::from(reference);
        max_abs = max_abs.max(error.abs());
    }
    let relative_l2 = (error_sq / reference_sq.max(f64::EPSILON)).sqrt();
    eprintln!("SD U-Net f32 coop/scalar: relative_l2={relative_l2:.6e}, max_abs={max_abs:.6e}");
    assert!(
        relative_l2 < 1e-4,
        "f32 cooperative path diverged from scalar: relative_l2={relative_l2:.6e}, max_abs={max_abs:.6e}"
    );
}
