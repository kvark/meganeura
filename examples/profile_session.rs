//! Produce a structured, repeatable GPU profile for a small inference graph.
//!
//! Run with:
//! `MEGANEURA_GPU_TIMING=1 cargo run --release --example profile_session -- gap-profile.json`

use meganeura::{
    Graph,
    profiler::{CaptureOptions, capture_session_profile, save_session_profile_json},
};
use std::time::Instant;

const ROWS: usize = 512;
const WIDTH: usize = 512;
const WARMUPS: usize = 5;
const SAMPLES: usize = 20;

fn main() {
    if std::env::var_os("MEGANEURA_GPU_TIMING").is_none() {
        eprintln!("set MEGANEURA_GPU_TIMING=1 before starting the example");
        std::process::exit(2);
    }
    let output_path = std::env::args_os()
        .nth(1)
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| "gap-profile.json".into());

    let mut graph = Graph::new();
    let input = graph.input("input", &[ROWS, WIDTH]);
    let weight = graph.parameter("weight", &[WIDTH, WIDTH]);
    let bias = graph.parameter("bias", &[WIDTH]);
    let projected = graph.matmul(input, weight);
    let biased = graph.bias_add(projected, bias);
    let activated = graph.gelu(biased);
    let probabilities = graph.softmax(activated);
    graph.set_outputs(vec![probabilities]);

    let mut session = meganeura::build(&graph, meganeura::SessionConfig::inference_from_env()).0;
    let input_data: Vec<f32> = (0..ROWS * WIDTH)
        .map(|index| (index as f32 * 0.001).sin())
        .collect();
    let weight_data: Vec<f32> = (0..WIDTH * WIDTH)
        .map(|index| (index as f32 * 0.003).cos() * 0.02)
        .collect();
    let bias_data = vec![0.0; WIDTH];
    session.set_parameter("weight", &weight_data);
    session.set_parameter("bias", &bias_data);

    for _ in 0..WARMUPS {
        prepare(&mut session, &input_data);
        session.step();
        session.wait();
    }
    let mut normal_samples_ms = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        prepare(&mut session, &input_data);
        let start = Instant::now();
        session.step();
        session.wait();
        normal_samples_ms.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    normal_samples_ms.sort_by(f64::total_cmp);
    let middle = normal_samples_ms.len() / 2;
    let unprofiled_median_ms = if normal_samples_ms.len().is_multiple_of(2) {
        (normal_samples_ms[middle - 1] + normal_samples_ms[middle]) * 0.5
    } else {
        normal_samples_ms[middle]
    };

    let profile = capture_session_profile(
        &mut session,
        |session| prepare(session, &input_data),
        CaptureOptions {
            samples: 5,
            unprofiled_median_ms: Some(unprofiled_median_ms),
            include_pipeline_statistics: true,
        },
    )
    .expect("capture structured GPU profile");
    save_session_profile_json(&output_path, &profile).expect("save profile JSON");

    println!(
        "wrote {} dispatches and {} samples to {}",
        profile.plan.dispatch_count,
        profile.measurement.sample_count,
        output_path.display()
    );
}

fn prepare(session: &mut meganeura::Session, input: &[f32]) {
    session.set_input("input", input);
}
