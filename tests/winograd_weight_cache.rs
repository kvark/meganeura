//! Inference sessions keep a 3×3 convolution's Winograd weight transform
//! between steps; every way of changing the weight must transform it again.
use meganeura::{Graph, Mode, Session, SessionConfig, optimize};

const CHANNELS: u32 = 64;
const SIDE: u32 = 8;
/// Elements of the input and of the output.
const IMAGE: usize = (CHANNELS * SIDE * SIDE) as usize;
const KERNEL: usize = (CHANNELS * CHANNELS * 9) as usize;

fn graph() -> Graph {
    let mut g = Graph::new();
    let x = g.input("x", &[IMAGE]);
    let w = g.parameter("w", &[KERNEL]);
    let y = g.conv2d(x, w, 1, CHANNELS, SIDE, SIDE, CHANNELS, 3, 3, 1, 1);
    g.set_outputs(vec![y]);
    g
}

fn session(direct: bool) -> Session {
    meganeura::build(
        &graph(),
        SessionConfig {
            mode: Mode::Inference,
            gpu: Some(crate::support::gpu::gpu()),
            optimize: optimize::OptimizeConfig {
                no_winograd: direct,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .0
}

fn weights(seed: usize) -> Vec<f32> {
    (0..KERNEL)
        .map(|i| ((i * 13 + seed * 7) % 29) as f32 * 0.001 - 0.014)
        .collect()
}

fn input(seed: usize) -> Vec<f32> {
    (0..IMAGE)
        .map(|i| ((i + seed * 5) % 31) as f32 / 31.0)
        .collect()
}

fn run(session: &mut Session, x: &[f32]) -> Vec<f32> {
    session.set_input("x", x);
    session.step();
    session.wait();
    session.read_output(IMAGE)
}

fn assert_close(winograd: &[f32], direct: &[f32], what: &str) {
    let error = winograd
        .iter()
        .zip(direct)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(error < 1e-4, "{what}: {error}");
}

#[test]
fn changed_weights_are_transformed_again() {
    let (mut winograd, mut direct) = (session(false), session(true));
    for s in [&mut winograd, &mut direct] {
        s.set_parameter("w", &weights(1));
    }
    assert_close(
        &run(&mut winograd, &input(1)),
        &run(&mut direct, &input(1)),
        "first step",
    );
    // Same weights, new input: the kept transform serves the second step.
    assert_close(
        &run(&mut winograd, &input(2)),
        &run(&mut direct, &input(2)),
        "kept weights",
    );

    for s in [&mut winograd, &mut direct] {
        s.set_parameter("w", &weights(2));
    }
    assert_close(
        &run(&mut winograd, &input(2)),
        &run(&mut direct, &input(2)),
        "set_parameter",
    );

    direct.set_parameter("w", &weights(3));
    let path = std::env::temp_dir().join(format!(
        "winograd-weight-cache-{}.safetensors",
        std::process::id()
    ));
    direct.save_checkpoint(&path).unwrap();
    winograd.load_checkpoint(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    assert_close(
        &run(&mut winograd, &input(3)),
        &run(&mut direct, &input(3)),
        "checkpoint",
    );
}

#[test]
fn shared_weights_changed_by_the_donor_are_transformed_again() {
    let (mut donor, mut shared, mut direct) = (session(false), session(false), session(true));
    donor.set_parameter("w", &weights(1));
    shared.share_parameter_from(&mut donor, "w").unwrap();
    direct.set_parameter("w", &weights(1));
    assert_close(
        &run(&mut shared, &input(1)),
        &run(&mut direct, &input(1)),
        "shared",
    );

    // The donor writes the storage both sessions read.
    donor.set_parameter("w", &weights(2));
    direct.set_parameter("w", &weights(2));
    assert_close(
        &run(&mut shared, &input(1)),
        &run(&mut direct, &input(1)),
        "donor write",
    );
    assert_close(
        &run(&mut donor, &input(2)),
        &run(&mut direct, &input(2)),
        "donor",
    );
}
