//! `Session::record` puts a step into a caller's command encoder.
//!
//! An application that already drives the GPU (a renderer, a video
//! pipeline) wants the model in its own submissions, next to its own
//! passes, rather than in a queue submission per step. These tests record
//! between the application's own passes and check the results against
//! `Session::step`.

use std::sync::Arc;

use blade_graphics as bg;
use meganeura::{Graph, RecordError, Session, SessionConfig};

/// Every layer depends on the previous one, so a missing barrier anywhere
/// in the recorded step shows up as a stale read.
fn chain(layers: usize, rows: usize, dim: usize) -> Graph {
    let mut g = Graph::new();
    let mut x = g.input("x", &[rows, dim]);
    for i in 0..layers {
        let w = g.parameter(&format!("w{i}"), &[dim, dim]);
        let b = g.parameter(&format!("b{i}"), &[dim]);
        x = g.matmul(x, w);
        x = g.bias_add(x, b);
        x = g.gelu(x);
    }
    g.set_outputs(vec![x]);
    g
}

fn seed(session: &mut Session, layers: usize, dim: usize) {
    for i in 0..layers {
        let w: Vec<f32> = (0..dim * dim)
            .map(|k| ((k + i * 7) as f32 * 0.017).sin() * 0.2)
            .collect();
        session.set_parameter(&format!("w{i}"), &w);
        let b: Vec<f32> = (0..dim)
            .map(|k| ((k + i) as f32 * 0.03).cos() * 0.1)
            .collect();
        session.set_parameter(&format!("b{i}"), &b);
    }
}

fn app_encoder(gpu: &bg::Context) -> bg::CommandEncoder {
    gpu.create_command_encoder(bg::CommandEncoderDesc {
        name: "application",
        buffer_count: 1,
        manual_barriers: false,
    })
}

fn shared_buffer(gpu: &bg::Context, name: &str, data: &[f32]) -> bg::Buffer {
    let buffer = gpu.create_buffer(bg::BufferDesc {
        name,
        size: std::mem::size_of_val(data) as u64,
        memory: bg::Memory::Shared,
    });
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr(), buffer.data().cast::<f32>(), data.len());
    }
    buffer
}

fn read_shared(buffer: bg::Buffer, len: usize) -> Vec<f32> {
    unsafe { std::slice::from_raw_parts(buffer.data().cast::<f32>(), len).to_vec() }
}

/// The application uploads the input with its own copy, records two model
/// applications with a copy feeding the first output back as the second
/// input, and copies the result out — one encoder, one submission.
#[test]
fn recorded_inference_matches_step_between_application_passes() {
    const LAYERS: usize = 6;
    const ROWS: usize = 16;
    const DIM: usize = 64;
    let len = ROWS * DIM;
    let bytes = (len * 4) as u64;
    let x: Vec<f32> = (0..len).map(|i| (i as f32 * 0.011).sin()).collect();

    let gpu = crate::support::gpu::gpu();
    let config = SessionConfig {
        gpu: Some(Arc::clone(&gpu)),
        ..crate::support::gpu::inference_config()
    };
    let mut session = meganeura::build(&chain(LAYERS, ROWS, DIM), config).0;
    seed(&mut session, LAYERS, DIM);

    // Reference: two host-driven steps.
    let mut first = vec![0.0; len];
    session.set_input("x", &x);
    session.step();
    session.wait();
    session.read_output_by_index(0, &mut first);
    let mut expected = vec![0.0; len];
    session.set_input("x", &first);
    session.step();
    session.wait();
    session.read_output_by_index(0, &mut expected);
    // Clear the input so a skipped upload cannot pass.
    session.set_input("x", &vec![0.0; len]);

    let source = shared_buffer(&gpu, "app_source", &x);
    let result = shared_buffer(&gpu, "app_result", &vec![f32::NAN; len]);
    let input = session.input_buffer("x").unwrap();
    let output = session.output_buffer(0).unwrap();
    let mut encoder = app_encoder(&gpu);
    encoder.start();
    encoder
        .transfer("upload")
        .copy_buffer_to_buffer(source.at(0), input, bytes);
    session.record(&mut encoder).unwrap();
    encoder
        .transfer("feedback")
        .copy_buffer_to_buffer(output, input, bytes);
    session.record(&mut encoder).unwrap();
    encoder
        .transfer("download")
        .copy_buffer_to_buffer(output, result.at(0), bytes);
    let sync = gpu.submit(&mut encoder);
    session.track_submission(sync);
    session.wait();

    let got = read_shared(result, len);
    assert!(expected.iter().all(|v| v.is_finite()));
    // The same kernels on the same data: bit for bit.
    assert_eq!(
        got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );

    gpu.destroy_command_encoder(&mut encoder);
    gpu.destroy_buffer(source);
    gpu.destroy_buffer(result);
}

fn training_graph() -> Graph {
    let mut g = Graph::new();
    let x = g.input("x", &[4, 8]);
    let w0 = g.parameter("w0", &[8, 8]);
    let w1 = g.parameter("w1", &[8, 8]);
    let h = g.matmul(x, w0);
    let h = g.gelu(h);
    let y = g.matmul(h, w1);
    let y = g.mul(y, y);
    let loss = g.sum_all(y);
    g.set_outputs(vec![loss]);
    g
}

fn training_session(gpu: &Arc<bg::Context>, adam: bool) -> Session {
    let config = SessionConfig {
        gpu: Some(Arc::clone(gpu)),
        ..crate::support::gpu::config()
    };
    let mut session = meganeura::build(&training_graph(), config).0;
    for (name, phase) in [("w0", 0.3), ("w1", 1.1)] {
        let w: Vec<f32> = (0..64)
            .map(|k| (k as f32 * 0.21 + phase).sin() * 0.3)
            .collect();
        session.set_parameter(name, &w);
    }
    let x: Vec<f32> = (0..32).map(|k| (k as f32 * 0.37).cos()).collect();
    session.set_input("x", &x);
    if adam {
        session.set_adam(0.01, 0.9, 0.999, 1e-8);
    } else {
        session.set_learning_rate(0.01);
    }
    session
}

/// Several training steps recorded into one encoder update the parameters
/// exactly as the same number of `step()` calls, for SGD and for Adam,
/// whose step count is baked into each recording.
#[test]
fn recorded_training_steps_match_step() {
    const STEPS: usize = 3;
    let gpu = crate::support::gpu::gpu();
    for adam in [false, true] {
        let mut reference = training_session(&gpu, adam);
        for _ in 0..STEPS {
            reference.step();
        }
        reference.wait();

        let mut recorded = training_session(&gpu, adam);
        let mut encoder = app_encoder(&gpu);
        encoder.start();
        for _ in 0..STEPS {
            recorded.record(&mut encoder).unwrap();
        }
        let sync = gpu.submit(&mut encoder);
        recorded.track_submission(sync);
        recorded.wait();
        gpu.destroy_command_encoder(&mut encoder);

        let mut initial = training_session(&gpu, adam);
        initial.wait();
        for name in ["w0", "w1"] {
            let mut want = vec![0.0; 64];
            let mut got = vec![0.0; 64];
            let mut before = vec![0.0; 64];
            reference.read_param(name, &mut want);
            recorded.read_param(name, &mut got);
            initial.read_param(name, &mut before);
            assert_ne!(want, before, "adam={adam}: {name} did not train");
            assert_eq!(got, want, "adam={adam}: {name}");
        }
        if adam {
            assert_eq!(recorded.adam_step_count(), STEPS as u32);
        }
    }
}

#[test]
fn gradient_accumulation_cannot_be_recorded() {
    let gpu = crate::support::gpu::gpu();
    let mut session = training_session(&gpu, false);
    session.set_grad_accumulate(2);
    let mut encoder = app_encoder(&gpu);
    encoder.start();
    assert_eq!(
        session.record(&mut encoder),
        Err(RecordError::GradAccumulation)
    );
    let sync = gpu.submit(&mut encoder);
    let _ = gpu.wait_for(&sync, !0);
    gpu.destroy_command_encoder(&mut encoder);
}
