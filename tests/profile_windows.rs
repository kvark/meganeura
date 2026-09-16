//! Windowed profiling must preserve a complete replay and stitch one timing
//! sample per dispatch. A small artificial timestamp budget reaches the same
//! path as a production plan that exceeds Blade's limit.

use meganeura::profiler::{CaptureOptions, capture_session_profile};
use meganeura::train::SessionConfig;
use meganeura::{GpuOptions, Graph, Session, init_gpu_context_with};
use std::sync::{Arc, OnceLock};

fn timing_context() -> Option<Arc<blade_graphics::Context>> {
    static CONTEXT: OnceLock<Option<Arc<blade_graphics::Context>>> = OnceLock::new();
    CONTEXT
        .get_or_init(|| {
            init_gpu_context_with(GpuOptions {
                timing: true,
                ..GpuOptions::default()
            })
            .ok()
            .map(Arc::new)
        })
        .clone()
}

fn ordinary_context() -> Arc<blade_graphics::Context> {
    static CONTEXT: OnceLock<Arc<blade_graphics::Context>> = OnceLock::new();
    Arc::clone(CONTEXT.get_or_init(|| {
        Arc::new(init_gpu_context_with(GpuOptions::default()).expect("GPU context"))
    }))
}

const LAYERS: usize = 6;
const ROWS: usize = 32;
const DIM: usize = 32;

fn build_chain() -> Graph {
    let mut g = Graph::new();
    let mut x = g.input("x", &[ROWS, DIM]);
    for i in 0..LAYERS {
        let w = g.parameter(&format!("w{i}"), &[DIM, DIM]);
        let b = g.parameter(&format!("b{i}"), &[DIM]);
        x = g.matmul(x, w);
        x = g.bias_add(x, b);
        x = g.gelu(x);
        let ln_w = g.parameter(&format!("ln{i}.w"), &[DIM]);
        let ln_b = g.parameter(&format!("ln{i}.b"), &[DIM]);
        x = g.layer_norm(x, ln_w, ln_b, 1e-5);
    }
    g.set_outputs(vec![x]);
    g
}

fn seed(session: &mut Session) {
    for i in 0..LAYERS {
        let w: Vec<f32> = (0..DIM * DIM)
            .map(|k| ((k + i * 7) as f32 * 0.017).sin() * 0.1)
            .collect();
        session.set_parameter(&format!("w{i}"), &w);
        let b: Vec<f32> = (0..DIM)
            .map(|k| ((k + i) as f32 * 0.03).cos() * 0.01)
            .collect();
        session.set_parameter(&format!("b{i}"), &b);
        session.set_parameter(&format!("ln{i}.w"), &[1.0f32; DIM]);
        session.set_parameter(&format!("ln{i}.b"), &[0.0f32; DIM]);
    }
}

fn input() -> Vec<f32> {
    (0..ROWS * DIM)
        .map(|i| (i as f32 * 0.011).sin() * 0.5)
        .collect()
}

fn build_session() -> Session {
    let mut config = SessionConfig::inference_from_env_on(ordinary_context());
    config.runtime.gpu_timing = false;
    let (mut session, _) = meganeura::train::build(&build_chain(), config);
    seed(&mut session);
    session.set_input("x", &input());
    session
}

fn build_timed_session() -> Option<Session> {
    let base = SessionConfig::inference_from_env_on(timing_context()?);
    let config = SessionConfig {
        runtime: meganeura::SessionOptions {
            gpu_timing: true,
            ..base.runtime
        },
        ..base
    };
    let (mut session, _) = meganeura::train::build(&build_chain(), config);
    seed(&mut session);
    session.set_input("x", &input());
    Some(session)
}

fn run_unprofiled() -> Vec<f32> {
    let mut session = build_session();
    session.step();
    session.wait();
    let mut out = vec![0.0f32; ROWS * DIM];
    session.read_output_by_index(0, &mut out);
    out
}

#[test]
fn a_windowed_step_is_a_complete_replay() {
    sanity_check_reference();
    let mut reference_session = build_session();
    let mut session = build_session();
    let dispatch_count = session.plan().dispatches.len();
    let mut out = vec![0.0f32; ROWS * DIM];
    let mut reference = vec![0.0f32; ROWS * DIM];

    let middle = dispatch_count / 2;
    let windows = [
        0..1,
        1..4.min(dispatch_count),
        middle.saturating_sub(1)..(middle + 2).min(dispatch_count),
        dispatch_count - 1..dispatch_count,
        0..dispatch_count,
    ];
    for (replay, window) in windows.into_iter().enumerate() {
        let x = varied_input(replay as u32 + 1);
        reference_session.set_input("x", &x);
        reference_session.step();
        reference_session.wait();
        reference_session.read_output_by_index(0, &mut reference);

        session.set_profiling_window(Some(window.clone()));
        session.set_input("x", &x);
        session.step();
        session.wait();
        session.read_output_by_index(0, &mut out);
        assert_eq!(
            out, reference,
            "window {window:?} of {dispatch_count} changed the result"
        );
    }
    session.set_profiling_window(None);
}

/// A different activation per replay, still in the range the fixture uses.
fn varied_input(replay: u32) -> Vec<f32> {
    (0..ROWS * DIM)
        .map(|i| ((i as u32 * 7 + replay * 31) % 97) as f32 * 0.01 - 0.48)
        .collect()
}

fn sanity_check_reference() -> Vec<f32> {
    let reference = run_unprofiled();
    assert!(
        reference.iter().all(|v| v.is_finite()),
        "reference output is not finite"
    );
    let spread = reference.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
        - reference.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    assert!(
        spread > 0.1,
        "reference output is nearly constant: {spread}"
    );
    reference
}

#[test]
fn windowed_capture_times_every_dispatch_and_preserves_the_result() {
    let Some(mut session) = build_timed_session() else {
        eprintln!("SKIP: selected device cannot timestamp GPU passes");
        return;
    };
    let reference = sanity_check_reference();
    let dispatch_count = session.plan().dispatches.len();
    assert!(
        dispatch_count > 8,
        "the chain compiled to only {dispatch_count} dispatches; \
         it can no longer exercise multiple windows"
    );

    const SAMPLES: usize = 2;
    let profile = capture_session_profile(
        &mut session,
        |session| session.set_input("x", &input()),
        CaptureOptions {
            samples: SAMPLES,
            include_pipeline_statistics: false,
            max_timed_passes_per_replay: Some(5),
            ..CaptureOptions::default()
        },
    )
    .expect("windowed capture");

    let expected_windows = dispatch_count.div_ceil(3);
    assert_eq!(
        profile.measurement.window_count, expected_windows,
        "{dispatch_count} dispatches at three per replay"
    );
    assert!(
        profile.measurement.window_count > 1,
        "this test is only meaningful with more than one window"
    );
    assert!(
        profile.measurement.max_window_dispatches <= 3,
        "a window exceeded the requested budget: {}",
        profile.measurement.max_window_dispatches
    );
    assert_eq!(
        profile.measurement.profiled_wall_samples_ms.len(),
        SAMPLES,
        "one summed wall-time entry per sample"
    );
    let gpu_share = profile
        .measurement
        .timestamped_gpu_share_of_profiled_wall_pct;
    assert!(
        (0.0..=100.1).contains(&gpu_share),
        "GPU time from all windows was compared with only one replay's wall time: {gpu_share}%"
    );

    assert_eq!(profile.dispatches.len(), dispatch_count);
    for (index, dispatch) in profile.dispatches.iter().enumerate() {
        assert_eq!(dispatch.index, index);
        assert_eq!(
            dispatch.timing_samples_ms.len(),
            SAMPLES,
            "dispatch {index} ({}) was timed {} times",
            dispatch.label,
            dispatch.timing_samples_ms.len()
        );
        assert!(
            dispatch.timing_samples_ms.iter().all(|ms| ms.is_finite()),
            "dispatch {index} ({}) has a non-finite timing",
            dispatch.label
        );
        assert_eq!(
            dispatch.timestamp_label, dispatch.label,
            "dispatch {index} was attributed to the wrong pass"
        );
    }

    let mut out = vec![0.0f32; ROWS * DIM];
    session.read_output_by_index(0, &mut out);
    assert_eq!(out, reference, "windowed profiling changed the result");
}

#[test]
fn a_plan_within_the_budget_is_captured_in_one_replay() {
    let Some(mut session) = build_timed_session() else {
        eprintln!("SKIP: selected device cannot timestamp GPU passes");
        return;
    };
    let dispatch_count = session.plan().dispatches.len();

    let profile = capture_session_profile(
        &mut session,
        |session| session.set_input("x", &input()),
        CaptureOptions {
            samples: 1,
            include_pipeline_statistics: false,
            ..CaptureOptions::default()
        },
    )
    .expect("single-window capture");

    assert_eq!(profile.measurement.window_count, 1);
    assert_eq!(
        profile.measurement.max_window_dispatches, dispatch_count,
        "the single window must cover the plan"
    );
    assert_eq!(profile.measurement.profiled_wall_samples_ms.len(), 1);
    for dispatch in &profile.dispatches {
        assert_eq!(dispatch.timing_samples_ms.len(), 1);
        assert_eq!(dispatch.timestamp_label, dispatch.label);
    }
}

#[test]
fn a_failed_capture_leaves_the_session_unprofiled() {
    let mut config = SessionConfig::inference_from_env_on(ordinary_context());
    config.runtime.gpu_timing = false;
    let (mut session, _) = meganeura::train::build(&build_chain(), config);
    seed(&mut session);
    session.set_input("x", &input());

    let error = capture_session_profile(
        &mut session,
        |session| session.set_input("x", &input()),
        CaptureOptions {
            samples: 1,
            include_pipeline_statistics: false,
            ..CaptureOptions::default()
        },
    )
    .expect_err("a context without timestamp pools cannot be captured");
    eprintln!("capture failed as expected: {error}");

    assert_eq!(
        session.profiling_window(),
        None,
        "the session was left timing a window after a failed capture"
    );

    let mut out = vec![0.0f32; ROWS * DIM];
    session.set_input("x", &input());
    session.step();
    session.wait();
    session.read_output_by_index(0, &mut out);
    assert!(out.iter().all(|v| v.is_finite()));
    assert_eq!(out, run_unprofiled(), "the later step did not match");
}
