//! Profiling a plan larger than Blade's timestamp budget.
//!
//! Blade writes at most `limits::PASS_COUNT` timestamps per submission and
//! silently drops the rest, so the profiler used to refuse outright any plan
//! with more dispatches than that — and a decode or training step of a real
//! model runs well into the thousands. It now measures such a plan as several
//! deterministic replays, each timing one window of dispatch indices and
//! batching everything outside the window into a grouped pass on either
//! side.
//!
//! That only produces a usable measurement if the grouped passes keep the
//! plan's barriers, so these tests check the two things that could go wrong:
//! the replays must still compute what an unprofiled run computes, and the
//! windows must between them time every dispatch exactly `samples` times.
//!
//! The window size is forced down through `max_timed_passes_per_replay` so
//! the multi-window path is reachable on a graph small enough to run in CI;
//! nothing else about it differs from hitting the real limit.

use meganeura::profiler::{CaptureOptions, capture_session_profile};
use meganeura::train::SessionConfig;
use meganeura::{GpuOptions, Graph, Session, init_gpu_context_with};
use std::sync::{Arc, OnceLock};

/// Timestamp pools have to be requested before the context exists, and one
/// context is enough for the whole binary.
fn timing_context() -> Arc<blade_graphics::Context> {
    static CONTEXT: OnceLock<Arc<blade_graphics::Context>> = OnceLock::new();
    Arc::clone(CONTEXT.get_or_init(|| {
        Arc::new(
            init_gpu_context_with(GpuOptions {
                timing: true,
                ..GpuOptions::default()
            })
            .expect("GPU context with timestamp pools"),
        )
    }))
}

const LAYERS: usize = 6;
const ROWS: usize = 32;
const DIM: usize = 32;

/// Every op consumes the previous one's output, so the plan has many barrier
/// groups and a window boundary necessarily lands in the middle of a real
/// dependency rather than at a convenient seam.
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
    let (mut session, _) = meganeura::train::build(
        &build_chain(),
        SessionConfig::inference_from_env_on(timing_context()),
    );
    seed(&mut session);
    session.set_input("x", &input());
    session
}

fn run_unprofiled() -> Vec<f32> {
    let mut session = build_session();
    session.step();
    session.wait();
    let mut out = vec![0.0f32; ROWS * DIM];
    session.read_output_by_index(0, &mut out);
    out
}

/// Every window on its own must still execute the entire plan.
///
/// Checking only the stitched capture would not establish this: the windows
/// tile the plan in order, so a bug that ran nothing but the window would
/// still leave the right answer in the buffers once the last replay finished,
/// each replay having carried the chain a little further. The guarantee that
/// matters is per replay, so this steps one window at a time and compares
/// after each one.
///
/// The input changes on every replay, against a reference session stepped on
/// the same input. Reusing one input would let a stale but valid result from
/// the previous replay stand in for work this one skipped.
#[test]
fn a_windowed_step_is_a_complete_replay() {
    sanity_check_reference();
    let mut reference_session = build_session();
    let mut session = build_session();
    let dispatch_count = session.plan().dispatches.len();
    let mut out = vec![0.0f32; ROWS * DIM];
    let mut reference = vec![0.0f32; ROWS * DIM];

    // Widths that divide the plan and widths that do not, so windows land on
    // barrier-group seams and inside groups alike. `None` is the unprofiled
    // path, and a window wider than the plan is the `set_profiling(true)` one.
    let mut replay = 0u32;
    for width in [1, 2, 3, 5, dispatch_count] {
        for start in (0..dispatch_count).step_by(width) {
            let window = start..(start + width).min(dispatch_count);
            replay += 1;
            let x = varied_input(replay);

            reference_session.set_input("x", &x);
            reference_session.step();
            reference_session.wait();
            reference_session.read_output_by_index(0, &mut reference);

            session.set_profiling_window(Some(window.clone()));
            session.set_input("x", &x);
            session.step();
            session.wait();
            session.read_output_by_index(0, &mut out);

            // The same kernels ran on the same data in the same order, so
            // this is exact equality. Comparing bit patterns rather than a
            // tolerance is deliberate: `f32::max` ignores NaN, so folding
            // absolute differences would let an all-NaN output pass as zero
            // error.
            assert_eq!(
                out, reference,
                "window {window:?} of {dispatch_count} changed the result"
            );
        }
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
    // A chain of layer norms should not collapse to a constant, or the
    // comparisons against it would pass for the wrong reason.
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
    let reference = sanity_check_reference();
    let mut session = build_session();
    let dispatch_count = session.plan().dispatches.len();
    // Enough dispatches that a five-pass budget forces several windows,
    // otherwise this test would silently degrade to the single-window path.
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
            // Three timed dispatches per replay: one pass each, plus the two
            // grouped passes around them.
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
        SAMPLES * profile.measurement.window_count,
        "one wall-time entry per replay"
    );

    // Stitching is the whole point: after the replays, every dispatch must
    // carry a full set of samples, not just those in the last window.
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
        // The timestamp label comes from the pass the window opened for this
        // dispatch. A stitching error that shifted attribution by even one
        // position would show up here before it showed up in the numbers.
        assert_eq!(
            dispatch.timestamp_label, dispatch.label,
            "dispatch {index} was attributed to the wrong pass"
        );
    }

    // Untimed head and tail passes must leave the computation alone. The same
    // kernels ran on the same data in the same order, so this is exact
    // equality: any difference means a grouped pass lost a barrier. Compared
    // as values rather than as a folded maximum difference, because
    // `f32::max` ignores NaN and would report an all-NaN output as agreeing.
    let mut out = vec![0.0f32; ROWS * DIM];
    session.read_output_by_index(0, &mut out);
    assert_eq!(out, reference, "windowed profiling changed the result");
}

/// A plan that fits in the budget must still be captured in exactly one
/// replay, timing every dispatch with no grouped passes in the way.
#[test]
fn a_plan_within_the_budget_is_captured_in_one_replay() {
    let mut session = build_session();
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
