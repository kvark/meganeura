//! Class races decide every challenger from shared rounds, qualify only an
//! installed winner and never touch live state. These check invariants, not
//! speed, so they run on any device.

use meganeura::{
    CoopPolicy, Graph, MachineProfile, Session, TuneCalibration, TuneDecision, TuneOptions,
    TuneReport, TuneStrategy,
};
use std::time::Duration;

fn race_options() -> TuneOptions {
    TuneOptions {
        strategy: TuneStrategy::Race,
        calibration: TuneCalibration::Off,
        max_time: Duration::from_secs(60),
        max_classes: 1,
        sample_pairs: 4,
        ..Default::default()
    }
}

/// A product with exactly representable values, so that any tile and any
/// summation order produces the same output bits.
fn product(gpu: std::sync::Arc<blade_graphics::Context>, gpu_timing: bool) -> Session {
    let mut graph = Graph::new();
    let a = graph.input("a", &[48, 96]);
    let b = graph.parameter("b", &[96, 80]);
    let y = graph.matmul(a, b);
    graph.set_outputs(vec![y]);
    let mut config = meganeura::SessionConfig::inference_from_env_on(gpu);
    config.tune = false;
    config.runtime.coop = CoopPolicy::Disabled;
    config.runtime.gpu_timing = gpu_timing;
    let (mut session, _) = meganeura::build(&graph, config);
    let values = |len: usize, seed: usize| -> Vec<f32> {
        (0..len)
            .map(|i| ((i * seed % 17) as f32 - 8.0) / 16.0)
            .collect()
    };
    session.set_input("a", &values(48 * 96, 5));
    session.set_parameter("b", &values(96 * 80, 7));
    session
}

fn run(session: &mut Session) -> Vec<f32> {
    session.step();
    session.wait();
    session.read_output(48 * 80)
}

fn check_race(report: &TuneReport) {
    assert_eq!(report.races.len(), 1, "{report:#?}");
    let race = &report.races[0];
    assert!(race.complete, "{race:#?}");
    assert!(race.rounds >= 4, "{race:#?}");
    assert!(race.incumbent_ms.is_some_and(|ms| ms > 0.0), "{race:#?}");
    for outcome in &report.outcomes {
        assert!(
            matches!(
                outcome.decision,
                TuneDecision::FasterCandidate
                    | TuneDecision::KeepBaseline
                    | TuneDecision::Eliminated
                    | TuneDecision::InvalidOutput
                    | TuneDecision::ShaderRejected
            ),
            "{outcome:#?}"
        );
        // Only an installed winner pays for qualification.
        assert_eq!(
            outcome.qualified,
            outcome.decision == TuneDecision::FasterCandidate,
            "{outcome:#?}"
        );
        if outcome.decision != TuneDecision::ShaderRejected {
            assert!(!outcome.candidate_ms.is_empty(), "{outcome:#?}");
            assert_eq!(outcome.baseline_ms.len(), outcome.candidate_ms.len());
        }
    }
    let winners: Vec<_> = report
        .outcomes
        .iter()
        .filter(|outcome| outcome.decision == TuneDecision::FasterCandidate)
        .collect();
    assert!(winners.len() <= 1, "{winners:#?}");
    assert_eq!(
        race.selected,
        winners
            .first()
            .map_or(race.initial, |outcome| outcome.candidate)
    );
}

#[test]
fn race_decides_every_challenger_and_keeps_live_outputs() {
    let mut session = product(crate::support::gpu::gpu(), false);
    let before = run(&mut session);
    let report = session.tune_with(race_options()).unwrap();
    check_race(&report);
    assert!(!report.races[0].gpu_timestamps);
    assert_eq!(report.outcomes.len() + 1, report.races[0].candidates);
    assert_eq!(run(&mut session), before, "tuning changed the live program");
}

#[test]
fn host_encoder_scratch_is_bounded_and_a_larger_search_can_retry() {
    let mut session = product(crate::support::gpu::gpu(), false);
    let before = run(&mut session);
    // The bindings, staging and shared encoder fit, but the candidate host
    // encoders do not. This used to run anyway, exceeding the caller's limit.
    let report = session
        .tune_with(TuneOptions {
            max_scratch_bytes: 2 << 20,
            ..race_options()
        })
        .unwrap();
    assert_eq!(report.races.len(), 1, "{report:#?}");
    assert!(!report.races[0].complete);
    assert_eq!(report.races[0].rounds, 0);
    assert!(
        report
            .outcomes
            .iter()
            .any(|o| o.decision == TuneDecision::ScratchLimit)
    );
    assert_eq!(report.scratch.unwrap().peak_bytes, 0);
    assert_eq!(run(&mut session), before);
    let report = session.tune_with(race_options()).unwrap();
    check_race(&report);
    // All live candidate encoders plus the upload/qualification encoder.
    assert!(report.scratch.unwrap().peak_bytes > (report.races[0].candidates + 1) * (1 << 20));
    assert_eq!(run(&mut session), before);
}

#[test]
fn a_later_tuning_call_completes_its_cached_profile() {
    let gpu = std::sync::Arc::new(
        meganeura::init_gpu_context_with(meganeura::GpuOptions {
            timing: false,
            ..meganeura::GpuOptions::from_env()
        })
        .unwrap(),
    );
    let mut session = product(gpu, false);
    let first = session
        .tune_with(TuneOptions {
            calibration: TuneCalibration::Measure,
            max_scratch_bytes: 1 << 20,
            ..race_options()
        })
        .unwrap();
    assert!(first.machine_profile.unwrap().bandwidth.is_none());
    assert!(first.calibration_time.is_zero());
    let second = session
        .tune_with(TuneOptions {
            calibration: TuneCalibration::Measure,
            ..race_options()
        })
        .unwrap();
    assert!(
        second.machine_profile.as_ref().unwrap().bandwidth.is_some(),
        "{second:#?}"
    );
    assert!(!second.calibration_time.is_zero());
    check_race(&second);
}

#[test]
fn races_use_gpu_timestamps_when_the_context_records_them() {
    let options = meganeura::GpuOptions::from_env();
    let supported = meganeura::init_gpu_context_with(options.clone())
        .map(|context| context.capabilities().timing)
        .unwrap_or(false);
    if !supported {
        eprintln!("GPU timestamps unavailable; skipping");
        return;
    }
    let gpu = std::sync::Arc::new(
        meganeura::init_gpu_context_with(meganeura::GpuOptions {
            timing: true,
            ..options
        })
        .unwrap(),
    );
    let mut session = product(gpu, true);
    let before = run(&mut session);
    // Timestamp rounds share the scratch encoder, so they fit the same small
    // allowance that cannot accommodate per-candidate host encoders.
    let report = session
        .tune_with(TuneOptions {
            max_scratch_bytes: 2 << 20,
            ..race_options()
        })
        .unwrap();
    check_race(&report);
    assert!(report.races[0].gpu_timestamps);
    assert!(report.scratch.unwrap().peak_bytes <= 2 << 20);
    assert_eq!(run(&mut session), before, "tuning changed the live program");
}

#[test]
fn timestamp_races_keep_typed_split_sequences_and_repetitions_ordered() {
    typed_split_race(true);
}

#[test]
fn host_races_keep_typed_split_sequences_and_repetitions_ordered() {
    typed_split_race(false);
}

fn typed_split_race(gpu_timing: bool) {
    let options = meganeura::GpuOptions::from_env();
    let supported = crate::support::gpu::gpu();
    if (gpu_timing && !supported.capabilities().timing)
        || !meganeura::runtime::typed_matrix_supported(&supported, meganeura::DType::F16)
    {
        eprintln!("F16 matrices or timestamps unavailable; skipping typed race");
        return;
    }
    let gpu = std::sync::Arc::new(
        meganeura::init_gpu_context_with(meganeura::GpuOptions {
            timing: gpu_timing,
            ..options
        })
        .unwrap(),
    );
    let mut graph = Graph::new();
    let a = graph.parameter_f16("a", &[33, 1031]);
    let b = graph.parameter_f16("b", &[1031, 65]);
    let y = graph.matmul(a, b);
    graph.set_outputs(vec![y]);
    let mut config = meganeura::SessionConfig::inference_from_env_on(gpu);
    config.tune = false;
    config.runtime.gpu_timing = gpu_timing;
    let (mut session, _) = meganeura::build(&graph, config);
    session.set_parameter("a", &[0.125; 33 * 1031]);
    session.set_parameter("b", &[0.25; 1031 * 65]);
    session.step();
    session.wait();
    let expected = session.read_output(33 * 65);
    assert!(expected.iter().all(|&x| x == 1031.0 / 32.0));
    let report = session
        .tune_with(TuneOptions {
            // With no warmup/fixed repeats, exercise barriers between sequences as
            // well as the producer -> SumRows dependency within each sequence.
            warmup_runs: 0,
            dispatches_per_sample: 4,
            target_sample_time: None,
            ..race_options()
        })
        .unwrap();
    check_race(&report);
    assert!(report.races[0].class.split_k, "{report:#?}");
    assert!(
        report
            .outcomes
            .iter()
            .all(|o| o.sample_repetitions == Some(4))
    );
    assert_eq!(report.races[0].gpu_timestamps, gpu_timing);
    session.step();
    session.wait();
    assert_eq!(session.read_output(33 * 65), expected);
}

/// With a supplied profile, the classes predicted largest are visited first
/// and a class predicted below 0.1% of the plan is not probed.
#[test]
fn supplied_profile_orders_classes_and_skips_negligible_ones() {
    let mut graph = Graph::new();
    let a = graph.input("a", &[32, 32]);
    let b = graph.parameter("b", &[32, 32]);
    let c = graph.input("c", &[256, 256]);
    let d = graph.parameter("d", &[256, 256]);
    let e = graph.parameter("e", &[256, 512]);
    let small = graph.matmul(a, b);
    let square = graph.matmul(c, d);
    let wide = graph.matmul(c, e);
    graph.set_outputs(vec![small, square, wide]);
    let mut config = crate::support::gpu::inference_config();
    config.tune = false;
    config.runtime.coop = CoopPolicy::Disabled;
    let (mut session, _) = meganeura::build(&graph, config);
    session.set_input("a", &[0.25; 32 * 32]);
    session.set_parameter("b", &[0.5; 32 * 32]);
    session.set_input("c", &[0.25; 256 * 256]);
    session.set_parameter("d", &[0.5; 256 * 256]);
    session.set_parameter("e", &[0.5; 256 * 512]);
    // About 65 KFLOP against 100 MFLOP: the 32-cubed product is predicted at
    // well under 0.1% of the plan.
    let profile = MachineProfile {
        bandwidth: Some(1e10),
        matmul_flops: Some(1e10),
        parallel_width: Some(1),
        ..Default::default()
    };
    let report = session
        .tune_with(TuneOptions {
            calibration: TuneCalibration::Profile(profile.clone()),
            max_classes: 8,
            ..race_options()
        })
        .unwrap();
    assert_eq!(report.machine_profile, Some(profile));
    assert!(report.predicted_plan_ms.is_some_and(|ms| ms > 0.0));
    assert!(report.share_cutoff, "{report:#?}");
    let raced: Vec<(u32, u32)> = report
        .races
        .iter()
        .map(|race| (race.class.m, race.class.n))
        .collect();
    // The wide product is predicted twice as long, so it races first.
    assert_eq!(raced, [(256, 512), (256, 256)], "{report:#?}");
    assert!(report.races.iter().all(|race| race.predicted_ms.is_some()));
    session.step();
    session.wait();
    let mut small_out = vec![0.0; 32 * 32];
    let mut square_out = vec![0.0; 256 * 256];
    let mut wide_out = vec![0.0; 256 * 512];
    session.read_output_by_index(0, &mut small_out);
    session.read_output_by_index(1, &mut square_out);
    session.read_output_by_index(2, &mut wide_out);
    assert!(small_out.iter().all(|&x| x == 32.0 / 8.0));
    assert!(square_out.iter().all(|&x| x == 256.0 / 8.0));
    assert!(wide_out.iter().all(|&x| x == 256.0 / 8.0));
}
