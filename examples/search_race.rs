//! Compare kernel-search strategies on identical, freshly built sessions.
//!
//! `cargo run --release --example search_race -- [budget_seconds] [strategy...]`
//!
//! Strategies are `sequential` (one challenger at a time, structural class
//! order), `race` (shared rounds with a measured machine profile) and
//! `race-uncalibrated`; the default runs `sequential race`. The graph has
//! the projections of one SmolLM2-135M block at 128 tokens plus one decode
//! row. Each strategy gets a new session, so its decisions start from the
//! compiled plan. Set `MEGANEURA_GPU_TIMING=1` to time races with GPU
//! timestamps. Retained programs are then compared in interleaved, balanced
//! order. Set `MEGANEURA_SEARCH_VERBOSE=1` for raw candidate timings, or
//! `MEGANEURA_SEARCH_SHARED_CONTEXT=1` to share the context (and calibration).
//! A two-strategy shared-context run also swaps the selected kernels and
//! repeats the held-out comparison. This compares search policies, not engines.

use meganeura::{
    Graph, Session, SessionConfig, TuneCalibration, TuneDecision, TuneOptions, TuneReport,
    TuneStrategy,
};
use std::time::{Duration, Instant};

const TOKENS: usize = 128;
const HIDDEN: usize = 576;
const INTERMEDIATE: usize = 1536;
const STEP_WARMUPS: usize = 10;
const STEP_SAMPLES: usize = 30;

fn graph() -> Graph {
    let mut g = Graph::new();
    let x = g.input("x", &[TOKENS, HIDDEN]);
    let mut outputs = Vec::new();
    for name in ["q", "k", "v", "o"] {
        let weight = g.parameter(name, &[HIDDEN, HIDDEN]);
        outputs.push(g.matmul(x, weight));
    }
    let up = g.parameter("up", &[HIDDEN, INTERMEDIATE]);
    let gate = g.parameter("gate", &[HIDDEN, INTERMEDIATE]);
    let down = g.parameter("down", &[INTERMEDIATE, HIDDEN]);
    let up_x = g.matmul(x, up);
    let gate_x = g.matmul(x, gate);
    let gated = g.silu(gate_x);
    let hidden = g.mul(gated, up_x);
    outputs.push(g.matmul(hidden, down));
    let token = g.input("token", &[1, HIDDEN]);
    outputs.push(g.matmul(token, up));
    g.set_outputs(outputs);
    g
}

fn values(len: usize, seed: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((i * seed % 251) as f32 / 251.0 - 0.5) * 0.1)
        .collect()
}

fn session(graph: &Graph, gpu: Option<std::sync::Arc<blade_graphics::Context>>) -> Session {
    let mut config = match gpu {
        Some(gpu) => SessionConfig::inference_from_env_on(gpu),
        None => SessionConfig::inference_from_env(),
    };
    config.tune = false;
    let (mut session, _) = meganeura::build(graph, config);
    session.set_input("x", &values(TOKENS * HIDDEN, 7));
    session.set_input("token", &values(HIDDEN, 11));
    for (index, (name, len)) in [
        ("q", HIDDEN * HIDDEN),
        ("k", HIDDEN * HIDDEN),
        ("v", HIDDEN * HIDDEN),
        ("o", HIDDEN * HIDDEN),
        ("up", HIDDEN * INTERMEDIATE),
        ("gate", HIDDEN * INTERMEDIATE),
        ("down", INTERMEDIATE * HIDDEN),
    ]
    .into_iter()
    .enumerate()
    {
        session.set_parameter(name, &values(len, 13 + 2 * index));
    }
    session
}

/// Median wall time of `step` plus `wait`, after warming up.
fn step_ms(session: &mut Session) -> f64 {
    for _ in 0..STEP_WARMUPS {
        session.step();
        session.wait();
    }
    let mut samples: Vec<f64> = (0..STEP_SAMPLES)
        .map(|_| {
            let start = Instant::now();
            session.step();
            session.wait();
            start.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

/// Revisit retained programs in balanced order, independently of the search
/// that last warmed the device. Immediate post-search timings are diagnostic,
/// not a paired comparison of the selected programs.
fn held_out(strategies: &[String], sessions: &mut [Session]) {
    let mut samples = vec![Vec::new(); sessions.len()];
    for round in 0..STEP_WARMUPS + STEP_SAMPLES {
        let mut order: Vec<_> = (0..sessions.len()).collect();
        let shift = (round / 2) % order.len();
        order.rotate_left(shift);
        if round % 2 == 1 {
            order.reverse();
        }
        for index in order {
            let start = Instant::now();
            sessions[index].step();
            sessions[index].wait();
            if round >= STEP_WARMUPS {
                samples[index].push(start.elapsed().as_secs_f64() * 1e3);
            }
        }
    }
    println!("== held-out interleaved steps");
    for (name, times) in strategies.iter().zip(&mut samples) {
        times.sort_by(f64::total_cmp);
        println!("{name}: {:.4} ms", times[times.len() / 2]);
    }
}

fn options(strategy: &str, budget: Duration) -> TuneOptions {
    let (strategy, calibration) = match strategy {
        "sequential" => (TuneStrategy::Sequential, TuneCalibration::Off),
        "race" => (TuneStrategy::Race, TuneCalibration::Measure),
        "race-uncalibrated" => (TuneStrategy::Race, TuneCalibration::Off),
        other => panic!("unknown strategy {other}; use sequential, race or race-uncalibrated"),
    };
    TuneOptions {
        strategy,
        calibration,
        max_time: budget,
        ..Default::default()
    }
}

fn summarize(name: &str, report: &TuneReport, before_ms: f64, after_ms: f64, wall: Duration) {
    let count = |decision: TuneDecision| {
        report
            .outcomes
            .iter()
            .filter(|outcome| outcome.decision == decision)
            .count()
    };
    let qualified = report.outcomes.iter().filter(|o| o.qualified).count();
    println!("== {name}");
    println!(
        "search {:.3}s (call {:.3}s), calibration {:.3}s",
        report.elapsed.as_secs_f64(),
        wall.as_secs_f64(),
        report.calibration_time.as_secs_f64()
    );
    println!(
        "classes visited {}/{}, comparisons {}, qualified {}, faster {}, eliminated {}, \
         rejected {}, time budget {}, class limit {}, share cutoff {}",
        report.visited_classes,
        report.eligible_classes,
        report.outcomes.len(),
        qualified,
        count(TuneDecision::FasterCandidate),
        count(TuneDecision::Eliminated),
        count(TuneDecision::ShaderRejected) + count(TuneDecision::InvalidOutput),
        count(TuneDecision::TimeBudget),
        report.class_limit_reached,
        report.share_cutoff,
    );
    if std::env::var_os("MEGANEURA_SEARCH_VERBOSE").is_some() {
        for outcome in &report.outcomes {
            println!(
                "  {:?} {}x{}x{}: {:?} vs {:?}: {:?}, reps {:?}, medians {:?} / {:?}, noise {:?}; samples {:?} / {:?}",
                outcome.class.shader,
                outcome.class.m,
                outcome.class.n,
                outcome.class.k,
                outcome.initial,
                outcome.candidate,
                outcome.decision,
                outcome.sample_repetitions,
                outcome.baseline_median_ms,
                outcome.candidate_median_ms,
                outcome.noise_margin_ms,
                outcome.baseline_ms,
                outcome.candidate_ms,
            );
        }
    }
    for race in &report.races {
        println!(
            "  race {:?} {}x{}x{} x{}: {} candidates, {} rounds, {:?} -> {:?}, incumbent {:.4} ms \
             (predicted {}), compile {:.3}s, race {:.3}s, qualify {:.3}s, timestamps {}",
            race.class.shader,
            race.class.m,
            race.class.n,
            race.class.k,
            race.dispatches,
            race.candidates,
            race.rounds,
            race.initial,
            race.selected,
            race.incumbent_ms.unwrap_or(f64::NAN),
            race.predicted_ms
                .map_or_else(|| "none".to_owned(), |ms| format!("{ms:.4} ms")),
            race.compile_time.as_secs_f64(),
            race.racing.as_secs_f64(),
            race.qualification.as_secs_f64(),
            race.gpu_timestamps,
        );
    }
    if let Some(ms) = report.predicted_plan_ms {
        println!("predicted plan {ms:.4} ms");
    }
    println!(
        "step {before_ms:.4} -> {after_ms:.4} ms ({:+.1}%)",
        (after_ms / before_ms - 1.0) * 100.0
    );
}

fn main() {
    let mut args = std::env::args().skip(1);
    let budget = Duration::from_secs_f64(
        args.next()
            .map_or(10.0, |seconds| seconds.parse().expect("budget in seconds")),
    );
    let mut strategies: Vec<String> = args.collect();
    if strategies.is_empty() {
        strategies = vec!["sequential".to_owned(), "race".to_owned()];
    }
    let graph = graph();
    let mut sessions = Vec::new();
    let shared = std::env::var_os("MEGANEURA_SEARCH_SHARED_CONTEXT").is_some();
    let mut gpu = None;
    for name in &strategies {
        let options = options(name, budget);
        let mut session = session(&graph, gpu.clone());
        if shared {
            gpu = Some(session.context());
        }
        let before_ms = step_ms(&mut session);
        let start = Instant::now();
        let report = session.tune_with(options).expect("valid tuning options");
        let wall = start.elapsed();
        let after_ms = step_ms(&mut session);
        summarize(name, &report, before_ms, after_ms, wall);
        sessions.push(session);
    }
    held_out(&strategies, &mut sessions);
    if shared && sessions.len() == 2 {
        let (left, right) = sessions.split_at_mut(1);
        println!(
            "swapped {} dispatch choices",
            left[0].swap_tuning_with(&mut right[0]).unwrap()
        );
        held_out(
            &[strategies[1].clone(), strategies[0].clone()],
            &mut sessions,
        );
    }
}
