//! CPU-only replay of the retained tuning experiment. This checks record
//! consistency and decision arithmetic, not GPU correctness or performance.

#[path = "support/archives.rs"]
mod archives;

use meganeura::{MatmulTile, TuneDecision, TuneReport};
use serde_json::Value;

fn median(values: &[f64]) -> f64 {
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[mid - 1] + values[mid]) * 0.5
    } else {
        values[mid]
    }
}

fn samples(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect()
}

fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-12, "{a} != {b}");
}

fn decision(a: &[f64], b: &[f64], margin: f64) -> (bool, f64, f64) {
    assert_eq!(a.len(), b.len());
    assert!(!a.is_empty());
    assert!(a.iter().chain(b).all(|v| v.is_finite() && *v > 0.0));
    let differences: Vec<_> = a.iter().zip(b).map(|(a, b)| a - b).collect();
    let gain = median(&differences);
    let noise = 2.0
        * median(
            &differences
                .iter()
                .map(|d| (d - gain).abs())
                .collect::<Vec<_>>(),
        );
    let required = median(a) * margin + noise;
    (
        median(a) - median(b) > required && gain > required,
        gain,
        noise,
    )
}

#[test]
fn retained_scalar_tuning_runs_replay() {
    let records: [&[u8]; 5] = [
        include_bytes!("../docs/experiments/tuning-2026-09-05/run-01.json.gz"),
        include_bytes!("../docs/experiments/tuning-2026-09-05/run-02.json.gz"),
        include_bytes!("../docs/experiments/tuning-2026-09-05/run-03.json.gz"),
        include_bytes!("../docs/experiments/tuning-2026-09-05/run-04.json.gz"),
        include_bytes!("../docs/experiments/tuning-2026-09-05/run-05.json.gz"),
    ];
    let mut comparisons = 0;
    let mut accepted_kernels = 0;
    let mut accepted_steps = 0;
    let mut speedups = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for record in records {
        let data: Value = archives::json(record);
        let metadata = &data["metadata"];
        assert_eq!(
            metadata["revision"],
            "0e27b68e92bfb72745eb2f582107236cd541c409"
        );
        assert_eq!(metadata["tracked_source_clean"], true);
        assert_eq!(metadata["device"]["f32_tile"], 0);
        assert_eq!(metadata["cooperative_policy"], "Disabled");
        let cases = data["cases"].as_array().unwrap();
        assert_eq!(cases.len(), 4);
        for (index, case) in cases.iter().enumerate() {
            assert_eq!(case["live_output_unchanged_by_search"], true);
            assert_eq!(case["native_f32_qualified"], false);
            assert_eq!(case["final_parity"]["passed"], true);
            assert_eq!(case["final_parity"]["relative_l2"], 0.0);
            let baseline = samples(&case["baseline_ms"]);
            let tuned = samples(&case["tuned_ms"]);
            assert_eq!(baseline.len(), 40);
            let (win, gain, noise) = decision(&baseline, &tuned, 0.05);
            close(
                median(&baseline),
                case["baseline_median_ms"].as_f64().unwrap(),
            );
            close(median(&tuned), case["tuned_median_ms"].as_f64().unwrap());
            close(gain, case["paired_difference_median_ms"].as_f64().unwrap());
            close(noise, case["paired_noise_margin_ms"].as_f64().unwrap());
            let speedup = median(&baseline) / median(&tuned);
            close(speedup, case["speedup"].as_f64().unwrap());
            speedups[index].push(speedup);
            assert_eq!(
                win,
                case["whole_step_improvement_exceeds_guard"]
                    .as_bool()
                    .unwrap()
            );
            accepted_steps += usize::from(win);
            let before = case["baseline_pipeline_keys"].as_array().unwrap();
            let after = case["tuned_pipeline_keys"].as_array().unwrap();
            assert_eq!(before.len(), after.len());
            let changed = before.iter().zip(after).filter(|(a, b)| a != b).count();
            assert_eq!(changed as u64, case["dispatches_changed"].as_u64().unwrap());

            let report: TuneReport = serde_json::from_value(case["search"].clone()).unwrap();
            assert!(!report.class_limit_reached && !report.time_budget_exhausted);
            assert_eq!(report.eligible_classes, 3);
            assert_eq!(report.visited_classes, 3);
            assert_eq!(report.outcomes.len(), 3);
            assert_eq!(report.excluded_dispatches, 0);
            for outcome in report.outcomes {
                comparisons += 1;
                assert!(outcome.qualified && outcome.failure.is_none());
                assert_eq!(outcome.baseline_ms.len(), report.options.sample_pairs);
                assert_eq!(outcome.candidate_ms.len(), report.options.sample_pairs);
                assert!(matches!(
                    outcome.initial,
                    MatmulTile::Tile32 | MatmulTile::Tile64
                ));
                assert_ne!(outcome.initial, outcome.candidate);
                let (win, _, noise) = decision(
                    &outcome.baseline_ms,
                    &outcome.candidate_ms,
                    report.options.min_improvement,
                );
                close(
                    median(&outcome.baseline_ms),
                    outcome.baseline_median_ms.unwrap(),
                );
                close(
                    median(&outcome.candidate_ms),
                    outcome.candidate_median_ms.unwrap(),
                );
                close(noise, outcome.noise_margin_ms.unwrap());
                assert_eq!(
                    outcome.decision,
                    if win {
                        TuneDecision::FasterCandidate
                    } else {
                        TuneDecision::KeepBaseline
                    }
                );
                assert_eq!(
                    outcome.selected,
                    if win {
                        outcome.candidate
                    } else {
                        outcome.initial
                    }
                );
                accepted_kernels += usize::from(win);
            }
        }
    }
    assert_eq!(
        (comparisons, accepted_kernels, accepted_steps),
        (60, 30, 10)
    );
    for (index, values) in speedups.iter().enumerate() {
        println!(
            "case {index}: median process speedup {:.6}x",
            median(values)
        );
    }
}
