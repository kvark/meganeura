//! Consistency checks for retained, bit-exact scalar evidence cohorts.

use super::measurement::{PairedTiming, TensorComparison};
use meganeura::{MatmulTile, TuneDecision, TuneReport};
use serde_json::Value;
use std::collections::{BTreeMap, HashSet};

pub struct Expected {
    pub name: &'static str,
    pub work: &'static str,
    pub dispatches: usize,
    pub eligible: usize,
    pub excluded: usize,
    pub parameters: (usize, usize),
    pub gradients: (usize, usize),
    pub output_elements: usize,
}

pub fn close(a: f64, b: f64) {
    assert!(a.is_finite() && b.is_finite());
    assert!(
        (a - b).abs() <= 1e-12 * a.abs().max(b.abs()).max(1e-30),
        "{a} != {b}"
    );
}

pub fn check_timing(expected: &PairedTiming, recorded: &Value) {
    let replayed = serde_json::to_value(expected).unwrap();
    assert_eq!(
        replayed.as_object().unwrap().len(),
        recorded.as_object().unwrap().len()
    );
    for (key, value) in replayed.as_object().unwrap() {
        if let Some(number) = value.as_f64() {
            close(number, recorded[key].as_f64().unwrap());
        } else {
            assert_eq!(value, &recorded[key]);
        }
    }
}

pub fn check_comparison(
    value: &Value,
    expected: &Expected,
    stage: &str,
    steps: u32,
) -> BTreeMap<String, usize> {
    assert_eq!(value["stage"], stage);
    let count = if expected.work == "Adam" { steps } else { 0 };
    for key in [
        "expected_adam_step",
        "baseline_adam_step",
        "tuned_adam_step",
    ] {
        assert_eq!(value[key], count);
    }
    assert_eq!(value["same_moment_allocation"], true);
    let tensors: Vec<TensorComparison> = serde_json::from_value(value["tensors"].clone()).unwrap();
    let mut roster = BTreeMap::new();
    for tensor in &tensors {
        assert!(tensor.elements > 0);
        assert!(
            roster
                .insert(tensor.name.clone(), tensor.elements)
                .is_none()
        );
        for number in [
            tensor.reference_sum_sq,
            tensor.candidate_sum_sq,
            tensor.error_sum_sq,
            tensor.max_abs_error,
        ] {
            assert!(number.is_finite() && number >= 0.0);
        }
        let relative = (tensor.error_sum_sq / tensor.reference_sum_sq.max(1e-30)).sqrt();
        close(relative, tensor.relative_l2);
        assert_eq!(
            tensor.passed,
            tensor.nonfinite_pairs == 0 && relative <= 2e-4 && tensor.elementwise_failures == 0
        );
        assert!(tensor.passed);
        if tensor.exact {
            assert_eq!(tensor.error_sum_sq, 0.0);
            assert_eq!(tensor.max_abs_error, 0.0);
            close(tensor.reference_sum_sq, tensor.candidate_sum_sq);
        }
    }
    assert_eq!(value["passed"], tensors.iter().all(|t| t.passed));
    assert_eq!(value["exact"], tensors.iter().all(|t| t.exact));
    // An observation of this cohort, not a requirement for every strict-f32 implementation.
    assert_eq!(value["exact"], true);
    let group = |prefix: &str| {
        roster
            .iter()
            .filter(|(name, _)| name.starts_with(prefix))
            .fold((0, 0), |(count, elements), (_, size)| {
                (count + 1, elements + size)
            })
    };
    assert_eq!(group("parameter."), expected.parameters);
    assert_eq!(group("gradient."), expected.gradients);
    let moments = if expected.work == "Adam" {
        expected.gradients
    } else {
        (0, 0)
    };
    assert_eq!(group("adam_m."), moments);
    assert_eq!(group("adam_v."), moments);
    let outputs = if expected.work == "Inference" {
        assert_eq!(roster.get("output"), Some(&expected.output_elements));
        1
    } else {
        assert_eq!(roster.get("loss"), Some(&1));
        assert_eq!(roster.get("loss_partials"), Some(&expected.output_elements));
        assert!(
            tensors
                .iter()
                .any(|t| t.name.starts_with("gradient.") && t.reference_sum_sq > 0.0)
        );
        2
    };
    assert_eq!(
        roster.len(),
        expected.parameters.0 + expected.gradients.0 + 2 * moments.0 + outputs
    );
    for (name, elements) in &roster {
        if let Some(parameter) = name.strip_prefix("gradient.") {
            assert_eq!(
                roster.get(&format!("parameter.{parameter}")),
                Some(elements)
            );
            if expected.work == "Adam" {
                for prefix in ["adam_m", "adam_v"] {
                    assert_eq!(roster.get(&format!("{prefix}.{parameter}")), Some(elements));
                }
            }
        }
    }
    roster
}

pub fn check_memory(memory: &Value, expected: &Expected) {
    let bytes = |name: &str| memory[name].as_u64().unwrap();
    assert_eq!(
        bytes("resident_buffer_requests"),
        bytes("graph_allocation_bytes")
            + bytes("adam_bytes")
            + bytes("accumulator_bytes")
            + bytes("auxiliary_bytes")
    );
    assert!(bytes("device_local_bytes") <= bytes("resident_buffer_requests"));
    assert!(bytes("graph_allocation_bytes") <= bytes("plan_capacity_bytes"));
    assert_eq!(
        bytes("adam_bytes"),
        if expected.work == "Adam" {
            8 * expected.gradients.1 as u64
        } else {
            0
        }
    );
    assert_eq!(bytes("accumulator_bytes"), 0);
}

pub fn check_search(report: &TuneReport, expected: &Expected, phases_expected: bool) -> usize {
    assert_eq!(report.options.max_classes, 8);
    assert_eq!(report.options.max_scratch_bytes, 64 * 1024 * 1024);
    assert_eq!(report.options.max_time.as_secs_f64(), 10.0);
    assert_eq!(report.options.sample_pairs, 6);
    assert_eq!(report.options.dispatches_per_sample, 16);
    assert_eq!(report.options.warmup_runs, 1);
    assert_eq!(report.options.min_improvement, 0.05);
    assert_eq!(report.eligible_classes, expected.eligible);
    assert_eq!(report.excluded_dispatches, expected.excluded);
    assert_eq!(report.visited_classes, expected.eligible.min(8));
    assert_eq!(report.outcomes.len(), report.visited_classes);
    assert_eq!(report.class_limit_reached, expected.eligible > 8);
    assert!(!report.time_budget_exhausted);
    assert!(!report.elapsed.is_zero());
    let mut changed_by_report = 0;
    let mut classes = HashSet::new();
    for outcome in &report.outcomes {
        assert!(classes.insert(outcome.class.clone()));
        assert!(outcome.qualified && outcome.failure.is_none());
        assert!(matches!(
            outcome.initial,
            MatmulTile::Tile32 | MatmulTile::Tile64
        ));
        assert!(matches!(
            outcome.candidate,
            MatmulTile::Tile32 | MatmulTile::Tile64
        ));
        assert_ne!(outcome.initial, outcome.candidate);
        assert_eq!(outcome.baseline_ms.len(), 6);
        assert_eq!(outcome.candidate_ms.len(), 6);
        let choice = PairedTiming::new(&outcome.baseline_ms, &outcome.candidate_ms).unwrap();
        close(
            choice.baseline_median_ms,
            outcome.baseline_median_ms.unwrap(),
        );
        close(choice.tuned_median_ms, outcome.candidate_median_ms.unwrap());
        close(
            choice.paired_noise_margin_ms,
            outcome.noise_margin_ms.unwrap(),
        );
        let win = choice.improvement_exceeds_guard;
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
        changed_by_report += if win { outcome.dispatches } else { 0 };
        match (&outcome.phase_times, phases_expected) {
            (None, false) => {}
            (Some(phases), true) => {
                let values = [
                    phases.preparation,
                    phases.qualification,
                    phases.warmup,
                    phases.sampling,
                ];
                let total: std::time::Duration = values.into_iter().map(Option::unwrap).sum();
                assert!(outcome.compile_time <= phases.preparation.unwrap());
                assert!(total <= outcome.elapsed);
            }
            _ => panic!("unexpected phase-timer availability"),
        }
        assert!(outcome.compile_time <= outcome.elapsed && outcome.elapsed <= report.elapsed);
    }

    changed_by_report
}

pub fn check_pipeline_changes(before: &[Value], after: &[Value], report: &TuneReport) {
    assert_eq!(before.len(), after.len());
    let mut observed = BTreeMap::new();
    for (a, b) in before.iter().zip(after).filter(|(a, b)| a != b) {
        *observed
            .entry((
                a.as_str().unwrap().to_owned(),
                b.as_str().unwrap().to_owned(),
            ))
            .or_insert(0) += 1;
    }
    let mut selected = BTreeMap::new();
    for outcome in report.outcomes.iter().filter(|o| o.initial != o.selected) {
        let key = |tile| {
            let variant = match tile {
                MatmulTile::Tile32 => "small-tile",
                MatmulTile::Tile64 => "scalar",
                _ => panic!("these evidence cohorts contain only scalar choices"),
            };
            format!("{:?}:{variant}", outcome.class.shader)
        };
        *selected
            .entry((key(outcome.initial), key(outcome.selected)))
            .or_insert(0) += outcome.dispatches;
    }
    assert_eq!(observed, selected);
}
