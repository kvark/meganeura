//! CPU-only replay of the fixed holdout cohort, not a new GPU measurement.
//! Full vectors were compared by the runner but only summaries are retained.

#[path = "../examples/support/tuning_measurement.rs"]
mod measurement;

#[path = "support/tuning_evidence.rs"]
mod evidence;

use evidence::*;
use measurement::{PairedTiming, TensorComparison, median};
use meganeura::TuneReport;
use serde_json::{Value, json};
use std::collections::HashSet;

const RECORDS: [&str; 5] = [
    include_str!("../docs/experiments/holdouts-2026-09-06/run-01.json"),
    include_str!("../docs/experiments/holdouts-2026-09-06/run-02.json"),
    include_str!("../docs/experiments/holdouts-2026-09-06/run-03.json"),
    include_str!("../docs/experiments/holdouts-2026-09-06/run-04.json"),
    include_str!("../docs/experiments/holdouts-2026-09-06/run-05.json"),
];

const CASES: [Expected; 6] = [
    Expected {
        name: "mlp-inference",
        work: "Inference",
        dispatches: 5,
        eligible: 2,
        excluded: 3,
        parameters: (4, 410496),
        gradients: (0, 0),
        output_elements: 32512,
    },
    Expected {
        name: "mlp-adam",
        work: "Adam",
        dispatches: 18,
        eligible: 5,
        excluded: 13,
        parameters: (4, 410496),
        gradients: (4, 410496),
        output_elements: 127,
    },
    Expected {
        name: "smollm2-inference",
        work: "Inference",
        dispatches: 83,
        eligible: 4,
        excluded: 58,
        parameters: (82, 1845376),
        gradients: (0, 0),
        output_elements: 8128,
    },
    Expected {
        name: "smollm2-adam",
        work: "Adam",
        dispatches: 284,
        eligible: 11,
        excluded: 194,
        parameters: (82, 1845376),
        gradients: (66, 1321088),
        output_elements: 127,
    },
    Expected {
        name: "whisper-sgd",
        work: "Sgd",
        dispatches: 227,
        eligible: 9,
        excluded: 187,
        parameters: (67, 7651584),
        gradients: (66, 7632384),
        output_elements: 1,
    },
    Expected {
        name: "resnet50-flb",
        work: "ForwardLossBackward",
        dispatches: 512,
        eligible: 1,
        excluded: 511,
        parameters: (108, 25530472),
        gradients: (108, 25530472),
        output_elements: 1,
    },
];

fn samples(value: &Value) -> Vec<f64> {
    serde_json::from_value(value.clone()).unwrap()
}

fn validate(record: &Value) -> Vec<PairedTiming> {
    assert_eq!(record["schema_version"], 1);
    assert_eq!(record["status"], "complete");
    let metadata = &record["metadata"];
    assert_eq!(
        metadata["revision"],
        "8bcd6b924e6dbb2d40b825171f89e8d19785d3d8"
    );
    assert_eq!(
        metadata["cargo_lock_sha256"],
        "4a84951f05631821a4dedb57f87195f82e59dad6eaf93bca015848b8a44eb874"
    );
    assert_eq!(
        metadata["executable_sha256"],
        "7ada9ad886d956fe6da80818d95d03bbe8b75833632c35237bc3fbccefa0db57"
    );
    assert_eq!(metadata["tracked_source_clean"], true);
    assert_eq!(metadata["device"]["f32_tile"], 0);
    assert_eq!(metadata["device"]["f16_tile"], 16);
    assert_eq!(metadata["device"]["name"], "NVIDIA GeForce RTX 5070");
    assert_eq!(metadata["device"]["driver_info"], "595.71.05");
    assert_eq!(metadata["cooperative_policy"], "Disabled");
    assert_eq!(metadata["prefix_steps"], 3);
    assert_eq!(metadata["warmups_per_session"], 30);
    assert_eq!(metadata["settling_steps"], 5);
    assert_eq!(metadata["sample_pairs"], 40);
    assert_eq!(metadata["skip_full_optimize"], false);
    assert_eq!(metadata["build_cache"], false);
    assert_eq!(metadata["rustc"], "rustc 1.98.0 (88d9e12ae 2026-08-18)");
    assert_eq!(metadata["os"], "linux");
    assert_eq!(metadata["arch"], "x86_64");
    assert_eq!(
        metadata["runtime_options"],
        "SessionOptions { debug: false, coop: Disabled, no_alias: false, no_device_local: false, serial_dispatch: false, dump_plan: false, pin_buffers: None }"
    );
    assert_eq!(
        metadata["compile_options"],
        json!({
            "flash_forward_coop": false, "flash_backward_coop": false,
            "fuse_dispatches": true, "use_schedule_pointwise": true, "use_schedule_reduction": true,
            "knobs": {"flash_ept_cap": 32, "flash_grad_kv_ept_cap": 32, "flash_grad_q_ept_cap": 32},
        })
    );
    assert_eq!(
        metadata["optimize_config"],
        json!({"mode": "Greedy", "extraction_cost": "TensorTraffic", "no_winograd": false, "saturation_cutoff": 300})
    );
    assert_eq!(
        metadata["optimizer"],
        json!({
            "adam_lr": 1e-4, "sgd_lr": 1e-3, "beta1": 0.9, "beta2": 0.999,
            "epsilon": 1e-8, "clip_norm": 1.0, "clip_every": 1,
            "accumulation": false, "weight_decay": 0.0,
        })
    );
    for flag in ["flash_forward_coop", "flash_backward_coop"] {
        assert_eq!(metadata[flag], false);
        assert_eq!(metadata["compile_options"][flag], false);
    }
    assert!(
        record["finished_unix_seconds"].as_u64().unwrap()
            > metadata["started_unix_seconds"].as_u64().unwrap()
    );
    let cases = record["cases"].as_array().unwrap();
    assert_eq!(cases.len(), CASES.len());
    cases
        .iter()
        .zip(&CASES)
        .map(|(case, expected)| {
            assert_eq!(case["name"], expected.name);
            assert_eq!(case["work"], expected.work);
            assert_eq!(case["status"], "complete");
            assert_eq!(case["dispatches"], expected.dispatches);
            let mut roster = None;
            for (key, stage, steps) in [
                ("prefix_comparison", "prefix", 3),
                ("search_state_comparison", "before_after_search", 3),
                ("warmup_comparison", "warmup", 33),
                ("final_comparison", "final", 78),
            ] {
                let current = check_comparison(&case[key], expected, stage, steps);
                if let Some(ref previous) = roster {
                    assert_eq!(previous, &current);
                }
                roster = Some(current);
            }
            let trajectory = case["loss_trajectory"].as_array().unwrap();
            assert_eq!(
                trajectory.len(),
                if expected.work == "Inference" { 0 } else { 40 }
            );
            for (index, loss) in trajectory.iter().enumerate() {
                assert_eq!(loss["step"], 39 + index);
                let a = loss["baseline"].as_f64().unwrap() as f32;
                let b = loss["tuned"].as_f64().unwrap() as f32;
                assert!(TensorComparison::new("loss".into(), &[a], &[b]).passed);
            }
            assert_eq!(case["loss_trajectory_passed"], true);
            for key in [
                "baseline_memory",
                "tuned_memory_before_search",
                "tuned_memory_after_search",
            ] {
                check_memory(&case[key], expected);
                for field in [
                    "plan_capacity_bytes",
                    "graph_allocation_bytes",
                    "adam_bytes",
                    "auxiliary_bytes",
                    "resident_buffer_requests",
                    "device_local_bytes",
                ] {
                    assert_eq!(case[key][field], case["baseline_memory"][field]);
                }
            }
            let baseline = samples(&case["baseline_ms"]);
            let tuned = samples(&case["tuned_ms"]);
            assert_eq!(baseline.len(), 40);
            let timing = PairedTiming::new(&baseline, &tuned).unwrap();
            check_timing(&timing, &case["timing"]);
            let before = case["baseline_pipeline_keys"].as_array().unwrap();
            let after = case["tuned_pipeline_keys"].as_array().unwrap();
            assert_eq!(before.len(), expected.dispatches);
            assert_eq!(after.len(), before.len());
            let changed = before.iter().zip(after).filter(|(a, b)| a != b).count();
            assert_eq!(case["dispatches_changed"], changed);

            let report: TuneReport = serde_json::from_value(case["search"].clone()).unwrap();
            assert_eq!(changed, check_search(&report, expected, false));
            check_pipeline_changes(before, after, &report);
            timing
        })
        .collect()
}

#[test]
fn retained_holdout_runs_replay() {
    let records: Vec<Value> = RECORDS
        .iter()
        .map(|text| serde_json::from_str(text).unwrap())
        .collect();
    let timings: Vec<_> = records.iter().map(validate).collect();
    let ids: HashSet<_> = records
        .iter()
        .map(|r| r["metadata"]["process_id"].as_u64().unwrap())
        .collect();
    assert_eq!(ids.len(), 5);
    let comparisons: Vec<_> = records
        .iter()
        .flat_map(|r| r["cases"].as_array().unwrap())
        .flat_map(|c| c["search"]["outcomes"].as_array().unwrap())
        .collect();
    assert_eq!(comparisons.len(), 140);
    assert_eq!(
        comparisons
            .iter()
            .filter(|c| c["decision"] == "FasterCandidate")
            .count(),
        31
    );
    for pair in records.windows(2) {
        assert!(
            pair[0]["finished_unix_seconds"].as_u64().unwrap()
                <= pair[1]["metadata"]["started_unix_seconds"]
                    .as_u64()
                    .unwrap()
        );
    }
    for (index, expected) in CASES.iter().enumerate() {
        let ratios: Vec<_> = timings.iter().map(|t| t[index].speedup).collect();
        let baseline: Vec<_> = timings
            .iter()
            .map(|t| t[index].baseline_median_ms)
            .collect();
        let tuned: Vec<_> = timings.iter().map(|t| t[index].tuned_median_ms).collect();
        let search: Vec<_> = records
            .iter()
            .map(|r| {
                serde_json::from_value::<TuneReport>(r["cases"][index]["search"].clone())
                    .unwrap()
                    .elapsed
                    .as_secs_f64()
                    * 1e3
            })
            .collect();
        println!(
            "{}: {:.6} -> {:.6} ms, median {:.6}x, range {:.6}..{:.6}, search {:.3} ms",
            expected.name,
            median(&baseline),
            median(&tuned),
            median(&ratios),
            ratios.iter().copied().reduce(f64::min).unwrap(),
            ratios.iter().copied().reduce(f64::max).unwrap(),
            median(&search)
        );
    }
    assert!(
        timings
            .iter()
            .flatten()
            .all(|t| !t.improvement_exceeds_guard && !t.regression_exceeds_guard)
    );
}

#[test]
fn replay_rejects_mutated_status_samples_state_and_decisions() {
    let original: Value = serde_json::from_str(RECORDS[0]).unwrap();
    for (path, replacement) in [
        ("/status", json!("running")),
        ("/metadata/settling_steps", json!(0)),
        ("/metadata/optimizer/clip_norm", json!(0.0)),
        ("/metadata/compile_options/fuse_dispatches", json!(false)),
        ("/cases/0/status", json!("final_parity_failed")),
        ("/cases/0/baseline_ms/0", json!(0)),
        ("/cases/0/timing/improvement_exceeds_guard", json!(true)),
        ("/cases/0/dispatches_changed", json!(0)),
        (
            "/cases/0/final_comparison/tensors/0/nonfinite_pairs",
            json!(1),
        ),
        (
            "/cases/0/final_comparison/tensors/0/relative_l2",
            json!(0.01),
        ),
        (
            "/cases/1/search_state_comparison/expected_adam_step",
            json!(4),
        ),
        ("/cases/1/search_state_comparison/tuned_adam_step", json!(4)),
        ("/cases/1/final_comparison/tensors/0/elements", json!(1)),
        ("/cases/1/loss_trajectory/0/baseline", Value::Null),
        ("/cases/1/loss_trajectory/0/tuned", json!(-1.0)),
        ("/cases/0/search/outcomes/0/selected", json!("Tile64")),
        ("/cases/3/search/class_limit_reached", json!(false)),
    ] {
        let mut changed = original.clone();
        *changed.pointer_mut(path).unwrap() = replacement;
        assert!(
            std::panic::catch_unwind(|| validate(&changed)).is_err(),
            "mutation accepted at {path}"
        );
    }
    let mut missing = original;
    missing["cases"].as_array_mut().unwrap().pop();
    assert!(std::panic::catch_unwind(|| validate(&missing)).is_err());
}

#[test]
fn unchanged_control_ratio_is_not_a_paired_gain() {
    let record: Value = serde_json::from_str(RECORDS[2]).unwrap();
    let timing = &validate(&record)[5];
    assert_eq!(record["cases"][5]["dispatches_changed"], 0);
    assert!(timing.speedup > 1.07);
    assert!(timing.paired_gain_median_ms < 0.02);
    assert!(!timing.improvement_exceeds_guard);
}
