//! CPU replay of the fixed Shared/Download cohort; full vectors are not archived.
#[path = "support/tuning_evidence.rs"]
mod evidence;
#[path = "../examples/support/tuning_measurement.rs"]
mod measurement;
#[path = "../examples/support/readback_measurement.rs"]
mod readback;
#[path = "support/telemetry.rs"]
mod telemetry;

use evidence::{DIAGNOSTIC_CASES as CASES, *};
use measurement::{PairedTiming, TensorComparison, median};
use meganeura::{TuneReport, TuneStaging};
use readback::{FINAL, MIDDLE, PREFIX, PROCESSES, costs, search_order};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, HashSet},
    time::Duration,
};

const RECORDS: [&str; PROCESSES] = [
    include_str!("../docs/experiments/readback-2026-09-06/run-01.json"),
    include_str!("../docs/experiments/readback-2026-09-06/run-02.json"),
    include_str!("../docs/experiments/readback-2026-09-06/run-03.json"),
    include_str!("../docs/experiments/readback-2026-09-06/run-04.json"),
    include_str!("../docs/experiments/readback-2026-09-06/run-05.json"),
    include_str!("../docs/experiments/readback-2026-09-06/run-06.json"),
];
const SUMMARY: &str = include_str!("../docs/experiments/readback-2026-09-06/summary.json");
const REVISION: &str = "2abeff93b6ae5d9714a698cdc64d942ca965d2ff";
type Costs = BTreeMap<String, f64>;

fn equal_memory(a: &Value, b: &Value) {
    for key in [
        "plan_capacity_bytes",
        "graph_allocation_bytes",
        "adam_bytes",
        "accumulator_bytes",
        "auxiliary_bytes",
        "resident_buffer_requests",
        "device_local_bytes",
    ] {
        assert_eq!(a[key], b[key]);
    }
}

fn validate(record: &Value, seed: usize) -> Vec<[Costs; 2]> {
    assert_eq!(record["schema_version"], 1);
    assert_eq!(record["protocol"], "readback-2026-09-06");
    assert_eq!(record["status"], "complete");
    let metadata = &record["metadata"];
    assert_eq!(metadata["seed"], seed);
    assert_eq!(metadata["revision"], REVISION);
    assert_eq!(
        metadata["cargo_lock_sha256"],
        "72cecddd0f090e82d5db46bc6f7f80429a058b2afa0eaec84a9b27e7340a4a80"
    );
    assert_eq!(
        metadata["executable_sha256"],
        "306e52acccefb9ad8d346aae7370f213d953c8e91c581aa201c53395e7644ce1"
    );
    assert_eq!(metadata["tracked_source_clean"], true);
    assert_eq!(
        metadata["device"],
        json!({"name": "NVIDIA GeForce RTX 5070", "driver": "595.71.05", "f32_tile": 0, "f16_tile": 16})
    );
    assert_eq!(metadata["rustc"], "rustc 1.98.0 (88d9e12ae 2026-08-18)");
    assert!(metadata["rustflags"].is_null());
    assert_eq!(metadata["cooperative_policy"], "Disabled");
    assert_eq!(metadata["prefix"], PREFIX);
    assert_eq!(metadata["middle"], MIDDLE);
    assert_eq!(metadata["final"], FINAL);
    assert_eq!(
        metadata["runtime_options"],
        "SessionOptions { debug: false, coop: Disabled, no_alias: false, no_device_local: false, serial_dispatch: false, dump_plan: false, pin_buffers: None }"
    );
    assert_eq!(
        metadata["compile_options"],
        json!({
            "flash_forward_coop": false, "flash_backward_coop": false, "fuse_dispatches": true,
            "use_schedule_pointwise": true, "use_schedule_reduction": true,
            "knobs": {"flash_ept_cap": 32, "flash_grad_kv_ept_cap": 32, "flash_grad_q_ept_cap": 32}
        })
    );
    assert_eq!(
        metadata["optimize"],
        json!({"mode": "Greedy", "extraction_cost": "TensorTraffic", "no_winograd": false, "saturation_cutoff": 300})
    );
    assert_eq!(
        metadata["optimizer"],
        json!({"adam_lr": 1e-4, "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8, "clip_norm": 1.0, "clip_every": 1, "accumulation": false, "decay": 0.0})
    );
    telemetry::check_telemetry(record);
    let mut previous_case = metadata["started_unix_ms"].as_u64().unwrap();
    let end = record["finished_unix_ms"].as_u64().unwrap();
    let cases = record["cases"].as_array().unwrap();
    assert_eq!(cases.len(), CASES.len());
    cases
        .iter()
        .zip(&CASES)
        .enumerate()
        .map(|(case_index, (case, expected))| {
            assert_eq!(case["name"], expected.name);
            assert_eq!(case["work"], expected.work);
            assert_eq!(case["status"], "complete");
            assert_eq!(case["final_age"], FINAL);
            let start = case["host_start"]["unix_ms"].as_u64().unwrap();
            let finish = case["host_finish"]["unix_ms"].as_u64().unwrap();
            assert!(previous_case <= start && start < finish && finish <= end);
            previous_case = finish;
            let roster = check_comparison(
                &case["prefix_comparison"],
                expected,
                "prefix",
                PREFIX as u32,
            );
            for (key, stage, age) in [
                ("middle_comparison", "middle", MIDDLE),
                ("final_comparison", "final", FINAL),
            ] {
                assert_eq!(
                    roster,
                    check_comparison(&case[key], expected, stage, age as u32)
                );
            }
            let loss = case["loss_trajectory"].as_array().unwrap();
            assert_eq!(
                loss.len(),
                if expected.work == "Inference" {
                    0
                } else {
                    FINAL - PREFIX
                }
            );
            for (index, pair) in loss.iter().enumerate() {
                assert_eq!(pair["step"], PREFIX + 1 + index);
                let a = pair["shared"].as_f64().unwrap() as f32;
                let b = pair["download"].as_f64().unwrap() as f32;
                assert!(TensorComparison::new("loss".into(), &[a], &[b]).passed);
            }
            let order = search_order(seed, case_index);
            assert_eq!(case["search_order"], json!(order));
            let searches = case["searches"].as_array().unwrap();
            assert_eq!(searches.len(), 2);
            let mut previous_search = start;
            for index in order {
                previous_search = telemetry::time_window(&searches[index], previous_search, finish);
            }
            let initial = case["initial_pipeline_keys"].as_array().unwrap();
            assert_eq!(initial.len(), expected.dispatches);
            let mut classes = None;
            std::array::from_fn(|side| {
                let search = &searches[side];
                let report: TuneReport = serde_json::from_value(search["report"].clone()).unwrap();
                let staging = [TuneStaging::Shared, TuneStaging::Download][side];
                assert_eq!(report.options.staging, staging);
                assert_eq!(search["staging"], serde_json::to_value(staging).unwrap());
                let selected = search["selected_pipeline_keys"].as_array().unwrap();
                assert_eq!(selected.len(), initial.len());
                let changed = initial.iter().zip(selected).filter(|(a, b)| a != b).count();
                assert_eq!(changed, check_search(&report, expected, true));
                check_pipeline_changes(initial, selected, &report);
                let coverage: Vec<_> = report
                    .outcomes
                    .iter()
                    .map(|o| (o.class.clone(), o.dispatches, o.initial, o.candidate))
                    .collect();
                if let Some(ref previous) = classes {
                    assert_eq!(&coverage, previous);
                }
                classes = Some(coverage);
                for outcome in &report.outcomes {
                    let phases = outcome.phase_times.unwrap();
                    let details = phases.qualification_breakdown.unwrap();
                    let pieces = [
                        details.input_preparation,
                        details.upload_host_copy,
                        details.upload_transfer,
                        details.dispatch,
                        details.readback_transfer,
                        details.readback_host_copy,
                        details.validation,
                    ];
                    assert!(pieces.iter().all(|p| p.is_some_and(|p| !p.is_zero())));
                    let total: Duration = pieces.into_iter().map(Option::unwrap).sum();
                    assert!(total <= phases.qualification.unwrap());
                }
                assert_eq!(
                    roster,
                    check_comparison(
                        &search["state_comparison"],
                        expected,
                        "search",
                        PREFIX as u32
                    )
                );
                for memory in [
                    &search["memory_before"],
                    &search["memory_after"],
                    &case["final_memory"][side],
                ] {
                    check_memory(memory, expected);
                    equal_memory(memory, &searches[0]["memory_before"]);
                }
                let times = costs(&report).unwrap();
                let recorded: Costs = serde_json::from_value(search["cost_ms"].clone()).unwrap();
                assert_eq!(
                    times.keys().collect::<Vec<_>>(),
                    recorded.keys().collect::<Vec<_>>()
                );
                for (key, value) in &times {
                    close(*value, recorded[key]);
                }
                times
            })
        })
        .collect()
}

fn check_cohort(records: &[Value], summary: &Value) {
    assert_eq!(records.len(), PROCESSES);
    assert_eq!(summary["schema_version"], 1);
    assert_eq!(summary["measured_revision"], REVISION);
    assert_eq!(summary["cases"].as_array().unwrap().len(), CASES.len());
    let replayed: Vec<_> = records
        .iter()
        .enumerate()
        .map(|(i, record)| validate(record, i + 1))
        .collect();
    let ids: HashSet<_> = records
        .iter()
        .map(|r| r["metadata"]["process_id"].as_u64().unwrap())
        .collect();
    assert_eq!(ids.len(), PROCESSES);
    for pair in records.windows(2) {
        assert!(
            pair[0]["finished_unix_ms"].as_u64().unwrap()
                <= pair[1]["metadata"]["started_unix_ms"].as_u64().unwrap()
        );
    }
    let mut accepted = true;
    for (index, case) in CASES.iter().enumerate() {
        let recorded = &summary["cases"][index];
        assert_eq!(recorded["name"], case.name);
        assert_eq!(
            recorded["costs"].as_object().unwrap().len(),
            replayed[0][index][0].len()
        );
        for key in replayed[0][index][0].keys() {
            let shared: Vec<_> = replayed.iter().map(|r| r[index][0][key]).collect();
            let download: Vec<_> = replayed.iter().map(|r| r[index][1][key]).collect();
            let timing = PairedTiming::new(&shared, &download).unwrap();
            check_timing(&timing, &recorded["costs"][key]);
            if index == 2
                && matches!(
                    key.as_str(),
                    "elapsed" | "qualification" | "readback_host_copy"
                )
            {
                accepted &= timing.improvement_exceeds_guard;
            } else if index < 2 && key == "elapsed" {
                accepted &= !timing.regression_exceeds_guard;
            }
            println!(
                "{} {key}: {:.6} -> {:.6} ms; guard gain={} regression={}",
                case.name,
                timing.baseline_median_ms,
                timing.tuned_median_ms,
                timing.improvement_exceeds_guard,
                timing.regression_exceeds_guard
            );
        }
        let ratios: Vec<_> = replayed
            .iter()
            .map(|r| r[index][0]["elapsed"] / r[index][1]["elapsed"])
            .collect();
        let stored: Vec<f64> =
            serde_json::from_value(recorded["process_search_ratios"].clone()).unwrap();
        assert_eq!(stored.len(), PROCESSES);
        for (a, b) in ratios.iter().zip(stored) {
            close(*a, b);
        }
        close(
            median(&ratios),
            recorded["median_process_search_ratio"].as_f64().unwrap(),
        );
        for first in [0, 1] {
            let subgroup: Vec<_> = ratios
                .iter()
                .enumerate()
                .filter(|(i, _)| search_order(i + 1, index)[0] == first)
                .map(|(_, r)| *r)
                .collect();
            assert_eq!(subgroup.len(), 3);
            println!(
                "{} first={first}: median ratio {:.6}, range {:.6}..{:.6}",
                case.name,
                median(&subgroup),
                subgroup.iter().copied().reduce(f64::min).unwrap(),
                subgroup.iter().copied().reduce(f64::max).unwrap()
            );
        }
    }
    assert_eq!(summary["default_download_accepted"], accepted);
    assert!(accepted);
    let comparisons: Vec<_> = records
        .iter()
        .flat_map(|r| r["cases"].as_array().unwrap())
        .flat_map(|c| c["searches"].as_array().unwrap())
        .flat_map(|s| s["report"]["outcomes"].as_array().unwrap())
        .collect();
    assert_eq!(comparisons.len(), 108);
}

#[test]
fn retained_readback_runs_and_promotion_gates_replay() {
    let records: Vec<_> = RECORDS
        .iter()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect();
    check_cohort(&records, &serde_json::from_str(SUMMARY).unwrap());
}

#[test]
fn replay_rejects_mutated_policy_coverage_costs_state_and_summary() {
    let original: Value = serde_json::from_str(RECORDS[0]).unwrap();
    for (path, value) in [
        ("/metadata/revision", json!("newer source")),
        ("/metadata/seed", json!(2)),
        ("/metadata/optimizer/clip_norm", json!(0.0)),
        ("/metadata/final", json!(78)),
        ("/telemetry/samples", json!([])),
        ("/cases/0/search_order/0", json!(0)),
        ("/cases/0/searches/0/staging", json!("Download")),
        (
            "/cases/0/searches/0/report/options/staging",
            json!("Download"),
        ),
        ("/cases/0/searches/0/report/options/sample_pairs", json!(4)),
        (
            "/cases/0/searches/0/report/outcomes/0/qualified",
            json!(false),
        ),
        (
            "/cases/0/searches/1/report/outcomes/0/class/device_local/0",
            json!(false),
        ),
        (
            "/cases/0/searches/0/report/outcomes/0/phase_times/qualification_breakdown",
            Value::Null,
        ),
        (
            "/cases/0/searches/0/report/outcomes/0/phase_times/qualification_breakdown/readback_host_copy/secs",
            json!(100),
        ),
        ("/cases/0/searches/0/cost_ms/readback_host_copy", json!(0.0)),
        (
            "/cases/0/searches/0/selected_pipeline_keys/0",
            json!("MatMul:unsupported"),
        ),
        (
            "/cases/1/searches/0/state_comparison/expected_adam_step",
            json!(0),
        ),
        ("/cases/1/final_comparison/tuned_adam_step", json!(177)),
        (
            "/cases/1/final_comparison/tensors/0/nonfinite_pairs",
            json!(1),
        ),
        ("/cases/1/final_comparison/tensors/0/elements", json!(1)),
        ("/cases/1/loss_trajectory/0/download", Value::Null),
        ("/cases/1/loss_trajectory/0/shared", json!(-1.0)),
        ("/cases/1/searches/1/memory_after/adam_bytes", json!(0)),
    ] {
        let mut changed = original.clone();
        *changed.pointer_mut(path).unwrap() = value;
        assert!(
            std::panic::catch_unwind(|| validate(&changed, 1)).is_err(),
            "accepted mutation at {path}"
        );
    }
    for path in [
        "/cases",
        "/cases/0/searches",
        "/cases/1/loss_trajectory",
        "/cases/2/searches/0/report/outcomes",
    ] {
        let mut changed = original.clone();
        changed
            .pointer_mut(path)
            .unwrap()
            .as_array_mut()
            .unwrap()
            .pop();
        assert!(
            std::panic::catch_unwind(|| validate(&changed, 1)).is_err(),
            "accepted missing row at {path}"
        );
    }
    let records: Vec<_> = RECORDS
        .iter()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect();
    let mut summary: Value = serde_json::from_str(SUMMARY).unwrap();
    summary["default_download_accepted"] = json!(false);
    assert!(std::panic::catch_unwind(|| check_cohort(&records, &summary)).is_err());
    summary = serde_json::from_str(SUMMARY).unwrap();
    summary["cases"][2]["costs"]["elapsed"]["improvement_exceeds_guard"] = json!(false);
    assert!(std::panic::catch_unwind(|| check_cohort(&records, &summary)).is_err());
}
