//! Shared validity checks for the matched private-staging diagnostics.
use super::{
    evidence::{DIAGNOSTIC_CASES as CASES, *},
    measurement::TensorComparison,
    readback::{FINAL, MIDDLE, PREFIX, search_order},
    telemetry,
};
use meganeura::{TuneReport, TuneStaging, TuneStagingReuse};
use serde_json::{Value, json};
use std::{collections::BTreeMap, time::Duration};

pub struct Protocol {
    pub name: &'static str,
    pub revision: &'static str,
    pub executable: &'static str,
    pub labels: [&'static str; 2],
    pub staging: [TuneStaging; 2],
    pub reuse: [TuneStagingReuse; 2],
    pub costs: fn(&TuneReport) -> Result<Costs, &'static str>,
}

pub type Costs = BTreeMap<String, f64>;

pub fn equal_memory(a: &Value, b: &Value) {
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

pub fn validate(record: &Value, seed: usize, protocol: &Protocol) -> Vec<[Costs; 2]> {
    assert_eq!(record["schema_version"], 1);
    assert_eq!(record["protocol"], protocol.name);
    assert_eq!(record["status"], "complete");
    let metadata = &record["metadata"];
    assert_eq!(metadata["seed"], seed);
    assert_eq!(metadata["revision"], protocol.revision);
    assert_eq!(
        metadata["cargo_lock_sha256"],
        "72cecddd0f090e82d5db46bc6f7f80429a058b2afa0eaec84a9b27e7340a4a80"
    );
    assert_eq!(metadata["executable_sha256"], protocol.executable);
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
    telemetry::check_telemetry(record, false);
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
                let a = pair[protocol.labels[0]].as_f64().unwrap() as f32;
                let b = pair[protocol.labels[1]].as_f64().unwrap() as f32;
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
                let staging = protocol.staging[side];
                assert_eq!(report.options.staging_reuse, protocol.reuse[side]);
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
                let times = (protocol.costs)(&report).unwrap();
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
