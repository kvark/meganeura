//! CPU replay of the fixed Shared/Download cohort; full vectors are not archived.
#[path = "support/staging_evidence.rs"]
mod diagnostic;
#[path = "support/tuning_evidence.rs"]
mod evidence;
#[path = "../examples/support/tuning_measurement.rs"]
mod measurement;
#[path = "../examples/support/readback_measurement.rs"]
mod readback;
#[path = "support/telemetry.rs"]
mod telemetry;

use evidence::{DIAGNOSTIC_CASES as CASES, *};
use measurement::{PairedTiming, median};
use meganeura::TuneStaging;
use readback::{PROCESSES, costs, search_order};
use serde_json::{Value, json};
use std::collections::HashSet;

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
const PROTOCOL: diagnostic::Protocol = diagnostic::Protocol {
    name: "readback-2026-09-06",
    revision: REVISION,
    executable: "306e52acccefb9ad8d346aae7370f213d953c8e91c581aa201c53395e7644ce1",
    labels: ["shared", "download"],
    staging: [TuneStaging::Shared, TuneStaging::Download],
    reuse: [meganeura::TuneStagingReuse::Fresh; 2],
    costs,
};

fn validate(record: &Value, seed: usize) -> Vec<[diagnostic::Costs; 2]> {
    diagnostic::validate(record, seed, &PROTOCOL)
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
