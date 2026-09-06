//! CPU replay of allocation localization and the separately retained reuse cohort.
#[path = "support/staging_evidence.rs"]
mod diagnostic;
#[path = "support/tuning_evidence.rs"]
mod evidence;
#[path = "../examples/support/tuning_measurement.rs"]
mod measurement;
#[path = "../examples/support/preparation_measurement.rs"]
mod preparation;
#[path = "../examples/support/readback_measurement.rs"]
mod readback;
#[path = "support/telemetry.rs"]
mod telemetry;

#[path = "support/archives.rs"]
mod archives;

use evidence::{DIAGNOSTIC_CASES as CASES, check_timing, close};
use measurement::{PairedTiming, median};
use meganeura::{TuneReport, TuneScratchStats, TuneScratchUsage, TuneStaging, TuneStagingReuse};
use serde_json::{Value, json};
use std::{collections::HashSet, time::Duration};

const PROFILE: &[u8] =
    include_bytes!("../docs/experiments/staging-reuse-2026-09-06/profile-01.json.gz");
const PROFILE_PROTOCOL: diagnostic::Protocol = diagnostic::Protocol {
    name: "readback-2026-09-06",
    revision: "42fda56d568293e82b3bc92defe57025bf78c800",
    executable: "e1f4a4c36c314195173a3448ca3fb416616f594056006a2c26b49a98bdfd82cd",
    labels: ["shared", "download"],
    staging: [TuneStaging::Shared, TuneStaging::Download],
    reuse: [TuneStagingReuse::Fresh; 2],
    costs: readback::costs,
};

const RECORDS: [&[u8]; readback::PROCESSES] = [
    include_bytes!("../docs/experiments/staging-reuse-2026-09-06/run-01.json.gz"),
    include_bytes!("../docs/experiments/staging-reuse-2026-09-06/run-02.json.gz"),
    include_bytes!("../docs/experiments/staging-reuse-2026-09-06/run-03.json.gz"),
    include_bytes!("../docs/experiments/staging-reuse-2026-09-06/run-04.json.gz"),
    include_bytes!("../docs/experiments/staging-reuse-2026-09-06/run-05.json.gz"),
    include_bytes!("../docs/experiments/staging-reuse-2026-09-06/run-06.json.gz"),
];
const SUMMARY: &str = include_str!("../docs/experiments/staging-reuse-2026-09-06/summary.json");
const PROTOCOL: diagnostic::Protocol = diagnostic::Protocol {
    name: "staging-reuse-2026-09-06",
    revision: "875c501af4d9b9b5e63274758fa5cb610a629c3e",
    executable: "13d937c3fdade2488889c26ee1280f6188db35e76d629a6b05b309c5e34fcb78",
    labels: ["fresh", "reuse"],
    staging: [TuneStaging::Download; 2],
    reuse: [TuneStagingReuse::Fresh, TuneStagingReuse::SameSize],
    costs: preparation::costs,
};

fn check_scratch(report: &TuneReport) -> Vec<TuneScratchUsage> {
    let mut expected = TuneScratchStats::default();
    let mut previous = 0;
    let reuse = report.options.staging_reuse == TuneStagingReuse::SameSize;
    let usages = report
        .outcomes
        .iter()
        .map(|outcome| {
            let usage = outcome.scratch.clone().unwrap();
            let (m, n, k) = (
                outcome.class.m as usize,
                outcome.class.n as usize,
                outcome.class.k as usize,
            );
            let mut bindings = vec![m * k * 4, k * n * 4];
            if outcome.class.shader == meganeura::compile::ShaderEntry::FusedMatMulAdd {
                bindings.push(m * n * 4);
            }
            bindings.push(m * n * 4);
            assert_eq!(usage.binding_bytes, bindings);
            let staging = *bindings.iter().max().unwrap();
            assert_eq!(usage.staging_bytes, staging);
            assert_eq!(usage.staging_reused, reuse && previous == staging);
            expected.staging_allocations += usize::from(!usage.staging_reused);
            expected.staging_reuses += usize::from(usage.staging_reused);
            previous = staging;
            let total = staging + bindings.iter().sum::<usize>();
            assert!(total <= report.options.max_scratch_bytes);
            expected.peak_bytes = expected.peak_bytes.max(total);
            usage
        })
        .collect();
    expected.staging_releases = expected.staging_allocations;
    assert_eq!(report.scratch.unwrap(), expected);
    assert_eq!(
        report.final_cleanup.is_some(),
        reuse && expected.staging_allocations > 0
    );
    if let Some(time) = report.final_cleanup {
        assert!(!time.is_zero());
    }
    usages
}

fn validate(record: &Value, seed: usize) -> Vec<[diagnostic::Costs; 2]> {
    let costs = diagnostic::validate(record, seed, &PROTOCOL);
    assert_eq!(record["metadata"]["arms"], json!(PROTOCOL.labels));
    assert_eq!(
        record["metadata"]["search_options"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    for (index, case) in record["cases"].as_array().unwrap().iter().enumerate() {
        let mut previous: Option<Vec<TuneScratchUsage>> = None;
        let mut peak = None;
        for (side, search) in case["searches"].as_array().unwrap().iter().enumerate() {
            assert_eq!(search["arm"], PROTOCOL.labels[side]);
            assert_eq!(
                search["staging_reuse"],
                serde_json::to_value(PROTOCOL.reuse[side]).unwrap()
            );
            assert_eq!(
                record["metadata"]["search_options"][side],
                search["report"]["options"]
            );
            let report: TuneReport = serde_json::from_value(search["report"].clone()).unwrap();
            check_preparation(&report);
            let usage = check_scratch(&report);
            let stats = report.scratch.unwrap();
            assert_eq!(
                stats.staging_allocations,
                [[3, 1], [5, 2], [1, 1]][index][side]
            );
            if let Some(previous) = &previous {
                for (a, b) in previous.iter().zip(&usage) {
                    assert_eq!(a.binding_bytes, b.binding_bytes);
                    assert_eq!(a.staging_bytes, b.staging_bytes);
                }
                assert_eq!(previous.len(), usage.len());
                assert_eq!(peak.unwrap(), stats.peak_bytes);
            }
            previous = Some(usage);
            peak = Some(stats.peak_bytes);
        }
    }
    costs
}

fn check_cohort(records: &[Value], summary: &Value) {
    assert_eq!(records.len(), readback::PROCESSES);
    assert_eq!(summary["schema_version"], 1);
    assert_eq!(summary["measured_revision"], PROTOCOL.revision);
    assert_eq!(summary["cases"].as_array().unwrap().len(), CASES.len());
    let costs: Vec<_> = records
        .iter()
        .enumerate()
        .map(|(i, r)| validate(r, i + 1))
        .collect();
    let ids: HashSet<_> = records
        .iter()
        .map(|r| r["metadata"]["process_id"].as_u64().unwrap())
        .collect();
    assert_eq!(ids.len(), readback::PROCESSES);
    let profile: Value = archives::json(PROFILE);
    let mut previous = profile["finished_unix_ms"].as_u64().unwrap();
    for record in records {
        assert!(previous <= record["metadata"]["started_unix_ms"].as_u64().unwrap());
        previous = record["finished_unix_ms"].as_u64().unwrap();
    }
    let mut accepted = true;
    for (index, case) in CASES.iter().enumerate() {
        let stored = &summary["cases"][index];
        assert_eq!(stored["name"], case.name);
        assert_eq!(
            stored["costs"].as_object().unwrap().len(),
            costs[0][index][0].len()
        );
        for key in costs[0][index][0].keys() {
            let fresh: Vec<_> = costs.iter().map(|c| c[index][0][key]).collect();
            let reuse: Vec<_> = costs.iter().map(|c| c[index][1][key]).collect();
            let timing = PairedTiming::new(&fresh, &reuse).unwrap();
            check_timing(&timing, &stored["costs"][key]);
            if index < 2 && matches!(key.as_str(), "elapsed" | "staging_and_cleanup") {
                accepted &= timing.improvement_exceeds_guard;
            } else if index == 2 && key == "elapsed" {
                accepted &= !timing.regression_exceeds_guard;
            }
            println!(
                "{} {key}: {:.6} -> {:.6} ms, gain={}, regression={}",
                case.name,
                timing.baseline_median_ms,
                timing.tuned_median_ms,
                timing.improvement_exceeds_guard,
                timing.regression_exceeds_guard
            );
        }
        let ratios: Vec<_> = costs
            .iter()
            .map(|c| c[index][0]["elapsed"] / c[index][1]["elapsed"])
            .collect();
        let stored_ratios: Vec<f64> =
            serde_json::from_value(stored["process_search_ratios"].clone()).unwrap();
        assert_eq!(ratios.len(), stored_ratios.len());
        for (a, b) in ratios.iter().zip(stored_ratios) {
            close(*a, b);
        }
        close(
            median(&ratios),
            stored["median_process_search_ratio"].as_f64().unwrap(),
        );
        for first in [0, 1] {
            let subgroup: Vec<_> = ratios
                .iter()
                .enumerate()
                .filter(|(i, _)| readback::search_order(i + 1, index)[0] == first)
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
    assert_eq!(summary["default_same_size_accepted"], accepted);
    assert!(accepted);
    let count: usize = records
        .iter()
        .flat_map(|r| r["cases"].as_array().unwrap())
        .flat_map(|c| c["searches"].as_array().unwrap())
        .map(|s| s["report"]["outcomes"].as_array().unwrap().len())
        .sum();
    assert_eq!(count, 108);
}

#[test]
fn retained_reuse_cohort_and_promotion_gates_replay() {
    let records: Vec<_> = RECORDS.iter().map(|r| archives::json(r)).collect();
    check_cohort(&records, &serde_json::from_str(SUMMARY).unwrap());
}

#[test]
fn replay_rejects_changed_reuse_lifetime_bytes_timings_state_and_summary() {
    let original: Value = archives::json(RECORDS[0]);
    for (path, value) in [
        ("/metadata/revision", json!("other source")),
        ("/metadata/arms/0", json!("reuse")),
        (
            "/metadata/search_options/0/staging_reuse",
            json!("SameSize"),
        ),
        ("/cases/0/searches/1/staging_reuse", json!("Fresh")),
        (
            "/cases/0/searches/1/report/options/staging_reuse",
            json!("Fresh"),
        ),
        (
            "/cases/0/searches/1/report/scratch/staging_allocations",
            json!(2),
        ),
        (
            "/cases/0/searches/1/report/scratch/staging_releases",
            json!(0),
        ),
        (
            "/cases/0/searches/1/report/scratch/staging_reuses",
            json!(1),
        ),
        (
            "/cases/0/searches/1/report/scratch/retained_staging_bytes",
            json!(1048576),
        ),
        (
            "/cases/0/searches/1/report/scratch/peak_bytes",
            json!(2621441),
        ),
        (
            "/cases/0/searches/1/report/outcomes/1/scratch/staging_reused",
            json!(false),
        ),
        (
            "/cases/0/searches/1/report/outcomes/0/scratch/staging_bytes",
            json!(2097152),
        ),
        (
            "/cases/0/searches/1/report/outcomes/0/scratch/binding_bytes/0",
            json!(4),
        ),
        ("/cases/0/searches/1/report/final_cleanup", Value::Null),
        (
            "/cases/0/searches/0/report/final_cleanup",
            json!({"secs":0,"nanos":1}),
        ),
        (
            "/cases/0/searches/1/report/outcomes/0/phase_times/cleanup/secs",
            json!(10),
        ),
        (
            "/cases/0/searches/1/report/outcomes/0/phase_times/preparation_breakdown/staging/secs",
            json!(10),
        ),
        ("/cases/0/searches/1/cost_ms/cleanup", json!(0.0)),
        ("/cases/1/final_comparison/tuned_adam_step", json!(177)),
        ("/cases/1/loss_trajectory/0/reuse", Value::Null),
        ("/cases/1/searches/1/memory_after/adam_bytes", json!(0)),
    ] {
        let mut changed = original.clone();
        *changed.pointer_mut(path).unwrap() = value;
        assert!(
            std::panic::catch_unwind(|| validate(&changed, 1)).is_err(),
            "accepted mutation at {path}"
        );
    }
    let records: Vec<_> = RECORDS.iter().map(|r| archives::json(r)).collect();
    let summary: Value = serde_json::from_str(SUMMARY).unwrap();
    assert!(std::panic::catch_unwind(|| check_cohort(&records[..5], &summary)).is_err());
    let mut changed = summary.clone();
    changed["default_same_size_accepted"] = json!(false);
    assert!(std::panic::catch_unwind(|| check_cohort(&records, &changed)).is_err());
    changed = summary;
    changed["cases"][0]["costs"]["elapsed"]["improvement_exceeds_guard"] = json!(false);
    assert!(std::panic::catch_unwind(|| check_cohort(&records, &changed)).is_err());
}

fn check_preparation(report: &TuneReport) {
    for outcome in &report.outcomes {
        let phases = outcome.phase_times.unwrap();
        let prep = phases.preparation_breakdown.unwrap();
        let parts = [
            prep.checks,
            prep.pipelines,
            prep.buffers,
            prep.staging,
            prep.encoder,
            prep.bindings,
        ];
        assert!(parts.iter().all(|p| p.is_some_and(|p| !p.is_zero())));
        let total: Duration = parts.into_iter().map(Option::unwrap).sum();
        assert_eq!(outcome.compile_time, prep.pipelines.unwrap());
        assert!(total <= phases.preparation.unwrap());
        assert!(!phases.cleanup.unwrap().is_zero());
        let total: Duration = [
            phases.preparation,
            phases.qualification,
            phases.warmup,
            phases.sampling,
            phases.cleanup,
        ]
        .into_iter()
        .map(Option::unwrap)
        .sum();
        assert!(total <= outcome.elapsed);
    }
    let total: Duration = report.outcomes.iter().map(|o| o.elapsed).sum();
    assert!(total + report.final_cleanup.unwrap_or_default() <= report.elapsed);
}

#[test]
fn retained_allocation_profile_replays_without_relabeling_the_readback_cohort() {
    let record: Value = archives::json(PROFILE);
    diagnostic::validate(&record, 1, &PROFILE_PROTOCOL);
    for case in record["cases"].as_array().unwrap() {
        for search in case["searches"].as_array().unwrap() {
            let report: TuneReport = serde_json::from_value(search["report"].clone()).unwrap();
            check_preparation(&report);
            assert_eq!(report.scratch, None);
            assert_eq!(report.final_cleanup, None);
            let costs = preparation::costs(&report).unwrap();
            println!(
                "{} {}: preparation {:.6}, staging {:.6}, buffers {:.6}, encoder {:.6}, cleanup {:.6} ms",
                case["name"],
                search["staging"],
                costs["preparation"],
                costs["prep_staging"],
                costs["prep_buffers"],
                costs["prep_encoder"],
                costs["cleanup"]
            );
        }
    }
}
