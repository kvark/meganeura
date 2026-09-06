//! CPU consistency replay, not a replacement for the unarchived full vectors.
#[path = "../examples/support/crossover_measurement.rs"]
mod crossover;
#[path = "support/tuning_evidence.rs"]
mod evidence;
#[path = "../examples/support/tuning_measurement.rs"]
mod measurement;

use crossover::{Block, Confirmation, Pair};
use evidence::*;
use measurement::{TensorComparison, median};
use meganeura::TuneReport;
use serde_json::{Value, json};
use std::{collections::HashSet, time::Duration};

const RECORDS: [&str; 6] = [
    include_str!("../docs/experiments/crossover-2026-09-06/run-01.json"),
    include_str!("../docs/experiments/crossover-2026-09-06/run-02.json"),
    include_str!("../docs/experiments/crossover-2026-09-06/run-03.json"),
    include_str!("../docs/experiments/crossover-2026-09-06/run-04.json"),
    include_str!("../docs/experiments/crossover-2026-09-06/run-05.json"),
    include_str!("../docs/experiments/crossover-2026-09-06/run-06.json"),
];

const CASES: [Expected; 3] = [
    Expected {
        name: "dense-inference",
        work: "Inference",
        dispatches: 8,
        eligible: 3,
        excluded: 0,
        parameters: (8, 2097152),
        gradients: (0, 0),
        output_elements: 65536,
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

fn check_pairs(pairs: &[Pair], count: usize, first_step: usize, order: usize, work: &str) {
    assert_eq!(pairs.len(), count);
    for (index, pair) in pairs.iter().enumerate() {
        assert_eq!(pair.step, first_step + index);
        assert_eq!(pair.first_session, (order + index) % 2);
        assert!(pair.left_ms.is_finite() && pair.left_ms > 0.0);
        assert!(pair.right_ms.is_finite() && pair.right_ms > 0.0);
        if work == "Inference" {
            assert!(pair.left_loss.is_none() && pair.right_loss.is_none());
        } else {
            assert!(
                TensorComparison::new(
                    "loss".into(),
                    &[pair.left_loss.unwrap()],
                    &[pair.right_loss.unwrap()]
                )
                .passed
            );
        }
    }
}

fn time_window(value: &Value, earliest: u64, latest: u64) -> u64 {
    let host = value["host_start"]["unix_ms"].as_u64().unwrap();
    let start = value["started_unix_ms"].as_u64().unwrap();
    let end = value["finished_unix_ms"].as_u64().unwrap();
    assert!(earliest <= host && host <= start && start < end && end <= latest);
    end
}

fn check_telemetry(record: &Value) {
    let telemetry = &record["telemetry"];
    assert!(telemetry["error"].is_null());
    assert_eq!(telemetry["requested_interval_ms"], 250);
    assert_eq!(telemetry["sample_cap"], 40000);
    assert_eq!(telemetry["cap_reached"], false);
    assert_eq!(
        telemetry["fields"],
        "timestamp,uuid,utilization.gpu,memory.used,clocks.gr,clocks.mem,power.draw,temperature.gpu,pstate"
    );
    let samples = telemetry["samples"].as_array().unwrap();
    assert!(!samples.is_empty());
    let mut previous = 0;
    for sample in samples {
        let time = sample["received_unix_ms"].as_u64().unwrap();
        assert!(time >= previous);
        previous = time;
        let fields: Vec<_> = sample["csv"]
            .as_str()
            .unwrap()
            .split(',')
            .map(str::trim)
            .collect();
        assert_eq!(fields.len(), 9);
        assert_eq!(fields[1], "GPU-705c613d-97a2-2380-4fd9-49006cebab54");
        for field in &fields[2..8] {
            let number = field.parse::<f64>().unwrap();
            assert!(number.is_finite() && number >= 0.0);
        }
        assert!(fields[2].parse::<f64>().unwrap() <= 100.0);
        assert!(fields[8].strip_prefix('P').unwrap().parse::<u32>().is_ok());
    }
    assert!(
        samples[0]["received_unix_ms"].as_u64().unwrap()
            <= record["metadata"]["started_unix_ms"].as_u64().unwrap()
    );
    assert!(previous <= record["finished_unix_ms"].as_u64().unwrap());
}

fn validate(record: &Value, seed: usize) -> Vec<Confirmation> {
    assert_eq!(record["schema_version"], 1);
    assert_eq!(record["protocol"], "crossover-2026-09-06");
    assert_eq!(record["status"], "complete");
    let metadata = &record["metadata"];
    assert_eq!(metadata["seed"], seed);
    assert_eq!(
        metadata["revision"],
        "c789bfa95dd2e645022e1c58b5bffab128f66fac"
    );
    assert_eq!(
        metadata["cargo_lock_sha256"],
        "4a84951f05631821a4dedb57f87195f82e59dad6eaf93bca015848b8a44eb874"
    );
    assert_eq!(
        metadata["executable_sha256"],
        "264f1fcaa8364ddbd23b19464f4396f9c21f0888b0b30a620fa026de3390d2b4"
    );
    assert_eq!(metadata["tracked_source_clean"], true);
    assert_eq!(
        metadata["device"],
        json!({"name": "NVIDIA GeForce RTX 5070", "driver": "595.71.05", "f32_tile": 0, "f16_tile": 16})
    );
    assert_eq!(metadata["rustc"], "rustc 1.98.0 (88d9e12ae 2026-08-18)");
    assert!(metadata["rustflags"].is_null());
    assert_eq!(metadata["cooperative_policy"], "Disabled");
    for (key, value) in [
        ("prefix", crossover::PREFIX),
        ("warmup", crossover::WARMUP),
        ("settling", crossover::SETTLING),
        ("control_pairs", crossover::CONTROL_PAIRS),
        ("block_pairs", crossover::BLOCK_PAIRS),
    ] {
        assert_eq!(metadata[key], value);
    }
    assert_eq!(
        metadata["compile_options"],
        json!({
            "flash_forward_coop": false, "flash_backward_coop": false,
            "fuse_dispatches": true, "use_schedule_pointwise": true, "use_schedule_reduction": true,
            "knobs": {"flash_ept_cap": 32, "flash_grad_kv_ept_cap": 32, "flash_grad_q_ept_cap": 32}
        })
    );
    assert_eq!(
        metadata["runtime_options"],
        "SessionOptions { debug: false, coop: Disabled, no_alias: false, no_device_local: false, serial_dispatch: false, dump_plan: false, pin_buffers: None }"
    );
    assert_eq!(
        metadata["optimize"],
        json!({"mode": "Greedy", "extraction_cost": "TensorTraffic", "no_winograd": false, "saturation_cutoff": 300})
    );
    assert_eq!(
        metadata["optimizer"],
        json!({"adam_lr": 1e-4, "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8, "clip_norm": 1.0, "clip_every": 1, "accumulation": false, "decay": 0.0})
    );
    check_telemetry(record);
    let end = record["finished_unix_ms"].as_u64().unwrap();
    let mut previous_case = metadata["started_unix_ms"].as_u64().unwrap();
    let cases = record["cases"].as_array().unwrap();
    assert_eq!(cases.len(), CASES.len());
    cases
        .iter()
        .zip(&CASES)
        .enumerate()
        .map(|(case_index, (case, expected))| {
            let order = seed + case_index;
            assert_eq!(case["seed"], order);
            assert_eq!(case["name"], expected.name);
            assert_eq!(case["work"], expected.work);
            assert_eq!(case["status"], "complete");
            assert_eq!(case["final_age"], 178);
            let start = case["host_start"]["unix_ms"].as_u64().unwrap();
            let finish = case["host_finish"]["unix_ms"].as_u64().unwrap();
            assert!(previous_case <= start && start < finish && finish <= end);
            previous_case = finish;
            let roster = check_comparison(&case["prefix_comparison"], expected, "prefix", 3);
            for (value, stage, age) in [
                (&case["warmup_comparison"], "warmup", 33),
                (&case["control"]["comparison"], "control", 78),
                (&case["search_state_comparison"], "search", 78),
            ] {
                assert_eq!(roster, check_comparison(value, expected, stage, age));
            }
            let control: Vec<Pair> =
                serde_json::from_value(case["control"]["pairs"].clone()).unwrap();
            check_pairs(&control, 40, 39, order, expected.work);
            let mut time = time_window(&case["control"], start, finish);
            let baseline = case["baseline_pipeline_keys"].as_array().unwrap();
            let winner = case["winner_pipeline_keys"].as_array().unwrap();
            assert_eq!(baseline.len(), expected.dispatches);
            assert_eq!(winner.len(), baseline.len());
            let changed = baseline.iter().zip(winner).filter(|(a, b)| a != b).count();
            assert_eq!(case["dispatches_changed"], changed);
            let report: TuneReport = serde_json::from_value(case["search"].clone()).unwrap();
            assert_eq!(changed, check_search(&report, expected, true));
            check_pipeline_changes(baseline, winner, &report);
            let records = case["blocks"].as_array().unwrap();
            assert_eq!(records.len(), crossover::BLOCKS);
            let blocks: Vec<Block> = records
                .iter()
                .enumerate()
                .map(|(index, block)| {
                    assert_eq!(block["index"], index);
                    time = time_window(block, time, finish);
                    let samples: Block = serde_json::from_value(block["samples"].clone()).unwrap();
                    assert_eq!(
                        samples.winner_session,
                        crossover::winner_order(order % 2)[index]
                    );
                    check_pairs(
                        &samples.pairs,
                        20,
                        84 + index * 25,
                        order + index,
                        expected.work,
                    );
                    assert_eq!(
                        roster,
                        check_comparison(
                            &block["comparison"],
                            expected,
                            "crossover",
                            103 + index as u32 * 25
                        )
                    );
                    if index == 1 || index == 3 {
                        assert!(block["swap"]["elapsed_ms"].as_f64().unwrap() > 0.0);
                        for (side, stage) in [("left", "swap_left"), ("right", "swap_right")] {
                            assert_eq!(
                                roster,
                                check_comparison(
                                    &block["swap"][side],
                                    expected,
                                    stage,
                                    78 + index as u32 * 25
                                )
                            );
                        }
                    } else {
                        assert!(block["swap"].is_null());
                    }
                    samples
                })
                .collect();
            for side in 0..2 {
                let initial = &case["initial_memory"][side];
                let final_memory = &case["final_memory"][side];
                for memory in [initial, final_memory] {
                    check_memory(memory, expected);
                    assert_eq!(
                        memory["graph_allocation_bytes"],
                        initial["graph_allocation_bytes"]
                    );
                    assert_eq!(
                        memory["resident_buffer_requests"],
                        initial["resident_buffer_requests"]
                    );
                }
            }
            let confirmation = Confirmation::new(&control, &blocks, changed, true).unwrap();
            let recorded = &case["confirmation"];
            assert_eq!(recorded["decision"], confirmation.decision);
            assert_eq!(recorded["control_stable"], confirmation.control_stable);
            for (key, value) in [
                ("control", &confirmation.control),
                ("left_winner", &confirmation.left_winner),
                ("right_winner", &confirmation.right_winner),
                ("pooled", &confirmation.pooled),
            ] {
                check_timing(value, &recorded[key]);
            }
            confirmation
        })
        .collect()
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e3
}

#[test]
fn retained_crossover_runs_replay() {
    let records: Vec<Value> = RECORDS
        .iter()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect();
    let replayed: Vec<_> = records
        .iter()
        .enumerate()
        .map(|(i, r)| validate(r, i + 1))
        .collect();
    let ids: HashSet<_> = records
        .iter()
        .map(|r| r["metadata"]["process_id"].as_u64().unwrap())
        .collect();
    assert_eq!(ids.len(), 6);
    for pair in records.windows(2) {
        assert!(
            pair[0]["finished_unix_ms"].as_u64().unwrap()
                <= pair[1]["metadata"]["started_unix_ms"].as_u64().unwrap()
        );
    }
    let mut comparisons = 0;
    let mut winners = 0;
    for (index, expected) in CASES.iter().enumerate() {
        let ratios: Vec<_> = replayed.iter().map(|r| r[index].pooled.speedup).collect();
        let baseline: Vec<_> = replayed
            .iter()
            .map(|r| r[index].pooled.baseline_median_ms)
            .collect();
        let tuned: Vec<_> = replayed
            .iter()
            .map(|r| r[index].pooled.tuned_median_ms)
            .collect();
        let mut phase_totals = [
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ];
        let mut amortization = Vec::new();
        for (process, record) in records.iter().enumerate() {
            let report: TuneReport =
                serde_json::from_value(record["cases"][index]["search"].clone()).unwrap();
            comparisons += report.outcomes.len();
            let mut totals = [0.0; 6];
            totals[0] = milliseconds(report.elapsed);
            for outcome in report.outcomes {
                winners += usize::from(outcome.initial != outcome.selected);
                let phases = outcome.phase_times.unwrap();
                for (i, phase) in [
                    phases.preparation,
                    phases.qualification,
                    phases.warmup,
                    phases.sampling,
                    Some(outcome.compile_time),
                ]
                .into_iter()
                .enumerate()
                {
                    totals[i + 1] += milliseconds(phase.unwrap());
                }
            }
            for (values, total) in phase_totals.iter_mut().zip(totals) {
                values.push(total);
            }
            if replayed[process][index].decision == "confirmed_gain" {
                amortization.push(totals[0] / (baseline[process] - tuned[process]));
            }
        }
        println!(
            "{}: {:.6} -> {:.6} ms, {:.6}x [{:.6}, {:.6}]; decisions {:?}",
            expected.name,
            median(&baseline),
            median(&tuned),
            median(&ratios),
            ratios.iter().copied().reduce(f64::min).unwrap(),
            ratios.iter().copied().reduce(f64::max).unwrap(),
            replayed
                .iter()
                .map(|r| &r[index].decision)
                .collect::<Vec<_>>()
        );
        println!(
            "  ms: search/preparation/qualification/warmup/sampling/compile {:?}",
            phase_totals.map(|v| median(&v))
        );
        if !amortization.is_empty() {
            println!(
                "  search-only break-even steps: median {:.1}, range {:.1}..{:.1}",
                median(&amortization),
                amortization.iter().copied().reduce(f64::min).unwrap(),
                amortization.iter().copied().reduce(f64::max).unwrap()
            );
        }
    }
    assert_eq!(comparisons, 54);
    assert_eq!(winners, 30);
    for (process, row) in replayed.iter().enumerate() {
        assert_eq!(row[0].decision, "confirmed_gain");
        assert_eq!(
            row[1].decision,
            if process < 4 {
                "inconclusive"
            } else {
                "unstable_control"
            }
        );
        assert_eq!(row[2].decision, "unchanged_selection");
    }
}

#[test]
fn replay_rejects_missing_data_bad_roles_counters_timings_and_phases() {
    let original: Value = serde_json::from_str(RECORDS[0]).unwrap();
    for (path, replacement) in [
        ("/metadata/revision", json!("newer source")),
        ("/metadata/seed", json!(2)),
        ("/metadata/optimizer/clip_norm", json!(0.0)),
        ("/metadata/compile_options/fuse_dispatches", json!(false)),
        ("/telemetry/samples", json!([])),
        ("/cases/0/status", json!("running")),
        ("/cases/0/control/pairs/0/step", json!(40)),
        ("/cases/0/blocks/0/samples/pairs/0/first_session", json!(0)),
        ("/cases/0/blocks/0/samples/winner_session", json!(0)),
        ("/cases/0/blocks/0/samples/pairs/0/left_ms", json!(0.0)),
        ("/cases/0/blocks/1/swap/left/exact", json!(false)),
        ("/cases/0/blocks/1/swap", Value::Null),
        ("/cases/0/confirmation/control_stable", json!(false)),
        ("/cases/0/confirmation/decision", json!("inconclusive")),
        ("/cases/0/dispatches_changed", json!(0)),
        (
            "/cases/0/winner_pipeline_keys/0",
            json!("MatMul:unsupported"),
        ),
        (
            "/cases/0/search/outcomes/0/phase_times/qualification/secs",
            json!(100),
        ),
        (
            "/cases/0/search/outcomes/0/phase_times/sampling",
            Value::Null,
        ),
        ("/cases/0/search/outcomes/0/selected", json!("Tile64")),
        (
            "/cases/1/search_state_comparison/expected_adam_step",
            json!(3),
        ),
        ("/cases/1/blocks/3/comparison/tuned_adam_step", json!(177)),
        (
            "/cases/1/blocks/3/comparison/tensors/0/nonfinite_pairs",
            json!(1),
        ),
        ("/cases/1/blocks/3/comparison/tensors/0/elements", json!(1)),
        ("/cases/1/blocks/3/samples/pairs/0/left_loss", Value::Null),
        ("/cases/1/blocks/3/samples/pairs/0/right_loss", json!(-1.0)),
    ] {
        let mut changed = original.clone();
        *changed.pointer_mut(path).unwrap() = replacement;
        assert!(
            std::panic::catch_unwind(|| validate(&changed, 1)).is_err(),
            "accepted mutation at {path}"
        );
    }
    for path in [
        "/cases",
        "/cases/0/control/pairs",
        "/cases/1/blocks",
        "/cases/2/blocks/0/samples/pairs",
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
}
