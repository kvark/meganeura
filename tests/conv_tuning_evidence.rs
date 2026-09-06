//! Replay both cohorts, retaining and explicitly disqualifying the vacuous controls.
#[path = "../examples/support/crossover_measurement.rs"]
mod crossover;
#[path = "support/tuning_evidence.rs"]
#[allow(dead_code)]
mod evidence;
#[path = "../examples/support/tuning_measurement.rs"]
mod measurement;
#[path = "support/telemetry.rs"]
mod telemetry;

use crossover::{Block, Confirmation, Pair};
use evidence::{Expected, check_comparison, check_memory, check_search, check_timing, close};
use measurement::{TensorComparison, median};
use meganeura::{
    MatmulTile, TuneReport, TuneScope, TuneStaging, TuneStagingReuse,
    compile::{Dispatch, ShaderEntry},
};
use serde_json::{Value, json};
use std::{collections::HashSet, time::Duration};

const CASES: [Expected; 3] = [
    Expected {
        name: "resnet50-flb",
        work: "ForwardLossBackward",
        dispatches: 512,
        eligible: 45,
        excluded: 407,
        parameters: (108, 25530472),
        gradients: (108, 25530472),
        output_elements: 1,
    },
    Expected {
        name: "conv-edges-adam",
        work: "Adam",
        dispatches: 11,
        eligible: 3,
        excluded: 8,
        parameters: (2, 3876),
        gradients: (2, 3876),
        output_elements: 1,
    },
    Expected {
        name: "conv-edges-sgd",
        work: "Sgd",
        dispatches: 11,
        eligible: 3,
        excluded: 8,
        parameters: (2, 3876),
        gradients: (2, 3876),
        output_elements: 1,
    },
];

fn records(corrected: bool) -> Vec<Value> {
    let directory = if corrected {
        "conv-tiles-corrected-2026-09-06"
    } else {
        "conv-tiles-2026-09-06"
    };
    (1..=6)
        .map(|seed| {
            let path = format!(
                "{}/docs/experiments/{directory}/run-{seed:02}.json",
                env!("CARGO_MANIFEST_DIR")
            );
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
        })
        .collect()
}

fn metadata(record: &Value, seed: usize, corrected: bool) {
    assert_eq!(record["schema_version"], 1);
    assert_eq!(record["status"], "complete");
    let (protocol, revision, executable) = if corrected {
        (
            "conv-tiles-corrected-2026-09-06",
            "e2def382b4971cd800d93bf408f0dd575025a3a7",
            "c747cba9644622477f33326bbf79956f833011c331f049ed3c3dc122b478378d",
        )
    } else {
        (
            "conv-tiles-2026-09-06",
            "fc151ac30626991e9d9d4611df0433dbe9e59a43",
            "f6361f97aa1b5839975eaea39200d25fcf1b4c9bacf77ed0b2e6615522fe66ef",
        )
    };
    assert_eq!(record["protocol"], protocol);
    let m = &record["metadata"];
    assert_eq!(m["revision"], revision);
    assert_eq!(m["executable_sha256"], executable);
    assert_eq!(
        m["cargo_lock_sha256"],
        "72cecddd0f090e82d5db46bc6f7f80429a058b2afa0eaec84a9b27e7340a4a80"
    );
    assert_eq!(m["tracked_source_clean"], true);
    assert_eq!(m["scope"], "ConvDerivatives");
    assert_eq!(m["seed"], seed);
    assert_eq!(
        m["device"],
        json!({"name":"NVIDIA GeForce RTX 5070", "driver":"595.71.05", "f32_tile":0, "f16_tile":16})
    );
    assert_eq!(m["cooperative_policy"], "Disabled");
    assert_eq!(m["rustc"], "rustc 1.98.0 (88d9e12ae 2026-08-18)");
    assert!(m["rustflags"].is_null());
    assert_eq!(
        m["compile_options"],
        json!({"flash_forward_coop":false, "flash_backward_coop":false,
        "fuse_dispatches":true, "use_schedule_pointwise":true, "use_schedule_reduction":true,
        "knobs":{"flash_ept_cap":32,"flash_grad_kv_ept_cap":32,"flash_grad_q_ept_cap":32}})
    );
    assert_eq!(
        m["runtime_options"],
        "SessionOptions { debug: false, coop: Disabled, no_alias: false, no_device_local: false, serial_dispatch: false, dump_plan: false, pin_buffers: None }"
    );
    assert_eq!(
        m["optimize"],
        json!({"mode":"Greedy","extraction_cost":"TensorTraffic","no_winograd":false,"saturation_cutoff":300})
    );
    assert_eq!(
        m["optimizer"],
        json!({"adam_lr":1e-4,"sgd_lr":1e-3,"beta1":0.9,"beta2":0.999,
        "epsilon":1e-8,"clip_norm":1.0,"clip_every":1,"accumulation":false,"decay":0.0})
    );
    for (key, value) in [
        ("prefix", crossover::PREFIX),
        ("warmup", crossover::WARMUP),
        ("settling", crossover::SETTLING),
        ("control_pairs", crossover::CONTROL_PAIRS),
        ("block_pairs", crossover::BLOCK_PAIRS),
    ] {
        assert_eq!(m[key], value);
    }
    telemetry::check_telemetry(record, false);
    assert_eq!(record["cases"].as_array().unwrap().len(), 3);
}

fn pairs(pairs: &[Pair], count: usize, first: usize, order: usize) {
    assert_eq!(pairs.len(), count);
    for (index, pair) in pairs.iter().enumerate() {
        assert_eq!(pair.step, first + index);
        assert_eq!(pair.first_session, (order + index) % 2);
        assert!(pair.left_ms.is_finite() && pair.left_ms > 0.0);
        assert!(pair.right_ms.is_finite() && pair.right_ms > 0.0);
        let a = pair.left_loss.unwrap();
        let b = pair.right_loss.unwrap();
        assert!(a != 0.0 && b != 0.0);
        let comparison = TensorComparison::new("loss".into(), &[a], &[b]);
        assert!(comparison.passed && comparison.exact);
    }
}

fn search_contracts(case: &Value, expected: &Expected) -> TuneReport {
    let report: TuneReport = serde_json::from_value(case["search"].clone()).unwrap();
    assert_eq!(report.options.scope, TuneScope::ConvDerivatives);
    assert_eq!(report.options.staging, TuneStaging::Download);
    assert_eq!(report.options.staging_reuse, TuneStagingReuse::SameSize);
    let changed = check_search(&report, expected, true);
    assert_eq!(case["dispatches_changed"], changed);
    let before: Vec<Dispatch> =
        serde_json::from_value(case["baseline_dispatches"].clone()).unwrap();
    let after: Vec<Dispatch> = serde_json::from_value(case["selected_dispatches"].clone()).unwrap();
    let buffers: Vec<usize> = serde_json::from_value(case["plan_buffers"].clone()).unwrap();
    assert_eq!(before.len(), expected.dispatches);
    assert_eq!(after.len(), before.len());
    let before_keys = case["baseline_pipeline_keys"].as_array().unwrap();
    let after_keys = case["winner_pipeline_keys"].as_array().unwrap();
    assert_eq!(before_keys.len(), before.len());
    assert_eq!(after_keys.len(), before.len());
    let mut reconstructed = before.clone();
    let mut visited = HashSet::new();
    let (mut previous_stage, mut allocations, mut reuses, mut peak) = (0, 0, 0, 0);
    let mut total = report.final_cleanup.unwrap();
    for outcome in &report.outcomes {
        let c = &outcome.class;
        let s = c.conv2d.unwrap();
        let dx = c.shader == ShaderEntry::Conv2dGradInputGemm;
        assert!(dx || c.shader == ShaderEntry::Conv2dGradWeightGemm);
        assert!(c.requires_full_precision);
        let p = vec![
            s.batch,
            s.in_channels,
            s.in_h,
            s.in_w,
            s.out_channels,
            s.kernel_h,
            s.kernel_w,
            s.stride,
            s.padding_h,
            s.out_h,
            s.out_w,
            s.padding_w,
        ];
        assert_eq!(
            (s.in_h + 2 * s.padding_h - s.kernel_h) / s.stride + 1,
            s.out_h
        );
        assert_eq!(
            (s.in_w + 2 * s.padding_w - s.kernel_w) / s.stride + 1,
            s.out_w
        );
        let kernel = s.kernel_h * s.kernel_w;
        assert_eq!(
            (c.m, c.n, c.k),
            if dx {
                (s.in_channels, s.in_h * s.in_w, s.out_channels * kernel)
            } else {
                (
                    s.out_channels,
                    s.in_channels * kernel,
                    s.batch * s.out_h * s.out_w,
                )
            }
        );
        let (input, weight, upstream) = (
            s.batch * s.in_channels * s.in_h * s.in_w,
            s.out_channels * s.in_channels * kernel,
            s.batch * s.out_channels * s.out_h * s.out_w,
        );
        let sizes = if dx {
            vec![upstream, weight, input]
        } else {
            vec![upstream, input, weight]
        };
        let sizes: Vec<_> = sizes.into_iter().map(|n| n as usize * 4).collect();
        assert_eq!(c.binding_bytes, sizes); // Observed exact capacities, no im2col padding.
        assert!(c.device_local[0] && !c.device_local[2] && c.device_local[3]);
        if dx {
            assert!(!c.device_local[1]);
        }
        let entry = |tile| match tile {
            MatmulTile::Tile64 => c.shader.clone(),
            MatmulTile::Tile32 if dx => ShaderEntry::Conv2dGradInputGemmSmall,
            MatmulTile::Tile32 => ShaderEntry::Conv2dGradWeightGemmSmall,
            _ => panic!("no cooperative convolution in this cohort"),
        };
        let mut members = 0;
        for (index, d) in before
            .iter()
            .enumerate()
            .filter(|(_, d)| d.shader == entry(outcome.initial) && d.params == p)
        {
            let declared: Vec<_> = d
                .input_buffers
                .iter()
                .chain(std::iter::once(&d.output_buffer))
                .map(|b| buffers[b.0 as usize])
                .collect();
            if declared != c.binding_bytes {
                continue;
            }
            assert!(visited.insert(index));
            members += 1;
            assert_eq!(d.input_buffers.len(), 2);
            assert!(
                d.input_buffers[0] != d.input_buffers[1]
                    && !d.input_buffers.contains(&d.output_buffer)
            );
            assert!(
                d.requires_full_precision
                    && !d.use_coop
                    && !d.use_coop_compensated
                    && !d.use_small_tiles
            );
            assert!(
                d.scalar_fallback.is_none()
                    && d.matmul_prologue.is_none()
                    && d.matmul_epilogue.is_none()
                    && d.gemv_rmsnorm.is_none()
                    && d.pointwise.is_none()
                    && d.reduction.is_none()
                    && d.extra_outputs.is_empty()
                    && d.epilogue.is_empty()
                    && d.epilogue_buffers.is_empty()
            );
            assert_eq!(d.horizontal_batch, 0);
            assert_eq!(d.weight_format, meganeura::compile::WeightFormat::F32);
            for (tile, dispatch, key) in [
                (outcome.initial, d, &before_keys[index]),
                (outcome.selected, &after[index], &after_keys[index]),
            ] {
                let side = if tile == MatmulTile::Tile32 { 32 } else { 64 };
                assert_eq!(
                    dispatch.workgroups,
                    [
                        c.n.div_ceil(side),
                        c.m.div_ceil(side),
                        if dx { s.batch } else { 1 }
                    ]
                );
                assert_eq!(key, &json!(format!("{:?}:scalar", entry(tile))));
            }
            reconstructed[index].shader = entry(outcome.selected);
            reconstructed[index].workgroups = after[index].workgroups;
        }
        assert_eq!(members, outcome.dispatches);
        let scratch = outcome.scratch.as_ref().unwrap();
        let stage = *sizes.iter().max().unwrap();
        assert_eq!(scratch.binding_bytes, sizes);
        assert_eq!(scratch.staging_bytes, stage);
        assert_eq!(scratch.staging_reused, stage == previous_stage);
        if stage == previous_stage {
            reuses += 1;
        } else {
            allocations += 1;
        }
        previous_stage = stage;
        let bytes = sizes.iter().sum::<usize>() + stage;
        assert!(bytes <= report.options.max_scratch_bytes);
        peak = peak.max(bytes);
        let phases = outcome.phase_times.unwrap();
        let prep = phases.preparation_breakdown.unwrap();
        let prep_sum: Duration = [
            prep.checks,
            prep.pipelines,
            prep.buffers,
            prep.staging,
            prep.encoder,
            prep.bindings,
        ]
        .into_iter()
        .map(Option::unwrap)
        .sum();
        assert_eq!(prep.pipelines.unwrap(), outcome.compile_time);
        assert!(prep_sum <= phases.preparation.unwrap());
        let q = phases.qualification_breakdown.unwrap();
        let q_sum: Duration = [
            q.input_preparation,
            q.upload_host_copy,
            q.upload_transfer,
            q.dispatch,
            q.readback_transfer,
            q.readback_host_copy,
            q.validation,
        ]
        .into_iter()
        .map(Option::unwrap)
        .sum();
        assert!(q_sum <= phases.qualification.unwrap());
        let sum: Duration = [
            phases.preparation,
            phases.qualification,
            phases.warmup,
            phases.sampling,
            phases.cleanup,
        ]
        .into_iter()
        .map(Option::unwrap)
        .sum();
        assert!(sum <= outcome.elapsed);
        total += outcome.elapsed;
    }
    assert!(total <= report.elapsed);
    assert_eq!(reconstructed, after); // All non-choice fields and nonmembers stay fixed.
    let mut observed = 0;
    for index in 0..before.len() {
        assert_eq!(
            before[index] != after[index],
            before_keys[index] != after_keys[index]
        );
        observed += usize::from(before[index] != after[index]);
    }
    assert_eq!(observed, changed);
    let scratch = report.scratch.unwrap();
    assert_eq!(scratch.peak_bytes, peak);
    assert_eq!(scratch.retained_staging_bytes, 0);
    assert_eq!(scratch.staging_allocations, allocations);
    assert_eq!(scratch.staging_releases, allocations);
    assert_eq!(scratch.staging_reuses, reuses);
    report
}

fn case_record(
    case: &Value,
    index: usize,
    seed: usize,
    corrected: bool,
    earliest: u64,
    latest: u64,
) -> Confirmation {
    let e = &CASES[index];
    let order = seed + index;
    assert_eq!(case["name"], e.name);
    assert_eq!(case["work"], e.work);
    assert_eq!(case["seed"], order);
    assert_eq!(case["status"], "complete");
    assert_eq!(case["final_age"], 178);
    assert_eq!(
        case["description"],
        if index == 0 {
            json!({"builder":"build_resnet50_training","batch":1,"image":[3,224,224],"classes":1000,
            "batch_norm":"folded, not running-statistic training","loss":"cross entropy"})
        } else {
            json!({"batch":2,"input":[5,17,19],"channels":[17,33],"kernels":[[3,2],[2,3]],
            "strides":[1,2],"padding":[[2,0],[0,1]],"activation":null,"loss":"mean squared output"})
        }
    );
    let start = case["host_start"]["unix_ms"].as_u64().unwrap();
    let end = case["host_finish"]["unix_ms"].as_u64().unwrap();
    assert!(earliest <= start && start < end && end <= latest);
    assert!(
        case["baseline_dispatches"]
            .as_array()
            .unwrap()
            .iter()
            .all(|d| d["workgroups"]
                .as_array()
                .unwrap()
                .iter()
                .all(|n| n.as_u64().unwrap() > 0))
    );
    if corrected {
        assert_eq!(case["prefix_training_signal"], true);
        if index > 0 {
            assert_eq!(case["parameters_updated"], json!([true, true]));
        }
    }
    let roster = check_comparison(&case["prefix_comparison"], e, "prefix", 3);
    for (value, stage, age) in [
        (&case["warmup_comparison"], "warmup", 33),
        (&case["control"]["comparison"], "control", 78),
        (&case["search_state_comparison"], "search", 78),
    ] {
        assert_eq!(check_comparison(value, e, stage, age), roster);
    }
    let control: Vec<Pair> = serde_json::from_value(case["control"]["pairs"].clone()).unwrap();
    pairs(&control, 40, 39, order);
    let mut time = telemetry::time_window(&case["control"], start, end);
    let report = search_contracts(case, e);
    let records = case["blocks"].as_array().unwrap();
    assert_eq!(records.len(), 4);
    let blocks: Vec<Block> = records
        .iter()
        .enumerate()
        .map(|(i, b)| {
            assert_eq!(b["index"], i);
            time = telemetry::time_window(b, time, end);
            let samples: Block = serde_json::from_value(b["samples"].clone()).unwrap();
            assert_eq!(
                samples.winner_session,
                crossover::winner_order(order % 2)[i]
            );
            pairs(&samples.pairs, 20, 84 + i * 25, order + i);
            assert_eq!(
                check_comparison(&b["comparison"], e, "crossover", 103 + i as u32 * 25),
                roster
            );
            if i == 1 || i == 3 {
                assert!(b["swap"]["elapsed_ms"].as_f64().unwrap() > 0.0);
                for (side, stage) in [("left", "swap_left"), ("right", "swap_right")] {
                    assert_eq!(
                        check_comparison(&b["swap"][side], e, stage, 78 + i as u32 * 25),
                        roster
                    );
                }
            } else {
                assert!(b["swap"].is_null());
            }
            samples
        })
        .collect();
    if index > 0 {
        let first = case["prefix_comparison"]["tensors"].as_array().unwrap();
        let last = case["blocks"][3]["comparison"]["tensors"]
            .as_array()
            .unwrap();
        assert!(first.iter().zip(last).any(|(a, b)| {
            a["name"].as_str().unwrap().starts_with("parameter.")
                && a["reference_sum_sq"] != b["reference_sum_sq"]
        }));
        if index == 1 {
            assert!(
                first
                    .iter()
                    .filter(|t| t["name"].as_str().unwrap().starts_with("adam_"))
                    .all(|t| t["reference_sum_sq"].as_f64().unwrap() > 0.0)
            );
        }
    }
    for side in 0..2 {
        for m in [&case["initial_memory"][side], &case["final_memory"][side]] {
            check_memory(m, e);
            for field in [
                "plan_capacity_bytes",
                "graph_allocation_bytes",
                "adam_bytes",
                "accumulator_bytes",
                "auxiliary_bytes",
                "resident_buffer_requests",
                "device_local_bytes",
            ] {
                assert_eq!(m[field], case["initial_memory"][0][field]);
            }
        }
    }
    let changed = report
        .outcomes
        .iter()
        .filter(|o| o.selected != o.initial)
        .map(|o| o.dispatches)
        .sum();
    let confirmation = Confirmation::new(&control, &blocks, changed, true).unwrap();
    assert_eq!(case["confirmation"]["decision"], confirmation.decision);
    assert_eq!(
        case["confirmation"]["control_stable"],
        confirmation.control_stable
    );
    for (key, value) in [
        ("control", &confirmation.control),
        ("left_winner", &confirmation.left_winner),
        ("right_winner", &confirmation.right_winner),
        ("pooled", &confirmation.pooled),
    ] {
        check_timing(value, &case["confirmation"][key]);
    }
    confirmation
}

fn summary(records: &[Value]) -> Value {
    assert_eq!(records.len(), 6);
    let mut ids = HashSet::new();
    let mut previous = 0;
    for (index, record) in records.iter().enumerate() {
        metadata(record, index + 1, true);
        let start = record["metadata"]["started_unix_ms"].as_u64().unwrap();
        let end = record["finished_unix_ms"].as_u64().unwrap();
        assert!(previous <= start && start < end);
        previous = end;
        assert!(ids.insert(record["metadata"]["process_id"].as_u64().unwrap()));
        let mut case_start = start;
        for (i, case) in record["cases"].as_array().unwrap().iter().enumerate() {
            case_record(case, i, index + 1, true, case_start, end);
            case_start = case["host_finish"]["unix_ms"].as_u64().unwrap();
        }
    }
    let cases: Vec<_> = CASES.iter().enumerate().map(|(i,e)| {
        let extract = |field| records.iter().map(|r| r["cases"][i]["confirmation"]["pooled"][field].as_f64().unwrap()).collect::<Vec<_>>();
        let ratios = extract("speedup");
        let search: Vec<_> = records.iter().map(|r| serde_json::from_value::<TuneReport>(r["cases"][i]["search"].clone()).unwrap()).collect();
        let phases: Value = ["preparation","qualification","warmup","sampling"].into_iter().map(|name| {
            let values: Vec<_> = records.iter().map(|r| r["cases"][i]["search"]["outcomes"].as_array().unwrap().iter()
                .map(|o| serde_json::from_value::<Duration>(o["phase_times"][name].clone()).unwrap().as_secs_f64()*1e3).sum::<f64>()).collect();
            (name.to_owned(),json!(median(&values)))
        }).collect::<serde_json::Map<_,_>>().into();
        json!({"name":e.name,"baseline_median_ms":median(&extract("baseline_median_ms")),
            "selected_median_ms":median(&extract("tuned_median_ms")),"median_process_ratio":median(&ratios),
            "process_ratios":ratios,"decisions":records.iter().map(|r| &r["cases"][i]["confirmation"]["decision"]).collect::<Vec<_>>(),
            "stable_controls":records.iter().filter(|r| r["cases"][i]["confirmation"]["control_stable"] == true).count(),
            "search_median_ms":median(&search.iter().map(|r| r.elapsed.as_secs_f64()*1e3).collect::<Vec<_>>()),
            "phase_medians_ms":phases,"changed_dispatches":records.iter().map(|r| &r["cases"][i]["dispatches_changed"]).collect::<Vec<_>>(),
            "eligible_classes":e.eligible,"visited_classes":e.eligible.min(8),
            "scratch_peak_bytes":search[0].scratch.unwrap().peak_bytes})
    }).collect();
    json!({"protocol":"conv-tiles-corrected-2026-09-06","processes":6,"qualified_comparisons":84,
        "original_small_controls_disqualified":12,"cases":cases})
}

#[test]
fn corrected_cohort_replays_complete_contracts_and_state() {
    let actual = summary(&records(true));
    println!("{}", serde_json::to_string_pretty(&actual).unwrap());
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/docs/experiments/conv-tiles-corrected-2026-09-06/summary.json"
    );
    let expected: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    fn compare(a: &Value, b: &Value) {
        match (a, b) {
            (Value::Number(_), Value::Number(_)) => close(a.as_f64().unwrap(), b.as_f64().unwrap()),
            (Value::Object(a), Value::Object(b)) => {
                assert_eq!(a.len(), b.len());
                for (k, v) in a {
                    compare(v, &b[k]);
                }
            }
            (Value::Array(a), Value::Array(b)) => {
                assert_eq!(a.len(), b.len());
                for (a, b) in a.iter().zip(b) {
                    compare(a, b);
                }
            }
            _ => assert_eq!(a, b),
        }
    }
    compare(&actual, &expected);
}

#[test]
fn original_records_retain_their_disqualifying_zero_workloads() {
    for (i, r) in records(false).iter().enumerate() {
        metadata(r, i + 1, false);
        case_record(
            &r["cases"][0],
            0,
            i + 1,
            false,
            r["metadata"]["started_unix_ms"].as_u64().unwrap(),
            r["finished_unix_ms"].as_u64().unwrap(),
        );
        for index in 1..3 {
            let c = &r["cases"][index];
            assert_eq!(c["status"], "complete"); // Historical runner was insufficient.
            assert_eq!(c["baseline_dispatches"][0]["params"][0], 0);
            assert_eq!(c["baseline_dispatches"][0]["workgroups"][2], 0);
            let tensors: Vec<TensorComparison> =
                serde_json::from_value(c["prefix_comparison"]["tensors"].clone()).unwrap();
            assert!(
                tensors
                    .iter()
                    .filter(|t| !t.name.starts_with("parameter."))
                    .all(|t| t.reference_sum_sq == 0.0 && t.candidate_sum_sq == 0.0)
            );
            assert!(
                std::panic::catch_unwind(|| check_comparison(
                    &c["prefix_comparison"],
                    &CASES[index],
                    "prefix",
                    3
                ))
                .is_err()
            );
            search_contracts(c, &CASES[index]); // Private synthetic qualification is still meaningful.
        }
    }
}

#[test]
fn mutations_of_layout_evidence_signal_and_guards_are_rejected() {
    let records = records(true);
    for (path, value) in [
        ("/cases/0/search/options/scope", json!("Dense")),
        (
            "/cases/0/search/outcomes/0/class/conv2d/padding_h",
            json!(2),
        ),
        ("/cases/0/search/outcomes/0/scratch/staging_bytes", json!(4)),
        ("/cases/0/search/outcomes/0/class/binding_bytes/1", json!(4)),
        ("/cases/0/selected_dispatches/0/workgroups/2", json!(0)),
        ("/cases/0/confirmation/decision", json!("confirmed_gain")),
        ("/cases/1/prefix_training_signal", json!(false)),
        ("/cases/1/parameters_updated/0", json!(false)),
        ("/cases/1/blocks/0/comparison/expected_adam_step", json!(0)),
    ] {
        let mut bad = records.clone();
        *bad[0].pointer_mut(path).unwrap() = value;
        assert!(
            std::panic::catch_unwind(|| summary(&bad)).is_err(),
            "accepted {path}"
        );
    }
}
