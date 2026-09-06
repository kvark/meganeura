//! CPU replay of baseline localization; no inference of candidate speedups.
#[path = "support/tuning_evidence.rs"]
#[allow(dead_code)] // Reuse tensor roster checks, not the tuning-pair contract.
mod evidence;
#[path = "../examples/support/tuning_measurement.rs"]
mod measurement;
#[path = "support/telemetry.rs"]
#[allow(dead_code)] // This collector uses enclosing host windows, not pair blocks.
mod telemetry;

use evidence::{Expected, check_comparison, close};
use measurement::{TensorComparison, median};
use meganeura::{
    TuneReport, TuneScratchStats,
    compile::{Dispatch, ShaderEntry},
};
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashSet};

const RECORDS: [&str; 3] = [
    include_str!("../docs/experiments/training-profile-2026-09-06/run-01.json"),
    include_str!("../docs/experiments/training-profile-2026-09-06/run-02.json"),
    include_str!("../docs/experiments/training-profile-2026-09-06/run-03.json"),
];
const REVISION: &str = "48e23154419e2098db1109be6e59d8cd4d265b6a";
const EXECUTABLE: &str = "563f10841be0738ef1dc3546040c841a093ee83d3ba4847032b0260de4b114c4";
const SUMMARY: &str = include_str!("../docs/experiments/training-profile-2026-09-06/summary.json");
const CASES: [Expected; 3] = [
    Expected {
        name: "smollm2-flb",
        work: "ForwardLossBackward",
        dispatches: 284,
        eligible: 11,
        excluded: 194,
        parameters: (82, 1845376),
        gradients: (66, 1321088),
        output_elements: 127,
    },
    Expected {
        name: "whisper-flb",
        work: "ForwardLossBackward",
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

fn number(v: &Value) -> f64 {
    let n = v.as_f64().unwrap();
    assert!(n.is_finite() && n >= 0.0);
    n
}

fn samples(v: &Value, count: usize) -> Vec<f64> {
    let values: Vec<_> = v.as_array().unwrap().iter().map(number).collect();
    assert_eq!(values.len(), count);
    assert!(values.iter().all(|&x| x > 0.0));
    values
}

fn check_samples(expected: &[f64], actual: &Value) {
    let actual = samples(actual, expected.len());
    for (&a, b) in expected.iter().zip(actual) {
        close(a, b);
    }
}

fn check_normal(block: &Value, reference: &Value) {
    let ms = samples(&block["samples_ms"], 20);
    close(median(&ms), number(&block["median_ms"]));
    let losses: Vec<f32> = serde_json::from_value(block["losses"].clone()).unwrap();
    assert_eq!(losses.len(), 20);
    let expected = TensorComparison::new("loss_trajectory".into(), &[losses[0]; 20], &losses);
    assert!(expected.passed && expected.exact);
    assert_eq!(
        block["loss_comparison"],
        serde_json::to_value(expected).unwrap()
    );
    let loss = reference["tensors"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["name"] == "loss")
        .unwrap();
    close(
        f64::from(losses[0]).powi(2),
        number(&loss["reference_sum_sq"]),
    );
}

fn check_profile(case: &Value, expected: &Expected, forward: usize) {
    let p = &case["profile"];
    assert_eq!(p["schema_version"], 1);
    assert!(
        p["timing_contract"]
            .as_str()
            .unwrap()
            .contains("inter-pass barrier")
    );
    assert_eq!(p["device"]["backend"], "Vulkan");
    assert_eq!(p["device"]["software_emulated"], false);
    let plan = &p["plan"];
    assert_eq!(plan["dispatch_count"], expected.dispatches);
    assert_eq!(plan["forward_dispatch_count"], forward);
    assert_eq!(
        plan["backward_dispatch_count"],
        expected.dispatches - forward
    );
    for (key, memory_key) in [
        ("logical_buffer_bytes", "plan_capacity_bytes"),
        ("allocated_buffer_bytes", "graph_allocation_bytes"),
        ("resident_buffer_bytes", "resident_buffer_requests"),
        ("device_local_bytes", "device_local_bytes"),
        ("adam_state_bytes", "adam_bytes"),
        ("grad_accumulator_bytes", "accumulator_bytes"),
        ("optimizer_aux_bytes", "auxiliary_bytes"),
    ] {
        assert_eq!(plan[key], case["initial_memory"][memory_key]);
    }
    assert!(plan["barrier_group_count"].as_u64().unwrap() <= expected.dispatches as u64);
    assert!(plan["physical_allocation_count"].as_u64().unwrap() > 0);
    let contracts: Vec<Dispatch> =
        serde_json::from_value(case["dispatch_contracts"].clone()).unwrap();
    let dispatches = p["dispatches"].as_array().unwrap();
    assert_eq!(contracts.len(), expected.dispatches);
    assert_eq!(dispatches.len(), expected.dispatches);
    assert_eq!(
        case["pipeline_keys"].as_array().unwrap().len(),
        expected.dispatches
    );
    let mut total = vec![0.0; 5];
    let mut sum = 0.0;
    let mut families = BTreeMap::<(&str, &str), Vec<&Value>>::new();
    for (index, (d, c)) in dispatches.iter().zip(&contracts).enumerate() {
        assert_eq!(d["index"], index);
        assert_eq!(d["shader"], format!("{:?}", c.shader));
        assert_eq!(d["family"], c.profile_family());
        assert_eq!(d["label"], c.label);
        assert_eq!(d["timestamp_label"], c.label);
        assert_eq!(d["origin"], json!(c.origin));
        assert_eq!(d["workgroups"], json!(c.workgroups));
        assert_eq!(
            d["workgroup_count"],
            c.workgroups.iter().map(|&n| u64::from(n)).product::<u64>()
        );
        assert_eq!(d["cooperative"], false);
        assert!(!c.use_coop && !c.use_coop_compensated && !c.weight_format.uses_reduced_storage());
        assert_eq!(d["small_tile"], c.use_small_tiles);
        assert_eq!(d["requires_full_precision"], c.requires_full_precision);
        assert_eq!(d["weight_format"], "F32");
        assert_eq!(d["has_prologue"], c.matmul_prologue.is_some());
        assert_eq!(
            d["has_epilogue"],
            c.matmul_epilogue.is_some() || !c.epilogue.is_empty()
        );
        assert_eq!(d["pipeline"], case["pipeline_keys"][index]);
        check_conv_contract(d, c);
        assert_eq!(
            d["phase"],
            if index < forward {
                "forward"
            } else {
                "backward"
            }
        );
        assert!(number(&d["input_buffer_bytes"]) > 0.0 && number(&d["output_buffer_bytes"]) > 0.0);
        let mut ms = samples(&d["timing_samples_ms"], 5);
        for (total, &ms) in total.iter_mut().zip(&ms) {
            *total += ms;
        }
        ms.sort_by(f64::total_cmp);
        close(ms[1], number(&d["p25_ms"]));
        close(ms[2], number(&d["median_ms"]));
        close(ms[3], number(&d["p75_ms"]));
        sum += ms[2];
        families
            .entry((d["phase"].as_str().unwrap(), d["family"].as_str().unwrap()))
            .or_default()
            .push(d);
    }
    for d in dispatches {
        close(
            number(&d["median_ms"]) / sum * 100.0,
            number(&d["share_of_dispatch_median_sum_pct"]),
        );
    }
    assert_eq!(families.len(), p["families"].as_array().unwrap().len());
    let mut seen = HashSet::new();
    for family in p["families"].as_array().unwrap() {
        let key = (
            family["phase"].as_str().unwrap(),
            family["family"].as_str().unwrap(),
        );
        assert!(seen.insert(key));
        let members = &families[&key];
        assert_eq!(family["dispatch_count"], members.len());
        let median_sum: f64 = members.iter().map(|d| number(&d["median_ms"])).sum();
        let totals: Vec<f64> = (0..5)
            .map(|i| {
                members
                    .iter()
                    .map(|d| number(&d["timing_samples_ms"][i]))
                    .sum()
            })
            .collect();
        check_samples(&totals, &family["timing_samples_ms"]);
        close(median(&totals), number(&family["median_ms"]));
        close(median_sum, number(&family["dispatch_median_sum_ms"]));
        close(
            median_sum / sum * 100.0,
            number(&family["share_of_dispatch_median_sum_pct"]),
        );
    }
    let m = &p["measurement"];
    assert_eq!(m["sample_count"], 5);
    check_samples(&total, &m["gpu_total_samples_ms"]);
    close(median(&total), number(&m["gpu_total_median_ms"]));
    let wall = median(&samples(&m["profiled_wall_samples_ms"], 5));
    close(wall, number(&m["profiled_wall_median_ms"]));
    close(
        median(&total) / wall * 100.0,
        number(&m["timestamped_gpu_share_of_profiled_wall_pct"]),
    );
    assert_eq!(
        m["unprofiled_median_ms"],
        case["normal_before"]["median_ms"]
    );
    close(
        wall / number(&m["unprofiled_median_ms"]),
        number(&m["instrumentation_wall_ratio"]),
    );
    assert_eq!(p["pipeline_statistics"], json!([])); // Requested, but no records returned.
}

fn check_conv_contract(profile: &Value, dispatch: &Dispatch) {
    let small = matches!(
        dispatch.shader,
        ShaderEntry::Conv2dGemmSmall
            | ShaderEntry::Conv2dGradInputGemmSmall
            | ShaderEntry::Conv2dGradWeightGemmSmall
    );
    let direction = match dispatch.shader {
        ShaderEntry::Conv2dGemm | ShaderEntry::Conv2dGemmSmall => 0,
        ShaderEntry::Conv2dGradInputGemm | ShaderEntry::Conv2dGradInputGemmSmall => 1,
        ShaderEntry::Conv2dGradWeightGemm | ShaderEntry::Conv2dGradWeightGemmSmall => 2,
        _ => return,
    };
    let [batch, ci, h, w, co, kh, kw, stride, ph, oh, ow, pw]: [u32; 12] =
        dispatch.params.clone().try_into().unwrap();
    assert!(
        [batch, ci, h, w, co, kh, kw, stride, oh, ow]
            .iter()
            .all(|&n| n > 0)
    );
    assert_eq!(oh, (h + 2 * ph - kh) / stride + 1);
    assert_eq!(ow, (w + 2 * pw - kw) / stride + 1);
    let x = batch * ci * h * w * 4;
    let weight = co * ci * kh * kw * 4;
    let y = batch * co * oh * ow * 4;
    let tile = if small { 32 } else { 64 };
    let (input, output, groups) = match direction {
        0 => (
            x + weight,
            y,
            [(oh * ow).div_ceil(tile), co.div_ceil(tile), batch],
        ),
        1 => (
            y + weight,
            x,
            [(h * w).div_ceil(tile), ci.div_ceil(tile), batch],
        ),
        _ => (
            y + x,
            weight,
            [(ci * kh * kw).div_ceil(tile), co.div_ceil(tile), 1],
        ),
    };
    assert_eq!(dispatch.workgroups, groups);
    assert_eq!(profile["input_buffer_bytes"], input);
    assert_eq!(profile["output_buffer_bytes"], output);
    if direction == 1 && stride == 1 {
        // The later padding fix leaves these recorded cases' equations equal;
        // it does not relabel this cohort as a measurement of corrected code.
        assert_eq!(2 * ph, kh - 1);
        assert_eq!(2 * pw, kw - 1);
    }
}

fn validate(record: &Value, seed: usize) {
    validate_source(
        record,
        seed,
        "training-profile-2026-09-06",
        REVISION,
        EXECUTABLE,
    );
}

fn validate_source(record: &Value, seed: usize, protocol: &str, revision: &str, executable: &str) {
    assert_eq!(record["schema_version"], 1);
    assert_eq!(record["protocol"], protocol);
    assert_eq!(record["status"], "complete");
    let meta = &record["metadata"];
    assert_eq!(meta["revision"], revision);
    assert_eq!(meta["executable_sha256"], executable);
    assert_eq!(
        meta["cargo_lock_sha256"],
        "72cecddd0f090e82d5db46bc6f7f80429a058b2afa0eaec84a9b27e7340a4a80"
    );
    assert_eq!(meta["rustc"], "rustc 1.98.0 (88d9e12ae 2026-08-18)");
    assert!(meta["rustflags"].is_null());
    assert_eq!(meta["tracked_source_clean"], true);
    assert_eq!(meta["seed"], seed);
    assert_eq!(meta["gpu_timing"], true);
    assert_eq!(meta["cooperative_policy"], "Disabled");
    assert_eq!(
        meta["compile_options"],
        json!({"flash_backward_coop": false, "flash_forward_coop": false,
        "fuse_dispatches": true, "use_schedule_pointwise": true, "use_schedule_reduction": true,
        "knobs": {"flash_ept_cap": 32, "flash_grad_kv_ept_cap": 32, "flash_grad_q_ept_cap": 32}})
    );
    assert_eq!(
        meta["runtime_options"],
        "SessionOptions { debug: false, coop: Disabled, no_alias: false, no_device_local: false, serial_dispatch: false, dump_plan: false, pin_buffers: None }"
    );
    assert_eq!(
        meta["optimize"],
        json!({"extraction_cost":"TensorTraffic", "mode":"Greedy", "no_winograd":false, "saturation_cutoff":300})
    );
    assert_eq!(
        meta["device"],
        json!({"name":"NVIDIA GeForce RTX 5070", "driver":"595.71.05", "f32_tile":0, "f16_tile":16})
    );
    for (key, value) in [
        ("warmup", 30),
        ("settling", 5),
        ("normal_samples", 20),
        ("profile_samples", 5),
    ] {
        assert_eq!(meta[key], value);
    }
    let cases = record["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 3);
    let mut previous = meta["started_unix_ms"].as_u64().unwrap();
    for (position, case) in cases.iter().enumerate() {
        let index = (position + seed - 1) % 3;
        let expected = &CASES[index];
        assert_eq!(case["name"], expected.name);
        assert_eq!(meta["case_order"][position], expected.name);
        assert_eq!(case["work"], expected.work);
        assert_eq!(case["status"], "complete");
        for field in [
            "host_start",
            "normal_before/host_start",
            "normal_before/host_finish",
            "profile_host_start",
            "profile_host_finish",
            "normal_after/host_start",
            "normal_after/host_finish",
            "host_finish",
        ] {
            let time = case
                .pointer(&format!("/{field}/unix_ms"))
                .unwrap()
                .as_u64()
                .unwrap();
            assert!(time >= previous);
            previous = time;
        }
        let census: TuneReport = serde_json::from_value(case["search_census"].clone()).unwrap();
        if protocol == "training-profile-2026-09-06" {
            assert!(case["search_census"]["options"].get("scope").is_none());
        } else {
            assert_eq!(case["search_census"]["options"]["scope"], "Dense");
        }
        assert_eq!(
            serde_json::to_value(census.options).unwrap(),
            json!({"scope":"Dense", "max_time":{"secs":0,"nanos":0}, "max_classes":8,
                "max_scratch_bytes":67108864, "staging":"Download", "staging_reuse":"SameSize",
                "warmup_runs":1, "sample_pairs":6, "dispatches_per_sample":16, "min_improvement":0.05})
        );
        assert_eq!(census.eligible_classes, expected.eligible);
        assert_eq!(census.excluded_dispatches, expected.excluded);
        assert_eq!(census.class_limit_reached, expected.eligible > 8);
        assert!(census.time_budget_exhausted);
        assert_eq!(census.visited_classes, 0);
        assert!(census.outcomes.is_empty());
        assert_eq!(census.scratch, Some(TuneScratchStats::default()));
        assert_eq!(census.final_cleanup, None);
        for key in ["adam_bytes", "accumulator_bytes"] {
            assert_eq!(case["initial_memory"][key], 0);
        }
        assert_eq!(case["initial_memory"]["auxiliary_bytes"], 4);
        assert_eq!(case["same_memory"], true);
        for (key, value) in case["initial_memory"].as_object().unwrap() {
            if key != "process_api_sample" {
                assert_eq!(value, &case["final_memory"][key]);
            }
        }
        let reference = &case["reference_comparison"];
        let roster = check_comparison(reference, expected, "reference", 0);
        for stage in ["normal_before", "normal_after"] {
            assert_eq!(
                roster,
                check_comparison(&case[format!("{stage}_comparison")], expected, stage, 0)
            );
            check_normal(&case[stage], reference);
        }
        assert_eq!(case["profile_prepare_calls"], 15);
        let checks = case["profile_comparisons"].as_array().unwrap();
        assert_eq!(checks.len(), 5);
        for check in checks {
            assert_eq!(roster, check_comparison(check, expected, "profiled", 0));
        }
        check_profile(case, expected, [85, 98, 192][index]);
    }
    assert_eq!(meta["case_order"].as_array().unwrap().len(), 3);
    assert!(previous <= record["finished_unix_ms"].as_u64().unwrap());
    telemetry::check_telemetry(record, true);
}

fn cohort(records: &[Value]) -> Value {
    assert_eq!(records.len(), 3);
    let mut ids = HashSet::new();
    let mut previous = 0;
    for (i, r) in records.iter().enumerate() {
        validate(r, i + 1);
        assert!(ids.insert(r["metadata"]["process_id"].as_u64().unwrap()));
        assert!(previous <= r["metadata"]["started_unix_ms"].as_u64().unwrap());
        previous = r["finished_unix_ms"].as_u64().unwrap();
    }
    let cases: Vec<_> = CASES.iter().map(|expected| {
        let runs: Vec<_> = records.iter().map(|r| r["cases"].as_array().unwrap().iter().find(|c| c["name"] == expected.name).unwrap()).collect();
        for case in &runs[1..] {
            assert_eq!(case["description"], runs[0]["description"]);
            assert_eq!(case["dispatch_contracts"], runs[0]["dispatch_contracts"]);
            assert_eq!(case["pipeline_keys"], runs[0]["pipeline_keys"]);
        }
        let processes: Vec<_> = runs.iter().enumerate().map(|(i, c)| {
            let a = number(&c["normal_before"]["median_ms"]);
            let b = number(&c["normal_after"]["median_ms"]);
            json!({"seed":i+1,"normal_before_ms":a,"normal_after_ms":b,"normal_drift_pct":(b/a-1.0)*100.0,
                "profiled_wall_ms":c["profile"]["measurement"]["profiled_wall_median_ms"],
                "instrumentation_ratio":c["profile"]["measurement"]["instrumentation_wall_ratio"]})
        }).collect();
        let families: Vec<_> = runs[0]["profile"]["families"].as_array().unwrap().iter().map(|family| {
            let same: Vec<_> = runs.iter().map(|c| c["profile"]["families"].as_array().unwrap().iter()
                .find(|f| f["phase"]==family["phase"] && f["family"]==family["family"]).unwrap()).collect();
            let shares: Vec<_> = same.iter().map(|f| number(&f["share_of_dispatch_median_sum_pct"])).collect();
            json!({"phase":family["phase"], "family":family["family"], "dispatch_count":family["dispatch_count"],
                "median_dispatch_sum_ms":median(&same.iter().map(|f| number(&f["dispatch_median_sum_ms"])).collect::<Vec<_>>()),
                "median_share_pct":median(&shares),"share_min_pct":shares.iter().copied().reduce(f64::min).unwrap(),
                "share_max_pct":shares.iter().copied().reduce(f64::max).unwrap()})
        }).collect();
        json!({"name":expected.name,"processes":processes,"families":families})
    }).collect();
    json!({"schema_version":1,"measured_revision":REVISION,"cases":cases})
}

#[test]
fn retained_whole_step_profiles_replay() {
    let records: Vec<_> = RECORDS
        .iter()
        .map(|r| serde_json::from_str(r).unwrap())
        .collect();
    let summary = cohort(&records);
    check_summary(&summary, &serde_json::from_str(SUMMARY).unwrap());
    println!("{}", serde_json::to_string_pretty(&summary).unwrap());
}

fn check_summary(expected: &Value, recorded: &Value) {
    match expected {
        Value::Object(fields) => {
            assert_eq!(fields.len(), recorded.as_object().unwrap().len());
            for (key, value) in fields {
                check_summary(value, &recorded[key]);
            }
        }
        Value::Array(values) => {
            let actual = recorded.as_array().unwrap();
            assert_eq!(values.len(), actual.len());
            for (a, b) in values.iter().zip(actual) {
                check_summary(a, b);
            }
        }
        Value::Number(value) => close(value.as_f64().unwrap(), recorded.as_f64().unwrap()),
        _ => assert_eq!(expected, recorded),
    }
}

const INDEXING_ORDER: [(&str, usize); 6] = [
    ("baseline", 1),
    ("exact", 1),
    ("exact", 2),
    ("baseline", 2),
    ("baseline", 3),
    ("exact", 3),
];

fn indexing_records() -> Vec<Value> {
    INDEXING_ORDER
        .iter()
        .map(|(kind, seed)| {
            let path = format!(
                "{}/docs/experiments/conv-indexing-2026-09-06/{kind}-{seed:02}.json.gz",
                env!("CARGO_MANIFEST_DIR")
            );
            let file = std::fs::File::open(path).unwrap();
            serde_json::from_reader(flate2::read::GzDecoder::new(file)).unwrap()
        })
        .collect()
}

fn indexing_cohort(records: &[Value]) -> Value {
    assert_eq!(records.len(), INDEXING_ORDER.len());
    let mut previous = 0;
    let mut ids = HashSet::new();
    for (record, &(kind, seed)) in records.iter().zip(&INDEXING_ORDER) {
        let (revision, executable) = if kind == "baseline" {
            (
                "aa1344e784a37462b8261aed97cb804c11ce8ba3",
                "f4e681345ac30729ce45bfdbdbd0dd3fce395c4dbfa6f6a9fe8016fa5b015a59",
            )
        } else {
            (
                "45304b884d821875c476f6d28446051ad22fe35f",
                "7f0c7fbae450e8a16b553a4a6de3bfed2c8e3c4ae7ea505ca831371344feabde",
            )
        };
        validate_source(
            record,
            seed,
            "conv-indexing-2026-09-06",
            revision,
            executable,
        );
        let meta = &record["metadata"];
        assert!(ids.insert(meta["process_id"].as_u64().unwrap()));
        assert!(previous <= meta["started_unix_ms"].as_u64().unwrap());
        previous = record["finished_unix_ms"].as_u64().unwrap();
    }
    let cases: Vec<_> = CASES.iter().map(|expected| {
        let runs: Vec<_> = records.iter().map(|record| record["cases"].as_array().unwrap().iter()
            .find(|case| case["name"] == expected.name).unwrap()).collect();
        let reference = runs[0];
        let digest = reference["reference_state_sha256"].as_str().unwrap();
        assert_eq!(digest.len(), 64);
        assert!(digest.bytes().all(|c| c.is_ascii_hexdigit()));
        for case in &runs {
            assert_eq!(case["reference_state_sha256"], digest);
            assert_eq!(case["final_state_sha256"], digest);
            for key in ["description", "dispatch_contracts", "pipeline_keys", "reference_comparison"] {
                assert_eq!(case[key], reference[key]);
            }
            for (key, value) in reference["initial_memory"].as_object().unwrap() {
                if key != "process_api_sample" {
                    assert_eq!(&case["initial_memory"][key], value);
                }
            }
        }
        let conv_ms = |case: &Value| {
            let dispatches = case["profile"]["dispatches"].as_array().unwrap();
            let mut fields = serde_json::Map::new();
            for (direction, entries) in [
                ("forward", ["Conv2dGemm", "Conv2dGemmSmall"]),
                ("dx", ["Conv2dGradInputGemm", "Conv2dGradInputGemmSmall"]),
                ("dw", ["Conv2dGradWeightGemm", "Conv2dGradWeightGemmSmall"]),
            ] {
                let sum: f64 = dispatches.iter().filter(|d| entries.contains(&d["shader"].as_str().unwrap()))
                    .map(|d| number(&d["median_ms"])).sum();
                fields.insert(direction.into(), json!(sum));
            }
            Value::Object(fields)
        };
        let pairs: Vec<_> = (1..=3).map(|seed| {
            let get = |kind| runs[INDEXING_ORDER.iter().position(|&pair| pair == (kind, seed)).unwrap()];
            let baseline = get("baseline");
            let exact = get("exact");
            let bb = number(&baseline["normal_before"]["median_ms"]);
            let ba = number(&baseline["normal_after"]["median_ms"]);
            let eb = number(&exact["normal_before"]["median_ms"]);
            let ea = number(&exact["normal_after"]["median_ms"]);
            json!({"seed":seed, "baseline_before_ms":bb, "baseline_after_ms":ba,
                "exact_before_ms":eb, "exact_after_ms":ea,
                "baseline_to_exact_before_ratio":bb/eb, "baseline_to_exact_after_ratio":ba/ea,
                "baseline_drift_pct":(ba/bb-1.0)*100.0, "exact_drift_pct":(ea/eb-1.0)*100.0,
                "profile_baseline_conv_ms":conv_ms(baseline), "profile_exact_conv_ms":conv_ms(exact)})
        }).collect();
        json!({"name":expected.name, "state_sha256":digest,
            "same_dispatch_contracts_and_requested_memory":true, "process_pairs":pairs})
    }).collect();
    json!({"protocol":"conv-indexing-2026-09-06", "cases":cases,
        "processes":6, "normal_steps":720, "profiled_full_state_checks":90,
        "interpretation":"sequential process pairs, not paired-kernel or cross-engine gain evidence"})
}

#[test]
fn exact_convolution_indexing_cost_and_cross_source_states_replay() {
    let summary = indexing_cohort(&indexing_records());
    let recorded = std::fs::File::open(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/docs/experiments/conv-indexing-2026-09-06/summary.json"
    ))
    .unwrap();
    check_summary(&summary, &serde_json::from_reader(recorded).unwrap());
    println!(
        "INDEXING_SUMMARY={}",
        serde_json::to_string(&summary).unwrap()
    );
}

#[test]
fn indexing_replay_rejects_cross_source_state_drift() {
    let original = indexing_records();
    for key in ["reference_state_sha256", "final_state_sha256"] {
        let mut changed = original.clone();
        // Seed 1 starts with SmolLM2 in both revisions. Preserve the record's
        // in-process comparisons to isolate the new cross-source hash check.
        changed[1]["cases"][0][key] = json!("0".repeat(64));
        assert!(std::panic::catch_unwind(|| indexing_cohort(&changed)).is_err());
    }
}

#[test]
fn replay_rejects_misattribution_missing_states_and_changed_boundaries() {
    let original: Value = serde_json::from_str(RECORDS[0]).unwrap();
    for (path, value) in [
        ("/metadata/revision", json!("newer source")),
        ("/metadata/cooperative_policy", json!("Auto")),
        ("/metadata/profile_samples", json!(3)),
        ("/cases/0/work", json!("Adam")),
        ("/cases/1/dispatch_contracts/0/params/0", json!(2)),
        ("/cases/0/search_census/visited_classes", json!(1)),
        ("/cases/0/profile_prepare_calls", json!(10)),
        (
            "/cases/0/profile_comparisons/0/baseline_adam_step",
            json!(1),
        ),
        (
            "/cases/0/profile_comparisons/0/tensors/0/nonfinite_pairs",
            json!(1),
        ),
        (
            "/cases/0/profile_comparisons/0/tensors/0/elements",
            json!(1),
        ),
        ("/cases/0/normal_after/median_ms", json!(0.001)),
        ("/cases/0/normal_before/losses/0", json!(0.0)),
        ("/cases/0/final_memory/resident_buffer_requests", json!(4)),
        ("/cases/0/profile/dispatches/0/index", json!(1)),
        ("/cases/0/profile/dispatches/0/phase", json!("backward")),
        ("/cases/0/profile/dispatches/0/family", json!("matrix")),
        (
            "/cases/0/profile/dispatches/0/timestamp_label",
            json!("wrong pass"),
        ),
        ("/cases/0/profile/dispatches/0/workgroups/0", json!(1)),
        (
            "/cases/0/profile/dispatches/0/timing_samples_ms/0",
            json!(10.0),
        ),
        ("/cases/0/profile/families/0/dispatch_count", json!(1)),
        (
            "/cases/0/profile/families/0/share_of_dispatch_median_sum_pct",
            json!(100.0),
        ),
        (
            "/cases/0/profile/measurement/gpu_total_samples_ms/0",
            json!(0.1),
        ),
        (
            "/cases/0/profile/measurement/instrumentation_wall_ratio",
            json!(1.0),
        ),
    ] {
        let mut changed = original.clone();
        *changed.pointer_mut(path).unwrap() = value;
        assert!(
            std::panic::catch_unwind(|| validate(&changed, 1)).is_err(),
            "accepted {path}"
        );
    }
    let mut changed = original.clone();
    changed["cases"][0]["profile_comparisons"]
        .as_array_mut()
        .unwrap()
        .pop();
    assert!(std::panic::catch_unwind(|| validate(&changed, 1)).is_err());
    let records: Vec<_> = RECORDS
        .iter()
        .map(|r| serde_json::from_str(r).unwrap())
        .collect();
    assert!(std::panic::catch_unwind(|| cohort(&records[..2])).is_err());
    let mut summary: Value = serde_json::from_str(SUMMARY).unwrap();
    summary["cases"][0]["processes"][0]["normal_drift_pct"] = json!(0.0);
    assert!(std::panic::catch_unwind(|| check_summary(&cohort(&records), &summary)).is_err());
}
