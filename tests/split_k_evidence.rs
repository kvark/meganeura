//! CPU replay of full-sequence timing, including every rejected comparison.
#[path = "support/tuning_evidence.rs"]
#[allow(dead_code)]
mod evidence;
#[path = "../examples/support/tuning_measurement.rs"]
mod measurement;
#[path = "support/telemetry.rs"]
#[allow(dead_code)]
mod telemetry;

use evidence::close;
use measurement::{PairedTiming, TensorComparison, median};
use meganeura::{MatmulTile, TuneDecision, TuneReport, compile::ShaderEntry};
use serde_json::{Value, json};
use std::time::Duration;

const SOURCE: &str = "8d77f90d4565aa150b8b5f70a50a8d72331cabb4";
const EXECUTABLE: &str = "34f94f2fcdd3ba90ab7dc5f2de81416f5e6720520f9d45ee41d676133119d301";
const CASES: [(&str, [u32; 10], usize); 4] = [
    ("spatial-7x7", [1, 3, 224, 224, 64, 7, 7, 2, 3, 3], 23156236),
    ("pointwise", [1, 256, 56, 56, 64, 1, 1, 1, 0, 0], 11370508),
    ("rectangular-tail", [3, 5, 7, 9, 7, 2, 3, 1, 1, 0], 38400),
    ("long-tail", [2, 3, 1, 32771, 5, 1, 1, 1, 0, 0], 9962516),
];

fn record(seed: usize) -> Value {
    serde_json::from_str(
        &std::fs::read_to_string(format!(
            "{}/docs/experiments/split-k-sequence-2026-09-06/run-{seed:02}.json",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap(),
    )
    .unwrap()
}

fn replay(record: &Value, seed: usize) {
    assert_eq!(record["protocol"], "split-k-sequence-v1");
    assert_eq!(record["status"], "complete");
    let meta = &record["metadata"];
    assert_eq!(meta["revision"], SOURCE);
    assert_eq!(meta["executable_sha256"], EXECUTABLE);
    assert_eq!(
        meta["cargo_lock_sha256"],
        "72cecddd0f090e82d5db46bc6f7f80429a058b2afa0eaec84a9b27e7340a4a80"
    );
    assert_eq!(meta["tracked_source_clean"], true);
    assert_eq!(meta["seed"], seed);
    assert_eq!(meta["device"], "NVIDIA GeForce RTX 5070");
    assert_eq!(meta["driver"], "595.71.05");
    assert_eq!(meta["rustc"], "rustc 1.98.0 (88d9e12ae 2026-08-18)");
    assert!(meta["rustflags"].is_null());
    telemetry::check_telemetry(record, false);
    let mut previous = meta["started_unix_ms"].as_u64().unwrap();
    let finish = record["finished_unix_ms"].as_u64().unwrap();
    let cases = record["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 4);
    for (position, case) in cases.iter().enumerate() {
        let (name, shape, resident) = CASES[(position + seed - 1) % 4];
        assert_eq!(case["name"], name);
        assert_eq!(case["shape"], json!(shape));
        assert_eq!(case["status"], "complete");
        assert_eq!(case["plan_and_state_unchanged"], true);
        assert_eq!(case["resident_buffer_requests"], resident);
        assert_eq!(case["adam_step"], 0);
        assert_eq!(case["adam_bytes"], 0);
        let start = case["host_start"]["unix_ms"].as_u64().unwrap();
        let end = case["host_finish"]["unix_ms"].as_u64().unwrap();
        assert!(previous <= start && start < end && end <= finish);
        previous = end;
        let [batch, ci, h, w, co, kh, kw, stride, ph, pw] = shape;
        let (oh, ow) = (
            (h + 2 * ph - kh) / stride + 1,
            (w + 2 * pw - kw) / stride + 1,
        );
        let (m, n, k) = (co, ci * kh * kw, batch * oh * ow);
        let (nx, ny, nw) = (
            (batch * ci * h * w) as usize,
            (batch * co * oh * ow) as usize,
            (m * n) as usize,
        );
        let dispatch: meganeura::compile::Dispatch =
            serde_json::from_value(case["dispatch"].clone()).unwrap();
        assert_eq!(dispatch.shader, ShaderEntry::Conv2dGradWeightGemmSmall);
        assert_eq!(
            case["pipeline_keys"][case["dispatch_index"].as_u64().unwrap() as usize],
            "Conv2dGradWeightGemmSmall:scalar"
        );
        assert_eq!(
            dispatch.params,
            [batch, ci, h, w, co, kh, kw, stride, ph, oh, ow, pw]
        );
        assert_eq!(dispatch.workgroups, [n.div_ceil(32), m.div_ceil(32), 1]);
        assert!(
            dispatch.requires_full_precision && !dispatch.use_coop && !dispatch.use_small_tiles
        );
        let state: Vec<TensorComparison> =
            serde_json::from_value(case["state_comparisons"].clone()).unwrap();
        assert_eq!(state.len(), 6);
        for (tensor, (name, elements)) in state.iter().zip([
            ("input.x", nx),
            ("input.dy", ny),
            ("parameter.w", nw),
            ("gradient.w", nw),
            ("loss_partials", 1),
            ("loss", 1),
        ]) {
            assert_eq!(tensor.name, name);
            assert_eq!(tensor.elements, elements);
            assert!(tensor.exact && tensor.passed);
            assert_eq!(tensor.nonfinite_pairs, 0);
            assert_eq!(tensor.elementwise_failures, 0);
            assert_eq!(tensor.max_abs_error, 0.0);
            assert_eq!(tensor.error_sum_sq, 0.0);
            assert_eq!(tensor.relative_l2, 0.0);
            assert!(tensor.reference_sum_sq > 0.0);
            assert_eq!(tensor.reference_sum_sq, tensor.candidate_sum_sq);
        }
        let report: TuneReport = serde_json::from_value(case["report"].clone()).unwrap();
        assert_eq!(
            serde_json::to_value(&report.options).unwrap(),
            json!({
                "scope":"All","max_classes":8,"max_scratch_bytes":67108864,"max_time":{"secs":120,"nanos":0},
                "staging":"Download","staging_reuse":"SameSize","sample_pairs":6,"dispatches_per_sample":16,"warmup_runs":1,"min_improvement":0.05
            })
        );
        assert_eq!(report.eligible_classes, 1);
        assert_eq!(report.visited_classes, 1);
        assert_eq!(
            report.excluded_dispatches + 1,
            case["pipeline_keys"].as_array().unwrap().len()
        );
        assert!(!report.class_limit_reached && !report.time_budget_exhausted);
        assert_eq!(report.outcomes.len(), 4);
        assert!(report.elapsed.as_secs_f64() * 1000.0 <= (end - start + 1) as f64);
        let mut order = [2, 3, 4, 8];
        order.rotate_left(seed - 1);
        assert_eq!(case["splits"], json!(order));
        let (mut last_staging, mut allocations, mut peak) = (0, 0, 0);
        for (outcome, splits) in report.outcomes.iter().zip(order) {
            assert_eq!(outcome.candidate_split_k, Some(splits));
            assert_eq!(outcome.initial, MatmulTile::Tile32);
            assert_eq!(outcome.initial, outcome.candidate);
            assert_eq!(outcome.selected, outcome.initial);
            assert_eq!(outcome.dispatches, 1);
            let class = &outcome.class;
            assert_eq!(class.shader, ShaderEntry::Conv2dGradWeightGemm);
            assert_eq!((class.m, class.n, class.k), (m, n, k));
            assert_eq!(class.binding_bytes, [ny * 4, nx * 4, nw * 4]);
            assert_eq!(class.device_local, [true, false, false, true]);
            assert!(class.requires_full_precision);
            let s = class.conv2d.unwrap();
            assert_eq!(
                [
                    s.batch,
                    s.in_channels,
                    s.in_h,
                    s.in_w,
                    s.out_channels,
                    s.kernel_h,
                    s.kernel_w,
                    s.stride,
                    s.padding_h,
                    s.padding_w
                ],
                shape
            );
            assert_eq!((s.out_h, s.out_w), (oh, ow));
            let scratch = outcome.scratch.as_ref().unwrap();
            assert_eq!(
                scratch.binding_bytes,
                [ny * 4, nx * 4, nw * 4, nw * splits as usize * 4]
            );
            assert_eq!(
                scratch.staging_bytes,
                *scratch.binding_bytes.iter().max().unwrap()
            );
            assert_eq!(
                scratch.staging_reused,
                last_staging == scratch.staging_bytes
            );
            allocations += usize::from(!scratch.staging_reused);
            last_staging = scratch.staging_bytes;
            peak = peak.max(scratch.binding_bytes.iter().sum::<usize>() + scratch.staging_bytes);
            let phases = outcome.phase_times.unwrap();
            let total: Duration = [
                phases.preparation,
                phases.qualification,
                phases.warmup,
                phases.sampling,
                phases.cleanup,
            ]
            .into_iter()
            .flatten()
            .sum();
            assert!(total <= outcome.elapsed && outcome.elapsed <= report.elapsed);
            assert_eq!(
                outcome.compile_time,
                phases.preparation_breakdown.unwrap().pipelines.unwrap()
            );
            let prep = phases.preparation_breakdown.unwrap();
            let nested: Duration = [
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
            assert!(nested <= phases.preparation.unwrap());
            let q = phases.qualification_breakdown.unwrap();
            let nested: Duration = [
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
            assert!(nested <= phases.qualification.unwrap());
            if outcome.decision == TuneDecision::InvalidOutput {
                assert!(!outcome.qualified);
                assert!(outcome.failure.as_ref().is_some_and(|s| !s.is_empty()));
                assert!(outcome.baseline_ms.is_empty() && outcome.candidate_ms.is_empty());
                assert!(
                    outcome.baseline_median_ms.is_none()
                        && outcome.candidate_median_ms.is_none()
                        && outcome.noise_margin_ms.is_none()
                );
                assert!(phases.warmup.is_none() && phases.sampling.is_none());
            } else {
                assert!(outcome.qualified && outcome.failure.is_none());
                assert_eq!(outcome.baseline_ms.len(), 6);
                assert_eq!(outcome.candidate_ms.len(), 6);
                let timing =
                    PairedTiming::new(&outcome.baseline_ms, &outcome.candidate_ms).unwrap();
                close(
                    timing.baseline_median_ms,
                    outcome.baseline_median_ms.unwrap(),
                );
                close(timing.tuned_median_ms, outcome.candidate_median_ms.unwrap());
                close(
                    timing.paired_noise_margin_ms,
                    outcome.noise_margin_ms.unwrap(),
                );
                assert_eq!(
                    outcome.decision,
                    if timing.improvement_exceeds_guard {
                        TuneDecision::FasterCandidate
                    } else {
                        TuneDecision::KeepBaseline
                    }
                );
                assert!(phases.warmup.is_some() && phases.sampling.is_some());
            }
        }
        let stats = report.scratch.unwrap();
        assert_eq!(stats.peak_bytes, peak);
        assert!(peak <= report.options.max_scratch_bytes);
        assert_eq!(stats.retained_staging_bytes, 0);
        assert_eq!(stats.staging_allocations, allocations);
        assert_eq!(stats.staging_releases, allocations);
        assert_eq!(stats.staging_reuses, 4 - allocations);
        let elapsed: Duration = report.outcomes.iter().map(|o| o.elapsed).sum();
        assert!(elapsed + report.final_cleanup.unwrap() <= report.elapsed);
    }
}

#[test]
fn retained_sequences_replay_without_promoting_rejected_or_incomplete_work() {
    let records: Vec<_> = (1..=4).map(record).collect();
    let summary: Value = serde_json::from_str(include_str!(
        "../docs/experiments/split-k-sequence-2026-09-06/summary.json"
    ))
    .unwrap();
    assert_eq!(summary["source"], SOURCE);
    assert_eq!(summary["executable_sha256"], EXECUTABLE);
    for (i, r) in records.iter().enumerate() {
        replay(r, i + 1);
        if i > 0 {
            assert!(
                records[i - 1]["finished_unix_ms"].as_u64().unwrap()
                    < r["metadata"]["started_unix_ms"].as_u64().unwrap()
            );
        }
    }
    for (index, (name, _, resident)) in CASES.iter().enumerate() {
        let cases: Vec<_> = records
            .iter()
            .map(|r| {
                r["cases"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|c| c["name"] == *name)
                    .unwrap()
            })
            .collect();
        let reports: Vec<TuneReport> = cases
            .iter()
            .map(|c| serde_json::from_value(c["report"].clone()).unwrap())
            .collect();
        let s = &summary["cases"][index];
        assert_eq!(s["name"], *name);
        assert_eq!(s["resident_buffer_requests"], *resident);
        assert_eq!(
            s["scratch_peak_bytes"],
            reports[0].scratch.unwrap().peak_bytes
        );
        for (field, times) in [
            (
                "median_search_ms",
                reports
                    .iter()
                    .map(|r| r.elapsed.as_secs_f64() * 1e3)
                    .collect::<Vec<_>>(),
            ),
            (
                "median_qualification_ms",
                reports
                    .iter()
                    .map(|r| {
                        r.outcomes
                            .iter()
                            .map(|o| {
                                o.phase_times.unwrap().qualification.unwrap().as_secs_f64() * 1e3
                            })
                            .sum()
                    })
                    .collect(),
            ),
            (
                "median_validation_ms",
                reports
                    .iter()
                    .map(|r| {
                        r.outcomes
                            .iter()
                            .map(|o| {
                                o.phase_times
                                    .unwrap()
                                    .qualification_breakdown
                                    .unwrap()
                                    .validation
                                    .unwrap()
                                    .as_secs_f64()
                                    * 1e3
                            })
                            .sum()
                    })
                    .collect(),
            ),
        ] {
            close(median(&times), s[field].as_f64().unwrap());
        }
        for (i, splits) in [2, 3, 4, 8].into_iter().enumerate() {
            let outcomes: Vec<_> = reports
                .iter()
                .map(|r| {
                    r.outcomes
                        .iter()
                        .find(|o| o.candidate_split_k == Some(splits))
                        .unwrap()
                })
                .collect();
            let row = &s["comparisons"][i];
            assert_eq!(row["splits"], splits);
            assert_eq!(
                row["qualified"],
                outcomes.iter().filter(|o| o.qualified).count()
            );
            assert_eq!(
                row["wins"],
                outcomes
                    .iter()
                    .filter(|o| o.decision == TuneDecision::FasterCandidate)
                    .count()
            );
            for (seed, o) in outcomes.iter().enumerate() {
                let p = &row["processes"][seed];
                assert_eq!(p["seed"], seed + 1);
                assert_eq!(p["splits"], splits);
                assert_eq!(p["qualified"], o.qualified);
                assert_eq!(p["decision"], serde_json::to_value(o.decision).unwrap());
                assert_eq!(p["failure"], json!(o.failure));
                for (key, value) in [
                    ("baseline_ms", o.baseline_median_ms),
                    ("candidate_ms", o.candidate_median_ms),
                    ("noise_margin_ms", o.noise_margin_ms),
                ] {
                    match value {
                        Some(v) => close(v, p[key].as_f64().unwrap()),
                        None => assert!(p[key].is_null()),
                    }
                }
            }
            for (key, values) in [
                (
                    "median_baseline_ms",
                    outcomes
                        .iter()
                        .filter_map(|o| o.baseline_median_ms)
                        .collect::<Vec<_>>(),
                ),
                (
                    "median_candidate_ms",
                    outcomes
                        .iter()
                        .filter_map(|o| o.candidate_median_ms)
                        .collect(),
                ),
                (
                    "median_process_ratio",
                    outcomes
                        .iter()
                        .filter_map(|o| Some(o.baseline_median_ms? / o.candidate_median_ms?))
                        .collect(),
                ),
            ] {
                if values.is_empty() {
                    assert!(row[key].is_null());
                } else {
                    close(median(&values), row[key].as_f64().unwrap());
                }
            }
        }
    }
}

#[test]
fn replay_rejects_changed_sequence_identity_bytes_timings_and_state() {
    let original = record(1);
    for (pointer, value) in [
        ("/metadata/revision", json!("different")),
        ("/cases/0/pipeline_keys/7", json!("different")),
        ("/cases/0/report/outcomes/0/candidate_split_k", json!(8)),
        (
            "/cases/0/report/outcomes/0/scratch/binding_bytes/3",
            json!(4),
        ),
        ("/cases/0/report/outcomes/0/qualified", json!(true)),
        (
            "/cases/0/report/outcomes/0/baseline_ms",
            json!(vec![1.0; 6]),
        ),
        ("/cases/0/report/scratch/retained_staging_bytes", json!(4)),
        ("/cases/0/state_comparisons/0/error_sum_sq", json!(1.0)),
        ("/cases/2/report/outcomes/0/candidate_ms/0", json!(0.0)),
        ("/cases/3/report/outcomes/0/decision", json!("KeepBaseline")),
    ] {
        let mut changed = original.clone();
        *changed.pointer_mut(pointer).unwrap() = value;
        assert!(
            std::panic::catch_unwind(|| replay(&changed, 1)).is_err(),
            "accepted {pointer}"
        );
    }
}
