//! Real-device tile qualification. These tests execute GPU timings;
//! run explicitly on an idle device with `--ignored --test-threads=1`.

use meganeura::graph::Graph;
use meganeura::{
    CoopPolicy, MatmulTile, Mode, SessionConfig, SessionOptions, TuneDecision, TuneOptions,
    TuneStaging, TuneStagingReuse, build,
};
use std::time::Duration;

fn matmul_graph(shader: &meganeura::compile::ShaderEntry, m: usize, n: usize, k: usize) -> Graph {
    use meganeura::compile::ShaderEntry;
    let mut graph = Graph::new();
    let a = graph.input(
        "a",
        &if *shader == ShaderEntry::MatMulAT {
            [k, m]
        } else {
            [m, k]
        },
    );
    let b = graph.input(
        "b",
        &if *shader == ShaderEntry::MatMulBT {
            [n, k]
        } else {
            [k, n]
        },
    );
    let y = match shader {
        ShaderEntry::MatMulAT => graph.matmul_at(a, b),
        ShaderEntry::MatMulBT => graph.matmul_bt(a, b),
        _ => graph.matmul(a, b),
    };
    let y = if *shader == ShaderEntry::FusedMatMulAdd {
        let addend = graph.input("addend", &[m, n]);
        graph.add(y, addend)
    } else {
        y
    };
    graph.set_outputs(vec![y]);
    graph
}

#[test]
#[ignore = "GPU tuning requires an idle device"]
fn tune_preserves_correctness() {
    let mut g = Graph::new();
    let x = g.input("x", &[64, 64]);
    let w1 = g.parameter("w1", &[64, 64]);
    let w2 = g.parameter("w2", &[64, 64]);
    let h = g.matmul(x, w1);
    let y = g.matmul(h, w2);
    g.set_outputs(vec![y]);

    let (mut s, _) = build(
        &g,
        SessionConfig {
            mode: Mode::Inference,
            runtime: SessionOptions {
                coop: CoopPolicy::Disabled,
                ..Default::default()
            },
            ..Default::default()
        },
    );

    let x_data: Vec<f32> = (0..64 * 64).map(|i| ((i % 13) as f32) * 0.05).collect();
    let w_data: Vec<f32> = (0..64 * 64)
        .map(|i| ((i % 7) as f32) * 0.02 - 0.05)
        .collect();
    s.set_input("x", &x_data);
    s.set_parameter("w1", &w_data);
    s.set_parameter("w2", &w_data);

    s.step();
    s.wait();
    let before = s.read_output(64 * 64);

    let report = s
        .tune_with(TuneOptions {
            max_time: Duration::from_secs(60),
            ..Default::default()
        })
        .unwrap();
    assert!(!report.outcomes.is_empty());
    for o in &report.outcomes {
        assert!(o.dispatches > 0);
        assert!(o.qualified, "qualification failed: {o:?}");
        assert!(o.baseline_median_ms.unwrap() > 0.0);
        assert!(o.candidate_median_ms.unwrap() > 0.0);
        let phases = o.phase_times.expect("new reports must account for phases");
        let preparation = phases.preparation.unwrap();
        let qualification = phases.qualification.unwrap();
        let warmup = phases.warmup.unwrap();
        let sampling = phases.sampling.unwrap();
        assert!(o.compile_time <= preparation);
        assert!(
            preparation + qualification + warmup + sampling + phases.cleanup.unwrap() <= o.elapsed
        );
        let prep = phases.preparation_breakdown.unwrap();
        assert_eq!(prep.pipelines.unwrap(), o.compile_time);
        let sum: Duration = [
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
        assert!(sum <= preparation);
        assert!(!qualification.is_zero() && !sampling.is_zero());
        let detail = phases.qualification_breakdown.unwrap();
        let parts = [
            detail.input_preparation,
            detail.upload_host_copy,
            detail.upload_transfer,
            detail.dispatch,
            detail.readback_transfer,
            detail.readback_host_copy,
            detail.validation,
        ];
        let sum: Duration = parts.into_iter().map(Option::unwrap).sum();
        assert!(sum <= qualification);
    }
    // Tuning must not execute the graph or overwrite the previous output.
    assert_eq!(
        before.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        s.read_output(64 * 64)
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>()
    );

    s.step();
    s.wait();
    let after = s.read_output(64 * 64);
    for (i, (&b, &a)) in before.iter().zip(after.iter()).enumerate() {
        assert!(
            (b - a).abs() <= 1e-4 * b.abs().max(1.0),
            "output[{i}] changed after tune: {b} vs {a}"
        );
    }
}

#[test]
#[ignore = "GPU tuning requires an idle device"]
fn tune_preserves_training_parameters_gradients_and_moments() {
    for staging in [TuneStaging::Shared, TuneStaging::Download] {
        for reuse in [TuneStagingReuse::Fresh, TuneStagingReuse::SameSize] {
            preserves_training_state(staging, reuse);
        }
    }
}

fn preserves_training_state(staging: TuneStaging, staging_reuse: TuneStagingReuse) {
    let mut graph = Graph::new();
    let x = graph.input("x", &[33, 17]);
    let weight = graph.parameter("weight", &[17, 65]);
    let y = graph.matmul(x, weight);
    let loss = graph.mean_all(y);
    graph.set_outputs(vec![loss]);
    let (mut session, _) = build(
        &graph,
        SessionConfig {
            runtime: SessionOptions {
                debug: true,
                coop: CoopPolicy::Disabled,
                ..Default::default()
            },
            ..Default::default()
        },
    );
    session.set_input("x", &vec![0.25; 33 * 17]);
    session.set_parameter("weight", &vec![0.125; 17 * 65]);
    session.set_adam(0.001, 0.9, 0.999, 1.0e-8);
    session.set_grad_accumulate(2);
    session.set_grad_clip_norm(0.05);
    session.set_grad_clip_every(3);
    let (mut control, _) = build(
        &graph,
        SessionConfig {
            gpu: Some(session.context()),
            runtime: SessionOptions {
                debug: true,
                coop: CoopPolicy::Disabled,
                ..Default::default()
            },
            ..Default::default()
        },
    );
    control.set_input("x", &vec![0.25; 33 * 17]);
    control.set_parameter("weight", &vec![0.125; 17 * 65]);
    control.set_adam(0.001, 0.9, 0.999, 1.0e-8);
    control.set_grad_accumulate(2);
    control.set_grad_clip_norm(0.05);
    control.set_grad_clip_every(3);
    control.step();
    control.wait();
    session.step();
    session.wait();
    let snapshot = |s: &meganeura::Session| {
        let mut values = vec![s.adam_step_count()];
        for (index, &bytes) in s.plan().buffers.iter().enumerate() {
            let mut data = vec![0.0; bytes / 4];
            s.read_buffer(meganeura::compile::BufferRef(index as u32), &mut data);
            values.extend(data.into_iter().map(f32::to_bits));
        }
        for (m, v) in s.read_adam_states(&["weight"]) {
            values.extend(m.into_iter().chain(v).map(f32::to_bits));
        }
        values
    };
    let before = snapshot(&session);
    let report = session
        .tune_with(TuneOptions {
            staging,
            staging_reuse,
            max_time: Duration::from_secs(60),
            ..Default::default()
        })
        .unwrap();
    assert!(report.outcomes.iter().any(|o| o.qualified), "{report:?}");
    assert_eq!(before, snapshot(&session));

    // Observe hidden accumulator/clip-cadence state through subsequent updates.
    for value in [0.5, -0.125, 0.75] {
        for s in [&mut control, &mut session] {
            s.set_input("x", &vec![value; 33 * 17]);
            s.step();
            s.wait();
        }
        assert_eq!(session.adam_step_count(), control.adam_step_count());
        let state = |s: &meganeura::Session| {
            let mut data = s.read_params(&["weight"]).remove(0);
            for (m, v) in s.read_adam_states(&["weight"]) {
                data.extend(m.into_iter().chain(v));
            }
            data
        };
        for (index, (a, b)) in state(&session).into_iter().zip(state(&control)).enumerate() {
            assert!(
                a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-6 + b.abs() * 1e-4,
                "optimizer state[{index}]: {a} != {b}"
            );
        }
    }
}

#[test]
#[ignore = "GPU tuning requires an idle device"]
fn tune_qualifies_all_four_scalar_entries_on_rectangular_edges() {
    for staging in [TuneStaging::Shared, TuneStaging::Download] {
        for reuse in [TuneStagingReuse::Fresh, TuneStagingReuse::SameSize] {
            qualify_scalar_entries(staging, reuse);
        }
    }
}

fn qualify_scalar_entries(staging: TuneStaging, staging_reuse: TuneStagingReuse) {
    use meganeura::compile::ShaderEntry;
    for shader in [
        ShaderEntry::MatMul,
        ShaderEntry::MatMulAT,
        ShaderEntry::MatMulBT,
        ShaderEntry::FusedMatMulAdd,
    ] {
        for (m, n, k) in [(33, 65, 17), (64, 32, 1), (17, 16, 65)] {
            let g = matmul_graph(&shader, m, n, k);
            let (mut session, _) = build(
                &g,
                SessionConfig {
                    mode: Mode::Inference,
                    runtime: SessionOptions {
                        coop: CoopPolicy::Disabled,
                        ..Default::default()
                    },
                    ..Default::default()
                },
            );
            let report = session
                .tune_with(TuneOptions {
                    staging,
                    staging_reuse,
                    max_time: Duration::from_secs(60),
                    ..Default::default()
                })
                .unwrap();
            assert_eq!(
                report.outcomes.len(),
                1,
                "{shader:?} {m}x{n}x{k}: {report:?}"
            );
            let outcome = &report.outcomes[0];
            assert_eq!(report.scratch.unwrap().retained_staging_bytes, 0);
            assert_eq!(
                report.scratch.unwrap().staging_allocations,
                report.scratch.unwrap().staging_releases
            );
            assert!(report.scratch.unwrap().peak_bytes <= report.options.max_scratch_bytes);
            assert_eq!(outcome.class.shader, shader);
            assert!(outcome.qualified, "{outcome:?}");
            assert!(matches!(
                outcome.decision,
                TuneDecision::FasterCandidate | TuneDecision::KeepBaseline
            ));
        }
    }
}

#[test]
#[ignore = "GPU tuning requires an idle device"]
fn tune_budget_skips_leave_selection_unchanged() {
    let mut graph = Graph::new();
    let a = graph.input("a", &[33, 17]);
    let b = graph.input("b", &[17, 65]);
    let y = graph.matmul(a, b);
    graph.set_outputs(vec![y]);
    let (mut session, _) = build(
        &graph,
        SessionConfig {
            mode: Mode::Inference,
            runtime: SessionOptions {
                coop: CoopPolicy::Disabled,
                ..Default::default()
            },
            ..Default::default()
        },
    );
    let before = session.dispatch_pipeline_keys();
    for options in [
        TuneOptions {
            max_classes: 0,
            ..Default::default()
        },
        TuneOptions {
            max_time: Duration::ZERO,
            ..Default::default()
        },
        TuneOptions {
            max_scratch_bytes: 0,
            ..Default::default()
        },
    ] {
        let report = session.tune_with(options).unwrap();
        assert_eq!(report.scratch.unwrap(), Default::default());
        assert_eq!(report.final_cleanup, None);
        assert_eq!(report.eligible_classes, 1);
        assert!(
            report.class_limit_reached
                || report.time_budget_exhausted
                || report
                    .outcomes
                    .iter()
                    .all(|o| o.decision == TuneDecision::ScratchLimit)
        );
        assert_eq!(before, session.dispatch_pipeline_keys());
        for outcome in &report.outcomes {
            assert_eq!(outcome.decision, TuneDecision::ScratchLimit);
            let phases = outcome.phase_times.unwrap();
            assert!(phases.preparation.unwrap() <= outcome.elapsed);
            assert_eq!(phases.qualification, None);
            assert_eq!(phases.warmup, None);
            assert_eq!(phases.sampling, None);
            assert_eq!(phases.cleanup, None);
            let preparation = phases.preparation_breakdown.unwrap();
            assert!(preparation.checks.is_some());
            assert_eq!(preparation.buffers, None);
            assert_eq!(preparation.staging, None);
        }
    }
}

#[test]
#[ignore = "GPU tuning requires an idle device"]
fn tuning_releases_retained_staging_after_a_later_scratch_skip() {
    let mut graph = Graph::new();
    let a = graph.input("a", &[32, 4096]);
    let b = graph.parameter("b", &[4096, 32]);
    let small = graph.matmul(a, b);
    let x = graph.input("x", &[1024, 1]);
    let y = graph.parameter("y", &[1, 1024]);
    let large = graph.matmul(x, y);
    graph.set_outputs(vec![small, large]);
    let (mut session, _) = build(
        &graph,
        SessionConfig {
            mode: Mode::Inference,
            runtime: SessionOptions {
                coop: CoopPolicy::Disabled,
                ..Default::default()
            },
            ..Default::default()
        },
    );
    let report = session
        .tune_with(TuneOptions {
            staging_reuse: TuneStagingReuse::SameSize,
            max_scratch_bytes: 2 * 1024 * 1024,
            max_time: Duration::from_secs(60),
            ..Default::default()
        })
        .unwrap();
    assert_eq!(report.outcomes.len(), 2);
    assert!(report.outcomes[0].qualified);
    assert_eq!(report.outcomes[1].decision, TuneDecision::ScratchLimit);
    assert_eq!(report.outcomes[1].scratch, None);
    let stats = report.scratch.unwrap();
    assert_eq!(stats.staging_allocations, 1);
    assert_eq!(stats.staging_releases, 1);
    assert_eq!(stats.staging_reuses, 0);
    assert_eq!(stats.retained_staging_bytes, 0);
    assert_eq!(stats.peak_bytes, 3 * 32 * 4096 * 4 + 32 * 32 * 4);
    assert!(report.final_cleanup.is_some());
    let keys = session.dispatch_pipeline_keys();
    let report = session
        .tune_with(TuneOptions {
            staging_reuse: TuneStagingReuse::SameSize,
            max_time: Duration::ZERO,
            ..Default::default()
        })
        .unwrap();
    assert_eq!(report.scratch.unwrap(), Default::default());
    assert_eq!(report.final_cleanup, None);
    assert_eq!(keys, session.dispatch_pipeline_keys());
}

#[test]
#[ignore = "GPU tuning requires an idle device"]
fn tune_native_cooperative_f32() {
    use meganeura::compile::ShaderEntry;
    // Deliberately fail on unsupported hardware: a scalar fallback must not
    // masquerade as native-f32 qualification. Run this test only on a device
    // advertising f32 tiles, separately from the portable qualification tests.
    let gpu = std::sync::Arc::new(meganeura::init_gpu_context().unwrap());
    assert!(
        gpu.capabilities().cooperative_matrix.f32_tile > 0,
        "native f32 matrix hardware required; this test cannot qualify a scalar fallback"
    );
    for shader in [
        ShaderEntry::MatMul,
        ShaderEntry::MatMulAT,
        ShaderEntry::MatMulBT,
        ShaderEntry::FusedMatMulAdd,
    ] {
        // Below occupancy threshold, already-promoted padded rows, above
        // the static large-shape veto, respectively. Never force a winner.
        for (m, n, k) in [(32, 32, 17), (97, 128, 17), (2048, 32, 17)] {
            let graph = matmul_graph(&shader, m, n, k);
            let (mut session, _) = build(
                &graph,
                SessionConfig {
                    gpu: Some(std::sync::Arc::clone(&gpu)),
                    mode: Mode::Inference,
                    ..Default::default()
                },
            );
            let report = session
                .tune_with(TuneOptions {
                    max_time: Duration::from_secs(60),
                    ..Default::default()
                })
                .unwrap();
            assert_eq!(report.visited_classes, 1, "{report:?}");
            assert_eq!(report.outcomes.len(), 2, "{report:?}");
            let native = report
                .outcomes
                .iter()
                .find(|o| {
                    matches!(o.initial, MatmulTile::CooperativeF32 { .. })
                        || matches!(o.candidate, MatmulTile::CooperativeF32 { .. })
                })
                .expect(
                    "native-f32 implementation must participate despite profitability thresholds",
                );
            assert!(native.qualified, "{native:?}");
            assert!(native.candidate_median_ms.unwrap() > 0.0);
        }
    }
}

/// The decode kernels must be reachable by measurement, quantized ones
/// included.
///
/// They were not: the search rejected every GEMV entry and every
/// reduced-storage weight, which between them are most of what a decode step
/// runs. A kernel nobody can select is a kernel nobody can tune, so this
/// checks the plumbing all the way through — the class is collected, a shape
/// candidate qualifies against the incumbent on real bytes, and the winner is
/// installed on the dispatch.
///
/// Which shape wins is a property of the device and deliberately not asserted.
#[test]
#[ignore = "GPU tuning requires an idle device"]
fn tune_reaches_quantized_gemv_and_installs_a_shape() {
    use meganeura::compile::{ShaderEntry, WeightFormat};
    use meganeura::{DType, GemvReduction, GemvShape};

    fn expected_format(dtype: DType) -> WeightFormat {
        WeightFormat::from_dtype(dtype)
    }

    // Decode-shaped: one row against a [K, N] weight, K and N multiples of
    // 256 so every K-quant's superblock divides the column.
    const K: usize = 512;
    const N: usize = 256;
    for dtype in [
        DType::F32,
        DType::F16,
        DType::Q4_0,
        DType::Q8_0,
        DType::Q40,
        DType::Q4K,
        DType::Q6K,
        DType::Q5K,
        DType::Q3K,
    ] {
        let mut g = Graph::new();
        let a = g.input("a", &[1, K]);
        let b = match dtype {
            DType::F32 => g.parameter("b", &[K, N]),
            DType::F16 => g.parameter_f16("b", &[K, N]),
            DType::Q4_0 => g.parameter_q4("b", &[K, N]),
            DType::Q8_0 => g.parameter_q8("b", &[K, N]),
            DType::Q40 => g.parameter_q40("b", &[K, N]),
            DType::Q4K => g.parameter_q4k("b", &[K, N]),
            DType::Q6K => g.parameter_q6k("b", &[K, N]),
            DType::Q5K => g.parameter_q5k("b", &[K, N]),
            DType::Q3K => g.parameter_q3k("b", &[K, N]),
            other => panic!("unhandled {other:?}"),
        };
        let y = g.matmul(a, b);
        g.set_outputs(vec![y]);

        let (mut session, _) = build(
            &g,
            SessionConfig {
                mode: Mode::Inference,
                runtime: SessionOptions {
                    coop: CoopPolicy::Disabled,
                    ..Default::default()
                },
                ..Default::default()
            },
        );
        assert!(
            session
                .plan()
                .dispatches
                .iter()
                .any(|d| d.shader == ShaderEntry::MatMulGemv),
            "{dtype:?}: the plan did not route through the K-split GEMV"
        );

        let report = session
            .tune_with(TuneOptions {
                max_time: Duration::from_secs(120),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(
            report.eligible_classes, 1,
            "{dtype:?}: the GEMV class was not collected"
        );
        assert_eq!(report.excluded_dispatches, 0, "{dtype:?}");
        assert!(
            !report.outcomes.is_empty(),
            "{dtype:?}: no shape was challenged"
        );
        for outcome in &report.outcomes {
            assert_eq!(outcome.class.shader, ShaderEntry::MatMulGemv);
            assert_eq!(outcome.class.weight_format, expected_format(dtype));
            for tile in [outcome.initial, outcome.candidate, outcome.selected] {
                assert!(
                    matches!(tile, MatmulTile::Gemv(_)),
                    "{dtype:?}: {tile:?} is not a GEMV shape"
                );
            }
            // Qualification is what proves the candidate agrees with the
            // kernel the plan already runs, on the same bytes. Without it a
            // measurement is just two numbers.
            assert!(
                outcome.qualified,
                "{dtype:?}: a shape failed to qualify: {:?}",
                outcome.failure
            );
            assert!(matches!(
                outcome.decision,
                TuneDecision::FasterCandidate | TuneDecision::KeepBaseline
            ));
            assert!(outcome.candidate_median_ms.unwrap() > 0.0);
        }
        // Both axes have to be reachable, or half the space is unsearchable.
        let tried: Vec<GemvShape> = report
            .outcomes
            .iter()
            .map(|o| match o.candidate {
                MatmulTile::Gemv(shape) => shape,
                other => unreachable!("{other:?}"),
            })
            .collect();
        assert!(
            tried.iter().any(|s| s.reduction == GemvReduction::Subgroup)
                && tried.iter().any(|s| s.reduction == GemvReduction::Tree),
            "{dtype:?}: only one reduction was challenged: {tried:?}"
        );
        assert!(
            tried.iter().map(|s| s.threads).collect::<Vec<_>>().len() > 1,
            "{dtype:?}: only one width was challenged"
        );

        // The winner has to reach the dispatch, or the search changed nothing.
        let MatmulTile::Gemv(winner) = report.outcomes.last().unwrap().selected else {
            unreachable!("checked above")
        };
        let installed: Vec<Option<GemvShape>> = session
            .plan()
            .dispatches
            .iter()
            .filter(|d| d.shader == ShaderEntry::MatMulGemv)
            .map(|d| d.gemv_shape)
            .collect();
        assert_eq!(
            installed,
            vec![Some(winner)],
            "{dtype:?}: the selected shape was not installed"
        );
        let first = &report.outcomes[0];
        eprintln!(
            "{dtype:?}: {:?} {:.3}ms -> {winner:?} {:.3}ms",
            first.initial,
            first.baseline_median_ms.unwrap(),
            report.outcomes.last().unwrap().candidate_median_ms.unwrap(),
        );
    }
}

/// Transposed-B GEMV must be correct at every shape, not just its default.
///
/// Two separate things make this worth its own test. The widths are only
/// reachable through measurement, and this kernel strides in vec4s where the
/// rest of the family strides in elements — a difference that left it
/// counting overlapping ranges at every width but the declared one. And the
/// qualification oracle addresses B by row only for entries it knows are
/// transposed, so admitting this one without telling it made a *correct*
/// kernel fail.
///
/// Qualification compares each shape against an f64 reference dot as well as
/// against the incumbent, so this is a numerical check at every width, not
/// just a compile.
#[test]
#[ignore = "GPU tuning requires an idle device"]
fn tune_covers_every_transposed_gemv_shape() {
    use meganeura::compile::ShaderEntry;
    use meganeura::{DType, GemvReduction, GemvShape};

    // K a multiple of 4 routes through the K-split GEMV-BT; N stays small so
    // the whole tournament is quick.
    const K: usize = 512;
    const N: usize = 12;
    for dtype in [DType::F32, DType::F16] {
        let mut g = Graph::new();
        let a = g.input("a", &[1, K]);
        let b = match dtype {
            DType::F32 => g.parameter("b", &[N, K]),
            DType::F16 => g.parameter_f16("b", &[N, K]),
            other => panic!("unhandled {other:?}"),
        };
        let y = g.matmul_bt(a, b);
        g.set_outputs(vec![y]);

        let (mut session, _) = build(
            &g,
            SessionConfig {
                mode: Mode::Inference,
                runtime: SessionOptions {
                    coop: CoopPolicy::Disabled,
                    ..Default::default()
                },
                ..Default::default()
            },
        );
        assert!(
            session
                .plan()
                .dispatches
                .iter()
                .any(|d| d.shader == ShaderEntry::MatMulGemvBT),
            "{dtype:?}: the plan did not route through the transposed K-split GEMV"
        );

        let report = session
            .tune_with(TuneOptions {
                max_time: Duration::from_secs(120),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(report.eligible_classes, 1, "{dtype:?}");
        for outcome in &report.outcomes {
            assert_eq!(outcome.class.shader, ShaderEntry::MatMulGemvBT);
            assert!(
                outcome.qualified,
                "{dtype:?}: {:?} failed to qualify: {:?}",
                outcome.candidate, outcome.failure
            );
        }

        // Every width and both reductions must actually have been run. A
        // candidate that silently failed to be offered would otherwise leave
        // this test asserting nothing about the shapes it never saw.
        let mut seen: Vec<GemvShape> = report
            .outcomes
            .iter()
            .map(|o| match o.candidate {
                MatmulTile::Gemv(shape) => shape,
                other => unreachable!("{other:?}"),
            })
            .collect();
        if let MatmulTile::Gemv(initial) = report.outcomes[0].initial {
            seen.push(initial);
        }
        for threads in [32, 64, 128, 256] {
            for reduction in [GemvReduction::Tree, GemvReduction::Subgroup] {
                let shape = GemvShape { threads, reduction };
                assert!(
                    seen.contains(&shape),
                    "{dtype:?}: {shape:?} was never measured; saw {seen:?}"
                );
            }
        }
    }
}
