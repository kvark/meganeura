//! Explicit F16 and BF16 dense products share typed cooperative
//! kernel: the same schedule, F32 accumulation and F32 partials.

use meganeura::{
    CoopPolicy, DType, Graph, MatmulTile, TuneOptions,
    codegen::MatrixTypes,
    compile::{Kernel, ShaderEntry},
};

const M: usize = 33;
const N: usize = 65;
const K: usize = 1031;

/// Positive quarter-grid operands give an exact layout fixture. Exercise indexing,
/// vector bindings, output tails, addends, rounding and K-partial addressing.
#[test]
fn typed_staging_handles_vector_tails_and_split_output() {
    let gpu = crate::support::gpu::gpu();
    for storage in [DType::F16, DType::BF16] {
        if !meganeura::runtime::typed_matrix_supported(&gpu, storage) {
            continue;
        }
        // Aligned, vector-width and scalar row tails, a single element, and a
        // split K; the addend toggles on the scalar-tail shapes and rounding
        // on the plain layout.
        for (m, n, k, both_adds) in [
            (64, 64, 64, false),
            (36, 68, 35, false),
            (1, 1, 1, false),
            (33, 65, 31, true),
            (33, 65, 1031, true),
        ] {
            for layout in 0..3 {
                for add in [true, false] {
                    if !add && !both_adds {
                        continue;
                    }
                    for round in [false, true] {
                        if round && (storage != DType::BF16 || layout != 0) {
                            continue;
                        }
                        let g = crate::support::data::layout_graph(
                            storage,
                            layout,
                            add,
                            round,
                            [m, n, k],
                        );
                        let mut a = vec![0.0; m * k];
                        let b: Vec<f32> =
                            (0..k * n).map(|i| (i * 31 % 61 + 1) as f32 / 4.0).collect();
                        let bias: Vec<f32> = (0..m * n).map(|i| (i % 7) as f32 / 4.0).collect();
                        let mut expected = vec![0.0; m * n];
                        for row in 0..m {
                            let inner = (row * 137 + k - 1) % k;
                            a[if layout == 1 {
                                inner * m + row
                            } else {
                                row * k + inner
                            }] = 1.0;
                            for col in 0..n {
                                expected[row * n + col] =
                                    b[if layout == 2 {
                                        col * k + inner
                                    } else {
                                        inner * n + col
                                    }] + if add { bias[row * n + col] } else { 0.0 };
                            }
                        }
                        let mut session = meganeura::Session::with_context_opts(
                            meganeura::compile::compile(&g),
                            gpu.clone(),
                            meganeura::SessionOptions {
                                poison: true,
                                ..Default::default()
                            },
                        );
                        session.set_parameter("a", &a);
                        session.set_parameter("b", &b);
                        if add {
                            session.set_parameter("bias", &bias);
                        }
                        session.step();
                        session.wait();
                        let actual = session.read_output(m * n);
                        for (index, (&actual, &expected)) in
                            actual.iter().zip(&expected).enumerate()
                        {
                            assert_eq!(
                                actual, expected,
                                "{storage:?} {m}x{n}x{k} layout={layout} add={add} round={round} index={index}: {actual} != {expected}"
                            );
                        }
                    }
                }
            }
        }
    }
}

fn graph(storage: DType, layout: usize) -> meganeura::Graph {
    crate::support::data::layout_graph(storage, layout, false, false, [M, N, K])
}

/// K partitions accumulate F32 partials and reduce them before any BF16
/// rounding; a fused addend keeps the product whole and rounds its output.
#[test]
fn explicit_storage_selects_the_typed_schedule() {
    for (storage, add, round) in [
        (DType::F16, false, false),
        (DType::BF16, false, true),
        (DType::BF16, true, true),
    ] {
        for layout in 0..3 {
            let graph = crate::support::data::layout_graph(storage, layout, add, round, [M, N, K]);
            let plan = meganeura::compile::compile(&graph);
            let index = plan
                .dispatches
                .iter()
                .position(|d| matches!(d.kernel, Kernel::CooperativeTyped { .. }))
                .unwrap();
            let product = &plan.dispatches[index];
            let splits = if add { 1 } else { 4 };
            assert_eq!(
                product.shader,
                if add {
                    [
                        ShaderEntry::FusedMatMulAdd,
                        ShaderEntry::FusedMatMulATAdd,
                        ShaderEntry::FusedMatMulBTAdd,
                    ]
                } else {
                    [
                        ShaderEntry::MatMul,
                        ShaderEntry::MatMulAT,
                        ShaderEntry::MatMulBT,
                    ]
                }[layout]
            );
            assert_eq!(product.mnk(), Some((M as u32, N as u32, K as u32)));
            assert_eq!(product.workgroups, [2, 3, splits]);
            assert_eq!(
                product.kernel,
                Kernel::CooperativeTyped {
                    types: MatrixTypes {
                        output: if add { DType::BF16 } else { DType::F32 },
                        ..MatrixTypes::reduced(storage)
                    },
                    splits,
                    tile: meganeura::codegen::CheckedTile::BASE,
                }
            );
            if !add {
                let sum = &plan.dispatches[index + 1];
                assert_eq!(sum.shader, ShaderEntry::SumRows);
                assert_eq!(sum.params, [4, (M * N) as u32, 1, 0]);
                assert_eq!(sum.input_buffers, [product.output_buffer]);
                assert_eq!(
                    plan.buffers[product.output_buffer.0 as usize],
                    4 * M * N * 4
                );
                assert_ne!(sum.output_buffer, product.output_buffer);
            }
        }
    }
}

/// Values on a 1/64 grid are exact in both half formats. The oracle has no
/// rounding error, but native WMMA need not implement scalar F32 FMAs.
fn values(len: usize, multiplier: usize, modulus: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((i * multiplier % modulus) as f32 - (modulus / 2) as f32) / 64.0)
        .collect()
}

/// BF16 products and their partials meet the same budget in `bf16_split_k`.
#[test]
fn typed_products_meet_native_accuracy_budget() {
    let gpu = crate::support::gpu::gpu();
    crate::support::gpu::assert_protocol_headroom(&gpu);
    let a = values(M * K, 17, 61);
    let b = values(N * K, 31, 59);
    if !meganeura::runtime::typed_matrix_supported(&gpu, DType::F16) {
        eprintln!("native F16 matrices unavailable; skipping device-specific coverage");
        return;
    }
    let magnitude = |values: &[f32]| values.iter().map(|v| v.abs()).collect::<Vec<_>>();
    for layout in 0..3 {
        let mut config = crate::support::gpu::inference_config();
        config.tune = false;
        config.runtime.poison = true;
        let (mut session, _) = meganeura::build(&graph(DType::F16, layout), config);
        session.set_parameter("a", &a);
        session.set_parameter("b", &b);
        session.step();
        session.wait();
        let actual = session.read_output(M * N);
        let sums = crate::support::data::matmul_reference(layout, &a, &b, [M, N, K]);
        let sums_abs = crate::support::data::matmul_reference(
            layout,
            &magnitude(&a),
            &magnitude(&b),
            [M, N, K],
        );
        for (index, ((&got, sum), sum_abs)) in actual.iter().zip(sums).zip(sums_abs).enumerate() {
            let got = f64::from(got);
            let bound = meganeura::reference::native_matrix::error_bound(K, sum_abs).unwrap();
            assert!(
                got.is_finite() && (got - sum).abs() <= bound,
                "layout {layout} at {index}: {got} != {sum}, bound={bound}"
            );
        }
    }
}

/// A zero-valued product with a negative sign can perturb native WMMA even
/// though scalar IEEE arithmetic is exact. Exercise checked and staged kernels,
/// cancellation and scaling without making the regression AMD-specific.
#[test]
fn native_signed_products_and_cancellation() {
    let gpu = crate::support::gpu::gpu();
    for storage in [DType::F16, DType::BF16] {
        if !meganeura::runtime::typed_matrix_supported(&gpu, storage) {
            continue;
        }
        // Each kernel meets cancellation and both scale extremes.
        for (size, scale, cancel) in [
            (1, 2.0f32.powi(-10), false),
            (1, 2.0f32.powi(10), true),
            (16, 2.0f32.powi(-10), true),
            (16, 2.0f32.powi(10), false),
        ] {
            let mut g = Graph::new();
            let (a, b) = if storage == DType::F16 {
                (
                    g.parameter_f16("a", &[size, 16]),
                    g.parameter_f16("b", &[16, size]),
                )
            } else {
                let a = g.parameter("a", &[size, 16]);
                let b = g.parameter("b", &[16, size]);
                (g.to_bf16(a), g.to_bf16(b))
            };
            let y = g.matmul(a, b);
            g.set_outputs(vec![y]);
            let mut av = vec![0.; size * 16];
            let mut bv = vec![0.; 16 * size];
            for r in 0..size {
                av[r * 16] = scale;
                av[r * 16 + 1] = -scale;
            }
            for c in 0..size {
                bv[c] = 1.;
                bv[size + c] = f32::from(cancel);
            }
            let mut s = meganeura::Session::with_context_opts(
                meganeura::compile::compile(&g),
                gpu.clone(),
                meganeura::SessionOptions {
                    poison: true,
                    ..Default::default()
                },
            );
            s.set_parameter("a", &av);
            s.set_parameter("b", &bv);
            s.step();
            s.wait();
            let expected = if cancel { 0. } else { f64::from(scale) };
            let sum_abs = f64::from(scale) * if cancel { 2. } else { 1. };
            let bound = meganeura::reference::native_matrix::error_bound(16, sum_abs).unwrap();
            for got in s.read_output(size * size) {
                assert!(
                    got.is_finite() && (f64::from(got) - expected).abs() <= bound,
                    "{storage:?} size={size} scale={scale} cancel={cancel}: {got} != {expected}, bound={bound}"
                );
            }
        }
    }
}

/// The F16 storage contract includes subnormal operands. This is the 2^-15
/// case that B570 flushes without SPIR-V denormal preservation.
#[test]
fn f16_products_keep_subnormal_operands() {
    let gpu = crate::support::gpu::gpu();
    if !meganeura::runtime::typed_matrix_supported(&gpu, DType::F16) {
        eprintln!("native F16 matrices unavailable; skipping device-specific coverage");
        return;
    }
    let mut g = Graph::new();
    let a = g.parameter_f16("a", &[1, 32]);
    let b = g.parameter_f16("b", &[32, 1]);
    let y = g.matmul(a, b);
    g.set_outputs(vec![y]);
    let mut config = crate::support::gpu::inference_config();
    config.tune = false;
    let (mut session, _) = meganeura::build(&g, config);
    session.set_parameter("a", &[2.0f32.powi(-15); 32]);
    session.set_parameter("b", &[1.0; 32]);
    session.step();
    session.wait();
    assert_eq!(session.read_output(1), [2.0f32.powi(-10)]);
}

/// The message of the panic that rejects `plan` on `context`.
fn rejection(
    plan: meganeura::compile::ExecutionPlan,
    context: std::sync::Arc<blade_graphics::Context>,
) -> String {
    let panic = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        meganeura::Session::with_context(plan, context)
    })) {
        Ok(_) => panic!("unsupported device accepted a reduced-precision plan"),
        Err(panic) => panic,
    };
    panic
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_else(|| "non-string panic".into())
}

#[test]
fn unsupported_devices_report_the_first_missing_requirement() {
    let context = crate::support::gpu::gpu();
    let caps = context.capabilities();
    // F16: storage, a tile shape, then subnormals.
    if meganeura::runtime::typed_matrix_supported(&context, DType::F16) {
        eprintln!(
            "native F16 storage and matrices available; skipping unsupported-device coverage"
        );
    } else {
        let expected = if !caps.shader_float16 {
            "selected plan requires native F16 shader support"
        } else if meganeura::codegen::checked_cooperative_shape(
            &caps.cooperative_matrix.f16_f32_shapes,
        )
        .is_none()
        {
            "F16 matrix products require native 16x16x16, 8x16x16 or 8x8x8 F16/F32 cooperative matrices"
        } else {
            "keep subnormal F16 operands"
        };
        let plan = meganeura::compile::compile(&graph(DType::F16, 0));
        let message = rejection(plan, context.clone());
        assert!(message.contains(expected), "unexpected panic: {message}");
    }
    // BF16: storage, then a tile shape, before shader validation.
    if caps.shader_bfloat16
        && meganeura::codegen::checked_cooperative_shape(&caps.cooperative_matrix.bf16_f32_shapes)
            .is_some()
    {
        eprintln!(
            "native BF16 storage and matrices available; skipping unsupported-device coverage"
        );
        return;
    }
    let mut graph = Graph::new();
    let input = graph.input("x", &[3, 7]);
    let narrow = graph.to_bf16(input);
    let (output, expected) = if caps.shader_bfloat16 {
        let weight = graph.parameter("w", &[7, 5]);
        let weight = graph.to_bf16(weight);
        (
            graph.matmul(narrow, weight),
            "BF16 matrix products require native 16x16x16, 8x16x16 or 8x8x8 BF16/F32 cooperative matrices",
        )
    } else {
        (
            graph.from_bf16(narrow),
            "selected plan requires native BF16 shader support",
        )
    };
    graph.set_outputs(vec![output]);
    let message = rejection(meganeura::compile::compile(&graph), context.clone());
    assert!(message.contains(expected), "unexpected panic: {message}");
}

#[test]
fn typed_partitions_tune_without_changing_precision() {
    let gpu = crate::support::gpu::gpu();
    // Positive exact fixtures retain a bit-exact partition-equivalence check.
    // Signed arithmetic and its native error budget are covered separately.
    let a: Vec<_> = values(M * K, 17, 61).into_iter().map(f32::abs).collect();
    let b: Vec<_> = values(N * K, 31, 59).into_iter().map(f32::abs).collect();
    let mut probes = 0;
    for storage in [DType::F16, DType::BF16] {
        if !meganeura::runtime::typed_matrix_supported(&gpu, storage) {
            eprintln!("native {storage:?} matrices unavailable; skipping device-specific coverage");
            continue;
        }
        crate::support::gpu::assert_protocol_headroom(&gpu);
        let mut config = crate::support::gpu::inference_config();
        config.tune = false;
        config.runtime.poison = true;
        let (mut session, _) = meganeura::build(&graph(storage, 0), config);
        session.set_parameter("a", &a);
        session.set_parameter("b", &b);
        session.step();
        session.wait();
        let expected = session.read_output(M * N);
        // One bounded qualification of the class's partition counts, not a
        // performance sweep. Tuning runs on private typed scratch.
        let report = session
            .tune_with(TuneOptions {
                max_time: std::time::Duration::from_secs(15),
                max_classes: 1,
                warmup_runs: 1,
                sample_pairs: 4,
                dispatches_per_sample: 1,
                target_sample_time: None,
                ..crate::support::sequential_tuning()
            })
            .unwrap();
        assert_eq!(report.eligible_classes, 1);
        assert!(!report.outcomes.is_empty());
        for outcome in &report.outcomes {
            assert!(outcome.qualified, "{outcome:?}");
            assert!(matches!(
                outcome.candidate,
                MatmulTile::CooperativeChecked { .. }
            ));
            assert_eq!(
                outcome.class.matrix_types,
                Some(MatrixTypes::reduced(storage))
            );
            probes += 1;
        }
        // Exact values: any installed partition count gives the same output.
        session.step();
        session.wait();
        assert_eq!(session.read_output(M * N), expected);
    }
    println!("typed_partition_probes_qualified={probes}");
}

/// An unsplit ragged typed product has no candidate. Under a class limit
/// of one, it must not keep tuning away from an ordinary product.
#[test]
fn untunable_typed_classes_leave_the_class_limit_to_others() {
    let gpu = crate::support::gpu::gpu();
    if !meganeura::runtime::typed_matrix_supported(&gpu, DType::BF16) {
        eprintln!("native BF16 matrices unavailable; skipping device-specific coverage");
        return;
    }
    crate::support::gpu::assert_protocol_headroom(&gpu);
    let mut g = Graph::new();
    let a = g.input("a", &[513, 512]);
    let b = g.parameter("b", &[512, 512]);
    let (a, b) = (g.to_bf16(a), g.to_bf16(b));
    let typed = g.matmul(a, b);
    let x = g.input("x", &[33, 17]);
    let w = g.parameter("w", &[17, 65]);
    let ordinary = g.matmul(x, w);
    g.set_outputs(vec![typed, ordinary]);
    let mut config = crate::support::gpu::inference_config();
    config.tune = false;
    config.runtime.coop = CoopPolicy::Disabled;
    let (mut session, _) = meganeura::build(&g, config);
    let report = session
        .tune_with(TuneOptions {
            max_time: std::time::Duration::from_secs(15),
            max_classes: 1,
            warmup_runs: 1,
            sample_pairs: 4,
            dispatches_per_sample: 1,
            target_sample_time: None,
            // The 512-cubed BF16 product dominates the predicted plan, so a
            // time-share cutoff would rightly skip the small ordinary class.
            calibration: meganeura::TuneCalibration::Off,
            ..Default::default()
        })
        .unwrap();
    assert_eq!(report.eligible_classes, 1);
    assert!(!report.outcomes.is_empty());
    assert!(
        report
            .outcomes
            .iter()
            .all(|outcome| outcome.class.matrix_types.is_none())
    );
}
