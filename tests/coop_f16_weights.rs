//! f16 weight storage through f16-input cooperative tiles: plain and
//! transposed-B products must match an f64 reference that uses the same
//! f16-rounded weights, and select cooperative kernels on supported hardware.

use crate::support::data::relative_l2;
use meganeura::{CoopPolicy, Graph, Session};

// Enough output tiles for automatic f16 promotion, with partial M/K tiles and
// non-vector-aligned row strides. Keep the CPU oracle small.
const M: usize = 257;
const K: usize = 65;
const N: usize = 512;

fn build(transposed: bool, relu: bool, coop: CoopPolicy) -> Session {
    let mut graph = Graph::new();
    let x = graph.input("x", &[M, K]);
    let output = if transposed {
        let w = graph.parameter("w", &[N, K]);
        graph.matmul_bt(x, w)
    } else {
        let w = graph.parameter("w", &[K, N]);
        graph.matmul(x, w)
    };
    let output = if relu { graph.relu(output) } else { output };
    graph.set_outputs(vec![output]);
    assert_eq!(graph.store_weights_f16(), vec!["w".to_owned()]);
    let mut config = crate::support::gpu::inference_config();
    config.runtime.coop = coop;
    config.tune = false;
    meganeura::build(&graph, config).0
}

fn expects_cooperative_f16_weights(session: &Session) -> bool {
    let caps = meganeura::runtime::auto_tune(&session.context(), 0).coop_caps;
    eprintln!(
        "{}: f16 tile={}, f32 tile={}",
        session.context().device_information().device_name,
        caps.f16_tile,
        caps.f32_tile
    );
    // Session policy currently prefers native-f32 tiles, which do not read
    // stored f16 weights. Do not mistake a selection regression on f16-only
    // devices for missing hardware support.
    caps.f32_tile == 0 && caps.f16_tile != 0
}

fn reference(transposed: bool, x: &[f32], w: &[f32]) -> Vec<f32> {
    let w: Vec<f64> = w
        .iter()
        .map(|&v| f64::from(half::f16::from_f32(v).to_f32()))
        .collect();
    let mut out = vec![0.0f32; M * N];
    for i in 0..M {
        for j in 0..N {
            let mut sum = 0.0f64;
            for p in 0..K {
                let weight = if transposed {
                    w[j * K + p]
                } else {
                    w[p * N + j]
                };
                sum += f64::from(x[i * K + p]) * weight;
            }
            out[i * N + j] = sum as f32;
        }
    }
    out
}

#[test]
fn coop_matmul_reads_f16_weights() {
    let x: Vec<f32> = (0..M * K)
        .map(|i| ((i * 37 % 211) as f32 - 105.0) * 0.004)
        .collect();
    for (transposed, relu) in [(false, false), (true, false), (false, true), (true, true)] {
        let w: Vec<f32> = (0..K * N)
            .map(|i| ((i * 13 % 97) as f32 - 48.0) * 0.003)
            .collect();
        let mut want = reference(transposed, &x, &w);
        if relu {
            want.iter_mut().for_each(|v| *v = v.max(0.0));
        }
        for coop in [CoopPolicy::AllowF16, CoopPolicy::Disabled] {
            let mut session = build(transposed, relu, coop);
            let cooperative = session.plan().dispatches.iter().any(|dispatch| {
                dispatch.use_coop()
                    && dispatch.weight_format == meganeura::compile::WeightFormat::F16
            });
            assert_eq!(
                cooperative,
                coop == CoopPolicy::AllowF16 && expects_cooperative_f16_weights(&session),
                "unexpected kernel selection: {:?}",
                session.dispatch_pipeline_keys()
            );
            assert_eq!(
                session.dispatch_pipeline_keys()[0].contains("cooperative"),
                cooperative,
                "the selected pipeline must implement the plan's kernel"
            );
            if relu && cooperative {
                assert!(session.plan().dispatches[0].matmul_epilogue.is_some());
            }
            session.set_parameter("w", &w);
            session.set_input("x", &x);
            session.step();
            session.wait();
            let mut got = vec![0.0f32; M * N];
            session.read_output_by_index(0, &mut got);
            let error = relative_l2(&got, &want);
            eprintln!(
                "f16 weights: transposed={transposed} relu={relu} coop={coop:?} selected={cooperative} relative_l2={error:.3e}"
            );
            // f16-input tiles also round x; scalar kernels compute in f32.
            let bound = if cooperative { 2e-3 } else { 1e-5 };
            assert!(
                error < bound,
                "transposed={transposed} relu={relu} coop={coop:?}: relative L2 {error:.3e} >= {bound:.0e}"
            );
        }
    }
}

#[test]
fn mixed_storage_rmsnorm_prologues_use_distinct_pipelines() {
    let mut graph = Graph::new();
    let x = graph.input("x", &[M, K]);
    let mut outputs = Vec::new();
    let mut full_weight = None;
    for name in ["full", "half"] {
        let norm = graph.parameter(&format!("{name}.norm"), &[K]);
        let normalized = graph.rms_norm(x, norm, 1e-5);
        let weight = graph.parameter(&format!("{name}.weight"), &[K, N]);
        outputs.push(graph.matmul(normalized, weight));
        if name == "full" {
            full_weight = Some(weight);
        }
    }
    // Exposed parameters must stay f32; only the other projection is converted.
    outputs.push(full_weight.unwrap());
    graph.set_outputs(outputs);
    assert_eq!(graph.store_weights_f16(), ["half.weight"]);
    let create = |coop| {
        let mut config = crate::support::gpu::inference_config();
        config.runtime.coop = coop;
        config.tune = false;
        meganeura::build(&graph, config).0
    };
    let mut scalar = create(CoopPolicy::Disabled);
    let mut candidate = create(CoopPolicy::AllowF16);
    if expects_cooperative_f16_weights(&candidate) {
        let keys = candidate.dispatch_pipeline_keys();
        let prologue_keys: Vec<_> = candidate
            .plan()
            .dispatches
            .iter()
            .zip(&keys)
            .filter_map(|(d, key)| d.matmul_prologue.as_ref().map(|_| key))
            .collect();
        assert_eq!(prologue_keys.len(), 2, "both projections must fuse");
        assert!(
            prologue_keys
                .iter()
                .all(|key| key.contains("cooperative-prologue"))
        );
        assert_ne!(prologue_keys[0], prologue_keys[1]);
    }
    for update in 0..2 {
        let x: Vec<_> = (0..M * K)
            .map(|i| ((i * 37 + update * 3) % 211) as f32 * 0.004 - 0.42)
            .collect();
        for session in [&mut scalar, &mut candidate] {
            session.set_input("x", &x);
            for (seed, name) in ["full", "half"].into_iter().enumerate() {
                let norm: Vec<_> = (0..K)
                    .map(|i| 0.75 + ((i + seed + update) % 13) as f32 * 0.03)
                    .collect();
                let weight: Vec<_> = (0..K * N)
                    .map(|i| ((i * 13 + seed * 7 + update) % 97) as f32 * 0.003 - 0.144)
                    .collect();
                session.set_parameter(&format!("{name}.norm"), &norm);
                session.set_parameter(&format!("{name}.weight"), &weight);
            }
            session.step();
            session.wait();
        }
        for output in 0..2 {
            let mut want = vec![0.0; M * N];
            let mut got = vec![0.0; M * N];
            scalar.read_output_by_index(output, &mut want);
            candidate.read_output_by_index(output, &mut got);
            let error = relative_l2(&got, &want);
            assert!(
                error < 2e-3,
                "output={output}, update={update}: {error:.3e}"
            );
        }
    }
}

/// Block-quantized prefill matmuls are admitted as subgroup-tiled cooperative
/// F16 GEMM tuner candidates under explicit `CoopPolicy::AllowF16` on supported
/// hardware (aligned direct-store and ragged M/N, plain, fused-add, and
/// unary-epilogue forms) and match the scalar F32 reference within F16 operand
/// tolerance. Untuned sessions and default `CoopPolicy::Auto` (even when tuned)
/// stay on the exact-F32 scalar path so large finite activations (`|x| > 65504`)
/// never overflow F16 staging.
#[test]
fn coop_quantized_prefill_matches_scalar_reference() {
    use meganeura::compile::{Kernel, WeightFormat};
    use meganeura::tune::{MatmulTile, TuneOptions, TuneStrategy};
    use std::time::Duration;

    let gpu = crate::support::gpu::gpu();
    let caps = gpu.capabilities();
    // Match `native_subgroup_shape` + `supports_quantized`:
    // requires qualified typed F16 matrices, a supported subgroup width (16/32/64),
    // and enough workgroup memory for the staged operand tiles + scratch.
    let supports_f16 = meganeura::runtime::typed_matrix_supported(&gpu, meganeura::DType::F16)
        && matches!(caps.subgroup_size, 16 | 32 | 64)
        && caps.max_compute_shared_memory_size >= 16 * 1024;

    let tune_opts = |strategy: TuneStrategy| TuneOptions {
        strategy,
        max_time: Duration::from_secs(10),
        warmup_runs: 1,
        sample_pairs: 4,
        target_sample_time: Some(Duration::from_nanos(1)),
        min_improvement: 0.01,
        ..Default::default()
    };

    // 0. Large-finite-input oracle under default `CoopPolicy::Auto` (untuned
    //    AND tuned): `131072.0` exceeds F16 max (`65504`), so Auto must never
    //    admit quantized F16 cooperative staging and must produce exact finite
    //    `16384.0` in every output.
    {
        let (m, k, n) = (64usize, 256usize, 512usize);
        let mut graph = Graph::new();
        let x = graph.input("x", &[m, k]);
        let w = graph.parameter_q4("w", &[k, n]);
        let y = graph.matmul(x, w);
        graph.set_outputs(vec![y]);
        let mut config = crate::support::gpu::inference_config();
        config.runtime.coop = CoopPolicy::Auto;
        config.tune = false;
        let mut session = meganeura::build(&graph, config).0;
        let report = session
            .tune_with(tune_opts(TuneStrategy::Sequential))
            .unwrap();
        assert!(
            report
                .outcomes
                .iter()
                .all(|o| !matches!(o.candidate, MatmulTile::CooperativeWorkgroup { .. })),
            "Auto tuning must not admit quantized cooperative F16 candidates: {:?}",
            report.outcomes
        );
        assert!(
            session
                .plan()
                .dispatches
                .iter()
                .all(|d| !matches!(d.kernel, Kernel::CooperativeWorkgroup { .. })),
            "Auto must not select quantized F16 cooperative staging: {:?}",
            session.dispatch_pipeline_keys()
        );
        let mut x_data = vec![0.0f32; m * k];
        for row in 0..m {
            x_data[row * k] = 131072.0;
        }
        let w_data = vec![0.125f32; k * n];
        session.set_input("x", &x_data);
        session.set_parameter("w", &w_data);
        session.step();
        session.wait();
        let got = session.read_output(m * n);
        assert!(
            got.iter()
                .all(|&v| v.is_finite() && (v - 16384.0).abs() < 1e-2),
            "large finite activation overflowed or diverged under Auto: first={:?}",
            &got[..got.len().min(8)]
        );
    }

    // 1. Q4 & Q8 under explicit `CoopPolicy::AllowF16`: aligned prefill
    //    (64×256×512, direct-store) and ragged M/N (65×256×368) with plain,
    //    fused-add, and ReLU epilogue. Untuned stays scalar; tuning qualifies
    //    quantized cooperative candidates on supported hardware.
    let cases = [
        (
            WeightFormat::Q4,
            64usize,
            256usize,
            512usize,
            TuneStrategy::Race,
            false,
            false,
        ),
        (
            WeightFormat::Q8,
            64,
            256,
            512,
            TuneStrategy::Sequential,
            false,
            false,
        ),
        (
            WeightFormat::Q4,
            65,
            256,
            368,
            TuneStrategy::Sequential,
            false,
            false,
        ),
        (
            WeightFormat::Q8,
            65,
            256,
            368,
            TuneStrategy::Sequential,
            true,
            false,
        ),
        (
            WeightFormat::Q4,
            65,
            256,
            368,
            TuneStrategy::Sequential,
            false,
            true,
        ),
    ];
    for (format, m, k, n, strategy, addend, relu) in cases {
        let mut graph = Graph::new();
        let x = graph.input("x", &[m, k]);
        let w = match format {
            WeightFormat::Q4 => graph.parameter_q4("w", &[k, n]),
            WeightFormat::Q8 => graph.parameter_q8("w", &[k, n]),
            _ => unreachable!(),
        };
        let mut y = graph.matmul(x, w);
        if addend {
            let d = graph.input("d", &[m, n]);
            y = graph.add(y, d);
        }
        if relu {
            y = graph.relu(y);
        }
        graph.set_outputs(vec![y]);

        let build_for = |coop: CoopPolicy| {
            let mut config = crate::support::gpu::inference_config();
            config.runtime.coop = coop;
            config.tune = false;
            meganeura::build(&graph, config).0
        };
        let mut scalar = build_for(CoopPolicy::Disabled);
        let mut candidate = build_for(CoopPolicy::AllowF16);

        // Untuned AllowF16 must remain scalar (no unconditional promotion).
        assert!(
            candidate
                .plan()
                .dispatches
                .iter()
                .all(|d| !matches!(d.kernel, Kernel::CooperativeWorkgroup { .. })),
            "{format:?} {m}x{k}x{n}: untuned AllowF16 should stay scalar"
        );

        let report = candidate.tune_with(tune_opts(strategy)).unwrap();
        let coop_outcomes: Vec<_> = report
            .outcomes
            .iter()
            .filter(|o| matches!(o.candidate, MatmulTile::CooperativeWorkgroup { .. }))
            .collect();
        assert_eq!(
            !coop_outcomes.is_empty(),
            supports_f16,
            "{format:?} {m}x{k}x{n} add={addend} relu={relu}: outcomes={:?}",
            report.outcomes
        );
        if supports_f16 && strategy == TuneStrategy::Sequential {
            assert!(
                coop_outcomes.iter().all(|o| o.qualified),
                "{format:?} {m}x{k}x{n} add={addend} relu={relu}: quantized coop failed qualification: {coop_outcomes:?}"
            );
        }
        let selected_coop = candidate.plan().dispatches.iter().any(|d| {
            d.weight_format == format
                && matches!(d.kernel, Kernel::CooperativeWorkgroup { types: None, .. })
        });
        if selected_coop {
            assert!(
                candidate
                    .dispatch_pipeline_keys()
                    .iter()
                    .any(|key| key.contains(&format!("workgroup-{format:?}"))),
                "expected quantized workgroup pipeline key, got {:?}",
                candidate.dispatch_pipeline_keys()
            );
        }

        let x_data: Vec<f32> = (0..m * k)
            .map(|i| ((i * 37 % 199) as f32 - 99.0) * 0.005)
            .collect();
        let w_data: Vec<f32> = (0..k * n)
            .map(|i| ((i * 19 % 131) as f32 - 65.0) * 0.004)
            .collect();
        let d_data: Vec<f32> = (0..m * n)
            .map(|i| ((i * 11 % 73) as f32 - 36.0) * 0.01)
            .collect();

        for session in [&mut scalar, &mut candidate] {
            session.set_input("x", &x_data);
            session.set_parameter("w", &w_data);
            if addend {
                session.set_input("d", &d_data);
            }
            session.step();
            session.wait();
        }
        let want = scalar.read_output(m * n);
        let got = candidate.read_output(m * n);
        assert!(got.iter().all(|v| v.is_finite()));
        let error = relative_l2(&got, &want);
        assert!(
            error < 2e-3,
            "{format:?} {m}x{k}x{n} add={addend} relu={relu}: relative_l2={error:.3e}"
        );
    }

    // 2. All five GGML packed formats (Q4_0, Q4_K, Q5_K, Q6_K, Q3_K) at ragged
    //    M=35, K=512, N=528 (two superblocks deep, ragged M & N).
    #[cfg(feature = "gguf")]
    {
        use meganeura::load::gguf::{GgmlType, GgufTensor};

        let mut lcg = 0x9E37_79B9u32;
        let mut next_u8 = || -> u8 {
            lcg = lcg.wrapping_mul(1664525).wrapping_add(1013904223);
            (lcg >> 16) as u8
        };
        let (m, k, n) = (35usize, 512usize, 528usize);
        let x_data: Vec<f32> = (0..m * k)
            .map(|i| ((i * 29 % 173) as f32 - 86.0) * 0.004)
            .collect();
        let gguf_formats = [
            (WeightFormat::Q40, GgmlType::Q4_0, 32usize, 18usize),
            (WeightFormat::Q4K, GgmlType::Q4K, 256, 144),
            (WeightFormat::Q5K, GgmlType::Q5K, 256, 176),
            (WeightFormat::Q6K, GgmlType::Q6K, 256, 210),
            (WeightFormat::Q3K, GgmlType::Q3K, 256, 110),
        ];
        for (format, ggml, block_elems, block_bytes) in gguf_formats {
            let num_blocks = (k / block_elems) * n;
            let mut raw = vec![0u8; num_blocks * block_bytes];
            for b in 0..num_blocks {
                let slice = &mut raw[b * block_bytes..(b + 1) * block_bytes];
                for byte in slice.iter_mut() {
                    *byte = next_u8();
                }
                let scale = half::f16::from_f32(0.004 + (b % 7) as f32 * 0.001)
                    .to_bits()
                    .to_le_bytes();
                let dmin = half::f16::from_f32(0.002 + (b % 5) as f32 * 0.0005)
                    .to_bits()
                    .to_le_bytes();
                match ggml {
                    GgmlType::Q4_0 => slice[0..2].copy_from_slice(&scale),
                    GgmlType::Q4K | GgmlType::Q5K => {
                        slice[0..2].copy_from_slice(&scale);
                        slice[2..4].copy_from_slice(&dmin);
                    }
                    GgmlType::Q6K => slice[208..210].copy_from_slice(&scale),
                    GgmlType::Q3K => slice[108..110].copy_from_slice(&scale),
                    _ => unreachable!(),
                }
            }
            let tensor = GgufTensor::new(vec![k, n], ggml, raw);
            let (_, packed) = tensor.to_packed().unwrap();

            let mut graph = Graph::new();
            let x = graph.input("x", &[m, k]);
            let w = match format {
                WeightFormat::Q40 => graph.parameter_q40("w", &[k, n]),
                WeightFormat::Q4K => graph.parameter_q4k("w", &[k, n]),
                WeightFormat::Q5K => graph.parameter_q5k("w", &[k, n]),
                WeightFormat::Q6K => graph.parameter_q6k("w", &[k, n]),
                WeightFormat::Q3K => graph.parameter_q3k("w", &[k, n]),
                _ => unreachable!(),
            };
            let y = graph.matmul(x, w);
            graph.set_outputs(vec![y]);

            let build_for = |coop: CoopPolicy| {
                let mut config = crate::support::gpu::inference_config();
                config.runtime.coop = coop;
                config.tune = false;
                meganeura::build(&graph, config).0
            };
            let mut scalar = build_for(CoopPolicy::Disabled);
            let mut candidate = build_for(CoopPolicy::AllowF16);

            assert!(
                candidate
                    .plan()
                    .dispatches
                    .iter()
                    .all(|d| !matches!(d.kernel, Kernel::CooperativeWorkgroup { .. })),
                "{format:?}: untuned AllowF16 should stay scalar"
            );
            let report = candidate
                .tune_with(tune_opts(TuneStrategy::Sequential))
                .unwrap();
            let coop_outcomes: Vec<_> = report
                .outcomes
                .iter()
                .filter(|o| matches!(o.candidate, MatmulTile::CooperativeWorkgroup { .. }))
                .collect();
            assert_eq!(
                !coop_outcomes.is_empty(),
                supports_f16,
                "{format:?}: unexpected candidate set {:?}",
                report.outcomes
            );
            if supports_f16 {
                assert!(
                    coop_outcomes.iter().all(|o| o.qualified),
                    "{format:?}: quantized coop failed qualification: {coop_outcomes:?}"
                );
            }

            for session in [&mut scalar, &mut candidate] {
                session.set_input("x", &x_data);
                session.set_parameter_packed("w", &packed);
                session.step();
                session.wait();
            }
            let want = scalar.read_output(m * n);
            let got = candidate.read_output(m * n);
            assert!(got.iter().all(|v| v.is_finite()), "{format:?}: non-finite");
            let error = relative_l2(&got, &want);
            assert!(
                error < 2e-3,
                "{format:?} {m}x{k}x{n}: relative_l2={error:.3e}"
            );
        }
    }
}
