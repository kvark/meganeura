//! f16 weight storage through f16-input cooperative tiles: plain and
//! transposed-B products must match an f64 reference that uses the same
//! f16-rounded weights, and select cooperative kernels on supported hardware.

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

fn relative_l2(got: &[f32], want: &[f32]) -> f64 {
    let num: f64 = got
        .iter()
        .zip(want)
        .map(|(&a, &b)| f64::from(a - b).powi(2))
        .sum();
    let den: f64 = want.iter().map(|&b| f64::from(b).powi(2)).sum();
    (num / den.max(f64::EPSILON)).sqrt()
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
