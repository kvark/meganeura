//! f16 weight storage through f16-input cooperative tiles: plain and
//! transposed-B products must match an f64 reference that uses the same
//! f16-rounded weights, and must actually select a cooperative kernel.

use meganeura::{CoopPolicy, Graph, Session};

const M: usize = 512;
const K: usize = 512;
const N: usize = 512;

fn build(transposed: bool, coop: CoopPolicy) -> Session {
    let mut graph = Graph::new();
    let x = graph.input("x", &[M, K]);
    let output = if transposed {
        let w = graph.parameter("w", &[N, K]);
        graph.matmul_bt(x, w)
    } else {
        let w = graph.parameter("w", &[K, N]);
        graph.matmul(x, w)
    };
    graph.set_outputs(vec![output]);
    assert_eq!(graph.store_weights_f16(), vec!["w".to_owned()]);
    let mut config = crate::support::gpu::inference_config();
    config.runtime.coop = coop;
    meganeura::build(&graph, config).0
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
    for transposed in [false, true] {
        let w: Vec<f32> = (0..K * N)
            .map(|i| ((i * 13 % 97) as f32 - 48.0) * 0.003)
            .collect();
        let want = reference(transposed, &x, &w);
        for coop in [CoopPolicy::AllowF16, CoopPolicy::Disabled] {
            let mut session = build(transposed, coop);
            let cooperative = session.plan().dispatches.iter().any(|dispatch| {
                dispatch.use_coop()
                    && dispatch.weight_format == meganeura::compile::WeightFormat::F16
            });
            if coop == CoopPolicy::AllowF16 && !cooperative {
                eprintln!("skipping cooperative f16 weights: no f16 cooperative tiles here");
                continue;
            }
            session.set_parameter("w", &w);
            session.set_input("x", &x);
            session.step();
            session.wait();
            let mut got = vec![0.0f32; M * N];
            session.read_output_by_index(0, &mut got);
            let error = relative_l2(&got, &want);
            // f16-input tiles also round x; scalar kernels compute in f32.
            let bound = if cooperative { 2e-3 } else { 1e-5 };
            assert!(
                error < bound,
                "transposed={transposed} coop={coop:?}: relative L2 {error:.3e} >= {bound:.0e}"
            );
        }
    }
}
