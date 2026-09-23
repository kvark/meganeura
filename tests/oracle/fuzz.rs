//! Random graphs through the whole pipeline — e-graph rewrites, fusion,
//! scheduling, memory planning — against the unoptimized reference.
//!
//! Single-op tests check kernels; these check what the compiler does to
//! compositions of them. Every failure prints its seed; rerun one with
//! `ORACLE_FUZZ_SEED=<seed>`, or search further with `ORACLE_FUZZ_COUNT`.

use meganeura::graph::TensorType;
use meganeura::reference::{Feeds, Rng, Tolerance, gpu, gradients};
use meganeura::{Graph, NodeId};

const DIMS: [usize; 5] = [1, 3, 4, 8, 17];

struct Builder {
    g: Graph,
    rng: Rng,
    nodes: Vec<NodeId>,
    params: usize,
}

impl Builder {
    fn new(seed: u64) -> Self {
        Self {
            g: Graph::new(),
            rng: Rng::new(seed),
            nodes: Vec::new(),
            params: 0,
        }
    }

    fn dim(&mut self) -> usize {
        DIMS[self.rng.below(DIMS.len() as u32) as usize]
    }

    fn shape(&self, id: NodeId) -> Vec<usize> {
        self.g.node(id).ty.shape.clone()
    }

    fn leaf(&mut self, shape: &[usize]) -> NodeId {
        self.params += 1;
        let id = self.g.parameter(&format!("p{}", self.params), shape);
        if shape.len() == 2 {
            self.nodes.push(id);
        }
        id
    }

    fn pick(&mut self) -> NodeId {
        // Prefer recent nodes so chains grow deep.
        let n = self.nodes.len() as u32;
        let back = self.rng.below(n.min(4));
        self.nodes[(n - 1 - back) as usize]
    }

    /// A node of `shape`, reusing one when available.
    fn with_shape(&mut self, shape: &[usize]) -> NodeId {
        let matches: Vec<NodeId> = self
            .nodes
            .iter()
            .copied()
            .filter(|&n| self.g.node(n).ty.shape == shape)
            .collect();
        if !matches.is_empty() && self.rng.below(3) != 0 {
            matches[self.rng.below(matches.len() as u32) as usize]
        } else {
            self.leaf(shape)
        }
    }

    fn step(&mut self) {
        let x = self.pick();
        let s = self.shape(x);
        let (m, n) = (s[0], s[1]);
        let y = match self.rng.below(18) {
            0 => self.g.relu(x),
            1 => self.g.sigmoid(x),
            2 => self.g.tanh(x),
            3 => self.g.neg(x),
            4 => self.g.silu(x),
            5 => self.g.gelu(x),
            6 => self.g.softplus(x, 1.0),
            7 => self.g.scale(x, 0.5),
            8 => {
                let other = self.with_shape(&[m, n]);
                self.g.add(x, other)
            }
            9 => {
                let other = self.with_shape(&[m, n]);
                self.g.mul(x, other)
            }
            10 => {
                let k = self.dim();
                let w = self.with_shape(&[n, k]);
                self.g.matmul(x, w)
            }
            11 => {
                let k = self.dim();
                let w = self.with_shape(&[k, n]);
                self.g.matmul_bt(x, w)
            }
            12 => {
                let b = self.with_shape(&[n]);
                self.g.bias_add(x, b)
            }
            13 => self.g.transpose(x),
            14 => self.g.softmax(x),
            15 => {
                let w = self.with_shape(&[n]);
                self.g.rms_norm(x, w, 1e-5)
            }
            16 => {
                let w = self.with_shape(&[n]);
                let b = self.with_shape(&[n]);
                self.g.layer_norm(x, w, b, 1e-5)
            }
            _ => {
                let sum = self.g.sum_inner(x);
                self.g.broadcast_inner(sum, n)
            }
        };
        self.nodes.push(y);
    }

    fn build(seed: u64, steps: usize) -> (Graph, NodeId) {
        let mut b = Self::new(seed);
        let (m, n) = (b.dim(), b.dim());
        b.leaf(&[m, n]);
        for _ in 0..steps {
            b.step();
        }
        let last = *b.nodes.last().unwrap();
        (b.g, last)
    }
}

/// `ORACLE_FUZZ_SEED` reruns one seed; `ORACLE_FUZZ_COUNT` widens a run.
fn seeds(count: u64) -> Vec<u64> {
    if let Ok(seed) = std::env::var("ORACLE_FUZZ_SEED") {
        return vec![seed.parse().expect("ORACLE_FUZZ_SEED must be an integer")];
    }
    let count = std::env::var("ORACLE_FUZZ_COUNT")
        .map(|c| c.parse().expect("ORACLE_FUZZ_COUNT must be an integer"))
        .unwrap_or(count);
    (1..=count).collect()
}

/// Errors compound through a chain, so the per-element bound of the final
/// op alone is not the whole budget.
fn chain_tolerance() -> Tolerance {
    Tolerance {
        rtol: 2e-3,
        floor: 1e-3,
    }
}

#[test]
fn random_inference_graphs() {
    let mut failures = Vec::new();
    for seed in seeds(40) {
        let (mut g, last) = Builder::build(seed, 10);
        // Output an intermediate as well, so its value must survive fusion.
        let mid = NodeId::try_from(g.nodes().len() / 2).unwrap();
        let outputs = if g.node(mid).ty.shape.len() == 2 && mid != last {
            vec![last, mid]
        } else {
            vec![last]
        };
        g.set_outputs(outputs);
        let mut feeds = Feeds::new();
        feeds.fill_random(&g, seed, 1.0);
        let options = gpu::Options {
            tolerance: chain_tolerance(),
            ..Default::default()
        };
        let report = gpu::check_inference(&g, &feeds, &options).unwrap();
        if !report.passed() {
            failures.push(format!("seed {seed}:\n{g}{report}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn random_training_graphs() {
    let mut failures = Vec::new();
    for seed in seeds(25) {
        let (mut g, last) = Builder::build(1000 + seed, 7);
        let loss = gradients::weighted_loss(&mut g, last, seed, 0.5);
        g.set_outputs(vec![loss]);
        let mut feeds = Feeds::new();
        feeds.fill_random(&g, seed, 1.0);
        let cpu = gradients::check(&g, &feeds, &gradients::Options::default()).unwrap();
        if !cpu.passed() {
            failures.push(format!("seed {seed} autodiff:\n{g}{cpu}"));
            continue;
        }
        let options = gpu::Options {
            tolerance: chain_tolerance(),
            ..Default::default()
        };
        let report = gpu::check_training(&g, &feeds, &options).unwrap();
        if !report.passed() {
            failures.push(format!("seed {seed} training:\n{g}{report}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn builder_graphs_are_well_formed() {
    for seed in 1..=5 {
        let (g, last) = Builder::build(seed, 10);
        assert_eq!(g.node(last).ty.shape.len(), 2);
        let _ = TensorType::f32(g.node(last).ty.shape.clone());
    }
}
