//! Composite ops against their decompositions into primitives.
//!
//! For every composite: the decomposition holds only primitives, computes
//! what the composite computes on the reference interpreter, and builds
//! exactly the same execution plan once recomposed, for inference and,
//! where the composite is differentiable, for training. The same plan is
//! the same performance.

use meganeura::graph::OpClass;
use meganeura::reference::{Feeds, evaluate_outputs, gradients};
use meganeura::{CompileOptions, Graph, Mode, NodeId, OptimizeConfig, compile_plan};

/// Everything about a plan that decides its cost, order-independent.
fn signature(graph: &Graph, mode: Mode) -> Vec<String> {
    let plan = compile_plan(
        graph,
        mode,
        OptimizeConfig::default(),
        &CompileOptions::default(),
    );
    let mut dispatches: Vec<String> = plan
        .dispatches
        .iter()
        .map(|d| {
            format!(
                "{:?} {:?} {:?} {:?} in={} out_extra={}",
                d.shader,
                d.kernel,
                d.params,
                d.workgroups,
                d.input_buffers.len(),
                d.extra_outputs.len()
            )
        })
        .collect();
    dispatches.sort();
    let mut buffers = plan.buffers.clone();
    buffers.sort_unstable();
    dispatches.push(format!("buffers {buffers:?}"));
    dispatches
}

/// One composite case: a graph whose last output applies the composite.
pub struct Case {
    pub what: String,
    pub graph: Graph,
    pub feeds: Feeds,
    pub trainable: bool,
}

pub fn case(
    what: impl Into<String>,
    trainable: bool,
    build: impl FnOnce(&mut Graph) -> NodeId,
    fix: impl FnOnce(&mut Feeds),
) -> Case {
    let what = what.into();
    let mut graph = Graph::new();
    let y = build(&mut graph);
    let out = if trainable {
        gradients::weighted_loss(&mut graph, y, what.len() as u64, 0.75)
    } else {
        y
    };
    graph.set_outputs(vec![out]);
    let mut feeds = Feeds::new();
    fix(&mut feeds);
    feeds.fill_random(&graph, 7 + what.len() as u64, 1.0);
    Case {
        what,
        graph,
        feeds,
        trainable,
    }
}

impl Case {
    pub fn check(&self) {
        let what = &self.what;
        let composites = self
            .graph
            .nodes()
            .iter()
            .filter(|n| n.op.class() == OpClass::Composite)
            .count();
        assert!(composites > 0, "{what}: the case has no composite");
        let decomposed = self.graph.decompose();
        for node in decomposed.nodes() {
            assert_eq!(
                node.op.class(),
                OpClass::Primitive,
                "{what}: {:?} left after decomposition",
                node.op
            );
        }

        // Same values on the reference interpreter.
        let want = evaluate_outputs(&self.graph, &self.feeds).unwrap();
        let got = evaluate_outputs(&decomposed, &self.feeds).unwrap();
        for (w, g) in want.iter().zip(&got) {
            assert_eq!(w.shape, g.shape, "{what}: output shape");
            let scale = w.data.iter().fold(1.0f64, |m, v| m.max(v.abs()));
            for (i, (a, b)) in w.data.iter().zip(&g.data).enumerate() {
                assert!(
                    (a - b).abs() <= 1e-6 * scale || (a.is_nan() && b.is_nan()),
                    "{what}: output {i} is {b}, want {a}"
                );
            }
        }

        // Same plans.
        let mut modes = vec![Mode::Inference];
        if self.trainable {
            modes.push(Mode::Training);
        }
        for mode in modes {
            let original = signature(&self.graph, mode);
            let rebuilt = signature(&decomposed, mode);
            if original != rebuilt {
                let only = |a: &[String], b: &[String]| {
                    a.iter()
                        .filter(|x| !b.contains(x))
                        .cloned()
                        .collect::<Vec<_>>()
                };
                panic!(
                    "{what} ({mode:?}): plans differ\n  only original: {:#?}\n  only decomposed: {:#?}\n  recomposed ops: {:?}",
                    only(&original, &rebuilt),
                    only(&rebuilt, &original),
                    decomposed
                        .recompose()
                        .nodes()
                        .iter()
                        .map(|n| format!("{:?}", n.op).chars().take(40).collect::<String>())
                        .collect::<Vec<_>>()
                );
            }
        }
    }
}

fn check_all(cases: Vec<Case>) {
    let mut failures = Vec::new();
    for case in cases {
        let what = case.what.clone();
        if let Err(e) = std::panic::catch_unwind(|| case.check()) {
            let msg = e
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default();
            failures.push(format!("{what}: {msg}"));
        }
    }
    assert!(
        failures.is_empty(),
        "{} composite case(s) failed:\n{}",
        failures.len(),
        failures.join("\n\n")
    );
}

fn positive(feeds: &mut Feeds, name: &str, n: usize) {
    let data: Vec<f32> = (0..n).map(|i| 0.3 + (i % 7) as f32 * 0.25).collect();
    feeds.set(name, &data);
}

#[test]
fn elementwise_and_norms() {
    check_all(vec![
        case(
            "softmax",
            true,
            |g| {
                let x = g.parameter("x", &[6, 10]);
                g.softmax(x)
            },
            |_| {},
        ),
        case(
            "log_softmax",
            true,
            |g| {
                let x = g.parameter("x", &[6, 10]);
                g.log_softmax(x)
            },
            |_| {},
        ),
        case(
            "rms_norm",
            true,
            |g| {
                let x = g.parameter("x", &[6, 32]);
                let w = g.parameter("w", &[32]);
                g.rms_norm(x, w, 1e-5)
            },
            |_| {},
        ),
        case(
            "layer_norm",
            true,
            |g| {
                let x = g.parameter("x", &[6, 32]);
                let w = g.parameter("w", &[32]);
                let b = g.parameter("b", &[32]);
                g.layer_norm(x, w, b, 1e-5)
            },
            |_| {},
        ),
        case(
            "silu",
            true,
            |g| {
                let x = g.parameter("x", &[6, 10]);
                g.silu(x)
            },
            |_| {},
        ),
        case(
            "gelu",
            true,
            |g| {
                let x = g.parameter("x", &[6, 10]);
                g.gelu(x)
            },
            |_| {},
        ),
        case(
            "softplus",
            true,
            |g| {
                let x = g.parameter("x", &[6, 10]);
                g.softplus(x, 2.0)
            },
            |_| {},
        ),
        case(
            "swiglu",
            true,
            |g| {
                let a = g.parameter("a", &[6, 10]);
                let b = g.parameter("b", &[6, 10]);
                g.swiglu(a, b)
            },
            |_| {},
        ),
        case(
            "geglu",
            true,
            |g| {
                let a = g.parameter("a", &[6, 10]);
                let b = g.parameter("b", &[6, 10]);
                g.geglu(a, b)
            },
            |_| {},
        ),
    ]);
}

#[test]
fn reductions_and_losses() {
    check_all(vec![
        case(
            "mean_all",
            true,
            |g| {
                let x = g.parameter("x", &[6, 10]);
                let m = g.mean_all(x);
                g.reshape(m, &[1, 1])
            },
            |_| {},
        ),
        case(
            "cross_entropy",
            true,
            |g| {
                let x = g.parameter("x", &[6, 10]);
                let labels = g.input("labels", &[6, 10]);
                let l = g.cross_entropy_loss(x, labels);
                g.reshape(l, &[1, 1])
            },
            |_| {},
        ),
        case(
            "bce",
            true,
            |g| {
                let x = g.parameter("x", &[6, 10]);
                let p = g.sigmoid(x);
                let labels = g.input("labels", &[6, 10]);
                let l = g.bce_loss(p, labels);
                g.reshape(l, &[1, 1])
            },
            |f| positive(f, "labels", 60),
        ),
        case(
            "normalize_inner_sum",
            true,
            |g| {
                let x = g.parameter("x", &[6, 10]);
                g.normalize_inner_sum(x, 0.5)
            },
            |f| positive(f, "x", 60),
        ),
        case(
            "pairwise_distance",
            true,
            |g| {
                let l = g.parameter("l", &[4, 6]);
                let r = g.parameter("r", &[12, 6]);
                g.pairwise_squared_distance(l, r, 3)
            },
            |_| {},
        ),
        case(
            "pairwise_rejection",
            true,
            |g| {
                let v = g.parameter("v", &[12, 6]);
                let u = g.parameter("u", &[4, 6]);
                g.pairwise_vector_rejection(v, u, 3)
            },
            |_| {},
        ),
        case(
            "cumsum",
            true,
            |g| {
                let x = g.parameter("x", &[4, 9]);
                g.exclusive_cumsum(x, false)
            },
            |_| {},
        ),
        case(
            "cumsum_reverse",
            true,
            |g| {
                let x = g.parameter("x", &[4, 9]);
                g.exclusive_cumsum(x, true)
            },
            |_| {},
        ),
        case(
            "shift_right",
            true,
            |g| {
                let x = g.parameter("x", &[4, 9]);
                g.shift_inner(x, 3)
            },
            |_| {},
        ),
        case(
            "shift_left",
            true,
            |g| {
                let x = g.parameter("x", &[4, 9]);
                g.shift_inner(x, -2)
            },
            |_| {},
        ),
        case(
            "shift_out",
            true,
            |g| {
                let x = g.parameter("x", &[4, 9]);
                g.shift_inner(x, 12)
            },
            |_| {},
        ),
    ]);
}

#[test]
fn vision() {
    check_all(vec![
        case(
            "global_avg_pool",
            true,
            |g| {
                let x = g.parameter("x", &[2 * 3 * 16]);
                g.global_avg_pool(x, 2, 3, 16)
            },
            |_| {},
        ),
        // Inference only, as the original op has no gradient.
        case(
            "mul_per_channel",
            false,
            |g| {
                let x = g.parameter("x", &[2 * 3 * 16]);
                let gate = g.parameter("gate", &[6]);
                g.mul_per_channel(x, gate, 3, 16)
            },
            |_| {},
        ),
        case(
            "add_per_channel",
            true,
            |g| {
                let x = g.parameter("x", &[2 * 3 * 16]);
                let b = g.parameter("b", &[3]);
                g.add_per_channel(x, b, 3, 16)
            },
            |_| {},
        ),
        case(
            "group_norm",
            true,
            |g| {
                let x = g.parameter("x", &[2 * 4 * 9]);
                let w = g.parameter("w", &[4]);
                let b = g.parameter("b", &[4]);
                g.group_norm(x, w, b, 2, 4, 9, 2, 1e-5)
            },
            |_| {},
        ),
        case(
            "upsample",
            true,
            |g| {
                let x = g.parameter("x", &[2 * 3 * 4 * 5]);
                g.upsample_2x(x, 2, 3, 4, 5)
            },
            |_| {},
        ),
    ]);
}
