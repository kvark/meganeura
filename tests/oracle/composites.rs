//! Composite ops against their decompositions into primitives.
//!
//! For every composite: the decomposition holds only primitives, computes
//! what the composite computes on the reference interpreter, and builds
//! exactly the same execution plan once recomposed: the same dispatches
//! reading the same values ([`ExecutionPlan::dataflow_digest`]), for
//! inference and, where the composite is differentiable, for training.
//!
//! [`ExecutionPlan::dataflow_digest`]: meganeura::compile::ExecutionPlan::dataflow_digest

use meganeura::graph::OpClass;
use meganeura::reference::{Feeds, evaluate_outputs, gradients};
use meganeura::{CompileOptions, Graph, Mode, NodeId, OptimizeConfig, compile_plan};

/// The plan's dispatch inventory, and the digest of what it computes.
fn signature(graph: &Graph, mode: Mode) -> (Vec<String>, u64) {
    let plan = compile_plan(
        graph,
        mode,
        OptimizeConfig::default(),
        &CompileOptions::default(),
    );
    (plan.dispatch_inventory(), plan.dataflow_digest())
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
            let scale = w
                .data
                .iter()
                .filter(|v| v.is_finite())
                .fold(1.0f64, |m, v| m.max(v.abs()));
            for (i, (a, b)) in w.data.iter().zip(&g.data).enumerate() {
                // The reference leaves some outputs unspecified (NaN).
                assert!(
                    (a - b).abs() <= 1e-6 * scale || a.is_nan(),
                    "{what}: output {i} is {b}, want {a}"
                );
            }
        }

        // Same gradients: training recognizes only composites whose
        // gradient is their decomposition's, so recognizing changes none.
        if self.trainable {
            let primitive = param_grads(&decomposed, &self.feeds);
            let recognized = param_grads(&decomposed.recompose_for_training(), &self.feeds);
            assert_eq!(primitive.len(), recognized.len(), "{what}: parameters");
            for ((name, p), (_, r)) in primitive.iter().zip(&recognized) {
                // A parameter without a gradient has a scalar zero.
                let zero = |g: &[f64]| g.iter().all(|&v| v == 0.0);
                if p.len() != r.len() && zero(p) && zero(r) {
                    continue;
                }
                assert_eq!(p.len(), r.len(), "{what}: gradient of {name}");
                let scale = p.iter().fold(1.0f64, |m, v| m.max(v.abs()));
                for (i, (a, b)) in p.iter().zip(r).enumerate() {
                    assert!(
                        (a - b).abs() <= 1e-6 * scale,
                        "{what}: gradient of {name}[{i}] is {b} recognized, {a} in primitives"
                    );
                }
            }
        }

        // Same plans.
        let mut modes = vec![Mode::Inference];
        if self.trainable {
            modes.push(Mode::Training);
        }
        for mode in modes {
            // Training recognizes only composites with exact gradients; the
            // others build as their primitives do.
            let written = match mode {
                Mode::Inference => self.graph.deep_clone(),
                Mode::Training => self
                    .graph
                    .decompose_where(|op| !op.differentiates_as_decomposed(1)),
            };
            let original = signature(&written, mode);
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
                    only(&original.0, &rebuilt.0),
                    only(&rebuilt.0, &original.0),
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

/// The loss gradient of each parameter of `forward`, on the reference
/// interpreter, by parameter name.
fn param_grads(forward: &Graph, feeds: &Feeds) -> Vec<(String, Vec<f64>)> {
    let names: Vec<String> = forward
        .nodes()
        .iter()
        .filter_map(|node| match node.op {
            meganeura::graph::Op::Parameter { ref name } => Some(name.clone()),
            _ => None,
        })
        .collect();
    let diff = meganeura::autodiff::differentiate(forward);
    let out = evaluate_outputs(&diff, feeds).unwrap();
    let mut grads: Vec<_> = names
        .into_iter()
        .zip(&out[forward.outputs().len()..])
        .map(|(name, grad)| (name, grad.data.clone()))
        .collect();
    grads.sort_by(|a, b| a.0.cmp(&b.0));
    grads
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
        case(
            "mul_per_channel",
            true,
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

#[test]
fn rotary_embeddings() {
    check_all(vec![
        case(
            "rope",
            true,
            |g| {
                let x = g.parameter("x", &[5, 16]);
                g.rope(x, 10_000.0, 8)
            },
            |_| {},
        ),
        case(
            "rope_offset",
            true,
            |g| {
                let x = g.parameter("x", &[5, 16]);
                g.rope_with_offset(x, 500.0, 3, 16)
            },
            |_| {},
        ),
        case(
            "rope_dynamic",
            false,
            |g| {
                let x = g.parameter("x", &[3, 16]);
                let pos = g.input_u32("pos", &[1]);
                g.rope_dynamic_offset(x, 10_000.0, pos, 8)
            },
            |f| {
                f.set_u32("pos", &[11]);
            },
        ),
        case(
            "rope_factors",
            false,
            |g| {
                let x = g.parameter("x", &[3, 16]);
                let pos = g.input_u32("pos", &[1]);
                let factors = g.input("factors", &[4]);
                g.rope_dynamic_offset_factors(x, 10_000.0, pos, 8, factors)
            },
            |f| {
                f.set_u32("pos", &[7]);
                positive(f, "factors", 4);
            },
        ),
        case(
            "rope_positions",
            false,
            |g| {
                let x = g.parameter("x", &[4, 16]);
                let pos = g.input_u32("pos", &[4]);
                g.rope_with_positions(x, 10_000.0, pos, 8)
            },
            |f| {
                f.set_u32("pos", &[9, 2, 30, 0]);
            },
        ),
    ]);
}

fn qkv(
    g: &mut Graph,
    rows: usize,
    kv_rows: usize,
    heads: usize,
    kv: usize,
    dim: usize,
) -> [NodeId; 3] {
    [
        g.parameter("q", &[rows, heads * dim]),
        g.parameter("k", &[kv_rows, kv * dim]),
        g.parameter("v", &[kv_rows, kv * dim]),
    ]
}

#[test]
fn attention() {
    check_all(vec![
        case(
            "causal",
            true,
            |g| {
                let [q, k, v] = qkv(g, 6, 6, 2, 2, 8);
                g.causal_attention(q, k, v, 2, 2, 8)
            },
            |_| {},
        ),
        case(
            "causal_gqa",
            true,
            |g| {
                let [q, k, v] = qkv(g, 6, 6, 4, 2, 8);
                g.causal_attention(q, k, v, 4, 2, 8)
            },
            |_| {},
        ),
        case(
            "sliding_window",
            true,
            |g| {
                let [q, k, v] = qkv(g, 7, 7, 2, 1, 8);
                g.sliding_window_attention(q, k, v, 2, 1, 8, 3)
            },
            |_| {},
        ),
        case(
            "sliding_window_spanning",
            true,
            |g| {
                let [q, k, v] = qkv(g, 5, 5, 2, 2, 8);
                g.sliding_window_attention(q, k, v, 2, 2, 8, 9)
            },
            |_| {},
        ),
        case(
            "full",
            true,
            |g| {
                let [q, k, v] = qkv(g, 5, 5, 3, 3, 8);
                g.full_attention(q, k, v, 3, 3, 8)
            },
            |_| {},
        ),
        case(
            "cross",
            false,
            |g| {
                let [q, k, v] = qkv(g, 4, 7, 2, 1, 8);
                g.cross_attention(q, k, v, 2, 1, 8)
            },
            |_| {},
        ),
        case(
            "multi_head_attn",
            true,
            |g| {
                let [q, k, v] = qkv(g, 4, 7, 2, 2, 8);
                g.multi_head_attn(q, k, v, 2, 2, 8, true)
            },
            |_| {},
        ),
        case(
            "multi_head_attn_self",
            true,
            |g| {
                let [q, k, v] = qkv(g, 5, 5, 2, 2, 8);
                g.multi_head_attn(q, k, v, 2, 2, 8, false)
            },
            |_| {},
        ),
        case(
            "cached",
            false,
            |g| {
                let [q, k, v] = qkv(g, 2, 9, 4, 2, 8);
                let pos = g.input_u32("pos", &[1]);
                g.cached_attention(q, k, v, pos, 4, 2, 8)
            },
            |f| {
                f.set_u32("pos", &[5]);
            },
        ),
        case(
            "cached_block",
            false,
            |g| {
                let [q, k, v] = qkv(g, 3, 10, 2, 1, 8);
                let pos = g.input_u32("pos", &[1]);
                let valid = g.input_u32("valid", &[1]);
                g.cached_block_attention(q, k, v, pos, valid, 2, 1, 8, 0)
            },
            |f| {
                f.set_u32("pos", &[4]);
                f.set_u32("valid", &[2]);
            },
        ),
        case(
            "cached_block_window",
            false,
            |g| {
                let [q, k, v] = qkv(g, 3, 10, 2, 2, 8);
                let pos = g.input_u32("pos", &[1]);
                let valid = g.input_u32("valid", &[1]);
                g.cached_block_attention(q, k, v, pos, valid, 2, 2, 8, 3)
            },
            |f| {
                f.set_u32("pos", &[5]);
                f.set_u32("valid", &[3]);
            },
        ),
        case(
            "biased",
            false,
            |g| {
                let [q, k, v] = qkv(g, 5, 7, 2, 1, 8);
                let bias = g.parameter("bias", &[2, 5, 7]);
                g.biased_attention([q, k, v, bias], 2, 1, 8, 1.0, false)
            },
            |_| {},
        ),
        case(
            "biased_causal",
            false,
            |g| {
                let [q, k, v] = qkv(g, 6, 6, 4, 2, 8);
                let bias = g.parameter("bias", &[4, 6, 6]);
                g.biased_attention([q, k, v, bias], 4, 2, 8, 0.5, true)
            },
            |_| {},
        ),
        case(
            "biased_cached",
            false,
            |g| {
                let [q, k, v] = qkv(g, 2, 9, 2, 1, 8);
                let pos = g.input_u32("pos", &[1]);
                let bias = g.parameter("bias", &[2, 9]);
                g.biased_cached_attention([q, k, v, pos, bias], 2, 1, 8, 1.0)
            },
            |f| {
                f.set_u32("pos", &[6]);
            },
        ),
        case(
            "chunked_relative",
            false,
            |g| {
                let [q, k, v] = qkv(g, 6, 6, 2, 2, 8);
                let rel = g.parameter("rel", &[4, 16]);
                g.chunked_relative_attention(q, k, v, rel, 2, 8, 4, 30.0)
            },
            |_| {},
        ),
    ]);
}

#[test]
fn column_sums() {
    check_all(vec![case(
        "sum_rows",
        true,
        |g| {
            let x = g.parameter("x", &[5, 7]);
            let ty = meganeura::TensorType::f32(vec![7]);
            let s = g.sum_rows(x, &ty);
            g.reshape(s, &[1, 7])
        },
        |_| {},
    )]);
}

/// Every composite the classification names has a case above.
#[test]
fn every_composite_is_covered() {
    let source = include_str!("../../src/graph/composite.rs");
    let start = source.find("Op::Softplus { .. }").expect("composite arm");
    let end = start
        + source[start..]
            .find("=> Composite")
            .expect("composite arm end");
    let names: Vec<&str> = source[start..end]
        .split("Op::")
        .skip(1)
        .map(|s| s.split(|c: char| !c.is_alphanumeric()).next().unwrap())
        .collect();
    let covered = [
        ("Softplus", "softplus"),
        ("Silu", "silu"),
        ("Gelu", "gelu"),
        ("SwiGLU", "swiglu"),
        ("GeGLU", "geglu"),
        ("MeanAll", "mean_all"),
        ("SumRows", "sum_rows"),
        ("GlobalAvgPool", "global_avg_pool"),
        ("NormalizeInnerSum", "normalize_inner_sum"),
        ("PairwiseSquaredDistance", "pairwise_distance"),
        ("PairwiseVectorRejection", "pairwise_rejection"),
        ("ShiftInner", "shift_right"),
        ("Softmax", "softmax"),
        ("LogSoftmax", "log_softmax"),
        ("CrossEntropyLoss", "cross_entropy"),
        ("BceLoss", "bce"),
        ("RmsNorm", "rms_norm"),
        ("LayerNorm", "layer_norm"),
        ("GroupNorm", "group_norm"),
        ("MulPerChannel", "mul_per_channel"),
        ("AddPerChannel", "add_per_channel"),
        ("Upsample2x", "upsample"),
        ("RoPE", "rope"),
        ("RoPEPositions", "rope_positions"),
        ("CausalAttention", "causal"),
        ("FullAttention", "full"),
        ("CrossAttention", "cross"),
        ("MultiHeadAttn", "multi_head_attn"),
        ("SlidingWindowAttention", "sliding_window"),
        ("CachedAttention", "cached"),
        ("CachedBlockAttention", "cached_block"),
        ("ChunkedRelativeAttention", "chunked_relative"),
        ("BiasedAttention", "biased"),
        ("BiasedCachedAttention", "biased_cached"),
    ];
    let this = include_str!("composites.rs");
    for name in &names {
        let case = covered
            .iter()
            .find(|c| c.0 == *name)
            .unwrap_or_else(|| panic!("composite {name} has no case"));
        assert!(
            this.contains(&format!("\"{}\",", case.1)),
            "composite {name}: no case named {}",
            case.1
        );
    }
    assert_eq!(names.len(), covered.len(), "{names:?}");
}

/// Why training builds leave the losses in primitives: their fused
/// gradients treat the labels as constant and differentiate through the
/// probability clamp, so recognizing them would change what a model with
/// trainable labels, or a saturated prediction, learns.
#[test]
fn loss_gradients_stay_exact_in_training() {
    let differentiated = |build: &dyn Fn(&mut Graph) -> NodeId, feeds: &Feeds| {
        let mut g = Graph::new();
        let loss = build(&mut g);
        g.set_outputs(vec![loss]);
        let recognized = g.decompose().recompose_for_training();
        assert!(
            recognized.nodes().iter().all(|n| !matches!(
                n.op,
                meganeura::graph::Op::CrossEntropyLoss | meganeura::graph::Op::BceLoss
            )),
            "a loss was recognized for training"
        );
        param_grads(&recognized, feeds)
    };

    // Cross entropy at logits [0, 0]: ∂/∂labels = -log_softmax = [ln 2, ln 2].
    let mut feeds = Feeds::new();
    feeds.set("logits", &[0.0, 0.0]);
    feeds.set("labels", &[0.5, 0.5]);
    let grads = differentiated(
        &|g| {
            let logits = g.parameter("logits", &[1, 2]);
            let labels = g.parameter("labels", &[1, 2]);
            g.cross_entropy_loss(logits, labels)
        },
        &feeds,
    );
    let (_, labels) = grads.iter().find(|(n, _)| n == "labels").unwrap();
    for &g in labels {
        assert!((g - std::f64::consts::LN_2).abs() < 1e-7, "{labels:?}");
    }

    // BCE below the clamp: the clamped prediction does not move the loss.
    let mut feeds = Feeds::new();
    feeds.set("pred", &[1e-8]);
    feeds.set("labels", &[1.0]);
    let grads = differentiated(
        &|g| {
            let pred = g.parameter("pred", &[1, 1]);
            let labels = g.input("labels", &[1, 1]);
            g.bce_loss(pred, labels)
        },
        &feeds,
    );
    let (_, pred) = grads.iter().find(|(n, _)| n == "pred").unwrap();
    assert_eq!(pred, &vec![0.0]);
}
