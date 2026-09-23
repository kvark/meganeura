//! Every `Op` must be exercised by this suite. Adding a variant fails
//! `every_op_is_covered` until it is listed here with the code that builds
//! it in a test, or with the reason it cannot be tested.

use Coverage::{Covered, Exempt};

enum Coverage {
    /// Text that appears in a test which builds the op.
    Covered(&'static str),
    /// Why no test builds the op.
    Exempt(&'static str),
}

const TABLE: &[(&str, Coverage)] = &[
    ("Parameter", Covered(".parameter(")),
    ("Input", Covered(".input(")),
    ("Constant", Covered(".constant(")),
    ("MatMul", Covered(".matmul(")),
    ("MatMulAT", Covered(".matmul_at(")),
    ("MatMulBT", Covered(".matmul_bt(")),
    ("BlockMatMul", Covered(".block_matmul(")),
    ("BlockMatMulAT", Covered(".block_matmul_at(")),
    ("BlockMatMulBT", Covered(".block_matmul_bt(")),
    ("Add", Covered(".add(")),
    ("Mul", Covered(".mul(")),
    ("Relu", Covered(".relu(")),
    ("Sigmoid", Covered(".sigmoid(")),
    ("Tanh", Covered(".tanh(")),
    ("Neg", Covered(".neg(")),
    ("Abs", Covered(".abs(")),
    ("Log", Covered(".log(")),
    ("Recip", Covered(".recip(")),
    ("Exp", Covered(".exp(")),
    ("Softplus", Covered(".softplus(")),
    ("SoftplusGrad", Covered("Op::SoftplusGrad")),
    ("Clamp", Covered(".clamp(")),
    ("Scale", Covered(".scale(")),
    ("SumAll", Covered(".sum_all(")),
    ("MeanAll", Covered(".mean_all(")),
    ("SumRows", Covered(".sum_rows(")),
    ("SumInner", Covered(".sum_inner(")),
    ("BroadcastInner", Covered(".broadcast_inner(")),
    ("NormalizeInnerSum", Covered(".normalize_inner_sum(")),
    (
        "NormalizeInnerSumGrad",
        Covered("Op::NormalizeInnerSumGrad"),
    ),
    (
        "PairwiseSquaredDistance",
        Covered(".pairwise_squared_distance("),
    ),
    (
        "PairwiseVectorRejection",
        Covered(".pairwise_vector_rejection("),
    ),
    ("PairwiseGrad", Covered("Op::PairwiseGrad")),
    ("ExclusiveCumsum", Covered(".exclusive_cumsum(")),
    ("ShiftInner", Covered(".shift_inner(")),
    ("Softmax", Covered(".softmax(")),
    ("CrossEntropyLoss", Covered(".cross_entropy_loss(")),
    ("BceLoss", Covered(".bce_loss(")),
    (
        "CrossEntropyLogitsGrad",
        Covered("Op::CrossEntropyLogitsGrad"),
    ),
    ("Greater", Covered(".greater(")),
    ("Transpose", Covered(".transpose(")),
    ("BiasAdd", Covered(".bias_add(")),
    ("BiasMul", Covered(".bias_mul(")),
    ("FusedMatMulAdd", Covered("Op::FusedMatMulAdd")),
    ("FusedMatMulATAdd", Covered("Op::FusedMatMulATAdd")),
    ("FusedMatMulBTAdd", Covered("Op::FusedMatMulBTAdd")),
    ("Identity", Covered(".reshape(")),
    ("Materialize", Covered(".materialize(")),
    ("StopGradient", Covered(".stop_gradient(")),
    ("LogSoftmax", Covered(".log_softmax(")),
    ("ScatterAdd", Covered(".scatter_add(")),
    ("Silu", Covered(".silu(")),
    ("SwiGLU", Covered(".swiglu(")),
    ("SwiGLUConcat", Covered(".swiglu_concat(")),
    ("SwiGLUConcatGrad", Covered("Op::SwiGLUConcatGrad")),
    ("GeGLU", Covered(".geglu(")),
    ("GeGLUConcat", Covered(".geglu_concat(")),
    ("GeGLUConcatGrad", Covered("Op::GeGLUConcatGrad")),
    ("SwiGLUGradGate", Covered(".swiglu_grad_gate(")),
    ("SwiGLUGradUp", Covered(".swiglu_grad_up(")),
    ("SiluGrad", Covered(".silu_grad(")),
    ("RmsNorm", Covered(".rms_norm(")),
    ("Embedding", Covered(".embedding(")),
    ("ToF16", Covered(".to_f16(")),
    ("RoPE", Covered(".rope(")),
    ("RoPEGrad", Covered(".rope_grad(")),
    ("RoPEPositions", Covered(".rope_with_positions(")),
    ("CausalAttention", Covered("Kind::Causal")),
    ("CausalAttentionRoPE", Covered("Kind::Rope")),
    ("Gelu", Covered(".gelu(")),
    ("LayerNorm", Covered(".layer_norm(")),
    ("FullAttention", Covered("Kind::Full")),
    ("CrossAttention", Covered("Kind::Cross")),
    ("MultiHeadAttn", Covered("multi_head_attn(")),
    ("MultiHeadAttnGradQ", Covered("Op::MultiHeadAttnGradQ")),
    ("MultiHeadAttnGradK", Covered("Op::MultiHeadAttnGradK")),
    ("MultiHeadAttnGradV", Covered("Op::MultiHeadAttnGradV")),
    ("RmsNormGradW", Covered(".rms_norm_grad_w(")),
    ("RmsNormGradX", Covered(".rms_norm_grad_x(")),
    ("LayerNormGradWB", Covered(".layer_norm_grad_wb(")),
    ("LayerNormGradX", Covered(".layer_norm_grad_x(")),
    ("Conv2d", Covered(".conv2d_hw(")),
    ("MulPerChannel", Covered(".mul_per_channel(")),
    ("AddPerChannel", Covered(".add_per_channel(")),
    ("Conv2dDw", Covered(".conv2d_dw(")),
    ("Conv2dGradInput", Covered(".conv2d_grad_input(")),
    ("Conv2dGradWeight", Covered(".conv2d_grad_weight(")),
    ("MaxPool2d", Covered(".max_pool_2d(")),
    ("MaxPool2dGrad", Covered("Op::MaxPool2dGrad")),
    ("GlobalAvgPool", Covered(".global_avg_pool(")),
    ("GlobalAvgPoolGrad", Covered("Op::GlobalAvgPoolGrad")),
    ("GroupNorm", Covered(".group_norm(")),
    ("WinogradConv2d", Covered("no_winograd = false")),
    ("GroupNormSilu", Covered("Op::GroupNormSilu")),
    ("GroupNormGradInput", Covered(".group_norm_grad_input(")),
    (
        "GroupNormGradWeightBias",
        Covered(".group_norm_grad_weight_bias("),
    ),
    ("Concat", Covered(".concat(")),
    ("SplitA", Covered(".split_a(")),
    ("SplitB", Covered(".split_b(")),
    ("Upsample2x", Covered(".upsample_2x(")),
    ("Upsample2xGrad", Covered(".upsample_2x_grad(")),
    (
        "SlidingWindowAttention",
        Covered(".sliding_window_attention("),
    ),
    ("CacheWrite", Covered(".cache_write(")),
    ("CacheWritePrefix", Covered(".cache_write_prefix(")),
    ("CachedAttention", Covered(".cached_attention(")),
    ("CachedBlockAttention", Covered(".cached_block_attention(")),
    (
        "ChunkedRelativeAttention",
        Covered(".chunked_relative_attention("),
    ),
    ("PrefixLast", Covered(".prefix_last(")),
    (
        "Nop",
        Exempt("a dead node left by fusion; it has no value to compare"),
    ),
];

const SUITE: &[&str] = &[
    include_str!("attention.rs"),
    include_str!("autodiff.rs"),
    include_str!("basic.rs"),
    include_str!("blocks.rs"),
    include_str!("fuzz.rs"),
    include_str!("losses.rs"),
    include_str!("norm.rs"),
    include_str!("regressions.rs"),
    include_str!("smoke.rs"),
    include_str!("vision.rs"),
];

/// Variant names of `pub enum Op`, read from the source.
fn op_variants() -> Vec<String> {
    let source = include_str!("../../src/graph.rs");
    let start = source.find("pub enum Op {").expect("Op enum");
    let body = &source[start..];
    let body = &body[..body.find("\n}\n").expect("end of Op enum")];
    body.lines()
        .filter_map(|line| {
            let name = line.strip_prefix("    ")?;
            let first = name.chars().next()?;
            if !first.is_ascii_uppercase() {
                return None;
            }
            Some(
                name.split(|c: char| !c.is_ascii_alphanumeric())
                    .next()?
                    .to_string(),
            )
        })
        .collect()
}

#[test]
fn every_op_is_covered() {
    let variants = op_variants();
    assert!(
        variants.len() > 100,
        "parsed only {} variants",
        variants.len()
    );
    let mut problems = Vec::new();
    for variant in &variants {
        match TABLE.iter().find(|entry| entry.0 == variant) {
            None => problems.push(format!("{variant} is not in the coverage table")),
            Some(&(_, Covered(needle))) => {
                if !SUITE.iter().any(|source| source.contains(needle)) {
                    problems.push(format!("{variant}: no test contains {needle:?}"));
                }
            }
            Some(&(_, Exempt(reason))) => {
                if reason.is_empty() {
                    problems.push(format!("{variant} is exempt without a reason"));
                }
            }
        }
    }
    for (name, _) in TABLE {
        if !variants.iter().any(|v| v == name) {
            problems.push(format!("{name} is listed but is no longer an Op"));
        }
    }
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}
