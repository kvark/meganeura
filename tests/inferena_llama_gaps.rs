//! Compile-plan locks for Meganeura vs llama.cpp Inferena gaps.
//!
//! These inspect the shipped `compile` / `compile_training_graph` path.
//! Cooperative selection is covered in `runtime` unit tests because it
//! needs `select_variants`.

use meganeura::compile::{
    ExecutionPlan, ShaderEntry, compile, elide_seq1_attention, fuse_horizontal_matmuls,
    fuse_rmsnorm_into_gemv,
};
use meganeura::graph::Op;
use meganeura::models::smollm2::{self, SmolLM2Config};
use meganeura::{Graph, compile_training_graph, optimize};

fn shader_counts(plan: &ExecutionPlan) -> std::collections::BTreeMap<String, usize> {
    let mut counts = std::collections::BTreeMap::new();
    for dispatch in &plan.dispatches {
        *counts.entry(format!("{:?}", dispatch.shader)).or_default() += 1;
    }
    counts
}

#[test]
fn ce_training_graph_broadcasts_scalar_not_k1_gemv() {
    let mut graph = Graph::new();
    let tokens = graph.input_u32("token_ids", &[4]);
    let embed = graph.parameter("embed", &[32, 8]);
    let x = graph.embedding(tokens, embed);
    let weight = graph.parameter("w", &[8, 16]);
    let logits = graph.matmul(x, weight);
    let labels = graph.input("labels", &[4, 16]);
    let loss = graph.cross_entropy_loss(logits, labels);
    graph.set_outputs(vec![loss]);

    let differentiated = meganeura::autodiff::differentiate(&optimize::optimize(&graph));
    assert!(
        differentiated
            .nodes()
            .iter()
            .any(|node| matches!(node.op, Op::Broadcast)),
        "scalar grad_output must be Op::Broadcast, not a ones-row matmul"
    );
    let (plan, _) = compile_training_graph(&graph);
    assert!(
        !plan.dispatches.iter().any(|dispatch| {
            matches!(
                dispatch.shader,
                ShaderEntry::MatMulGemv | ShaderEntry::MatMulGemvAdd
            ) && dispatch.params.get(1) == Some(&1)
        }),
        "K=1 must not select K-split GEMV: {:?}",
        plan.dispatches
            .iter()
            .filter(|d| matches!(
                d.shader,
                ShaderEntry::MatMulGemv | ShaderEntry::MatMulGemvAdd
            ))
            .map(|d| (&d.label, &d.params))
            .collect::<Vec<_>>()
    );
}

#[test]
fn smollm2_seq1_uses_gemv_and_fused_residual() {
    let config = SmolLM2Config::smollm2_135m();
    let mut graph = Graph::new();
    let logits = smollm2::build_graph(&mut graph, &config, 1);
    graph.set_outputs(vec![logits]);
    let mut plan = compile(&optimize::optimize(&graph));
    let counts = shader_counts(&plan);
    assert!(
        counts.get("MatMulGemv").copied().unwrap_or(0) >= 90,
        "seq=1 projections must use K-split GEMV, got {counts:?}"
    );
    fuse_rmsnorm_into_gemv(&mut plan);
    elide_seq1_attention(&mut plan);
    assert!(
        plan.dispatches.iter().any(|d| {
            !d.gemv_physical_bt
                && d.gemv_rmsnorm.is_some()
                && matches!(d.shader, ShaderEntry::MatMulGemv)
                && d.workgroups == [3072 / 4, 1, 1]
        }),
        "packed FFN-up must be K-split GEMV with folded RmsNorm, got {:?}",
        plan.dispatches
            .iter()
            .filter(|d| {
                matches!(
                    d.shader,
                    ShaderEntry::MatMulGemv | ShaderEntry::MatMulGemvBT
                )
            })
            .map(|d| {
                (
                    &d.label,
                    d.shader.clone(),
                    d.params.clone(),
                    d.gemv_physical_bt,
                    d.workgroups,
                    d.gemv_rmsnorm.is_some(),
                )
            })
            .collect::<Vec<_>>()
    );
    assert!(
        plan.dispatches.iter().any(|d| {
            !d.gemv_physical_bt
                && matches!(d.shader, ShaderEntry::MatMulGemvAdd)
                && d.workgroups == [576 / 4, 1, 1]
        }),
        "FFN-down residual must be K-split GEMV-add, got {:?}",
        plan.dispatches
            .iter()
            .filter(|d| {
                matches!(
                    d.shader,
                    ShaderEntry::MatMulGemvAdd | ShaderEntry::MatMulGemvBTAdd
                )
            })
            .map(|d| {
                (
                    &d.label,
                    d.shader.clone(),
                    d.params.clone(),
                    d.gemv_physical_bt,
                    d.workgroups,
                )
            })
            .collect::<Vec<_>>()
    );
    let n = plan.dispatches.len();
    let mut groups = vec![0..n];
    fuse_horizontal_matmuls(&mut plan.dispatches, &mut groups);
    assert!(
        !plan.dispatches.iter().any(|d| {
            matches!(
                d.shader,
                ShaderEntry::MultiHeadAttn
                    | ShaderEntry::FlashAttention
                    | ShaderEntry::FlashAttentionCoop
            )
        }),
        "seq=1 attention is softmax of one score and must not launch"
    );
    assert_eq!(
        plan.dispatches
            .iter()
            .filter(|d| matches!(d.shader, ShaderEntry::RepeatKv))
            .count(),
        0,
        "GQA V-repeat must fold into o_proj GEMV"
    );
    assert!(
        plan.dispatches
            .iter()
            .filter(|d| d.gemv_repeat_kv.is_some())
            .count()
            >= 30,
        "o_proj GEMV must load repeated V heads"
    );
    assert!(
        counts.get("MatMulGemvAdd").copied().unwrap_or(0)
            + counts.get("MatMulGemvBTAdd").copied().unwrap_or(0)
            >= 60,
        "seq=1 residuals must use fused GEMV-add, got {counts:?}"
    );
    assert!(
        counts.get("RmsNorm").copied().unwrap_or(0)
            + plan
                .dispatches
                .iter()
                .filter(|d| d.gemv_rmsnorm.is_some())
                .count()
            >= 30,
        "RMSNorm+scale must remain a fused kernel or GEMV prologue, got {counts:?}"
    );
    assert_eq!(
        counts.get("RoPE").copied().unwrap_or(0),
        0,
        "seq=1 pos=0 RoPE is the identity rotation and must not launch, got {counts:?}"
    );
}

#[test]
fn q1_kv_long_cross_attention_is_not_elided() {
    let mut graph = Graph::new();
    let q = graph.input("q", &[1, 32]);
    let k = graph.input("k", &[8, 16]);
    let v = graph.input("v", &[8, 16]);
    let y = graph.cross_attention(q, k, v, 4, 2, 8);
    graph.set_outputs(vec![y]);
    let mut plan = compile(&graph);
    assert!(
        plan.dispatches.iter().any(|d| {
            matches!(
                d.shader,
                ShaderEntry::MultiHeadAttn
                    | ShaderEntry::FlashAttention
                    | ShaderEntry::FlashAttentionCoop
            ) && d.params.first() == Some(&1)
                && d.params.get(1) == Some(&8)
        }),
        "cross-attn q=1 kv=8 must compile as attention, got {:?}",
        plan.dispatches
            .iter()
            .map(|d| (&d.label, d.shader.clone(), d.params.clone()))
            .collect::<Vec<_>>()
    );
    elide_seq1_attention(&mut plan);
    assert!(
        plan.dispatches.iter().any(|d| {
            matches!(
                d.shader,
                ShaderEntry::MultiHeadAttn
                    | ShaderEntry::FlashAttention
                    | ShaderEntry::FlashAttentionCoop
            )
        }),
        "q=1 kv>1 must not be replaced by repeat(V), got {:?}",
        plan.dispatches
            .iter()
            .map(|d| (
                &d.label,
                d.shader.clone(),
                d.params.clone(),
                d.gemv_repeat_kv
            ))
            .collect::<Vec<_>>()
    );
    assert!(
        plan.dispatches.iter().all(|d| d.gemv_repeat_kv.is_none()),
        "cross-attn must not fold repeat(V) into a GEMV"
    );
}

#[test]
fn smollm2_prefill_fuses_swiglu_and_residual() {
    let config = SmolLM2Config::smollm2_135m();
    let mut graph = Graph::new();
    let logits = smollm2::build_graph(&mut graph, &config, 128);
    graph.set_outputs(vec![logits]);
    let plan = compile(&optimize::optimize(&graph));
    let counts = shader_counts(&plan);
    assert_eq!(
        counts.get("SwiGLUConcat").copied().unwrap_or(0),
        30,
        "llama.cpp GLU equivalent: packed gate+up, got {counts:?}"
    );
    assert_eq!(
        counts.get("FusedMatMulAdd").copied().unwrap_or(0),
        60,
        "llama.cpp MUL_MAT+ADD equivalent: fused residual, got {counts:?}"
    );
    assert!(
        counts.get("RmsNorm").copied().unwrap_or(0) >= 60,
        "RMSNorm includes the scale (llama.cpp RMS_NORM+MUL), got {counts:?}"
    );
    assert_eq!(
        counts.get("RoPE").copied().unwrap_or(0),
        60,
        "seq=128 RoPE is not identity, got {counts:?}"
    );
}

#[test]
fn wide_inference_gemv_physically_transposes_b() {
    let mut graph = Graph::new();
    let a = graph.input("a", &[1, 576]);
    let b = graph.parameter("b", &[576, 3072]);
    let c = graph.matmul(a, b);
    graph.set_outputs(vec![c]);
    let plan = compile(&graph);
    let d = plan
        .dispatches
        .iter()
        .find(|d| matches!(d.shader, ShaderEntry::MatMulGemv))
        .expect("wide GEMV");
    assert!(
        !d.gemv_physical_bt,
        "physical-BT raised Inferena seq=1 wall; wide GEMV stays K-split"
    );
    assert_eq!(d.workgroups, [3072 / 4, 1, 1]);
    assert_eq!(d.params, vec![1, 576, 3072, 0]);
}

#[test]
fn longk_gemv_add_physically_transposes_b() {
    let mut graph = Graph::new();
    let a = graph.input("a", &[1, 1536]);
    let b = graph.parameter("b", &[1536, 576]);
    let residual = graph.input("d", &[1, 576]);
    let mm = graph.matmul(a, b);
    let out = graph.add(mm, residual);
    graph.set_outputs(vec![out]);
    let plan = compile(&optimize::optimize(&graph));
    let d = plan
        .dispatches
        .iter()
        .find(|d| matches!(d.shader, ShaderEntry::MatMulGemvAdd))
        .expect("long-K GEMV-add");
    assert!(!d.gemv_physical_bt);
    assert_eq!(d.workgroups, [576 / 4, 1, 1]);
}

#[test]
fn qproj_gemv_stays_k_split_so_kv_can_pack() {
    let mut graph = Graph::new();
    let a = graph.input("a", &[1, 576]);
    let b = graph.parameter("b", &[576, 576]);
    let c = graph.matmul(a, b);
    graph.set_outputs(vec![c]);
    let plan = compile(&graph);
    let d = plan
        .dispatches
        .iter()
        .find(|d| matches!(d.shader, ShaderEntry::MatMulGemv))
        .expect("Q-proj GEMV");
    assert!(
        !d.gemv_physical_bt,
        "N=576 stays K-split so K/V (N=192) keep the same family and pack"
    );
    assert_eq!(d.workgroups, [576 / 4, 1, 1]);
}

#[test]
fn training_m1_does_not_physically_transpose_shared_weights() {
    let mut graph = Graph::new();
    let tokens = graph.input_u32("token_ids", &[1]);
    let embed = graph.parameter("embed", &[32, 576]);
    let x = graph.embedding(tokens, embed);
    let w = graph.parameter("w", &[576, 3072]);
    let logits = graph.matmul(x, w);
    let labels = graph.input("labels", &[1, 3072]);
    let loss = graph.cross_entropy_loss(logits, labels);
    graph.set_outputs(vec![loss]);
    let (plan, _) = compile_training_graph(&graph);
    assert!(
        !plan.dispatches.iter().any(|d| d.gemv_physical_bt),
        "training reuses W as MatMulBT B; physical [N,K] would break dA, got {:?}",
        plan.dispatches
            .iter()
            .filter(|d| d.gemv_physical_bt
                || matches!(
                    d.shader,
                    ShaderEntry::MatMulGemv
                        | ShaderEntry::MatMulGemvBT
                        | ShaderEntry::MatMulGemvAdd
                        | ShaderEntry::MatMulGemvBTAdd
                ))
            .map(|d| (&d.label, d.shader.clone(), d.gemv_physical_bt, d.workgroups))
            .collect::<Vec<_>>()
    );
}
