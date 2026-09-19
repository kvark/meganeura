//! CPU dump of Inferena's five-model graphs: raw ops, optimizer fusions,
//! and compiled dispatch histograms. No GPU context.
//!
//! Usage:
//!   cargo run --release --example dump_inferena_graphs -- target/llama-compare/graphs.json

use std::collections::BTreeMap;
use std::path::PathBuf;

use meganeura::compile::{ExecutionPlan, ShaderEntry, fuse_rmsnorm_into_gemv};
use meganeura::graph::{Graph, Op};
use meganeura::models::{
    resnet,
    sd_unet::{self, SDUNetConfig},
    smollm2::{self, SmolLM2Config},
    smolvla::{self, SmolVLAConfig},
    whisper::{self, WhisperConfig},
};
use meganeura::optimize::{OptimizeReport, optimize_with_report};
use meganeura::{compile_training_graph, compile::compile};
use serde_json::{Value, json};

fn op_kind(op: &Op) -> String {
    let debug = format!("{op:?}");
    debug
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect()
}

fn histogram_ops(graph: &Graph) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for node in graph.nodes() {
        if matches!(node.op, Op::Nop) {
            continue;
        }
        *counts.entry(op_kind(&node.op)).or_default() += 1;
    }
    counts
}

fn shader_name(shader: &ShaderEntry) -> String {
    format!("{shader:?}")
}

fn histogram_shaders(plan: &ExecutionPlan) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for dispatch in &plan.dispatches {
        *counts.entry(shader_name(&dispatch.shader)).or_default() += 1;
    }
    counts
}

fn matmul_shapes(plan: &ExecutionPlan) -> Vec<Value> {
    let mut shapes: BTreeMap<(String, u32, u32, u32, u32, bool, bool, bool), usize> =
        BTreeMap::new();
    for dispatch in &plan.dispatches {
        let (m, n, k) = match dispatch.shader {
            ShaderEntry::MatMul
            | ShaderEntry::FusedMatMulAdd
            | ShaderEntry::MatMulGemv
            | ShaderEntry::MatMulGemvAdd => (dispatch.params[0], dispatch.params[2], dispatch.params[1]),
            ShaderEntry::MatMulAT
            | ShaderEntry::MatMulBT
            | ShaderEntry::FusedMatMulATAdd
            | ShaderEntry::FusedMatMulBTAdd
            | ShaderEntry::MatMulGemvBT => (dispatch.params[0], dispatch.params[1], dispatch.params[2]),
            _ => continue,
        };
        let key = (
            shader_name(&dispatch.shader),
            m,
            n,
            k,
            dispatch.horizontal_batch,
            dispatch.gemv_rmsnorm.is_some(),
            dispatch.matmul_prologue.is_some(),
            dispatch.matmul_epilogue.is_some(),
        );
        *shapes.entry(key).or_default() += 1usize;
    }
    shapes
        .into_iter()
        .map(
            |((shader, m, n, k, horizontal, gemv_rmsnorm, prologue, epilogue), count)| {
                json!({
                    "count": count,
                    "shader": shader,
                    "m": m,
                    "n": n,
                    "k": k,
                    "horizontal_batch": horizontal,
                    "gemv_rmsnorm": gemv_rmsnorm,
                    "matmul_prologue": prologue,
                    "matmul_epilogue": epilogue,
                })
            },
        )
        .collect()
}

fn fusion_counts(report: &OptimizeReport) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for (name, _) in &report.fusions_applied {
        *counts.entry(name.clone()).or_default() += 1;
    }
    counts
}

fn plan_json(plan: &ExecutionPlan) -> Value {
    let gemv = plan
        .dispatches
        .iter()
        .filter(|d| {
            matches!(
                d.shader,
                ShaderEntry::MatMulGemv | ShaderEntry::MatMulGemvAdd | ShaderEntry::MatMulGemvBT
            )
        })
        .count();
    let fused_add = plan
        .dispatches
        .iter()
        .filter(|d| {
            matches!(
                d.shader,
                ShaderEntry::FusedMatMulAdd
                    | ShaderEntry::FusedMatMulATAdd
                    | ShaderEntry::FusedMatMulBTAdd
                    | ShaderEntry::MatMulGemvAdd
            )
        })
        .count();
    let rmsnorm_folded = plan
        .dispatches
        .iter()
        .filter(|d| d.gemv_rmsnorm.is_some())
        .count();
    let prologues = plan
        .dispatches
        .iter()
        .filter(|d| d.matmul_prologue.is_some())
        .count();
    let epilogues = plan
        .dispatches
        .iter()
        .filter(|d| d.matmul_epilogue.is_some())
        .count();
    let horizontal = plan
        .dispatches
        .iter()
        .filter(|d| d.horizontal_batch >= 2)
        .count();
    json!({
        "dispatch_count": plan.dispatches.len(),
        "shaders": histogram_shaders(plan),
        "matmul_shapes": matmul_shapes(plan),
        "gemv_dispatches": gemv,
        "fused_add_dispatches": fused_add,
        "gemv_rmsnorm_dispatches": rmsnorm_folded,
        "matmul_prologue_dispatches": prologues,
        "matmul_epilogue_dispatches": epilogues,
        "horizontal_packed_dispatches": horizontal,
        "derived_params": plan.derived_params.len(),
    })
}

fn dump_inference(name: &str, mode: &str, graph: Graph) -> Value {
    let raw_ops = histogram_ops(&graph);
    let raw_nodes = graph
        .nodes()
        .iter()
        .filter(|n| !matches!(n.op, Op::Nop))
        .count();
    let (optimized, report) = optimize_with_report(&graph);
    let mut plan = compile(&optimized);
    fuse_rmsnorm_into_gemv(&mut plan);
    json!({
        "model": name,
        "mode": mode,
        "raw_nodes": raw_nodes,
        "raw_ops": raw_ops,
        "optimize": {
            "mode": report.mode.as_str(),
            "nodes_before": report.nodes_before,
            "nodes_after": report.nodes_after,
            "fusions": fusion_counts(&report),
            "outlined_regions": report.outlined_regions,
            "segments": report.segments,
            "extraction_failures": report.extraction_failures,
        },
        "optimized_ops": histogram_ops(&optimized),
        "plan": plan_json(&plan),
    })
}

fn dump_training(name: &str, graph: Graph) -> Value {
    let raw_ops = histogram_ops(&graph);
    let raw_nodes = graph
        .nodes()
        .iter()
        .filter(|n| !matches!(n.op, Op::Nop))
        .count();
    let (mut plan, report) = compile_training_graph(&graph);
    fuse_rmsnorm_into_gemv(&mut plan);
    json!({
        "model": name,
        "mode": "training",
        "raw_nodes": raw_nodes,
        "raw_ops": raw_ops,
        "optimize": {
            "mode": report.mode.as_str(),
            "nodes_before": report.nodes_before,
            "nodes_after": report.nodes_after,
            "fusions": fusion_counts(&report),
            "outlined_regions": report.outlined_regions,
            "segments": report.segments,
            "extraction_failures": report.extraction_failures,
        },
        "plan": plan_json(&plan),
    })
}

fn main() {
    let out = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("inferena-graphs.json"));

    let mut rows = Vec::new();

    {
        let config = SmolLM2Config::smollm2_135m();
        eprintln!("SmolLM2-135M inference seq=128");
        let mut g = Graph::new();
        let logits = smollm2::build_graph(&mut g, &config, 128);
        g.set_outputs(vec![logits]);
        rows.push(dump_inference("SmolLM2-135M", "inference", g));

        eprintln!("SmolLM2-135M latency seq=1");
        let mut g = Graph::new();
        let logits = smollm2::build_graph(&mut g, &config, 1);
        g.set_outputs(vec![logits]);
        rows.push(dump_inference("SmolLM2-135M", "latency", g));

        eprintln!("SmolLM2-135M training seq=128");
        rows.push(dump_training(
            "SmolLM2-135M",
            smollm2::build_training_graph(&config, 128),
        ));

        eprintln!("SmolLM2-135M decode max_seq=128 (not Inferena)");
        let mut g = Graph::new();
        let (logits, _, _) = smollm2::build_decode_graph(&mut g, &config, 128);
        g.set_outputs(vec![logits]);
        rows.push(dump_inference("SmolLM2-135M", "decode-kv", g));
    }

    {
        let config = SmolVLAConfig::smolvla_base();
        eprintln!("SmolVLA inference");
        let mut g = Graph::new();
        let pred = smolvla::build_action_expert(&mut g, &config, 50, 16);
        g.set_outputs(vec![pred]);
        rows.push(dump_inference("SmolVLA", "inference", g));

        eprintln!("SmolVLA training");
        rows.push(dump_training(
            "SmolVLA",
            smolvla::build_action_expert_training(&config, 50, 16),
        ));
    }

    {
        let config = SDUNetConfig::small();
        eprintln!("StableDiffusion inference");
        let mut g = Graph::new();
        let pred = sd_unet::build_unet(&mut g, &config);
        g.set_outputs(vec![pred]);
        rows.push(dump_inference("StableDiffusion", "inference", g));

        let mut lat = SDUNetConfig::small();
        lat.batch_size = 1;
        eprintln!("StableDiffusion latency");
        let mut g = Graph::new();
        let pred = sd_unet::build_unet(&mut g, &lat);
        g.set_outputs(vec![pred]);
        rows.push(dump_inference("StableDiffusion", "latency", g));

        eprintln!("StableDiffusion training");
        let mut g = Graph::new();
        let loss = sd_unet::build_training_graph(&mut g, &config);
        g.set_outputs(vec![loss]);
        rows.push(dump_training("StableDiffusion", g));
    }

    {
        eprintln!("ResNet-50 inference batch=4");
        let mut g = Graph::new();
        let logits = resnet::build_resnet50(&mut g, 4);
        g.set_outputs(vec![logits]);
        rows.push(dump_inference("ResNet-50", "inference", g));

        eprintln!("ResNet-50 latency batch=1");
        let mut g = Graph::new();
        let logits = resnet::build_resnet50(&mut g, 1);
        g.set_outputs(vec![logits]);
        rows.push(dump_inference("ResNet-50", "latency", g));

        eprintln!("ResNet-50 training batch=4");
        rows.push(dump_training("ResNet-50", resnet::build_resnet50_training(4)));
    }

    {
        let config = WhisperConfig::whisper_tiny();
        eprintln!("Whisper-tiny inference");
        let mut g = Graph::new();
        let out = whisper::build_encoder(&mut g, &config, 1, 3000);
        g.set_outputs(vec![out]);
        rows.push(dump_inference("Whisper-tiny", "inference", g));

        eprintln!("Whisper-tiny training");
        rows.push(dump_training(
            "Whisper-tiny",
            whisper::build_training_graph(&config, 1, 3000),
        ));
    }

    if let Some(parent) = out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).expect("create output dir");
        }
    }
    let file = std::fs::File::create(&out).expect("create dump");
    serde_json::to_writer_pretty(std::io::BufWriter::new(file), &rows).expect("write dump");
    eprintln!("wrote {} graphs to {}", rows.len(), out.display());
}
