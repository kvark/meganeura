//! Diagnostic pass: walks a compiled `ExecutionPlan` and identifies
//! fusion opportunities that the current compiler is leaving on the
//! table.
//!
//! Runs on the SmolVLA training graph by default. Prints a per-pattern
//! report with estimated barrier-overhead savings (~33µs per eliminated
//! dispatch on NVIDIA).
//!
//! Usage:
//!   cargo run --release --example diagnose_fusion

use std::collections::HashMap;

use meganeura::{
    compile::{BufferRef, Dispatch, ExecutionPlan, ShaderEntry},
    compile_training_graph,
    models::{
        smollm2::{self, SmolLM2Config},
        smolvla::{self, SmolVLAConfig},
    },
};

/// Approximate barrier cost per saved dispatch on NVIDIA (from the
/// existing fusion-pass docs).
const BARRIER_US: f64 = 33.0;

#[derive(Debug)]
struct Finding {
    kind: &'static str,
    producer_idx: usize,
    consumer_idx: usize,
    note: String,
}

fn shader_name(s: &ShaderEntry) -> String {
    format!("{:?}", s)
}

/// True if `shader` is any matmul variant that currently supports
/// epilogue absorption (scalar path only — coop matmul can't fuse).
fn is_scalar_matmul(d: &Dispatch) -> bool {
    use ShaderEntry::*;
    !d.use_coop
        && matches!(
            d.shader,
            MatMul | MatMulAT | MatMulBT | FusedMatMulAdd | FusedMatMulATAdd | FusedMatMulBTAdd
        )
}

fn is_rms_norm(d: &Dispatch) -> bool {
    matches!(d.shader, ShaderEntry::RmsNorm)
}

/// Elementwise-ish consumers whose work *could* be absorbed into a
/// matmul epilogue if their sole input came from a scalar matmul:
///   - unary: Relu/Sigmoid/Silu/Neg/Tanh/Abs (though only Relu/Silu/
///     Sigmoid/Neg are currently plumbed in fuse_epilogues).
///   - binary: Add/BiasAdd (only BiasAdd is broadcast-friendly).
fn absorbable_into_matmul(d: &Dispatch) -> Option<&'static str> {
    use ShaderEntry::*;
    match d.shader {
        Relu => Some("Relu"),
        Silu => Some("Silu"),
        Sigmoid => Some("Sigmoid"),
        Neg => Some("Neg"),
        Tanh => Some("Tanh (not currently in epilogue enum)"),
        Abs => Some("Abs (not currently in epilogue enum)"),
        Add => Some("Add (binary, extra buffer)"),
        BiasAdd => Some("BiasAdd"),
        _ => None,
    }
}

fn count_consumers(plan: &ExecutionPlan) -> HashMap<BufferRef, Vec<usize>> {
    let mut m: HashMap<BufferRef, Vec<usize>> = HashMap::new();
    for (i, d) in plan.dispatches.iter().enumerate() {
        for b in &d.input_buffers {
            m.entry(*b).or_default().push(i);
        }
        for b in &d.epilogue_buffers {
            m.entry(*b).or_default().push(i);
        }
    }
    m
}

fn producer_of(plan: &ExecutionPlan) -> HashMap<BufferRef, usize> {
    let mut m = HashMap::new();
    for (i, d) in plan.dispatches.iter().enumerate() {
        m.insert(d.output_buffer, i);
    }
    m
}

/// Walk plan and list unfused opportunities.
fn diagnose(plan: &ExecutionPlan) -> Vec<Finding> {
    let consumers = count_consumers(plan);
    let producer = producer_of(plan);

    // Anything that's read by "outside" the dispatch list (plan output,
    // parameter, etc.) must survive — we track these to avoid proposing
    // fusions that would eliminate an externally-visible buffer.
    let mut external: std::collections::HashSet<BufferRef> = Default::default();
    external.extend(plan.output_buffers.iter().copied());
    if let Some(b) = plan.loss_buffer {
        external.insert(b);
    }
    for (_, b) in &plan.param_buffers {
        external.insert(*b);
    }
    for (_, b) in &plan.input_buffers {
        external.insert(*b);
    }
    for (b, _) in &plan.constant_buffers {
        external.insert(*b);
    }
    for (_, b) in &plan.lse_buffers {
        external.insert(*b);
    }

    let mut out = Vec::new();

    for (i, d) in plan.dispatches.iter().enumerate() {
        // Pattern 1: absorb-into-matmul epilogue.
        if let Some(reason) = absorbable_into_matmul(d)
            && let Some(&primary) = d.input_buffers.first()
            && !external.contains(&primary)
            && let Some(&prod_i) = producer.get(&primary)
        {
            let prod = &plan.dispatches[prod_i];
            if is_scalar_matmul(prod) && consumers.get(&primary).map_or(0, |v| v.len()) == 1 {
                out.push(Finding {
                    kind: "matmul+elementwise epilogue not fused",
                    producer_idx: prod_i,
                    consumer_idx: i,
                    note: format!(
                        "{} → {} ({})",
                        shader_name(&prod.shader),
                        shader_name(&d.shader),
                        reason
                    ),
                });
            }
        }

        // Pattern 2: MatMul consumes an RmsNorm output — could in
        // principle be FusedRmsNormMatMul if not already.
        if matches!(
            d.shader,
            ShaderEntry::MatMul | ShaderEntry::MatMulBT | ShaderEntry::MatMulAT
        ) && !d.use_coop
            && d.input_buffers.len() >= 2
        {
            for (slot_idx, in_buf) in d.input_buffers[..2].iter().enumerate() {
                if let Some(&prod_i) = producer.get(in_buf)
                    && is_rms_norm(&plan.dispatches[prod_i])
                    && consumers.get(in_buf).map_or(0, |v| v.len()) == 1
                    && !external.contains(in_buf)
                {
                    out.push(Finding {
                        kind: "RmsNorm+MatMul not fused",
                        producer_idx: prod_i,
                        consumer_idx: i,
                        note: format!("RmsNorm → {} (slot {})", shader_name(&d.shader), slot_idx),
                    });
                }
            }
        }

        // Pattern 3: MatMul → (single consumer) Add/BiasAdd with a matmul-fused variant
        // already existing (FusedMatMulAdd). Count cases where this is being done in
        // two dispatches.
        if matches!(d.shader, ShaderEntry::Add | ShaderEntry::BiasAdd) && d.pointwise.is_none() {
            for (slot_idx, in_buf) in d.input_buffers.iter().enumerate() {
                if !external.contains(in_buf)
                    && let Some(&prod_i) = producer.get(in_buf)
                    && is_scalar_matmul(&plan.dispatches[prod_i])
                    && consumers.get(in_buf).map_or(0, |v| v.len()) == 1
                {
                    out.push(Finding {
                        kind: "MatMul+Add candidate for FusedMatMulAdd",
                        producer_idx: prod_i,
                        consumer_idx: i,
                        note: format!(
                            "{} → {} (producer at input slot {})",
                            shader_name(&plan.dispatches[prod_i].shader),
                            shader_name(&d.shader),
                            slot_idx
                        ),
                    });
                    break;
                }
            }
        }
    }

    out
}

fn print_report(label: &str, plan: &ExecutionPlan, findings: &[Finding]) {
    println!("\n=== {} ({} dispatches) ===", label, plan.dispatches.len());
    if findings.is_empty() {
        println!("  (no unfused patterns detected)");
        return;
    }
    // Group by kind.
    let mut by_kind: HashMap<&'static str, Vec<&Finding>> = HashMap::new();
    for f in findings {
        by_kind.entry(f.kind).or_default().push(f);
    }
    let total = findings.len();
    let saved_ms = total as f64 * BARRIER_US / 1000.0;
    println!(
        "  {} unfused opportunities. If all fused, saves ≈ {:.2} ms barrier overhead.",
        total, saved_ms,
    );
    let mut kinds: Vec<_> = by_kind.keys().copied().collect();
    kinds.sort();
    for k in kinds {
        let fs = &by_kind[k];
        println!("  [{}] {}", fs.len(), k);
        // Show up to 5 examples.
        for f in fs.iter().take(5) {
            println!(
                "      #{} → #{}: {}",
                f.producer_idx, f.consumer_idx, f.note
            );
        }
        if fs.len() > 5 {
            println!("      ... and {} more", fs.len() - 5);
        }
    }
}

fn dispatch_histogram(plan: &ExecutionPlan) -> Vec<(String, usize)> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for d in &plan.dispatches {
        *counts.entry(shader_name(&d.shader)).or_default() += 1;
    }
    let mut v: Vec<_> = counts.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1));
    v
}

fn analyze(label: &str, plan: &ExecutionPlan) {
    let findings = diagnose(plan);
    print_report(label, plan, &findings);
    println!("\n  dispatch histogram:");
    for (name, count) in dispatch_histogram(plan).iter().take(12) {
        println!("    {:>6}  {}", count, name);
    }
    println!(
        "\n  Barrier overhead assuming {:.0} µs/group × {} groups: ≈ {:.2} ms\n",
        BARRIER_US,
        plan.dispatches.len(),
        plan.dispatches.len() as f64 * BARRIER_US / 1000.0,
    );
}

fn main() {
    env_logger::init();

    // Pick model(s) to analyze from CLI; default: all.
    let args: Vec<String> = std::env::args().skip(1).collect();
    let models: &[&str] = if args.is_empty() {
        &["smolvla-train", "smollm2-train", "smollm2-prefill"]
    } else {
        // &args is Vec<String>, can't coerce to &[&str] directly; use a
        // simpler runtime dispatch below.
        &[]
    };

    let run = |name: &str| {
        match name {
            "smolvla-train" => {
                let config = SmolVLAConfig::smolvla_base();
                eprintln!(
                    "Building SmolVLA training (chunk={}, vlm_seq=16)",
                    config.chunk_size
                );
                let g = smolvla::build_action_expert_training(&config, config.chunk_size, 16);
                let (plan, _) = compile_training_graph(&g);
                analyze("SmolVLA training", &plan);
            }
            "smollm2-train" => {
                let config = SmolLM2Config::smollm2_135m();
                let seq = 128;
                eprintln!("Building SmolLM2-135M training (seq={})", seq);
                let g = smollm2::build_training_graph(&config, seq);
                let (plan, _) = compile_training_graph(&g);
                analyze("SmolLM2-135M training", &plan);
            }
            "smollm2-prefill" => {
                let config = SmolLM2Config::smollm2_135m();
                let seq = 128;
                eprintln!("Building SmolLM2-135M prefill (seq={})", seq);
                let mut g = meganeura::Graph::new();
                let (logits, _k, _v) = smollm2::build_prefill_graph(&mut g, &config, seq);
                g.set_outputs(vec![logits]);
                // Inference graph — compile directly; no autodiff.
                let optimized = meganeura::optimize::optimize(&g);
                let plan = meganeura::compile::compile(&optimized);
                analyze("SmolLM2-135M prefill", &plan);
            }
            "smollm2-decode" => {
                let config = SmolLM2Config::smollm2_135m();
                let max_seq = 256;
                eprintln!("Building SmolLM2-135M decode (max_seq={})", max_seq);
                let mut g = meganeura::Graph::new();
                let _ = smollm2::build_decode_graph(&mut g, &config, max_seq);
                let optimized = meganeura::optimize::optimize(&g);
                let plan = meganeura::compile::compile(&optimized);
                analyze("SmolLM2-135M decode", &plan);
            }
            other => eprintln!("unknown model: {}", other),
        }
    };

    if args.is_empty() {
        for m in models {
            run(m);
        }
    } else {
        for m in &args {
            run(m);
        }
    }
}
