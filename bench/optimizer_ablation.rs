//! CPU-only optimizer ablation on the model graphs used by Inferena.
//!
//! Examples:
//!   cargo run --release --example optimizer_ablation -- \
//!     --model SmolLM2-135M --phase training --mode egglog-outlined
//!   timeout 120 cargo run --release --example optimizer_ablation -- \
//!     --model SmolVLA --phase training --mode egglog-whole

use meganeura::{
    ExtractionCost, Graph, OptimizeConfig, OptimizeMode, OptimizeReport, autodiff,
    models::{resnet, sd_unet, smollm2, smolvla, whisper},
    optimize::optimize_with_config,
};
use serde_json::json;
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Inference,
    Training,
}

fn parse_mode(value: &str) -> OptimizeMode {
    match value {
        "off" => OptimizeMode::Off,
        "greedy" => OptimizeMode::Greedy,
        "egglog-windowed" | "windowed" => OptimizeMode::EgglogWindowed,
        "egglog-outlined" | "outlined" => OptimizeMode::EgglogOutlined,
        "egglog-whole" | "whole" => OptimizeMode::EgglogWhole,
        _ => panic!("unknown optimizer mode {value:?}"),
    }
}

fn parse_cost(value: &str) -> ExtractionCost {
    match value {
        "ast-size" | "ast" | "unit" => ExtractionCost::AstSize,
        "tensor-traffic" | "traffic" => ExtractionCost::TensorTraffic,
        _ => panic!("unknown extraction cost {value:?}"),
    }
}

fn build_model_graph(model: &str, phase: Phase) -> Graph {
    match (model, phase) {
        ("SmolLM2-135M", Phase::Inference) => {
            let config = smollm2::SmolLM2Config::smollm2_135m();
            let mut graph = Graph::new();
            let output = smollm2::build_graph(&mut graph, &config, 128);
            graph.set_outputs(vec![output]);
            graph
        }
        ("SmolLM2-135M", Phase::Training) => {
            smollm2::build_training_graph(&smollm2::SmolLM2Config::smollm2_135m(), 128)
        }
        ("SmolVLA", Phase::Inference) => {
            let config = smolvla::SmolVLAConfig::smolvla_base();
            let mut graph = Graph::new();
            let output = smolvla::build_action_expert(&mut graph, &config, 50, 16);
            graph.set_outputs(vec![output]);
            graph
        }
        ("SmolVLA", Phase::Training) => {
            smolvla::build_action_expert_training(&smolvla::SmolVLAConfig::smolvla_base(), 50, 16)
        }
        ("SD-style-UNet" | "StableDiffusion", Phase::Inference) => {
            let config = sd_unet::SDUNetConfig::small();
            let mut graph = Graph::new();
            let output = sd_unet::build_unet(&mut graph, &config);
            graph.set_outputs(vec![output]);
            graph
        }
        ("SD-style-UNet" | "StableDiffusion", Phase::Training) => {
            let config = sd_unet::SDUNetConfig::small();
            let mut graph = Graph::new();
            let loss = sd_unet::build_training_graph(&mut graph, &config);
            graph.set_outputs(vec![loss]);
            graph
        }
        ("ResNet-50", Phase::Inference) => {
            let mut graph = Graph::new();
            let output = resnet::build_resnet50(&mut graph, 4);
            graph.set_outputs(vec![output]);
            graph
        }
        ("ResNet-50", Phase::Training) => resnet::build_resnet50_training(4),
        ("Whisper-tiny", Phase::Inference) => {
            let config = whisper::WhisperConfig::whisper_tiny();
            let mut graph = Graph::new();
            let output = whisper::build_encoder(&mut graph, &config, 1, 3000);
            graph.set_outputs(vec![output]);
            graph
        }
        ("Whisper-tiny", Phase::Training) => {
            whisper::build_training_graph(&whisper::WhisperConfig::whisper_tiny(), 1, 3000)
        }
        _ => panic!("unsupported model/phase: {model} {phase:?}"),
    }
}

fn report_json(report: &OptimizeReport) -> serde_json::Value {
    json!({
        "mode": report.mode.as_str(),
        "extraction_cost": report.extraction_cost.as_str(),
        "nodes_before": report.nodes_before,
        "nodes_after": report.nodes_after,
        "fusions": report.fusions_applied.len(),
        "rules_fired": report.rules_fired,
        "egglog_ms": report.egglog_time.as_secs_f64() * 1000.0,
        "extract_ms": report.extract_time.as_secs_f64() * 1000.0,
        "eclasses": report.num_eclasses,
        "enodes": report.num_enodes,
        "outlined_regions": report.outlined_regions,
        "segments": report.segments,
        "max_segment_nodes": report.max_segment_nodes,
        "extraction_failures": report.extraction_failures,
    })
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    }
}

fn main() {
    let mut model = "SmolLM2-135M".to_string();
    let mut phase = Phase::Training;
    let mut mode = OptimizeMode::EgglogOutlined;
    let mut extraction_cost = ExtractionCost::TensorTraffic;
    let mut cutoff = 300usize;
    let mut repeats = 1usize;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => model = args.next().expect("--model requires a value"),
            "--phase" => {
                phase = match args.next().expect("--phase requires a value").as_str() {
                    "inference" => Phase::Inference,
                    "training" => Phase::Training,
                    value => panic!("unknown phase {value:?}"),
                }
            }
            "--mode" => mode = parse_mode(&args.next().expect("--mode requires a value")),
            "--cost" => {
                extraction_cost = parse_cost(&args.next().expect("--cost requires a value"))
            }
            "--cutoff" => {
                cutoff = args
                    .next()
                    .expect("--cutoff requires a value")
                    .parse()
                    .expect("--cutoff must be an integer")
            }
            "--repeats" => {
                repeats = args
                    .next()
                    .expect("--repeats requires a value")
                    .parse()
                    .expect("--repeats must be an integer")
            }
            value => panic!("unknown argument {value:?}"),
        }
    }
    assert!(cutoff > 0, "--cutoff must be positive");
    assert!(repeats > 0, "--repeats must be positive");

    let config = OptimizeConfig {
        mode,
        extraction_cost,
        saturation_cutoff: cutoff,
    };
    let source = build_model_graph(&model, phase);
    let source_nodes = source.nodes().len();
    let mut samples_ms = Vec::with_capacity(repeats);
    let mut last_forward = None;
    let mut last_full = None;
    let mut final_nodes = 0usize;

    for _ in 0..repeats {
        let start = Instant::now();
        let (optimized_forward, forward_report) = optimize_with_config(&source, config);
        let (final_graph, full_report) = if phase == Phase::Training {
            let differentiated = autodiff::differentiate(&optimized_forward.toposort());
            let (optimized, report) = optimize_with_config(&differentiated, config);
            (optimized, Some(report))
        } else {
            (optimized_forward, None)
        };
        samples_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        final_nodes = final_graph
            .nodes()
            .iter()
            .filter(|node| !matches!(node.op, meganeura::graph::Op::Nop))
            .count();
        last_forward = Some(forward_report);
        last_full = full_report;
    }

    let median_ms = median(&mut samples_ms.clone());
    println!(
        "{}",
        serde_json::to_string(&json!({
            "model": model,
            "phase": match phase {
                Phase::Inference => "inference",
                Phase::Training => "training",
            },
            "mode": mode.as_str(),
            "extraction_cost": extraction_cost.as_str(),
            "cutoff": cutoff,
            "repeats": repeats,
            "source_nodes": source_nodes,
            "final_active_nodes": final_nodes,
            "total_ms": {
                "median": median_ms,
                "samples": samples_ms,
            },
            "forward_optimization": report_json(last_forward.as_ref().unwrap()),
            "full_graph_optimization": last_full.as_ref().map(report_json),
        }))
        .unwrap()
    );
}
