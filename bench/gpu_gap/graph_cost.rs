use meganeura::{Graph, autodiff, graph::Op, models::{resnet, sd_unet, smollm2, smolvla, whisper}, optimize, outline};
use std::time::Instant;

fn timed<T>(f: impl FnOnce() -> T) -> (T, f64) {
    let start = Instant::now();
    let value = f();
    (value, start.elapsed().as_secs_f64() * 1000.0)
}

fn main() {
    for name in ["SmolLM2-135M", "SmolVLA", "Whisper-tiny", "StableDiffusion", "ResNet-50"] {
        let source = match name {
            "SmolLM2-135M" => smollm2::build_training_graph(&smollm2::Config::smollm2_135m(), 128),
            "SmolVLA" => smolvla::build_action_expert_training(&smolvla::Config::smolvla_base(), 50, 16),
            "Whisper-tiny" => whisper::build_training_graph(&whisper::Config::whisper_tiny(), 1, 3000),
            "ResNet-50" => resnet::build_resnet50_training(4),
            _ => {
                let mut graph = Graph::new();
                let out = sd_unet::build_training_graph(&mut graph, &sd_unet::Config::small());
                graph.set_outputs(vec![out]);
                graph
            }
        };
        for sample in 0..5 {
            let ((forward, _), forward_ms) = timed(|| optimize::optimize_with_report(&source));
            let (diff, ad_ms) = timed(|| autodiff::differentiate(&forward.toposort()));
            let (copy, copy_ms) = timed(|| diff.deep_clone());
            let (regions, outline_ms) = timed(|| outline::detect_repeated_regions(&diff));
            let ((result, report), optimize_ms) = timed(|| optimize::optimize_with_report(&diff));
            let constant_bytes: usize = diff.nodes().iter().map(|n| match n.op { Op::Constant { ref data } => data.len() * 4, _ => 0 }).sum();
            println!("{}", serde_json::json!({"model": name, "sample": sample, "nodes": diff.nodes().len(), "constant_bytes": constant_bytes,
                "forward_ms": forward_ms, "ad_ms": ad_ms, "copy_ms": copy_ms, "outline_ms": outline_ms, "optimize_ms": optimize_ms,
                "egglog_ms": report.egglog_time.as_secs_f64()*1000.0, "stamp_ms": report.extract_time.as_secs_f64()*1000.0, "regions": regions.len(), "optimized_nodes": report.nodes_after}));
            std::hint::black_box((copy, result));
        }
    }
}
