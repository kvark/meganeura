//! CPU survey of bounded alternatives inside model regions; no GPU measurement.
use meganeura::{
    Graph,
    models::{smolvla, whisper},
    optimize::search,
};
use std::time::Instant;

fn main() {
    let model = std::env::args().nth(1).expect("SmolVLA or Whisper-tiny");
    let mut graph = Graph::new();
    let output = match model.as_str() {
        "SmolVLA" => {
            smolvla::build_action_expert(&mut graph, &smolvla::Config::smolvla_base(), 50, 16)
        }
        "Whisper-tiny" => {
            whisper::build_encoder(&mut graph, &whisper::Config::whisper_tiny(), 1, 3000)
        }
        _ => panic!("SmolVLA or Whisper-tiny"),
    };
    graph.set_outputs(vec![output]);
    let regions = meganeura::outline::detect_repeated_regions(&graph);
    assert!(!regions.is_empty());
    let mut results = Vec::new();
    for region in regions {
        let start = Instant::now();
        let space =
            search::region_candidates(&graph, region.start..region.start + region.period, 8)
                .unwrap();
        results.push(serde_json::json!({
            "start": region.start, "nodes": region.period, "instances": region.count,
            "search_ms": start.elapsed().as_secs_f64() * 1000.0,
            "truncated": space.truncated,
            "expressions": space.candidates.iter().map(|c| &c.expression).collect::<Vec<_>>(),
            "candidate_nodes": space.candidates.iter().map(|c| c.graph.nodes().len()).collect::<Vec<_>>(),
        }));
    }
    println!(
        "{}",
        serde_json::json!({"model": model, "source_nodes": graph.nodes().len(), "regions": results})
    );
}
