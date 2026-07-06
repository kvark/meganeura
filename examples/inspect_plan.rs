use meganeura::models::magenta_rt::spectrostream::{
    build_decoder_graph_through, DecoderStage, SpectroStreamConfig,
};
use meganeura::{build_inference_session, Graph};

fn dump(stage: DecoderStage, name: &str) {
    let cfg = SpectroStreamConfig::default();
    let mut g = Graph::new();
    let out = build_decoder_graph_through(&mut g, &cfg, 50, stage);
    g.set_outputs(vec![out]);
    let s = build_inference_session(&g);
    let plan = s.plan();
    println!("=== {name}: {} dispatches ===", plan.dispatches.len());
    for (i, d) in plan.dispatches.iter().enumerate() {
        let wgs = d.workgroups;
        // Output bytes:
        let out_bytes = plan.buffers[d.output_buffer.0 as usize];
        println!("  d{i:3} {:?} wgs={:?} out={out_bytes}B params={:?}", d.shader, wgs, d.params);
    }
}

fn main() {
    let arg = std::env::args().nth(1).unwrap_or("5".into());
    let b: u8 = arg.parse().unwrap();
    dump(DecoderStage::Block(b), &format!("Block({b})"));
}
