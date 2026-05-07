//! Per-stage parity bisection — compare meganeura's output at each
//! `features.<i>` boundary against torchvision. The first stage that
//! exceeds tolerance localizes the bug.

use meganeura::data::safetensors::SafeTensorsModel;
use meganeura::models::efficientnet;
use meganeura::{Graph, build_inference_session};

const SAFETENSORS_PATH: &str = "bench/results/efficientnet_v2s.safetensors";
const REF_PATH: &str = "bench/results/efficientnet_v2s_reference.json";
const STAGES_REF_PATH: &str = "bench/results/efficientnet_v2s_stages_reference.json";

fn parse_f32_array(json: &str, key: &str) -> Vec<f32> {
    let needle = format!("\"{key}\": [");
    let start = json.find(&needle).expect(key) + needle.len();
    let end = start + json[start..].find(']').unwrap();
    json[start..end]
        .split(',')
        .map(|s| s.trim().parse::<f32>().unwrap())
        .collect()
}

#[test]
fn efficientnet_v2s_stages_match_torchvision() {
    if !std::path::Path::new(SAFETENSORS_PATH).exists()
        || !std::path::Path::new(STAGES_REF_PATH).exists()
        || !std::path::Path::new(REF_PATH).exists()
    {
        eprintln!("SKIP: missing reference files");
        return;
    }
    let stages_json = std::fs::read_to_string(STAGES_REF_PATH).unwrap();
    let main_json = std::fs::read_to_string(REF_PATH).unwrap();
    let input = parse_f32_array(&main_json, "input");

    let weights = SafeTensorsModel::load(SAFETENSORS_PATH.into()).unwrap();

    // Build the graph with all 6 stage outputs exposed.
    let mut g = Graph::new();
    let stage_nodes = efficientnet::build_graph_stage_outputs(&mut g, 1);
    g.set_outputs(stage_nodes.to_vec());
    let mut session = build_inference_session(&g);

    for name in efficientnet::weight_names() {
        let data = weights.tensor_f32(&name).unwrap();
        session.set_parameter(&name, &data);
    }
    session.set_input("image", &input);
    session.step();
    session.wait();

    let stage_lens = [
        24 * 96 * 96, // stage0
        24 * 96 * 96, // stage1
        48 * 48 * 48, // stage2
        64 * 24 * 24, // stage3
        128 * 12 * 12, // stage4
        160 * 12 * 12, // stage5
    ];

    let mut overall_pass = true;
    for (i, &len) in stage_lens.iter().enumerate() {
        let mut buf = vec![0.0f32; len];
        session.read_output_by_index(i, &mut buf);
        let key = format!("\"stage{i}\":");
        let stage_section_start = stages_json.find(&key).unwrap();
        let expected = parse_f32_array(&stages_json[stage_section_start..], "data");
        assert_eq!(expected.len(), len, "stage {} len mismatch", i);

        let mut max_abs = 0.0f32;
        let mut max_idx = 0;
        for (j, (&a, &e)) in buf.iter().zip(expected.iter()).enumerate() {
            let abs = (a - e).abs();
            if abs > max_abs {
                max_abs = abs;
                max_idx = j;
            }
        }
        let exp_abs_max = expected.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let rel = max_abs / exp_abs_max.max(1e-6);
        let ok = max_abs < 0.05 || rel < 1e-3;
        if !ok {
            overall_pass = false;
        }
        eprintln!(
            "stage {}: max_abs={:.4e} (got {:.4} vs {:.4} at idx {}) rel={:.4e} {}",
            i, max_abs, buf[max_idx], expected[max_idx], max_idx, rel,
            if ok { "OK" } else { "FAIL" }
        );
    }
    assert!(overall_pass, "at least one stage exceeded tolerance — see stderr");
}
