//! A session may contain attention blocks with different head widths (for
//! example, a multimodal vision encoder and text decoder). Each generated
//! attention pipeline must be specialized independently.

use meganeura::Graph;

fn values(len: usize, phase: f32) -> Vec<f32> {
    (0..len)
        .map(|i| ((i as f32 * 0.173) + phase).sin() * 0.4)
        .collect()
}

fn run_single(head_dim: usize, q: &[f32], k: &[f32], v: &[f32]) -> Vec<f32> {
    let seq = q.len() / head_dim;
    let mut graph = Graph::new();
    let qn = graph.input("q", &[seq, head_dim]);
    let kn = graph.input("k", &[seq, head_dim]);
    let vn = graph.input("v", &[seq, head_dim]);
    let out = graph.full_attention(qn, kn, vn, 1, 1, head_dim as u32);
    graph.set_outputs(vec![out]);

    let mut session = meganeura::build(&graph, meganeura::SessionConfig::inference_from_env()).0;
    session.set_input("q", q);
    session.set_input("k", k);
    session.set_input("v", v);
    session.step();
    session.wait();
    session.read_output(q.len())
}

#[test]
fn mixed_head_dims_match_independent_sessions() {
    let seq = 4;
    let (hd_a, hd_b) = (4, 8);
    let qa = values(seq * hd_a, 0.1);
    let ka = values(seq * hd_a, 0.7);
    let va = values(seq * hd_a, 1.3);
    let qb = values(seq * hd_b, 0.2);
    let kb = values(seq * hd_b, 0.8);
    let vb = values(seq * hd_b, 1.4);
    let expected_a = run_single(hd_a, &qa, &ka, &va);
    let expected_b = run_single(hd_b, &qb, &kb, &vb);

    let mut graph = Graph::new();
    let qa_node = graph.input("qa", &[seq, hd_a]);
    let ka_node = graph.input("ka", &[seq, hd_a]);
    let va_node = graph.input("va", &[seq, hd_a]);
    let qb_node = graph.input("qb", &[seq, hd_b]);
    let kb_node = graph.input("kb", &[seq, hd_b]);
    let vb_node = graph.input("vb", &[seq, hd_b]);
    let out_a = graph.full_attention(qa_node, ka_node, va_node, 1, 1, hd_a as u32);
    let out_b = graph.full_attention(qb_node, kb_node, vb_node, 1, 1, hd_b as u32);
    graph.set_outputs(vec![out_a, out_b]);

    let mut session = meganeura::build(&graph, meganeura::SessionConfig::inference_from_env()).0;
    for (name, data) in [
        ("qa", qa.as_slice()),
        ("ka", ka.as_slice()),
        ("va", va.as_slice()),
        ("qb", qb.as_slice()),
        ("kb", kb.as_slice()),
        ("vb", vb.as_slice()),
    ] {
        session.set_input(name, data);
    }
    session.step();
    session.wait();
    let mut actual_a = vec![0.0; expected_a.len()];
    let mut actual_b = vec![0.0; expected_b.len()];
    session.read_output_by_index(0, &mut actual_a);
    session.read_output_by_index(1, &mut actual_b);

    for (label, expected, actual) in [
        ("head_dim=4", expected_a, actual_a),
        ("head_dim=8", expected_b, actual_b),
    ] {
        let max_abs = expected
            .iter()
            .zip(&actual)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_abs < 1e-5, "{label} pipeline mismatch: {max_abs}");
    }
}
