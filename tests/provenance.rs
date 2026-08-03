//! Value identity and dispatch provenance: names attached via `Graph::named`
//! survive autodiff/toposort/compilation and surface in dispatch labels;
//! every dispatch carries the graph node ids it implements.

use meganeura::compile_training_graph;
use meganeura::graph::Graph;
use meganeura::nn::Linear;

fn build_named_mlp() -> Graph {
    let mut g = Graph::new();
    let x = g.input("x", &[8, 16]);
    let labels = g.input("labels", &[8, 4]);
    let fc1 = Linear::new(&mut g, "fc1", 16, 32);
    let fc2 = Linear::new(&mut g, "fc2", 32, 4);
    let h = fc1.forward(&mut g, x);
    let h = g.relu(h);
    let h = g.named(h, "fc1.act");
    let logits = fc2.forward(&mut g, h);
    let loss = g.cross_entropy_loss(logits, labels);
    let loss = g.named(loss, "loss");
    g.set_outputs(vec![loss]);
    g
}

#[test]
fn names_survive_pipeline_and_reach_labels() {
    let g = build_named_mlp();
    let (plan, _report) = compile_training_graph(&g);

    // Every dispatch carries provenance.
    for (i, d) in plan.dispatches.iter().enumerate() {
        assert!(
            !d.origin.is_empty(),
            "dispatch {i} ({}) has empty origin",
            d.label
        );
    }

    // Node names made it into the plan.
    let names: Vec<&str> = plan.node_names.iter().map(|(_, n)| n.as_str()).collect();
    for expected in ["fc1", "fc2", "fc1.act", "loss"] {
        assert!(names.contains(&expected), "missing node name {expected:?}");
    }

    // At least one dispatch label is prefixed with a user-facing name.
    assert!(
        plan.dispatches
            .iter()
            .any(|d| d.label.starts_with("fc1:") || d.label.starts_with("fc1.act:")),
        "no label carries a name prefix; labels: {:?}",
        plan.dispatches.iter().map(|d| &d.label).collect::<Vec<_>>()
    );

    // The node → buffer map covers at least every forward node (the
    // compiled graph adds backward nodes on top).
    assert!(plan.node_buffers.len() >= g.toposort().nodes().len());
}

#[test]
fn origins_reference_valid_nodes() {
    let g = build_named_mlp();
    let (plan, _report) = compile_training_graph(&g);
    // Origins must index into the compiled (post-toposort, post-autodiff)
    // node space, which node_buffers spans.
    let max_node = plan.node_buffers.iter().map(|&(n, _)| n).max().unwrap_or(0);
    for d in &plan.dispatches {
        for &n in &d.origin {
            assert!(
                n <= max_node,
                "dispatch {} origin {n} out of range (max {max_node})",
                d.label
            );
        }
    }
}
