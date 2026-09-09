use meganeura::{Graph, SessionConfig};

fn graph(rows: usize) -> Graph {
    let mut graph = Graph::new();
    let input = graph.input("x", &[1, rows]);
    let weight = graph.parameter("weight", &[rows, rows]);
    let output = graph.matmul(input, weight);
    graph.set_outputs(vec![output]);
    graph
}

#[test]
fn separately_compiled_sessions_can_share_parameter_lifetime() {
    let source_graph = graph(2);
    let mut source = meganeura::build(&source_graph, SessionConfig::inference_from_env()).0;
    source.set_parameter("weight", &[2.0, 0.0, 0.0, 3.0]);

    let target_graph = graph(2);
    let mut config = SessionConfig::inference_from_env();
    config.gpu = Some(source.context());
    let mut target = meganeura::build(&target_graph, config).0;
    target.share_parameter_from(&mut source, "weight").unwrap();
    drop(source);

    target.set_input("x", &[4.0, 5.0]);
    target.step();
    target.wait();
    assert_eq!(target.read_output(2), [8.0, 15.0]);
}
