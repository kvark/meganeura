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
    for (source_shared, target_shared) in [(false, true), (true, false)] {
        let source_graph = graph(2);
        let mut config = SessionConfig::inference_from_env();
        config.runtime.no_device_local = source_shared;
        let mut source = meganeura::build(&source_graph, config).0;
        source.set_parameter("weight", &[2.0, 0.0, 0.0, 3.0]);

        let target_graph = graph(2);
        let mut config = SessionConfig::inference_from_env();
        config.runtime.no_device_local = target_shared;
        config.gpu = Some(source.context());
        let mut target = meganeura::build(&target_graph, config).0;
        target.share_parameter_from(&mut source, "weight").unwrap();
        drop(source);

        let mut weights = [0.0; 4];
        target.read_param("weight", &mut weights);
        assert_eq!(weights, [2.0, 0.0, 0.0, 3.0]);
        target.set_input("x", &[4.0, 5.0]);
        target.step();
        target.wait();
        assert_eq!(target.read_output(2), [8.0, 15.0]);
        target.set_parameter("weight", &[3.0, 0.0, 0.0, 4.0]);
        target.step();
        target.wait();
        assert_eq!(target.read_output(2), [12.0, 20.0]);
    }
}
