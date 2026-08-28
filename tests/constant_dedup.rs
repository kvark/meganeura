use meganeura::{Graph, Mode, SessionConfig};

#[test]
fn deduplicated_constants_execute_and_remain_readable() {
    let mut graph = Graph::new();
    let input = graph.input("input", &[4]);
    let first = graph.constant(vec![1.0; 4], &[4]);
    let second = graph.constant(vec![1.0; 4], &[4]);
    let sum = graph.add(input, first);
    let output = graph.mul(sum, second);
    graph.set_outputs(vec![output]);

    let mut session = meganeura::build(
        &graph,
        SessionConfig {
            mode: Mode::Inference,
            ..SessionConfig::default()
        },
    )
    .0;
    session.set_input("input", &[2.0, -3.0, 0.5, 7.0]);
    session.step();
    session.wait();

    let mut actual = [0.0; 4];
    session.read_output_by_index(0, &mut actual);
    assert_eq!(actual, [3.0, -2.0, 1.5, 8.0]);
    assert_eq!(session.read_node(first).unwrap(), vec![1.0; 4]);
    assert_eq!(session.read_node(second).unwrap(), vec![1.0; 4]);
}
