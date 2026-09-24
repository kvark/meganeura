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

/// `loss = Σ (x · a) · weight` over two parameters, so `weight` sits at an
/// offset in the training session's parameter arena.
fn training_graph() -> Graph {
    let mut graph = Graph::new();
    let input = graph.input("x", &[1, 2]);
    let a = graph.parameter("a", &[2, 2]);
    let weight = graph.parameter("weight", &[2, 2]);
    let hidden = graph.matmul(input, a);
    let output = graph.matmul(hidden, weight);
    let loss = graph.sum_all(output);
    graph.set_outputs(vec![loss]);
    graph
}

/// One SGD step of `training_graph` at `x = [1, 2]`, on the CPU.
fn sgd_step(a: [f32; 4], weight: [f32; 4], lr: f32) -> ([f32; 4], [f32; 4]) {
    let x = [1.0, 2.0];
    let hidden = [x[0] * a[0] + x[1] * a[2], x[0] * a[1] + x[1] * a[3]];
    // d loss / d hidden[j] = Σ_k weight[j][k]
    let dh = [weight[0] + weight[1], weight[2] + weight[3]];
    let grad_a = [x[0] * dh[0], x[0] * dh[1], x[1] * dh[0], x[1] * dh[1]];
    let grad_w = [hidden[0], hidden[0], hidden[1], hidden[1]];
    (
        std::array::from_fn(|i| a[i] - lr * grad_a[i]),
        std::array::from_fn(|i| weight[i] - lr * grad_w[i]),
    )
}

#[test]
fn training_sessions_share_parameters_in_either_direction() {
    let a = [1.0, -1.0, 0.5, 2.0];
    let weight = [2.0, 0.0, 0.0, 3.0];
    let (next_a, next_weight) = sgd_step(a, weight, 0.125);
    for training_is_source in [true, false] {
        let mut trainer = meganeura::build(&training_graph(), SessionConfig::default()).0;
        let mut config = SessionConfig::inference_from_env();
        config.gpu = Some(trainer.context());
        let mut reader = meganeura::build(&graph(2), config).0;
        trainer.set_parameter("a", &a);
        if training_is_source {
            trainer.set_parameter("weight", &weight);
            reader.share_parameter_from(&mut trainer, "weight").unwrap();
        } else {
            // The trainer's `weight` leaves its arena slot for the reader's
            // allocation; the optimizer must still update it, and `a`.
            reader.set_parameter("weight", &weight);
            trainer.share_parameter_from(&mut reader, "weight").unwrap();
        }
        trainer.set_input("x", &[1.0, 2.0]);
        trainer.set_learning_rate(0.125);
        trainer.step();
        trainer.wait();

        let mut values = [0.0; 4];
        trainer.read_param("a", &mut values);
        assert_eq!(values, next_a, "source={training_is_source}");
        trainer.read_param("weight", &mut values);
        assert_eq!(values, next_weight, "source={training_is_source}");
        reader.read_param("weight", &mut values);
        assert_eq!(values, next_weight, "source={training_is_source}");
        reader.set_input("x", &[1.0, 0.0]);
        reader.step();
        reader.wait();
        assert_eq!(reader.read_output(2), [next_weight[0], next_weight[1]]);
    }
}
