use meganeura::{Graph, SessionConfig};

fn graph(rows: usize) -> Graph {
    let mut graph = Graph::new();
    let input = graph.input("x", &[1, rows]);
    let weight = graph.parameter("weight", &[rows, rows]);
    let output = graph.matmul(input, weight);
    graph.set_outputs(vec![output]);
    graph
}

fn graph_with_bias(rows: usize) -> Graph {
    let mut graph = Graph::new();
    let input = graph.input("x", &[1, rows]);
    let weight = graph.parameter("weight", &[rows, rows]);
    let bias = graph.parameter("bias", &[1, rows]);
    let product = graph.matmul(input, weight);
    let output = graph.add(product, bias);
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

#[test]
fn construction_shares_available_parameters_and_allocates_the_rest() {
    let mut source = meganeura::build(&graph(2), SessionConfig::inference_from_env()).0;
    source.set_parameter("weight", &[2.0, 0.0, 0.0, 3.0]);

    let mut config = SessionConfig::inference_from_env();
    config.share_parameters_from = Some(&mut source);
    let mut target = meganeura::build(&graph_with_bias(2), config).0;
    assert!(target.shares_parameter(&source, "weight"));
    drop(source);

    let mut values = [1.0; 4];
    target.read_param("weight", &mut values);
    assert_eq!(values, [2.0, 0.0, 0.0, 3.0]);
    let mut bias = [1.0; 2];
    target.read_param("bias", &mut bias);
    assert_eq!(bias, [0.0, 0.0]);
    target.set_input("x", &[4.0, 5.0]);
    target.step();
    target.wait();
    assert_eq!(target.read_output(2), [8.0, 15.0]);
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

#[test]
fn construction_preserves_a_donor_arena_offset() {
    let mut trainer = meganeura::build(&training_graph(), SessionConfig::default()).0;
    trainer.set_parameter("a", &[1.0, -1.0, 0.5, 2.0]);
    trainer.set_parameter("weight", &[2.0, 0.0, 0.0, 3.0]);

    let mut config = SessionConfig::inference_from_env();
    config.share_parameters_from = Some(&mut trainer);
    let mut reader = meganeura::build(&graph(2), config).0;
    drop(trainer);

    let mut values = [0.0; 4];
    reader.read_param("weight", &mut values);
    assert_eq!(values, [2.0, 0.0, 0.0, 3.0]);
    reader.set_input("x", &[4.0, 5.0]);
    reader.step();
    reader.wait();
    assert_eq!(reader.read_output(2), [8.0, 15.0]);
}

#[test]
fn construction_skips_a_parameter_stored_differently() {
    let mut source = meganeura::build(&graph(2), SessionConfig::inference_from_env()).0;
    source.set_parameter("weight", &[2.0, 0.0, 0.0, 3.0]);

    let mut config = SessionConfig::inference_from_env();
    config.share_parameters_from = Some(&mut source);
    let mut target = meganeura::build(&graph(3), config).0;
    assert!(!target.shares_parameter(&source, "weight"));
    let mut values = [1.0; 9];
    target.read_param("weight", &mut values);
    assert_eq!(values, [0.0; 9]);
    target.set_parameter("weight", &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    target.set_input("x", &[4.0, 5.0, 6.0]);
    target.step();
    target.wait();
    assert_eq!(target.read_output(3), [4.0, 5.0, 6.0]);
    // The source keeps its own values.
    let mut values = [0.0; 4];
    source.read_param("weight", &mut values);
    assert_eq!(values, [2.0, 0.0, 0.0, 3.0]);
}

/// A training session built on another session's parameter trains the
/// shared allocation, and its remaining parameters from its arena.
#[test]
fn construction_donates_into_a_training_arena() {
    let a = [1.0, -1.0, 0.5, 2.0];
    let weight = [2.0, 0.0, 0.0, 3.0];
    let (next_a, next_weight) = sgd_step(a, weight, 0.125);
    let mut reader = meganeura::build(&graph(2), SessionConfig::inference_from_env()).0;
    reader.set_parameter("weight", &weight);
    let config = SessionConfig {
        share_parameters_from: Some(&mut reader),
        ..Default::default()
    };
    let mut trainer = meganeura::build(&training_graph(), config).0;
    assert!(trainer.shares_parameter(&reader, "weight"));
    trainer.set_parameter("a", &a);
    trainer.set_input("x", &[1.0, 2.0]);
    trainer.set_learning_rate(0.125);
    trainer.step();
    trainer.wait();
    let mut values = [0.0; 4];
    trainer.read_param("a", &mut values);
    assert_eq!(values, next_a);
    trainer.read_param("weight", &mut values);
    assert_eq!(values, next_weight);
    reader.read_param("weight", &mut values);
    assert_eq!(values, next_weight);
}

/// With every parameter donated, the new session's parameter arena has no
/// tenants left and shrinks to a placeholder; training still updates the
/// donated allocations.
#[test]
fn construction_donates_a_whole_training_arena() {
    let a = [1.0, -1.0, 0.5, 2.0];
    let weight = [2.0, 0.0, 0.0, 3.0];
    let (next_a, next_weight) = sgd_step(a, weight, 0.125);
    let mut donor = meganeura::build(&training_graph(), SessionConfig::default()).0;
    donor.set_parameter("a", &a);
    donor.set_parameter("weight", &weight);
    let unshared = donor.memory_summary().allocated_buffer_bytes;
    let config = SessionConfig {
        share_parameters_from: Some(&mut donor),
        ..Default::default()
    };
    let mut trainer = meganeura::build(&training_graph(), config).0;
    assert!(trainer.memory_summary().allocated_buffer_bytes < unshared);
    trainer.set_input("x", &[1.0, 2.0]);
    trainer.set_learning_rate(0.125);
    trainer.step();
    trainer.wait();
    let mut values = [0.0; 4];
    for (name, expected) in [("a", next_a), ("weight", next_weight)] {
        assert!(trainer.shares_parameter(&donor, name));
        trainer.read_param(name, &mut values);
        assert_eq!(values, expected, "{name}");
        donor.read_param(name, &mut values);
        assert_eq!(values, expected, "{name}");
    }
}
