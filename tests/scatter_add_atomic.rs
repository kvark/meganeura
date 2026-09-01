use meganeura::{Graph, Mode, SessionConfig};

#[test]
fn large_scatter_add_matches_cpu_with_repeated_indices_and_zeros() {
    const VOCAB_SIZE: usize = 4097;
    const SEQ_LEN: usize = 256;
    const EMBED_DIM: usize = 3;

    let mut graph = Graph::new();
    let indices = graph.input_u32("indices", &[SEQ_LEN]);
    let src = graph.input("src", &[SEQ_LEN, EMBED_DIM]);
    let output = graph.scatter_add(indices, src, VOCAB_SIZE);
    graph.set_outputs(vec![output]);

    let (mut session, _) = meganeura::build(
        &graph,
        SessionConfig {
            mode: Mode::Inference,
            ..SessionConfig::default()
        },
    );

    let indices: Vec<u32> = (0..SEQ_LEN).map(|i| ((i * 17) % 31) as u32).collect();
    let src: Vec<f32> = (0..SEQ_LEN * EMBED_DIM)
        .map(|i| match i % 19 {
            0 => (i as f32 - 200.0) * 0.001,
            1 => -0.0,
            _ => 0.0,
        })
        .collect();
    let mut expected = vec![0.0_f32; VOCAB_SIZE * EMBED_DIM];
    for row in 0..SEQ_LEN {
        let output_row = indices[row] as usize;
        for column in 0..EMBED_DIM {
            expected[output_row * EMBED_DIM + column] += src[row * EMBED_DIM + column];
        }
    }

    session.set_input_u32("indices", &indices);
    session.set_input("src", &src);
    session.step();
    session.wait();
    let actual = session.read_output(expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            (actual - expected).abs() < 1.0e-5,
            "mismatch at {index}: actual={actual}, expected={expected}"
        );
    }

    let second_indices: Vec<u32> = (0..SEQ_LEN).map(|i| (i % 7) as u32).collect();
    let second_src: Vec<f32> = (0..SEQ_LEN * EMBED_DIM)
        .map(|i| (100.0 - i as f32) * 0.0005)
        .collect();
    let mut second_expected = vec![0.0_f32; VOCAB_SIZE * EMBED_DIM];
    for row in 0..SEQ_LEN {
        let output_row = second_indices[row] as usize;
        for column in 0..EMBED_DIM {
            second_expected[output_row * EMBED_DIM + column] +=
                second_src[row * EMBED_DIM + column];
        }
    }
    session.set_input_u32("indices", &second_indices);
    session.set_input("src", &second_src);
    session.step();
    session.wait();
    let second_actual = session.read_output(second_expected.len());
    for (index, (&actual, &expected)) in second_actual.iter().zip(&second_expected).enumerate() {
        assert!(
            (actual - expected).abs() < 1.0e-5,
            "second-step mismatch at {index}: actual={actual}, expected={expected}"
        );
    }
}

#[test]
fn scatter_add_source_gradient_gathers_repeated_output_rows() {
    const VOCAB_SIZE: usize = 5;
    const SEQ_LEN: usize = 4;
    const EMBED_DIM: usize = 3;

    let mut graph = Graph::new();
    let indices = graph.input_u32("indices", &[SEQ_LEN]);
    let src = graph.parameter("src", &[SEQ_LEN, EMBED_DIM]);
    let weights = graph.input("weights", &[VOCAB_SIZE, EMBED_DIM]);
    let output = graph.scatter_add(indices, src, VOCAB_SIZE);
    let weighted = graph.mul(output, weights);
    let mean = graph.mean_all(weighted);
    let scale = graph.scalar(1.5);
    let loss = graph.mul(mean, scale);
    graph.set_outputs(vec![loss]);

    let (mut session, _) = meganeura::build(&graph, SessionConfig::default());
    let indices = [1_u32, 4, 1, 0];
    let src = [
        0.1_f32, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0, 1.1, -1.2,
    ];
    let weights: Vec<f32> = (0..VOCAB_SIZE * EMBED_DIM)
        .map(|index| index as f32 * 0.1 - 0.4)
        .collect();

    session.set_input_u32("indices", &indices);
    session.set_input("weights", &weights);
    session.set_parameter("src", &src);
    session.step();
    session.wait();

    let mut actual = [0.0_f32; SEQ_LEN * EMBED_DIM];
    session.read_param_grad("src", &mut actual);
    let normalizer = 1.5 / (VOCAB_SIZE * EMBED_DIM) as f32;
    for row in 0..SEQ_LEN {
        for column in 0..EMBED_DIM {
            let expected = weights[indices[row] as usize * EMBED_DIM + column] * normalizer;
            let index = row * EMBED_DIM + column;
            assert!(
                (actual[index] - expected).abs() < 1.0e-6,
                "gradient mismatch at ({row}, {column}): actual={}, expected={expected}",
                actual[index],
            );
        }
    }
}
