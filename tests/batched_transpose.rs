use meganeura::reference::{Feeds, gradients};
use meganeura::{Graph, Mode, SessionConfig};

#[test]
fn cpu_batched_transpose_preserves_leading_axes_and_differentiates() {
    for shape in [vec![3, 17, 19], vec![2, 3, 4, 5]] {
        let mut graph = Graph::new();
        let x = graph.parameter("x", &shape);
        let y = graph.transpose(x);
        let mut expected = shape.clone();
        let rank = shape.len();
        expected.swap(rank - 2, rank - 1);
        assert_eq!(graph.node(y).ty.shape, expected);
        graph.set_outputs(vec![y]);
        for graph in [&graph, &meganeura::optimize::optimize(&graph)] {
            let plan = meganeura::compile::compile(graph);
            assert_eq!(plan.dispatches.len(), 1);
            assert_eq!(
                plan.dispatches[0].workgroups[2],
                shape[..rank - 2].iter().product::<usize>() as u32
            );
            assert_eq!(graph.node(graph.outputs()[0]).ty.shape, expected);
        }
        let loss = gradients::weighted_loss(&mut graph, y, 91, 0.75);
        graph.set_outputs(vec![loss]);
        let mut feeds = Feeds::new();
        feeds.fill_random(&graph, 103, 1.0);
        gradients::check(
            &graph,
            &feeds,
            &gradients::Options {
                max_elementwise: 0,
                ..Default::default()
            },
        )
        .unwrap()
        .assert_passed("batched transpose finite differences");
    }
}

#[test]
#[ignore = "requires GPU; includes ragged tile edges and multiple batch axes"]
fn gpu_batched_transpose_values_and_gradients() {
    for shape in [vec![3, 17, 19], vec![2, 3, 4, 5]] {
        let count = shape.iter().product::<usize>();
        let rank = shape.len();
        let (rows, columns) = (shape[rank - 2], shape[rank - 1]);
        let pixels = (0..count)
            .map(|i| i as f32 / count as f32)
            .collect::<Vec<_>>();
        let weights = (0..count)
            .map(|i| ((i * 17) % 43) as f32 - 20.0)
            .collect::<Vec<_>>();
        let mut graph = Graph::new();
        let x = graph.parameter("x", &shape);
        let y = graph.transpose(x);
        let weight = graph.constant(weights.clone(), &graph.node(y).ty.shape.clone());
        let product = graph.mul(y, weight);
        let loss = graph.mean_all(product);
        graph.set_outputs(vec![loss, y]);
        let (mut session, _) = meganeura::build(
            &graph,
            SessionConfig {
                mode: Mode::Training,
                ..SessionConfig::from_env()
            },
        );
        if let Ok(expected) = std::env::var("KINDLE_EXPECT_DEVICE_NAME") {
            assert_eq!(session.device_information().device_name, expected);
            assert!(!session.device_information().is_software_emulated);
            let memory = session.device_memory_stats().unwrap();
            assert!(memory.budget_bytes.saturating_sub(memory.usage_bytes) >= 2 << 30);
        }
        session.set_parameter("x", &pixels);
        session.step();
        session.wait();
        let mut output = vec![0.0; count];
        let mut gradient = vec![0.0; count];
        session.read_output_by_index(1, &mut output);
        session.read_param_grad("x", &mut gradient);
        for batch in 0..count / (rows * columns) {
            for row in 0..rows {
                for column in 0..columns {
                    let from = batch * rows * columns + row * columns + column;
                    let to = batch * rows * columns + column * rows + row;
                    assert_eq!(output[to], pixels[from]);
                    assert!((gradient[from] - weights[to] / count as f32).abs() < 1e-6);
                }
            }
        }
    }
}
