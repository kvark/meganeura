//! Window packing is the only image-specific primitive needed to compose
//! two-dimensional local attention from Meganeura's ordinary attention op.

use meganeura::{Graph, Mode, SessionConfig};

fn inference_config() -> SessionConfig<'static> {
    SessionConfig {
        mode: Mode::Inference,
        ..SessionConfig::default()
    }
}

fn cpu_partition(
    source: &[f32],
    batch: u32,
    channels: u32,
    height: u32,
    width: u32,
    window: u32,
    shift: u32,
) -> Vec<f32> {
    let windows_y = (height + shift).div_ceil(window);
    let windows_x = (width + shift).div_ceil(window);
    let window_count = windows_y * windows_x;
    let inner = batch * window_count * channels;
    let mut packed = vec![0.0; (window * window * inner) as usize];
    for token in 0..window * window {
        for column in 0..inner {
            let channel = column % channels;
            let batch_window = column / channels;
            let n = batch_window / window_count;
            let window_id = batch_window % window_count;
            let window_y = window_id / windows_x;
            let window_x = window_id % windows_x;
            let y = (window_y * window + token / window) as i32 - shift as i32;
            let x = (window_x * window + token % window) as i32 - shift as i32;
            if (0..height as i32).contains(&y) && (0..width as i32).contains(&x) {
                let index = ((n * channels + channel) * height + y as u32) * width + x as u32;
                packed[(token * inner + column) as usize] = source[index as usize];
            }
        }
    }
    packed
}

#[test]
fn window_partition_matches_cpu_with_batch_shift_and_padding() {
    let (batch, channels, height, width, window, shift) = (2, 3, 3, 5, 4, 2);
    let len = (batch * channels * height * width) as usize;
    let source: Vec<f32> = (0..len).map(|i| i as f32 + 0.25).collect();

    let mut graph = Graph::new();
    let input = graph.input("input", &[len]);
    let packed = graph.window_partition_2d(input, batch, channels, height, width, window, shift);
    let output_len = graph.node(packed).ty.num_elements();
    graph.set_outputs(vec![packed]);

    let (mut session, _) = meganeura::build(&graph, inference_config());
    session.set_input("input", &source);
    session.step();
    session.wait();
    let actual = session.read_output(output_len);
    let expected = cpu_partition(&source, batch, channels, height, width, window, shift);
    assert_eq!(actual, expected);
}

#[test]
fn window_partition_and_merge_are_an_exact_round_trip() {
    let (batch, channels, height, width, window, shift) = (2, 4, 7, 5, 4, 2);
    let len = (batch * channels * height * width) as usize;
    let source: Vec<f32> = (0..len).map(|i| (i as f32 * 0.17).sin()).collect();

    let mut graph = Graph::new();
    let input = graph.input("input", &[len]);
    let packed = graph.window_partition_2d(input, batch, channels, height, width, window, shift);
    let merged = graph.window_merge_2d(packed, batch, channels, height, width, window, shift);
    graph.set_outputs(vec![merged]);

    let (mut session, _) = meganeura::build(&graph, inference_config());
    session.set_input("input", &source);
    session.step();
    session.wait();
    assert_eq!(session.read_output(len), source);
}

#[test]
fn padding_has_no_gradient_but_every_image_element_does() {
    let (batch, channels, height, width, window, shift) = (1, 2, 3, 5, 4, 2);
    let len = (batch * channels * height * width) as usize;

    let mut graph = Graph::new();
    let input = graph.parameter("input", &[len]);
    let packed = graph.window_partition_2d(input, batch, channels, height, width, window, shift);
    let loss = graph.sum_all(packed);
    graph.set_outputs(vec![loss]);

    let mut session = meganeura::build_session(&graph);
    session.set_parameter("input", &vec![0.5; len]);
    session.set_learning_rate(0.0);
    session.step();
    session.wait();
    let mut gradient = vec![0.0; len];
    session.read_param_grad("input", &mut gradient);
    assert_eq!(gradient, vec![1.0; len]);
}
