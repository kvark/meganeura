//! A clamp fused into a batched f16-weight product keeps its bounds. The
//! other multimodal-encoder operators are checked by the `oracle` suite.

use meganeura::Graph;

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "output[{index}]={actual}, expected {expected}"
        );
    }
}

#[test]
fn batched_f16_matmul_bt_preserves_fused_clamp_bounds() {
    const M: usize = 13;
    const K: usize = 32;
    const N: usize = 20;
    let a: Vec<_> = (0..M * K)
        .map(|i| ((i * 7 % 29) as f32 - 14.0) * 0.125)
        .collect();
    let b: Vec<_> = (0..N * K)
        .map(|i| ((i * 11 % 31) as f32 - 15.0) * 0.1)
        .collect();
    let b_rounded: Vec<_> = b
        .iter()
        .map(|&value| half::f16::from_f32(value).to_f32())
        .collect();
    let mut expected = vec![0.0; M * N];
    for row in 0..M {
        for column in 0..N {
            expected[row * N + column] = (0..K)
                .map(|inner| a[row * K + inner] * b_rounded[column * K + inner])
                .sum::<f32>()
                .clamp(-3.25, 4.5);
        }
    }
    assert!(expected.contains(&-3.25));
    assert!(expected.contains(&4.5));
    assert!(expected.iter().any(|value| (-3.25..4.5).contains(value)));

    let mut graph = Graph::new();
    let an = graph.input("a", &[M, K]);
    let bn = graph.parameter_f16("b", &[N, K]);
    let product = graph.matmul_bt(an, bn);
    let output = graph.clamp(product, -3.25, 4.5);
    graph.set_outputs(vec![output]);
    let mut session = meganeura::build(&graph, meganeura::SessionConfig::inference_from_env()).0;
    session.set_input("a", &a);
    session.set_parameter("b", &b);
    session.step();
    session.wait();
    assert_close(&session.read_output(M * N), &expected, 5.0e-3);
}
