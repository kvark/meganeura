//! Correctness test for `Graph::upsample_nearest` (arbitrary scale_h × scale_w).

use meganeura::{Graph, build_inference_session};

fn run_upsample(
    input: &[f32],
    batch: u32,
    channels: u32,
    in_h: u32,
    in_w: u32,
    scale_h: u32,
    scale_w: u32,
) -> Vec<f32> {
    let in_size = (batch * channels * in_h * in_w) as usize;
    let out_size = (batch * channels * in_h * scale_h * in_w * scale_w) as usize;
    assert_eq!(input.len(), in_size);

    let mut g = Graph::new();
    let x = g.input("x", &[in_size]);
    let y = g.upsample_nearest(x, batch, channels, in_h, in_w, scale_h, scale_w);
    g.set_outputs(vec![y]);

    let mut s = build_inference_session(&g);
    s.set_input("x", input);
    s.step();
    s.wait();
    s.read_output(out_size)
}

#[test]
fn upsample_nearest_2x2_matches_upsample_2x() {
    // [1, 1, 2, 2] input, scale 2×2 → [1, 1, 4, 4] with each cell duplicated.
    let input = vec![1.0_f32, 2.0, 3.0, 4.0];
    let got = run_upsample(&input, 1, 1, 2, 2, 2, 2);
    let expected: Vec<f32> = vec![
        1.0, 1.0, 2.0, 2.0,
        1.0, 1.0, 2.0, 2.0,
        3.0, 3.0, 4.0, 4.0,
        3.0, 3.0, 4.0, 4.0,
    ];
    assert_eq!(got, expected);
}

#[test]
fn upsample_nearest_w_only_scale3() {
    // SpectroStream-shape: 1×3 W upsample (decoder_4 uses stride_w=3).
    let input = vec![10.0_f32, 20.0, 30.0, 40.0]; // [1, 1, 4, 1]
    let got = run_upsample(&input, 1, 1, 4, 1, 1, 3);
    // Each input element repeated 3× along W.
    let expected: Vec<f32> = vec![
        10.0, 10.0, 10.0,
        20.0, 20.0, 20.0,
        30.0, 30.0, 30.0,
        40.0, 40.0, 40.0,
    ];
    assert_eq!(got, expected);
}

#[test]
fn upsample_nearest_multichannel() {
    // [1, 2, 2, 2] input, scale 1×2 → [1, 2, 2, 4]. Each channel kept independent.
    let input: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let got = run_upsample(&input, 1, 2, 2, 2, 1, 2);
    let expected: Vec<f32> = vec![
        // channel 0
        0.0, 0.0, 1.0, 1.0,
        2.0, 2.0, 3.0, 3.0,
        // channel 1
        4.0, 4.0, 5.0, 5.0,
        6.0, 6.0, 7.0, 7.0,
    ];
    assert_eq!(got, expected);
}
