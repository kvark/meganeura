//! Correctness tests for `Graph::slice_2d` — spatial crop on NCHW.

use meganeura::{Graph, build_inference_session};

#[allow(clippy::too_many_arguments)]
fn run_slice(
    input: &[f32],
    batch: u32,
    channels: u32,
    in_h: u32,
    in_w: u32,
    start_h: u32,
    end_h: u32,
    start_w: u32,
    end_w: u32,
) -> Vec<f32> {
    let in_size = (batch * channels * in_h * in_w) as usize;
    let out_h = in_h - start_h - end_h;
    let out_w = in_w - start_w - end_w;
    let out_size = (batch * channels * out_h * out_w) as usize;
    assert_eq!(input.len(), in_size);

    let mut g = Graph::new();
    let x = g.input("x", &[in_size]);
    let y = g.slice_2d(x, batch, channels, in_h, in_w, start_h, end_h, start_w, end_w);
    g.set_outputs(vec![y]);

    let mut s = build_inference_session(&g);
    s.set_input("x", input);
    s.step();
    s.wait();
    s.read_output(out_size)
}

#[test]
fn slice2d_crops_w_only() {
    // [1, 1, 2, 4] → crop (start_w=1, end_w=1) → [1, 1, 2, 2]
    let input = vec![1.0, 2.0, 3.0, 4.0,
                     5.0, 6.0, 7.0, 8.0];
    let got = run_slice(&input, 1, 1, 2, 4, 0, 0, 1, 1);
    assert_eq!(got, vec![2.0, 3.0, 6.0, 7.0]);
}

#[test]
fn slice2d_crops_h_only() {
    // [1, 1, 4, 2] → crop (start_h=1, end_h=1) → [1, 1, 2, 2]
    let input = vec![1.0, 2.0,
                     3.0, 4.0,
                     5.0, 6.0,
                     7.0, 8.0];
    let got = run_slice(&input, 1, 1, 4, 2, 1, 1, 0, 0);
    assert_eq!(got, vec![3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn slice2d_crops_h_and_w_multichannel() {
    // [1, 2, 4, 4] → crop (1,1) H, (1,1) W → [1, 2, 2, 2]
    let input: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let got = run_slice(&input, 1, 2, 4, 4, 1, 1, 1, 1);
    // channel 0: rows 1, 2, cols 1, 2 → indices (1*4+1, 1*4+2, 2*4+1, 2*4+2) = 5,6,9,10
    // channel 1: same + 16 = 21, 22, 25, 26
    assert_eq!(got, vec![5.0, 6.0, 9.0, 10.0, 21.0, 22.0, 25.0, 26.0]);
}
