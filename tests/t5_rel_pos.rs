//! Test for `Graph::t5_rel_pos_bias`. Compares the GPU shader against a
//! pure-Rust reimplementation of flaxformer's RelativePositionBiases.

use meganeura::{Graph, build_inference_session};

/// Pure-Rust port of `flaxformer.components.relative_position_biases._relative_position_bucket`.
/// `q_minus_k = q_pos - k_pos`. Positive means looking back.
fn rel_pos_bucket(q_minus_k: i32, bidirectional: bool, num_buckets: u32, max_distance: u32) -> u32 {
    let mut n = q_minus_k;
    let mut ret = 0_u32;
    let mut nb = num_buckets;
    if bidirectional {
        nb /= 2;
        if n < 0 {
            ret = nb;
            n = -n;
        }
    } else if n < 0 {
        n = 0;
    }
    let max_exact = nb / 2;
    let n_u = n as u32;
    if n_u < max_exact {
        return ret + n_u;
    }
    let log_n = (n_u as f32 / max_exact as f32).ln();
    let log_max = (max_distance as f32 / max_exact as f32).ln();
    let val_large_f = max_exact as f32 + log_n / log_max * (nb - max_exact) as f32;
    let val_large = val_large_f as u32;
    let val_clamped = val_large.min(nb - 1);
    ret + val_clamped
}

fn rust_bias(
    table: &[f32],
    num_heads: u32,
    num_buckets: u32,
    max_distance: u32,
    bidirectional: bool,
    q_len: u32,
    kv_len: u32,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; (num_heads * q_len * kv_len) as usize];
    for h in 0..num_heads {
        for q in 0..q_len {
            for k in 0..kv_len {
                let bucket = rel_pos_bucket(q as i32 - k as i32, bidirectional, num_buckets, max_distance);
                let idx = ((h * q_len + q) * kv_len + k) as usize;
                let t_idx = (h * num_buckets + bucket) as usize;
                out[idx] = table[t_idx];
            }
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn run_gpu(
    table: &[f32],
    num_heads: u32,
    num_buckets: u32,
    max_distance: u32,
    bidirectional: bool,
    q_len: u32,
    kv_len: u32,
) -> Vec<f32> {
    let out_size = (num_heads * q_len * kv_len) as usize;
    let table_size = (num_heads * num_buckets) as usize;
    assert_eq!(table.len(), table_size);

    let mut g = Graph::new();
    let t = g.parameter("table", &[table_size]);
    let y = g.t5_rel_pos_bias(t, num_heads, num_buckets, max_distance, bidirectional, q_len, kv_len);
    g.set_outputs(vec![y]);

    let mut s = build_inference_session(&g);
    s.set_parameter("table", table);
    s.step();
    s.wait();
    s.read_output(out_size)
}

#[test]
fn t5_rel_pos_encoder_shape_matches_t5_paper() {
    // T5 encoder config: bidirectional, 32 buckets, max 128.
    let num_heads = 4;
    let num_buckets = 32;
    let max_distance = 128;
    let q_len = 64;
    let kv_len = 64;
    let table: Vec<f32> = (0..num_heads * num_buckets)
        .map(|i| (i as f32) * 0.01 - 0.5)
        .collect();
    let got = run_gpu(&table, num_heads, num_buckets, max_distance, true, q_len, kv_len);
    let want = rust_bias(&table, num_heads, num_buckets, max_distance, true, q_len, kv_len);
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let tol = w.abs() * 1e-5 + 1e-6;
        assert!(
            (g - w).abs() <= tol,
            "encoder bias mismatch at {i}: got {g}, expected {w}",
        );
    }
}

#[test]
fn t5_rel_pos_decoder_causal() {
    // T5 decoder self-attn: unidirectional, 32 buckets, max 128.
    let num_heads = 2;
    let num_buckets = 32;
    let max_distance = 128;
    let q_len = 50;
    let kv_len = 50;
    let table: Vec<f32> = (0..num_heads * num_buckets)
        .map(|i| ((i as f32) * 0.013).sin())
        .collect();
    let got = run_gpu(&table, num_heads, num_buckets, max_distance, false, q_len, kv_len);
    let want = rust_bias(&table, num_heads, num_buckets, max_distance, false, q_len, kv_len);
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let tol = w.abs() * 1e-5 + 1e-6;
        assert!(
            (g - w).abs() <= tol,
            "decoder bias mismatch at {i}: got {g}, expected {w}",
        );
    }
}

#[test]
fn t5_rel_pos_depthformer_small() {
    // Depthformer "depth" config: 16 buckets, max 16 (one bucket per level), bidirectional.
    let num_heads = 16;
    let num_buckets = 16;
    let max_distance = 16;
    let q_len = 16;
    let kv_len = 16;
    let table: Vec<f32> = (0..num_heads * num_buckets)
        .map(|i| (i as f32) * 0.05 - 1.0)
        .collect();
    let got = run_gpu(&table, num_heads, num_buckets, max_distance, true, q_len, kv_len);
    let want = rust_bias(&table, num_heads, num_buckets, max_distance, true, q_len, kv_len);
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let tol = w.abs() * 1e-5 + 1e-6;
        assert!(
            (g - w).abs() <= tol,
            "depthformer bias mismatch at {i}: got {g}, expected {w}",
        );
    }
}

#[test]
fn t5_rel_pos_bucket_function_corner_cases() {
    // Verify the Rust reference matches the documented T5 behavior on key positions.
    // bidirectional=True, 32 buckets, max 128.
    let bid = true;
    let nb = 32;
    let md = 128;

    // q == k → bucket 0 (positive half)
    assert_eq!(rel_pos_bucket(0, bid, nb, md), 0);
    // q - k = 1 → bucket 1 (small linear)
    assert_eq!(rel_pos_bucket(1, bid, nb, md), 1);
    // q - k = -1 → bucket 16 + 1 = 17 (negative half offset, then small linear)
    assert_eq!(rel_pos_bucket(-1, bid, nb, md), 17);
    // Very large distance saturates to the last bucket.
    assert_eq!(rel_pos_bucket(1000, bid, nb, md), 15);
    assert_eq!(rel_pos_bucket(-1000, bid, nb, md), 31);
}
