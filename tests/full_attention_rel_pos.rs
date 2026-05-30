//! Correctness test for `Graph::full_attention_with_rel_pos_bias`.
//!
//! Compares the fused shader against a pure-Rust reference implementation
//! (matmul + bias_add + softmax + matmul) for both bidirectional (encoder-style)
//! and causal (decoder-style) configurations.

use meganeura::{Graph, build_inference_session};

/// Pure-Rust reference. Operates on row-major flat tensors in the same
/// layout meganeura uses internally (Q/K/V are `[seq, num_heads*head_dim]`).
#[allow(clippy::too_many_arguments)]
fn ref_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    bias_table: &[f32],
    num_heads: usize,
    head_dim: usize,
    seq: usize,
    num_buckets: usize,
    max_distance: u32,
    bidirectional: bool,
    causal: bool,
) -> Vec<f32> {
    let bucket = |q_pos: usize, k_pos: usize| -> usize {
        let mut n = q_pos as i32 - k_pos as i32;
        let mut ret = 0_u32;
        let mut nb = num_buckets as u32;
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
            return (ret + n_u) as usize;
        }
        let log_n = (n_u as f32 / max_exact as f32).ln();
        let log_max = (max_distance as f32 / max_exact as f32).ln();
        let val_large = max_exact as f32 + log_n / log_max * (nb - max_exact) as f32;
        (ret + (val_large as u32).min(nb - 1)) as usize
    };

    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0.0_f32; seq * num_heads * head_dim];

    for h in 0..num_heads {
        for q_pos in 0..seq {
            // Per-(head, q_pos) softmax of scores over kv positions.
            let kv_end = if causal { q_pos + 1 } else { seq };
            // Online softmax (matches the kernel's numerics).
            let mut max_score = f32::NEG_INFINITY;
            let mut sum_exp = 0.0;
            let mut acc = vec![0.0_f32; head_dim];
            for k_pos in 0..kv_end {
                let mut dot = 0.0_f32;
                for d in 0..head_dim {
                    let qv = q[q_pos * num_heads * head_dim + h * head_dim + d];
                    let kv = k[k_pos * num_heads * head_dim + h * head_dim + d];
                    dot += qv * kv;
                }
                let b = bucket(q_pos, k_pos);
                let bias = bias_table[h * num_buckets + b];
                let score = dot * scale + bias;
                let new_max = max_score.max(score);
                let correction = (max_score - new_max).exp();
                let weight = (score - new_max).exp();
                for d in 0..head_dim {
                    let vv = v[k_pos * num_heads * head_dim + h * head_dim + d];
                    acc[d] = acc[d] * correction + weight * vv;
                }
                sum_exp = sum_exp * correction + weight;
                max_score = new_max;
            }
            let inv = if sum_exp == 0.0 { 1.0 } else { 1.0 / sum_exp };
            for d in 0..head_dim {
                out[q_pos * num_heads * head_dim + h * head_dim + d] = acc[d] * inv;
            }
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn run_gpu(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    bias_table: &[f32],
    seq: usize,
    num_heads: u32,
    head_dim: u32,
    num_buckets: u32,
    max_distance: u32,
    bidirectional: bool,
    causal: bool,
) -> Vec<f32> {
    let inner = (num_heads * head_dim) as usize;
    let out_size = seq * inner;
    let table_size = (num_heads * num_buckets) as usize;

    let mut g = Graph::new();
    let q_n = g.input("q", &[seq, inner]);
    let k_n = g.input("k", &[seq, inner]);
    let v_n = g.input("v", &[seq, inner]);
    let table_n = g.parameter("table", &[table_size]);
    let y = g.full_attention_with_rel_pos_bias(
        q_n, k_n, v_n, table_n,
        num_heads, num_heads, head_dim,
        num_buckets, max_distance, bidirectional, causal,
    );
    g.set_outputs(vec![y]);

    let mut s = build_inference_session(&g);
    s.set_input("q", q);
    s.set_input("k", k);
    s.set_input("v", v);
    s.set_parameter("table", bias_table);
    s.step();
    s.wait();
    s.read_output(out_size)
}

#[test]
fn full_attention_rel_pos_bidirectional_matches_reference() {
    // T5-encoder-shape: 2 heads × head_dim 8 (= 16 inner), seq 6, 8 buckets.
    let seq = 6usize;
    let num_heads = 2u32;
    let head_dim = 8u32;
    let num_buckets = 8u32;
    let max_distance = 32u32;
    let inner = (num_heads * head_dim) as usize;
    let q: Vec<f32> = (0..seq * inner).map(|i| (i as f32 * 0.013).sin()).collect();
    let k: Vec<f32> = (0..seq * inner).map(|i| (i as f32 * 0.017).cos()).collect();
    let v: Vec<f32> = (0..seq * inner).map(|i| (i as f32 * 0.011 - 0.5).tanh()).collect();
    let table: Vec<f32> = (0..num_heads * num_buckets)
        .map(|i| (i as f32) * 0.07 - 0.3)
        .collect();
    let got = run_gpu(&q, &k, &v, &table, seq, num_heads, head_dim, num_buckets, max_distance, true, false);
    let want = ref_attention(&q, &k, &v, &table, num_heads as usize, head_dim as usize, seq, num_buckets as usize, max_distance, true, false);
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let tol = w.abs() * 1e-4 + 1e-5;
        assert!((g - w).abs() <= tol, "bidir mismatch at {i}: got {g}, expected {w}");
    }
}

#[test]
fn full_attention_rel_pos_causal_matches_reference() {
    // T5-decoder-shape with causal mask: seq 5, 4 heads × head_dim 8.
    let seq = 5usize;
    let num_heads = 4u32;
    let head_dim = 8u32;
    let num_buckets = 8u32;
    let max_distance = 16u32;
    let inner = (num_heads * head_dim) as usize;
    let q: Vec<f32> = (0..seq * inner).map(|i| (i as f32 * 0.019).sin()).collect();
    let k: Vec<f32> = (0..seq * inner).map(|i| (i as f32 * 0.023).cos()).collect();
    let v: Vec<f32> = (0..seq * inner).map(|i| (i as f32 * 0.029 - 0.4).tanh()).collect();
    let table: Vec<f32> = (0..num_heads * num_buckets)
        .map(|i| (i as f32) * 0.05 - 0.2)
        .collect();
    let got = run_gpu(&q, &k, &v, &table, seq, num_heads, head_dim, num_buckets, max_distance, false, true);
    let want = ref_attention(&q, &k, &v, &table, num_heads as usize, head_dim as usize, seq, num_buckets as usize, max_distance, false, true);
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let tol = w.abs() * 1e-4 + 1e-5;
        assert!((g - w).abs() <= tol, "causal mismatch at {i}: got {g}, expected {w}");
    }
}

#[test]
fn full_attention_rel_pos_zero_bias_equals_plain_attention() {
    // Sanity: with bias_table=zeros, output should match full_attention exactly.
    let seq = 4usize;
    let num_heads = 2u32;
    let head_dim = 4u32;
    let inner = (num_heads * head_dim) as usize;
    let q: Vec<f32> = (0..seq * inner).map(|i| (i as f32) * 0.1).collect();
    let k: Vec<f32> = (0..seq * inner).map(|i| (i as f32) * 0.13).collect();
    let v: Vec<f32> = (0..seq * inner).map(|i| (i as f32) * 0.07).collect();

    // Run with-bias-zero.
    let table = vec![0.0_f32; (num_heads * 4) as usize];
    let with_bias = run_gpu(&q, &k, &v, &table, seq, num_heads, head_dim, 4, 32, true, false);

    // Run plain full_attention.
    let mut g = Graph::new();
    let q_n = g.input("q", &[seq, inner]);
    let k_n = g.input("k", &[seq, inner]);
    let v_n = g.input("v", &[seq, inner]);
    let y = g.full_attention(q_n, k_n, v_n, num_heads, num_heads, head_dim);
    g.set_outputs(vec![y]);
    let mut s = build_inference_session(&g);
    s.set_input("q", &q);
    s.set_input("k", &k);
    s.set_input("v", &v);
    s.step();
    s.wait();
    let plain = s.read_output(seq * inner);

    for (i, (a, b)) in with_bias.iter().zip(plain.iter()).enumerate() {
        let tol = b.abs() * 1e-5 + 1e-6;
        assert!((a - b).abs() <= tol, "zero-bias should equal plain at {i}: rel-pos {a}, plain {b}");
    }
}
