//! Regression test for the `cached_attention` subgroup-race bug.
//!
//! Drives a single query against a growing K/V cache (head_dim=64) and compares
//! each step to a CPU SDPA reference. `cached_attention`'s online-softmax loop
//! read the reduced score then overwrote the shared buffer next iteration with
//! no `workgroupBarrier` between, so it was exact only at kv_len=1 (trivial
//! softmax) and wrong for kv_len≥2. The repo's only prior coverage was a
//! finiteness smoke test, so this was never caught.

use meganeura::Graph;

fn cpu_sdpa(q: &[f32], k: &[f32], v: &[f32], kv_len: usize, heads: usize, hd: usize) -> Vec<f32> {
    let dim = heads * hd;
    let scale = 1.0 / (hd as f32).sqrt();
    let mut out = vec![0.0_f32; dim];
    for h in 0..heads {
        let off = h * hd;
        let mut scores = vec![0.0_f32; kv_len];
        for (j, sc) in scores.iter_mut().enumerate() {
            let mut s = 0.0;
            for d in 0..hd {
                s += q[off + d] * k[j * dim + off + d];
            }
            *sc = s * scale;
        }
        let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for s in scores.iter_mut() {
            *s = (*s - mx).exp();
            sum += *s;
        }
        for d in 0..hd {
            let mut acc = 0.0;
            for (j, &sc) in scores.iter().enumerate() {
                acc += (sc / sum) * v[j * dim + off + d];
            }
            out[off + d] = acc;
        }
    }
    out
}

#[test]
fn cached_attention_matches_cpu_sdpa() {
    let heads = 2usize;
    let hd = 64usize;
    let dim = heads * hd;
    let max_seq = 5usize;

    let mut g = Graph::new();
    let q_in = g.input("q", &[1, dim]);
    let k_in = g.input("k", &[1, dim]);
    let v_in = g.input("v", &[1, dim]);
    let kv_pos = g.input_u32("kv_pos", &[1]);
    let k_cache = g.parameter("k_cache", &[max_seq, dim]);
    let v_cache = g.parameter("v_cache", &[max_seq, dim]);
    let kc = g.cache_write(k_in, k_cache, kv_pos);
    let vc = g.cache_write(v_in, v_cache, kv_pos);
    let attn = g.cached_attention(q_in, kc, vc, kv_pos, heads as u32, heads as u32, hd as u32);
    g.set_outputs(vec![attn]);

    let mut s = meganeura::build_inference_session(&g);
    s.set_parameter("k_cache", &vec![0.0_f32; max_seq * dim]);
    s.set_parameter("v_cache", &vec![0.0_f32; max_seq * dim]);

    // Fixed query; per-position K/V.
    let q: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.017).sin() * 0.3).collect();
    let mut k_all = vec![0.0_f32; max_seq * dim];
    let mut v_all = vec![0.0_f32; max_seq * dim];
    for j in 0..max_seq {
        for d in 0..dim {
            k_all[j * dim + d] = ((j * dim + d) as f32 * 0.011).cos() * 0.3;
            v_all[j * dim + d] = ((j * dim + d) as f32 * 0.013).sin() * 0.3;
        }
    }

    s.set_input("q", &q);
    for t in 0..max_seq {
        s.set_input("k", &k_all[t * dim..(t + 1) * dim]);
        s.set_input("v", &v_all[t * dim..(t + 1) * dim]);
        s.set_input_u32("kv_pos", &[t as u32]);
        s.step();
        s.wait();
        let gpu = s.read_output(dim);
        let cpu = cpu_sdpa(&q, &k_all, &v_all, t + 1, heads, hd);
        let mut max_abs = 0.0_f32;
        for (a, b) in gpu.iter().zip(cpu.iter()) {
            max_abs = max_abs.max((a - b).abs());
        }
        eprintln!("kv_len={}: max abs diff {max_abs:.3e}", t + 1);
        assert!(
            max_abs <= 1e-4,
            "kv_len={}: cached_attention vs CPU diff {max_abs}",
            t + 1
        );
    }
}
