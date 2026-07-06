//! Probe: `cross_attention` (used by the MusicCoCa attention pool) at the real
//! `pool_head_dim = 256`, vs a CPU softmax-attention reference. The encoder/pool
//! attention kernels were only ever verified at head_dim≤64; the pool runs at
//! 256 (12 heads × 256). A head_dim-specific reduction bug would surface here.

use meganeura::Graph;

fn cpu_attn(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    qn: usize,
    kn: usize,
    h: usize,
    hd: usize,
) -> Vec<f32> {
    let dim = h * hd;
    let scale = 1.0 / (hd as f32).sqrt();
    let mut out = vec![0.0_f32; qn * dim];
    for head in 0..h {
        let off = head * hd;
        for i in 0..qn {
            let mut scores = vec![0.0_f32; kn];
            for (j, sc) in scores.iter_mut().enumerate() {
                let mut s = 0.0;
                for d in 0..hd {
                    s += q[i * dim + off + d] * k[j * dim + off + d];
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
                out[i * dim + off + d] = acc;
            }
        }
    }
    out
}

fn run(qn: usize, kn: usize, h: usize, hd: usize) -> f32 {
    run_op(qn, kn, h, hd, false)
}

fn run_op(qn: usize, kn: usize, h: usize, hd: usize, full: bool) -> f32 {
    let dim = h * hd;
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    let mut rng = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        ((seed >> 40) as f32 / (1u32 << 24) as f32 - 0.5) * 0.6
    };
    let q: Vec<f32> = (0..qn * dim).map(|_| rng()).collect();
    let k: Vec<f32> = (0..kn * dim).map(|_| rng()).collect();
    let v: Vec<f32> = (0..kn * dim).map(|_| rng()).collect();

    let mut g = Graph::new();
    let qn_node = g.input("q", &[qn, dim]);
    let kn_node = g.input("k", &[kn, dim]);
    let vn_node = g.input("v", &[kn, dim]);
    let out = if full {
        g.full_attention(qn_node, kn_node, vn_node, h as u32, h as u32, hd as u32)
    } else {
        g.cross_attention(qn_node, kn_node, vn_node, h as u32, h as u32, hd as u32)
    };
    g.set_outputs(vec![out]);
    let mut s = meganeura::build_inference_session(&g);
    s.set_input("q", &q);
    s.set_input("k", &k);
    s.set_input("v", &v);
    s.step();
    s.wait();
    let gpu = s.read_output(qn * dim);
    let cpu = cpu_attn(&q, &k, &v, qn, kn, h, hd);
    let mut m = 0.0_f32;
    for (a, b) in gpu.iter().zip(&cpu) {
        m = m.max((a - b).abs());
    }
    let kind = if full { "full" } else { "cross" };
    eprintln!("{kind}_attention qn={qn} kn={kn} h={h} hd={hd}: max abs diff {m:.3e}");
    m
}

#[test]
fn cross_attention_correct_at_pool_head_dim_256() {
    // Pool shape: 1 query, a few keys, 12 heads × 256.
    for kn in [1usize, 2, 3, 4] {
        assert!(
            run(1, kn, 12, 256) <= 1e-4,
            "pool head_dim=256 kn={kn} wrong"
        );
    }
    // Sanity at 64 (known-good).
    assert!(run(1, 4, 12, 64) <= 1e-4, "head_dim=64 attention wrong");
}

#[test]
fn full_attention_correct_at_encoder_head_dim_64_multiposition() {
    // Encoder self-attention shape: q_seq = kv_seq = a few, 12 heads × 64.
    assert!(
        run_op(2, 2, 12, 64, true) <= 1e-4,
        "full_attn 2pos hd=64 wrong"
    );
    assert!(
        run_op(4, 4, 12, 64, true) <= 1e-4,
        "full_attn 4pos hd=64 wrong"
    );
}
