//! Regression test for the head_dim=64 attention subgroup-race bug.
//!
//! `full_attention_with_rel_pos_bias` (zero bias) must match CPU SDPA at
//! head_dim=64 (the real Magenta-RT value) as well as head_dim=8. The generated
//! attention shader's online-softmax loops read the reduced score then overwrote
//! the shared buffer next iteration with no `workgroupBarrier` between — correct
//! at head_dim≤8 (one subgroup, lockstep) but racy at head_dim=64 (multiple
//! subgroups). All prior tests used head_dim≤8 and so never caught it.

use meganeura::Graph;

fn cpu_sdpa(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq: usize,
    heads: usize,
    hd: usize,
    causal: bool,
) -> Vec<f32> {
    let dim = heads * hd;
    let scale = 1.0 / (hd as f32).sqrt();
    let mut out = vec![0.0_f32; seq * dim];
    for h in 0..heads {
        let off = h * hd;
        for i in 0..seq {
            let lim = if causal { i + 1 } else { seq };
            let mut sc = vec![0.0_f32; lim];
            for (j, s) in sc.iter_mut().enumerate() {
                let mut acc = 0.0;
                for d in 0..hd {
                    acc += q[i * dim + off + d] * k[j * dim + off + d];
                }
                *s = acc * scale;
            }
            let mx = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;
            for s in sc.iter_mut() {
                *s = (*s - mx).exp();
                sum += *s;
            }
            for d in 0..hd {
                let mut acc = 0.0;
                for (j, &s) in sc.iter().enumerate() {
                    acc += (s / sum) * v[j * dim + off + d];
                }
                out[i * dim + off + d] = acc;
            }
        }
    }
    out
}

fn run(num_heads: u32, head_dim: u32, seq: usize, causal: bool) -> f32 {
    let dim = (num_heads * head_dim) as usize;
    let nb = 8u32;
    let mut g = Graph::new();
    let q = g.input("q", &[seq, dim]);
    let k = g.input("k", &[seq, dim]);
    let v = g.input("v", &[seq, dim]);
    let bias = g.parameter("bias", &[(num_heads * nb) as usize]);
    let attn = g.full_attention_with_rel_pos_bias(
        q, k, v, bias, num_heads, num_heads, head_dim, nb, 16, !causal, causal,
    );
    g.set_outputs(vec![attn]);
    let mut s = meganeura::build_inference_session(&g);
    s.set_parameter("bias", &vec![0.0_f32; (num_heads * nb) as usize]);
    let mk = |salt: f32| -> Vec<f32> {
        (0..seq * dim)
            .map(|i| ((i as f32 + salt) * 0.017).sin() * 0.3)
            .collect()
    };
    let (qd, kd, vd) = (mk(1.0), mk(2.0), mk(3.0));
    s.set_input("q", &qd);
    s.set_input("k", &kd);
    s.set_input("v", &vd);
    s.step();
    s.wait();
    let gpu = s.read_output(seq * dim);
    let cpu = cpu_sdpa(
        &qd,
        &kd,
        &vd,
        seq,
        num_heads as usize,
        head_dim as usize,
        causal,
    );
    gpu.iter()
        .zip(cpu.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn full_attention_head_dim_sweep() {
    for &causal in &[false, true] {
        let d8 = run(2, 8, 4, causal);
        let d64 = run(2, 64, 4, causal);
        eprintln!("causal={causal}: head_dim=8 diff {d8:.3e}, head_dim=64 diff {d64:.3e}");
        assert!(d8 <= 1e-3, "head_dim=8 causal={causal}: {d8}");
        assert!(d64 <= 1e-3, "head_dim=64 causal={causal}: {d64}");
    }
}
