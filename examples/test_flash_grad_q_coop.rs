//! Verify the coop FlashGradQ kernel produces the same dQ as the
//! scalar version. Builds a small attention forward + sum_all loss,
//! reads the dQ output, compares scalar vs coop.

use meganeura::{Graph, build_session};

fn build_and_run(seq: usize, num_heads: u32, head_dim: u32, causal: bool) -> Vec<f32> {
    let d = (num_heads * head_dim) as usize;
    let mut g = Graph::new();
    let q = g.parameter("q_param", &[seq, d]);
    let k = g.parameter("k_param", &[seq, d]);
    let v = g.parameter("v_param", &[seq, d]);
    let out = if causal {
        g.causal_attention(q, k, v, num_heads, num_heads, head_dim)
    } else {
        g.full_attention(q, k, v, num_heads, num_heads, head_dim)
    };
    let loss = g.sum_all(out);
    g.set_outputs(vec![loss]);

    // build_session differentiates internally (training session).
    let mut sess = build_session(&g);

    let total = seq * d;
    let qd: Vec<f32> = (0..total).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
    let kd: Vec<f32> = (0..total).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect();
    let vd: Vec<f32> = (0..total).map(|i| ((i % 19) as f32 - 9.0) * 0.05).collect();
    sess.set_parameter("q_param", &qd);
    sess.set_parameter("k_param", &kd);
    sess.set_parameter("v_param", &vd);
    sess.step();
    sess.wait();

    let mut dq = vec![0f32; total];
    sess.read_param_grad("q_param", &mut dq);
    eprintln!("    dQ[0..6] = {:?}", &dq[..6.min(dq.len())]);
    dq
}

fn main() {
    env_logger::init();

    // Cases — seq is multiple of 16 to match the coop tile size.
    let cases = [
        ("full   seq=16  heads=1 hd=64", 16usize, 1u32, 64u32, false),
        ("causal seq=16  heads=1 hd=64", 16, 1, 64, true),
        ("full   seq=64  heads=4 hd=64", 64, 4, 64, false),
        ("causal seq=64  heads=4 hd=64", 64, 4, 64, true),
    ];

    for (label, seq, heads, hd, causal) in cases {
        eprintln!("\n=== {label} ===");

        // Force scalar FORWARD in both runs so the only difference is
        // the backward kernel — otherwise the install_auto_tune from
        // the first iteration leaves coop forward on, and the test
        // compares (scalar+scalar) vs (coop+coop) which conflates
        // forward and backward errors.
        unsafe {
            std::env::set_var("MEGANEURA_FLASH_FWD_COOP", "0");
            std::env::remove_var("MEGANEURA_FLASH_BWD_COOP");
        }
        eprintln!("  scalar:");
        let scalar = build_and_run(seq, heads, hd, causal);

        let gpu = meganeura::runtime::init_gpu_context().expect("gpu");
        let result = meganeura::runtime::auto_tune(&gpu, hd);
        eprintln!(
            "  coop_matrix_available={}",
            result.coop_caps.is_supported()
        );
        meganeura::runtime::install_auto_tune(result);
        drop(gpu);
        unsafe {
            std::env::set_var("MEGANEURA_FLASH_BWD_COOP", "1");
        }
        eprintln!("  coop:");
        let coop = build_and_run(seq, heads, hd, causal);

        assert_eq!(scalar.len(), coop.len(), "{label}: shape mismatch");
        let mut max_abs = 0f32;
        let mut max_rel = 0f32;
        let mut max_idx = 0usize;
        for (i, (s, c)) in scalar.iter().zip(coop.iter()).enumerate() {
            let abs = (s - c).abs();
            let rel = abs / s.abs().max(1e-6);
            if abs > max_abs {
                max_abs = abs;
                max_idx = i;
            }
            max_rel = max_rel.max(rel);
        }
        let d = (heads * hd) as usize;
        let qrow = max_idx / d;
        let qcol = max_idx % d;
        eprintln!(
            "  max_abs_err={max_abs:.6e} (at idx={max_idx}, q_row={qrow}, q_col={qcol}, scalar={}, coop={})",
            scalar[max_idx], coop[max_idx]
        );
        eprintln!("  max_rel_err={max_rel:.6e}");
        // f16 staging widens tolerance vs pure-f32 scalar.
        assert!(
            max_abs < 1e-2,
            "{label}: coop diverges (max_abs={max_abs}, max_rel={max_rel})"
        );
    }

    eprintln!("\nALL OK — coop FlashGradQ matches scalar");
}
