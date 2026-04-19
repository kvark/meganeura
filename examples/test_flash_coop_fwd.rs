//! Verify the coop-matrix flash attention forward kernel produces
//! the same output as the scalar flash kernel.
//!
//! Runs the same forward graph twice — once with the coop kernel
//! (`MEGANEURA_FLASH_FWD_COOP=1` + auto-tune installed), once with the
//! scalar one — and asserts the outputs match within f16-input
//! tolerance.

use meganeura::{Graph, build_inference_session};

fn build_and_run(label: &str, seq: usize, num_heads: u32, head_dim: u32, causal: bool) -> Vec<f32> {
    let d = (num_heads * head_dim) as usize;
    let mut g = Graph::new();
    let q = g.input("q", &[seq, d]);
    let k = g.input("k", &[seq, d]);
    let v = g.input("v", &[seq, d]);
    let out = if causal {
        g.causal_attention(q, k, v, num_heads, num_heads, head_dim)
    } else {
        g.full_attention(q, k, v, num_heads, num_heads, head_dim)
    };
    g.set_outputs(vec![out]);
    let mut sess = build_inference_session(&g);

    let total = seq * d;
    let qd: Vec<f32> = (0..total).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
    let kd: Vec<f32> = (0..total).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect();
    let vd: Vec<f32> = (0..total).map(|i| ((i % 19) as f32 - 9.0) * 0.05).collect();
    sess.set_input("q", &qd);
    sess.set_input("k", &kd);
    sess.set_input("v", &vd);
    sess.step();
    sess.wait();
    let out = sess.read_output(total);
    eprintln!("  {label}: out[0..6] = {:?}", &out[..6.min(out.len())]);
    out
}

fn main() {
    env_logger::init();

    // Cases: (seq, num_heads, head_dim, causal).
    // SmolLM2-style: head_dim=64. Test both causal and full attention.
    let cases = [
        ("causal seq=128 heads=2 hd=64", 128, 2u32, 64u32, true),
        ("full   seq=128 heads=2 hd=64", 128, 2, 64, false),
        ("causal seq=64  heads=4 hd=64", 64, 4, 64, true),
        ("full   seq=32  heads=2 hd=64", 32, 2, 64, false),
    ];

    for (label, seq, heads, hd, causal) in cases {
        eprintln!("\n=== {label} ===");

        // Scalar baseline.
        unsafe {
            std::env::remove_var("MEGANEURA_FLASH_FWD_COOP");
        }
        let scalar = build_and_run("scalar", seq, heads, hd, causal);

        // Coop path.
        let gpu = meganeura::runtime::init_gpu_context().expect("gpu");
        let result = meganeura::runtime::auto_tune(&gpu, hd);
        eprintln!("  coop_matrix_available={}", result.coop_matrix_available);
        meganeura::runtime::install_auto_tune(result);
        drop(gpu);
        unsafe {
            std::env::set_var("MEGANEURA_FLASH_FWD_COOP", "1");
        }
        let coop = build_and_run("coop  ", seq, heads, hd, causal);

        assert_eq!(scalar.len(), coop.len(), "{label}: length mismatch");
        let mut max_abs = 0f32;
        let mut max_rel = 0f32;
        for (s, c) in scalar.iter().zip(coop.iter()) {
            let abs = (s - c).abs();
            let rel = abs / s.abs().max(1e-6);
            max_abs = max_abs.max(abs);
            max_rel = max_rel.max(rel);
        }
        eprintln!("  max_abs_err={max_abs:.6e}, max_rel_err={max_rel:.6e}");
        // f16 input → larger tolerance than pure f32 scalar.
        assert!(
            max_abs < 1e-2,
            "{label}: coop diverges (max_abs={max_abs}, max_rel={max_rel})"
        );
    }

    eprintln!("\nALL OK — coop flash forward matches scalar within tolerance");
}
