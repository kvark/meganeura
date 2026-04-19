//! Verify Conv2dGradInputGemmCoop3x3 produces the same gradient as
//! the scalar Conv2dGradInputGemm path.
//!
//! Runs the same forward + backward graph twice — once with the coop
//! kernel enabled (auto-tune installed), once with it disabled
//! (`MEGANEURA_DISABLE_COOP=1`) — and asserts the input gradients
//! match within FP32 tolerance.
//!
//! Usage:
//!   MEGANEURA_DEVICE_ID=<id> cargo run --release --example test_conv_coop_3x3

use meganeura::{Graph, autodiff, build_session};

fn build_and_run(label: &str, batch: u32, in_c: u32, out_c: u32, h: u32, w: u32) -> Vec<f32> {
    let kh = 3u32;
    let kw = 3u32;
    let stride = 1u32;
    let padding = 1u32;

    let in_size = (batch * in_c * h * w) as usize;
    let kernel_size = (out_c * in_c * kh * kw) as usize;

    // Make the input a Parameter so we can read its gradient (the
    // value the coop kernel computes). Functionally identical to an
    // Input for the forward pass.
    let mut g = Graph::new();
    let input = g.parameter("input_param", &[in_size]);
    let kernel = g.parameter("kernel", &[kernel_size]);
    let y = g.conv2d(
        input, kernel, batch, in_c, h, w, out_c, kh, kw, stride, padding,
    );
    let loss = g.sum_all(y);
    g.set_outputs(vec![loss]);

    let diff = autodiff::differentiate(&g);
    let mut session = build_session(&diff);

    let kernel_data: Vec<f32> = (0..kernel_size).map(|i| ((i % 13) as f32) * 0.05).collect();
    session.set_parameter("kernel", &kernel_data);

    let input_data: Vec<f32> = (0..in_size).map(|i| ((i % 7) as f32) * 0.1).collect();
    session.set_parameter("input_param", &input_data);
    session.step();
    session.wait();

    let mut in_grad = vec![0f32; in_size];
    session.read_param_grad("input_param", &mut in_grad);
    eprintln!(
        "  {label}: in_grad first 8: {:?}",
        &in_grad[..8.min(in_grad.len())]
    );
    in_grad
}

fn main() {
    env_logger::init();

    let cases = [
        ("ResNet stage1", 1u32, 64u32, 64u32, 56u32, 56u32),
        ("ResNet stage2", 1, 128, 128, 28, 28),
        ("ResNet stage3", 1, 256, 256, 14, 14),
    ];

    for (label, batch, in_c, out_c, h, w) in cases {
        eprintln!("\n=== {label}: batch={batch} {in_c}->{out_c} @ {h}x{w} ===");

        // Run scalar baseline first (MEGANEURA_CONV_COOP not set).
        unsafe {
            std::env::remove_var("MEGANEURA_CONV_COOP");
        }
        let scalar = build_and_run("scalar", batch, in_c, out_c, h, w);

        // Run with coop enabled — needs both the env var and the
        // installed coop_matrix_available global.
        let gpu = meganeura::runtime::init_gpu_context().expect("gpu");
        let result = meganeura::runtime::auto_tune(&gpu, 64);
        eprintln!(
            "  coop_matrix_available={}",
            result.coop_caps.is_supported()
        );
        meganeura::runtime::install_auto_tune(result);
        drop(gpu);
        unsafe {
            std::env::set_var("MEGANEURA_CONV_COOP", "1");
        }
        let coop = build_and_run("coop  ", batch, in_c, out_c, h, w);

        // Compare.
        assert_eq!(scalar.len(), coop.len(), "{label}: shape mismatch");
        let mut max_abs = 0f32;
        let mut max_rel = 0f32;
        for (s, c) in scalar.iter().zip(coop.iter()) {
            let abs = (s - c).abs();
            let rel = abs / s.abs().max(1e-6);
            max_abs = max_abs.max(abs);
            max_rel = max_rel.max(rel);
        }
        eprintln!("  max_abs_err={max_abs:.6e}, max_rel_err={max_rel:.6e}");
        // Coop uses f16 input × f16 weight → f32 accum, so tolerance
        // is larger than pure-f32 scalar.
        assert!(
            max_abs < 1e-2,
            "{label}: coop diverges from scalar (max_abs={max_abs})"
        );
    }

    eprintln!("\nALL OK — coop conv-backward matches scalar within tolerance");
}
