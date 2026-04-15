//! Matmul throughput microbench: measures GFlops achieved for
//! representative shapes, reports coop-vs-scalar path + %-of-peak.
//!
//! Goal: localize whether the 2–5× PyTorch gap is in matmul or elsewhere.
//! Prints per-shape: wall-clock, GFlops, % of theoretical fp32 peak, and
//! whether the coop-matrix (tensor-core) path is being used.
//!
//! Usage:
//!   cargo run --release --example matmul_throughput

use std::time::Instant;

use meganeura::{Graph, build_inference_session};

/// Theoretical fp32 peak for a few GPUs (TFLOPS). Rough; used only for
/// percentage reporting.
fn approximate_peak_tflops(device_name: &str) -> Option<f64> {
    // Rough numbers from published specs. fp32 shader-core peak.
    let d = device_name.to_lowercase();
    if d.contains("rtx 3050") {
        Some(7.4)
    } else if d.contains("rtx 3060") {
        Some(12.7)
    } else if d.contains("rtx 3070") {
        Some(20.3)
    } else if d.contains("rtx 3080") {
        Some(29.8)
    } else if d.contains("rtx 3090") {
        Some(35.6)
    } else if d.contains("rtx 4090") {
        Some(82.6)
    } else if d.contains("rtx 5080") {
        Some(56.3) // Blackwell, public est.
    } else if d.contains("rtx 5090") {
        Some(104.8)
    } else {
        None
    }
}

/// Theoretical fp16 tensor-core peak for the same GPUs (TFLOPS with fp32
/// accumulator — cuBLAS's usual mode). These are the numbers you'd
/// compare "coop matmul + f16 tiles" against.
fn approximate_fp16_tc_tflops(device_name: &str) -> Option<f64> {
    let d = device_name.to_lowercase();
    if d.contains("rtx 3050") {
        Some(36.7)
    } else if d.contains("rtx 3060") {
        Some(51.0)
    } else if d.contains("rtx 3070") {
        Some(81.3)
    } else if d.contains("rtx 3080") {
        Some(119.0)
    } else if d.contains("rtx 3090") {
        Some(142.0)
    } else if d.contains("rtx 4090") {
        Some(330.0)
    } else if d.contains("rtx 5080") {
        Some(225.0) // Blackwell fp16
    } else if d.contains("rtx 5090") {
        Some(419.0)
    } else {
        None
    }
}

fn bench_shape(
    m: usize,
    n: usize,
    k: usize,
    warmup: usize,
    iters: usize,
) -> (f64, u32, u32, u32, &'static str) {
    let mut g = Graph::new();
    let a = g.input("a", &[m, k]);
    let b = g.parameter("b", &[k, n]);
    let c = g.matmul(a, b);
    g.set_outputs(vec![c]);

    let mut session = build_inference_session(&g);
    let a_data = vec![0.01_f32; m * k];
    let b_data = vec![0.01_f32; k * n];
    session.set_input("a", &a_data);
    session.set_parameter("b", &b_data);

    // Inspect the compiled plan for coop / small-tile / GEMV usage.
    let plan = session.plan();
    let (use_coop, use_small, workgroups, kernel) = plan
        .dispatches
        .iter()
        .find(|d| {
            matches!(
                d.shader,
                meganeura::compile::ShaderEntry::MatMul
                    | meganeura::compile::ShaderEntry::MatMulGemv
            )
        })
        .map(|d| {
            let kernel = match d.shader {
                meganeura::compile::ShaderEntry::MatMulGemv => "gemv",
                _ if d.use_coop => "coop",
                _ if d.use_small_tiles => "small",
                _ => "tile",
            };
            (
                if d.use_coop { 1 } else { 0 },
                if d.use_small_tiles { 1 } else { 0 },
                d.workgroups[0] * d.workgroups[1] * d.workgroups[2],
                kernel,
            )
        })
        .unwrap_or((0, 0, 0, "?"));

    // Warmup.
    for _ in 0..warmup {
        session.step();
    }
    session.wait();

    // Timed loop — submit iters back-to-back then wait once.
    let t0 = Instant::now();
    for _ in 0..iters {
        session.step();
    }
    session.wait();
    let elapsed = t0.elapsed().as_secs_f64();
    let per_call = elapsed / iters as f64;

    (per_call, use_coop, use_small, workgroups, kernel)
}

fn main() {
    env_logger::init();

    // Build a throwaway session just to read the device name.
    let device_name = {
        let mut g = Graph::new();
        let x = g.input("x", &[4, 4]);
        let w = g.parameter("w", &[4, 4]);
        let y = g.matmul(x, w);
        g.set_outputs(vec![y]);
        let s = build_inference_session(&g);
        s.device_information().device_name.clone()
    };
    let peak_fp32 = approximate_peak_tflops(&device_name);
    let peak_fp16 = approximate_fp16_tc_tflops(&device_name);
    println!("device: {}", device_name);
    if let Some(p) = peak_fp32 {
        println!("  fp32 shader peak (rough): {:.1} TFLOPS", p);
    }
    if let Some(p) = peak_fp16 {
        println!(
            "  fp16 tensor-core peak (rough, fp32 accum): {:.1} TFLOPS",
            p
        );
    }
    println!();

    // Shapes chosen from real SmolVLA / SmolLM2 dispatch profiles + a
    // decode-style M=1 probe.
    let shapes: &[(usize, usize, usize, &str)] = &[
        // SmolVLA transformer forward hot sites
        (50, 720, 960, "SmolVLA Q/K/V-ish proj"),
        (50, 720, 2048, "SmolVLA MLP up (seq=50)"),
        (50, 4096, 720, "SmolVLA MLP down-ish"),
        (50, 320, 720, "SmolVLA KV proj (seq=50)"),
        (16, 320, 320, "SmolVLA VLM-side (seq=16)"),
        // SmolLM2-135M shapes (hidden=576, intermediate=1536, kv_dim=192)
        (128, 576, 576, "SmolLM2 Q-proj (seq=128)"),
        (128, 576, 1536, "SmolLM2 MLP up (seq=128)"),
        // Decode regime (batch=1)
        (1, 576, 576, "SmolLM2 decode Q-proj (M=1)"),
        (1, 576, 1536, "SmolLM2 decode MLP up (M=1)"),
        (1, 1536, 576, "SmolLM2 decode MLP down (M=1)"),
        (1, 720, 720, "SmolVLA decode proj (M=1)"),
        (1, 720, 2048, "SmolVLA decode MLP up (M=1)"),
    ];

    println!(
        "{:<42} {:>6} {:>7} {:>6} {:>6} {:>9} {:>9}",
        "shape (M×N×K)", "ms", "GFlops", "kernel", "wgs", "%fp32pk", "%fp16pk"
    );
    println!("{}", "-".repeat(120));

    for &(m, n, k, label) in shapes {
        let (per_call, _coop, _small, wgs, kernel) = bench_shape(m, n, k, 20, 200);
        let flops = 2.0 * (m * n * k) as f64;
        let gflops = flops / per_call / 1e9;
        let pct_fp32 = peak_fp32.map(|p| 100.0 * gflops / (p * 1000.0));
        let pct_fp16 = peak_fp16.map(|p| 100.0 * gflops / (p * 1000.0));
        println!(
            "{:<42} {:>6.3} {:>7.0} {:>6} {:>6} {:>8} {:>8}  {}",
            format!("{}×{}×{}", m, n, k),
            per_call * 1000.0,
            gflops,
            kernel,
            wgs,
            pct_fp32.map_or("n/a".to_string(), |p| format!("{:.1}%", p)),
            pct_fp16.map_or("n/a".to_string(), |p| format!("{:.1}%", p)),
            label,
        );
    }
}
