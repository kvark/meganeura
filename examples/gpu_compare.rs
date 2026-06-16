//! Raw hardware ceiling microbench: large compute-bound matmuls (fp32
//! GFLOPS) + a large elementwise pass (memory GB/s). Run on each GPU via
//! MEGANEURA_DEVICE_ID to get the *real* hardware delta, independent of the
//! per-dispatch launch latency that bottlenecks small graphs.
//!
//!   MEGANEURA_DEVICE_ID=12036 cargo run --release --example gpu_compare  # 5070
//!   MEGANEURA_DEVICE_ID=5710  cargo run --release --example gpu_compare  # 610M

use std::time::Instant;

use meganeura::{Graph, build_inference_session};

fn bench_matmul(n: usize, warmup: usize, iters: usize) -> (f64, &'static str) {
    // Square n×n×n matmul: 2 n^3 flops, O(n^2) memory — compute-bound for
    // n >= ~1024, so it saturates the shader cores.
    let mut g = Graph::new();
    let a = g.input("a", &[n, n]);
    let b = g.parameter("b", &[n, n]);
    let c = g.matmul(a, b);
    g.set_outputs(vec![c]);
    let mut s = build_inference_session(&g);

    let kernel = s
        .plan()
        .dispatches
        .iter()
        .find(|d| matches!(d.shader, meganeura::compile::ShaderEntry::MatMul))
        .map(|d| if d.use_coop { "coop" } else if d.use_small_tiles { "small" } else { "tile" })
        .unwrap_or("?");

    s.set_input("a", &vec![0.01_f32; n * n]);
    s.set_parameter("b", &vec![0.01_f32; n * n]);
    for _ in 0..warmup {
        s.step();
    }
    s.wait();
    let t0 = Instant::now();
    for _ in 0..iters {
        s.step();
    }
    s.wait();
    (t0.elapsed().as_secs_f64() / iters as f64, kernel)
}

fn bench_bandwidth(n: usize, warmup: usize, iters: usize) -> f64 {
    // Elementwise c = a + b over n floats: reads 2n, writes n => 12n bytes
    // of traffic, ~0 arithmetic intensity => memory-bandwidth bound.
    let mut g = Graph::new();
    let a = g.input("a", &[n]);
    let b = g.parameter("b", &[n]);
    let c = g.add(a, b);
    g.set_outputs(vec![c]);
    let mut s = build_inference_session(&g);
    s.set_input("a", &vec![1.0_f32; n]);
    s.set_parameter("b", &vec![2.0_f32; n]);
    for _ in 0..warmup {
        s.step();
    }
    s.wait();
    let t0 = Instant::now();
    for _ in 0..iters {
        s.step();
    }
    s.wait();
    t0.elapsed().as_secs_f64() / iters as f64
}

fn main() {
    env_logger::init();
    let device_name = {
        let mut g = Graph::new();
        let x = g.input("x", &[4, 4]);
        let w = g.parameter("w", &[4, 4]);
        let y = g.matmul(x, w);
        g.set_outputs(vec![y]);
        build_inference_session(&g).device_information().device_name.clone()
    };
    println!("device: {device_name}\n");

    println!("== compute: square fp32 matmul ==");
    println!("{:>6} {:>10} {:>9} {:>7}", "N", "ms", "GFLOP/s", "kernel");
    for &n in &[512usize, 1024, 2048, 4096] {
        let iters = if n >= 2048 { 30 } else { 100 };
        let (per_call, kernel) = bench_matmul(n, 10, iters);
        let gflops = 2.0 * (n as f64).powi(3) / per_call / 1e9;
        println!("{n:>6} {:>10.3} {gflops:>9.0} {kernel:>7}", per_call * 1000.0);
    }

    println!("\n== memory: elementwise add (12 bytes/elem) ==");
    println!("{:>10} {:>10} {:>9}", "Melem", "ms", "GB/s");
    for &n in &[1usize << 22, 1 << 24, 1 << 26] {
        let per_call = bench_bandwidth(n, 10, 100);
        let gbs = 12.0 * n as f64 / per_call / 1e9;
        println!("{:>10} {:>10.3} {gbs:>9.0}", n / (1 << 20), per_call * 1000.0);
    }
}
