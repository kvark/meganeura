//! Single-operation microbench: one GEMM or convolution per session, for
//! comparison with the vendor libraries through PyTorch
//! (`paper/p3hpc/presentation/data/op-bench/torch_ops.py`, same shape list).
//!
//! Prints one tab-separated row per shape: kind, name, median summed GPU pass
//! time of a profiled step (ms), median unprofiled wall time of a step (ms),
//! TFLOP/s from the GPU time (direct-convolution FLOPs for convolutions), and
//! the pass labels.
//!
//! Strict f32 with tile search, one shape per process (NVIDIA's driver fails
//! after about ten sessions in one process):
//!
//!   MEGANEURA_DEVICE_ID=<id> MEGANEURA_TUNE=1 MEGANEURA_DISABLE_COOP=1 \
//!   MEGANEURA_GPU_TIMING=1 cargo run --release --example op_bench -- \
//!       paper/p3hpc/presentation/data/op-bench/shapes.txt square-1024
use meganeura::Graph;
use std::time::Instant;

fn fill(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 31 + seed * 17) % 251) as f32 - 125.0) * 0.004)
        .collect()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let shapes = std::fs::read_to_string(std::env::args().nth(1).expect("shapes file")).unwrap();
    let only = std::env::args().nth(2);
    for line in shapes
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
    {
        let f: Vec<&str> = line.split_whitespace().collect();
        if only.as_deref().is_some_and(|o| o != f[1]) {
            continue;
        }
        let d: Vec<u32> = f[2..].iter().map(|x| x.parse().unwrap()).collect();
        let mut g = Graph::new();
        let (flops, inputs): (f64, Vec<(&str, usize)>) = if f[0] == "gemm" {
            let (m, k, n) = (d[0] as usize, d[1] as usize, d[2] as usize);
            let a = g.input("a", &[m, k]);
            let b = g.parameter("b", &[k, n]);
            let c = g.matmul(a, b);
            g.set_outputs(vec![c]);
            (2.0 * (m * k * n) as f64, vec![("a", m * k)])
        } else {
            let (ci, h, w, co, k, s, p) = (d[0], d[1], d[2], d[3], d[4], d[5], d[6]);
            let x = g.input("x", &[(ci * h * w) as usize]);
            let wt = g.parameter("w", &[(co * ci * k * k) as usize]);
            let y = g.conv2d(x, wt, 1, ci, h, w, co, k, k, s, p);
            g.set_outputs(vec![y]);
            let (oh, ow) = ((h + 2 * p - k) / s + 1, (w + 2 * p - k) / s + 1);
            (
                2.0 * (co * oh * ow * ci * k * k) as f64,
                vec![("x", (ci * h * w) as usize)],
            )
        };
        let mut session = meganeura::build(&g, meganeura::SessionConfig::inference_from_env()).0;
        for (i, (name, buf)) in session.plan().param_buffers.clone().into_iter().enumerate() {
            let n = session.plan().buffers[buf.0 as usize] / 4;
            session.set_parameter(&name, &fill(n, i + 1));
        }
        for (name, n) in &inputs {
            session.set_input(name, &fill(*n, 7));
        }
        let warm = Instant::now();
        while warm.elapsed().as_secs_f32() < 1.0 {
            session.step();
            session.wait();
        }
        let mut wall = Vec::new();
        for _ in 0..50 {
            let t = Instant::now();
            session.step();
            session.wait();
            wall.push(t.elapsed().as_secs_f64() * 1e3);
        }
        session.set_profiling(true);
        let mut gpu = Vec::new();
        let mut passes = String::new();
        for _ in 0..30 {
            session.step();
            session.wait();
            let t = session.gpu_timings();
            passes = t
                .iter()
                .map(|(n, _)| n.clone())
                .collect::<Vec<_>>()
                .join("+");
            gpu.push(t.iter().map(|(_, d)| d.as_secs_f64() * 1e3).sum());
        }
        session.set_profiling(false);
        let gpu_ms = median(gpu);
        println!(
            "{}\t{}\t{:.4}\t{:.4}\t{:.3}\t{}",
            f[0],
            f[1],
            gpu_ms,
            median(wall),
            flops / gpu_ms / 1e9,
            passes
        );
    }
}
