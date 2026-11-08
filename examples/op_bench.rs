//! Single-operation microbench: one GEMM or convolution shape per session, for
//! comparison with the vendor libraries through PyTorch
//! (`paper/p3hpc/presentation/data/op-bench/torch_ops.py`, same shape list).
//!
//! The reference times a graph of 30 serial calls on the same tensors. Here
//! the graph holds 30 copies of the operation on the same input and weight,
//! with a barrier after every dispatch, and one timestamped pass covers the
//! step. The GPU time therefore includes the gaps between dispatches but no
//! per-dispatch timestamps, which cost about 7 µs a pass on Mesa ANV.
//! Independent copies would run level by level, every copy's Winograd
//! transforms before any product, so a convolution that keeps its shape is
//! chained instead: each copy reads the previous output, as in a network.
//! Weight transforms read only the weight. An inference session pins that
//! output and skips the transform after the first step, so the timed steps
//! do not recompute it. Serial dispatch also
//! separates a Winograd convolution's weight and input transforms, which an
//! ordinary plan runs together.
//!
//! An `eltwise` row adds a parameter vector to a small input 30 times: the
//! dispatch floor. Chained, every add reads the previous sum, so the pass
//! holds a barrier after each dispatch; independent, the adds all read the
//! input and run as one barrier group, without barriers. The difference is
//! what a barrier costs. A single add per step makes the wall time one
//! submission's round trip, as PyTorch's wall time is for one eager call.
//!
//! The e-graph optimizer is off: a single operation has nothing to fuse,
//! and egglog 3.0 extracts chains of more than ten identical products on a
//! shared weight as shorter ones.
//!
//! Prints one tab-separated row per shape: kind, name, median GPU time of a
//! step divided by 30 (ms), median unprofiled wall time of a step divided by
//! 30 (ms), TFLOP/s from the GPU time (direct-convolution FLOPs for
//! convolutions), and the distinct dispatch labels.
//!
//! Strict f32 with tile search, one shape per process (NVIDIA's driver fails
//! after about ten sessions in one process):
//!
//!   MEGANEURA_DEVICE_ID=<id> MEGANEURA_TUNE=1 MEGANEURA_DISABLE_COOP=1 \
//!   MEGANEURA_GPU_TIMING=1 cargo run --release --example op_bench -- \
//!       paper/p3hpc/presentation/data/op-bench/shapes.txt square-1024
use meganeura::Graph;
use std::time::Instant;

const SERIAL: usize = 30;

fn fill(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 31 + seed * 17) % 251) as f32 - 125.0) * 0.004)
        .collect()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
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
        let mut serial = true;
        let mut copies = SERIAL;
        let (flops, input): (f64, (&str, usize)) = if f[0] == "eltwise" {
            // Mode 0: independent, 1: chained, 2: a single add.
            let (n, mode) = (d[0] as usize, d[1]);
            let x = g.input("x", &[n]);
            let b = g.parameter("b", &[n]);
            if mode == 2 {
                let y = g.add(x, b);
                g.set_outputs(vec![y]);
                copies = 1;
            } else if mode == 1 {
                // Every sum is an output, so pointwise fusion keeps each add
                // its own dispatch.
                let mut y = x;
                let outputs = (0..SERIAL)
                    .map(|_| {
                        y = g.add(y, b);
                        y
                    })
                    .collect();
                g.set_outputs(outputs);
            } else {
                let outputs = (0..SERIAL).map(|_| g.add(x, b)).collect();
                g.set_outputs(outputs);
                serial = false;
            }
            (n as f64, ("x", n))
        } else if f[0] == "gemm" {
            let (m, k, n) = (d[0] as usize, d[1] as usize, d[2] as usize);
            let a = g.input("a", &[m, k]);
            let b = g.parameter("b", &[k, n]);
            let outputs = (0..SERIAL).map(|_| g.matmul(a, b)).collect();
            g.set_outputs(outputs);
            (2.0 * (m * k * n) as f64, ("a", m * k))
        } else {
            let (ci, h, w, co, k, s, p) = (d[0], d[1], d[2], d[3], d[4], d[5], d[6]);
            let x = g.input("x", &[(ci * h * w) as usize]);
            let wt = g.parameter("w", &[(co * ci * k * k) as usize]);
            let (oh, ow) = ((h + 2 * p - k) / s + 1, (w + 2 * p - k) / s + 1);
            if (co, oh, ow) == (ci, h, w) {
                let mut y = x;
                for _ in 0..SERIAL {
                    y = g.conv2d(y, wt, 1, ci, h, w, co, k, k, s, p);
                }
                g.set_outputs(vec![y]);
            } else {
                let outputs = (0..SERIAL)
                    .map(|_| g.conv2d(x, wt, 1, ci, h, w, co, k, k, s, p))
                    .collect();
                g.set_outputs(outputs);
            }
            (
                2.0 * (co * oh * ow * ci * k * k) as f64,
                ("x", (ci * h * w) as usize),
            )
        };
        let mut config = meganeura::SessionConfig::inference_from_env();
        config.optimize.mode = meganeura::optimize::OptimizeMode::Off;
        config.runtime.serial_dispatch = serial;
        let mut session = meganeura::build(&g, config).0;
        let dispatches = &session.plan().dispatches;
        assert_eq!(dispatches.len() % copies, 0, "copies were merged");
        // The schedule interleaves independent copies; name each kind once.
        let mut labels = Vec::new();
        for d in dispatches {
            if !labels.contains(&d.label) {
                labels.push(d.label.clone());
            }
        }
        let labels = labels.join("+");
        for (i, (name, buf)) in session.plan().param_buffers.clone().into_iter().enumerate() {
            let n = session.plan().buffers[buf.0 as usize] / 4;
            session.set_parameter(&name, &fill(n, i + 1));
        }
        session.set_input(input.0, &fill(input.1, 7));
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
            wall.push(t.elapsed().as_secs_f64() * 1e3 / copies as f64);
        }
        // An empty window times every dispatch together in one grouped pass.
        session.set_profiling_window(Some(0..0));
        let mut gpu = Vec::new();
        for _ in 0..30 {
            session.step();
            session.wait();
            let t = session.gpu_timings();
            // Without MEGANEURA_GPU_TIMING, only the wall time is measured.
            assert!(t.len() <= 1, "expected one timed pass");
            gpu.push(
                t.first()
                    .map_or(f64::NAN, |t| t.1.as_secs_f64() * 1e3 / copies as f64),
            );
        }
        session.set_profiling_window(None);
        let gpu_ms = median(gpu);
        println!(
            "{}\t{}\t{:.5}\t{:.5}\t{:.3}\t{}",
            f[0],
            f[1],
            gpu_ms,
            median(wall),
            flops / gpu_ms / 1e9,
            labels
        );
    }
}
