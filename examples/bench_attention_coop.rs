//! What does cooperative flash attention forward cost, and what does it cost
//! in precision?
//!
//! Item 1 of `docs/open-items.md` is the only open item that is a defect in a
//! shipped path rather than a tidiness question. On NVIDIA the cooperative f16
//! forward loses about two decimal digits: the worst element of every failing
//! oracle comparison is exactly `f16(reference)`. Three policies are available,
//! and picking between them is a performance decision that had no measurement
//! behind it:
//!
//! * `Auto` — cooperative f16 forward. Fast, loses precision.
//! * `NativeF32` — the RTX 5070 advertises `f16_tile: 16, f32_tile: 0`, so
//!   there is no f32 cooperative tile and this falls back to the scalar kernel.
//! * coop forward gated off for inference.
//!
//! This measures the three on the shapes that decide it. It reports wall time
//! and the worst absolute error against a CPU f64 reference computed here, so
//! the speed/precision trade is one table rather than two arguments.
//!
//! The reference is deliberately independent of the device: softmax, scores and
//! the weighted sum in f64 on the CPU. It shares no code with the WGSL, so an
//! error against it is the kernel's, not a bug in a shared helper.
//!
//! A cooperative kernel with f16 operands should show an error bounded by
//! `f16(eps)`; one with f32 operands should not. The `exact_f16` column reports
//! how many results are *exactly* f16-representable, which distinguishes the two
//! ways an f16 error can arise: arithmetic error leaves a value near the
//! reference but off it, whereas a representation round lands on it.
//!
//! Run:
//!   cargo run --release --example bench_attention_coop

use std::time::{Duration, Instant};

use meganeura::{CoopPolicy, Graph, Mode, Session, SessionConfig, SessionOptions};

/// `(q_len, kv_len, heads, kv_heads, head_dim)`.
type Shape = (usize, usize, usize, usize, usize);

fn q_width(s: &Shape) -> usize {
    s.2 * s.4
}

fn kv_width(s: &Shape) -> usize {
    s.3 * s.4
}

fn causal_graph(s: &Shape) -> Graph {
    let mut g = Graph::new();
    let q = g.input("q", &[s.0, q_width(s)]);
    let k = g.input("k", &[s.1, kv_width(s)]);
    let v = g.input("v", &[s.1, kv_width(s)]);
    let o = g.causal_attention(q, k, v, s.2 as u32, s.3 as u32, s.4 as u32);
    g.set_outputs(vec![o]);
    g
}

/// The oracle's flash-shape set, which is where the failure shows up.
///
/// `head_dim` decides the kernel: 256 uses bq=32, 128 uses bq=64, 64 uses bq=128,
/// 32 and 16 fall to one lane per query. Shapes just below each threshold take
/// the scalar path, so they are the control — a policy that does not change
/// them is not changing the kernel selection.
const FLASH_SHAPES: &[(&str, Shape)] = &[
    ("scalar control hd=64", (31, 31, 2, 1, 64)),
    ("flash hd=256", (33, 33, 2, 1, 256)),
    ("flash hd=128 gqa", (64, 64, 4, 1, 128)),
    ("scalar control hd=128", (63, 63, 2, 2, 128)),
    ("flash hd=64", (130, 130, 2, 1, 64)),
    ("flash hd=32", (260, 260, 2, 2, 32)),
    ("flash hd=16", (257, 257, 1, 1, 16)),
];

/// A shape large enough to time rather than to check: 1024 queries, 8 heads.
const BIG: Shape = (1024, 1024, 8, 2, 64);

/// Deterministic feeds, so all three policies see identical input.
fn feeds(s: &Shape) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let fill = |n: usize, seed: u32| {
        (0..n)
            .map(|i| {
                // A cheap LCG: reproducible, no RNG dependency, and spread
                // enough that a precision difference is visible above the
                // reference's own noise.
                let x = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed);
                ((x >> 8) as f32 / 8_388_608.0) - 1.0
            })
            .collect()
    };
    (
        fill(s.0 * q_width(s), 1),
        fill(s.1 * kv_width(s), 2),
        fill(s.1 * kv_width(s), 3),
    )
}

/// Causal multi-head attention in f64 on the CPU.
///
/// Independent of the device by construction: it shares no code with the WGSL,
/// so an error against it is the kernel's, not a bug in a shared helper.
fn reference(s: &Shape, q: &[f32], k: &[f32], v: &[f32]) -> Vec<f32> {
    let (qlen, kvlen, heads, kv_heads, hd) = *s;
    let qw = heads * hd;
    let kw = kv_heads * hd;
    let mut out = vec![0f64; qlen * qw];
    let scale = 1.0 / (hd as f64).sqrt();

    for h in 0..heads {
        let kvh = h * kv_heads / heads;
        for i in 0..qlen {
            // Causal: query i attends to keys 0..=i, clamped to what exists.
            let hi = (i + 1).min(kvlen);
            let mut scores: Vec<f64> = (0..hi)
                .map(|j| {
                    let mut acc = 0f64;
                    for d in 0..hd {
                        acc += (q[i * qw + h * hd + d] as f64) * (k[j * kw + kvh * hd + d] as f64);
                    }
                    acc * scale
                })
                .collect();
            // Softmax, subtracting the max so the exponent cannot overflow.
            let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut sum = 0f64;
            for x in &mut scores {
                *x = (*x - max).exp();
                sum += *x;
            }
            for d in 0..hd {
                let mut acc = 0f64;
                for (j, &s) in scores.iter().enumerate() {
                    acc += (s / sum) * (v[j * kw + kvh * hd + d] as f64);
                }
                out[i * qw + h * hd + d] = acc;
            }
        }
    }
    out.iter().map(|&x| x as f32).collect()
}

/// One context for the whole process.
///
/// `from_env` builds a device-selected context per call and hands it to exactly
/// one session, so a sweep of 24 sessions exhausts the NVIDIA driver's
/// per-process budget (about ten) and the next one fails to open. Sharing is
/// what `from_env_with_gpu` is for; sessions still isolate their buffers and
/// plans.
fn shared_gpu() -> std::sync::Arc<blade_graphics::Context> {
    static GPU: std::sync::OnceLock<std::sync::Arc<blade_graphics::Context>> =
        std::sync::OnceLock::new();
    GPU.get_or_init(|| {
        std::sync::Arc::new(
            meganeura::init_gpu_context_with(meganeura::GpuOptions::from_env()).expect(
                "a GPU context. If MEGANEURA_DEVICE_ID names a device that cannot be opened, \
                 that is what this reports.",
            ),
        )
    })
    .clone()
}

fn session_for(g: &Graph, coop: CoopPolicy) -> Session {
    let base = SessionConfig::from_env_with_gpu(Some(shared_gpu()));
    let (session, _) = meganeura::build(
        g,
        SessionConfig {
            mode: Mode::Inference,
            runtime: SessionOptions {
                coop,
                ..base.runtime
            },
            ..base
        },
    );
    session
}

/// The *minimum* of `v`, not the median.
///
/// The NVIDIA driver takes a periodic ~9 ms stall that has nothing to do with
/// the graph: on a 200-step run of an 8x8 matmul, p10 is 0.06 ms, p50 is
/// 1.3 ms, p90 is 9.0 ms and max is 9.7 ms, identically for all three coop
/// policies. Which sample the median lands on therefore depends on where the
/// stall boundaries fall, which is why an earlier version of this benchmark
/// reported ~8.6 ms for one policy and ~1.6 ms for another and appeared to
/// show the scalar kernel six times faster than the cooperative one.
///
/// The minimum is stable to within 10% across runs and policies, so it is the
/// only statistic here that measures the kernel.
fn fastest(mut v: Vec<Duration>) -> Duration {
    v.sort_unstable();
    v[0]
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

/// Worst absolute difference, and the largest `f16` rounding of the reference.
///
/// The second column is the diagnostic that identifies item 1: when the worst
/// error equals the worst f16 rounding, the error *is* f16 precision and nothing
/// else.
struct Error {
    worst_abs: f32,
    worst_f16: f32,
    /// Fraction of results that are *exactly* f16-representable.
    ///
    /// This separates "the kernel accumulates in f16" from "the result was
    /// rounded to f16 on the way out". Arithmetic error leaves a value near the
    /// reference but not on it; a representation round lands exactly on it.
    /// The oracle's failing elements are all exactly `f16(reference)`, which
    /// points at the second, not the first.
    exactly_f16: f64,
}

fn errors(got: &[f32], want: &[f32]) -> Error {
    let mut worst_abs = 0f32;
    let mut worst_f16 = 0f32;
    let mut exact = 0usize;
    for (&g, &w) in got.iter().zip(want) {
        worst_abs = worst_abs.max((g - w).abs());
        // Compare against the gap the reference value itself rounds across,
        // which is what "the error is f16 precision" means concretely.
        let rounded = half::f16::from_f32(w).to_f32();
        worst_f16 = worst_f16.max((rounded - w).abs());
        if g == half::f16::from_f32(g).to_f32() {
            exact += 1;
        }
    }
    Error {
        worst_abs,
        worst_f16,
        exactly_f16: exact as f64 / got.len().max(1) as f64,
    }
}

struct Row {
    time_ms: f64,
    worst_abs: f32,
    worst_f16: f32,
    exactly_f16: f64,
}

fn run(g: &Graph, s: &Shape, feeds: &(Vec<f32>, Vec<f32>, Vec<f32>), coop: CoopPolicy) -> Row {
    let mut session = session_for(g, coop);
    session.set_input("q", &feeds.0);
    session.set_input("k", &feeds.1);
    session.set_input("v", &feeds.2);

    // Warm up: pipeline creation and shader compilation land on the first call.
    for _ in 0..3 {
        session.step();
    }
    session.wait();

    // Each step is timed with its own wait inside the region. Timing a bare
    // `step` measures queue submission, which pipelines across iterations and
    // reports whichever stall the driver happened to take — the result was
    // bimodal at 0.04 ms and 9 ms with no relation to the shape.
    let runs = 200;
    let mut times = Vec::with_capacity(runs);
    for _ in 0..runs {
        let start = Instant::now();
        session.step();
        session.wait();
        times.push(start.elapsed());
    }

    let mut got = vec![0f32; s.0 * q_width(s)];
    session.read_output_by_index(0, &mut got);
    let want = reference(s, &feeds.0, &feeds.1, &feeds.2);
    let e = errors(&got, &want);
    Row {
        time_ms: ms(fastest(times)),
        worst_abs: e.worst_abs,
        worst_f16: e.worst_f16,
        exactly_f16: e.exactly_f16,
    }
}

fn main() {
    println!("device: {:?}", {
        let s = session_for(&causal_graph(&BIG), CoopPolicy::Disabled);
        s.context().device_information().device_name.clone()
    });
    println!("(MEGANEURA_DEVICE_ID selects the adapter)");
    println!();

    let policies: [(&str, CoopPolicy); 3] = [
        ("Auto (f16 coop)", CoopPolicy::Auto),
        ("NativeF32", CoopPolicy::NativeF32),
        ("Disabled (scalar)", CoopPolicy::Disabled),
    ];

    println!("Precision: worst absolute error against an f64 CPU reference, and");
    println!("the largest f16 rounding of that reference. When the two columns");
    println!("match, the error *is* f16 rounding and nothing else.");
    println!();

    for (label, shape) in FLASH_SHAPES
        .iter()
        .copied()
        .chain(std::iter::once(("BIG", BIG)))
    {
        let g = causal_graph(&shape);
        let f = feeds(&shape);
        println!(
            "{label}  q={} kv={} heads={}/{} hd={}",
            shape.0, shape.1, shape.2, shape.3, shape.4
        );
        println!(
            "  {:<22} {:>9} {:>12} {:>12} {:>9}   {}",
            "policy", "time_ms", "worst_abs", "f16(ref)", "exact_f16", "verdict"
        );
        for (name, coop) in policies {
            let row = run(&g, &shape, &f, coop);
            // Equal to f16 precision means the difference is the rounding of the
            // reference itself; anything at or below it is free.
            let verdict = if row.worst_abs <= row.worst_f16 * 1.01 {
                "at f16 rounding"
            } else {
                "worse than f16"
            };
            println!(
                "  {:<22} {:>9.4} {:>12.3e} {:>12.3e} {:>8.1}%   {}",
                name,
                row.time_ms,
                row.worst_abs,
                row.worst_f16,
                row.exactly_f16 * 100.0,
                verdict
            );
        }
        println!();
    }
}
