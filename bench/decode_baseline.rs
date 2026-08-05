//! SmolLM2 single-token decode: latency, and where it goes.
//!
//! Reports wall-clock per token, splits CPU command recording from GPU
//! execution, and breaks device time down by kernel. `DECODE_CPU_REF=1`
//! additionally checks the result against an independent CPU
//! implementation of the same decode step, which is what makes a latency
//! number meaningful.
//!
//! Note that per-dispatch timestamps cost more than they measure here: the
//! instrumented run sums to more device time than the untimed token takes,
//! so the wall-clock figures are the ones to trust.
//!
//! Usage:
//!   MEGANEURA_DEVICE_ID=<id> MEGANEURA_GPU_TIMING=1 \
//!     cargo run --release --example decode_baseline

use meganeura::models::smollm2;
use meganeura::{Graph, Mode, SessionConfig};
use std::time::Instant;

fn decode_pos() -> u32 {
    std::env::var("DECODE_POS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(7)
}

fn main() {
    env_logger::init();
    let mut cfg = smollm2::SmolLM2Config::smollm2_135m();
    if let Ok(n) = std::env::var("DECODE_LAYERS") {
        cfg.num_hidden_layers = n.parse().expect("DECODE_LAYERS");
    }
    let max_seq = 512usize;

    let mut g = Graph::new();
    let (logits, _k, _v) = smollm2::build_decode_graph(&mut g, &cfg, max_seq);
    g.set_outputs(vec![logits]);

    let mut sc = SessionConfig::from_env();
    sc.mode = Mode::Inference;
    let mut sess = meganeura::build(&g, sc).0;

    // Deterministic synthetic weights: this measures time, not quality.
    //
    // Fusion can introduce a *derived* parameter, e.g. the concatenated
    // `gate_proj.weight+up_proj.weight` the SwiGLU rewrite creates. Filling
    // that from its own name would give the fused graph different weights
    // than the unfused one, so it has to be built from its sources.
    let synth = |name: &str, n: usize| -> Vec<f32> {
        (0..n)
            .map(|i| {
                let h = name.len().wrapping_mul(31).wrapping_add(i);
                ((h % 200) as f32 - 100.0) * 0.002
            })
            .collect()
    };
    for (name, buf_ref) in sess.plan().param_buffers.clone() {
        // Reduced-precision weights hold fewer bytes per element, so the
        // element count cannot be derived from the byte size alone.
        let bytes = sess.plan().buffers[buf_ref.0 as usize];
        let n = match sess.plan().weight_buffers.get(&buf_ref).map(|w| w.0) {
            Some(meganeura::compile::WeightFormat::F16) => bytes / 2,
            _ => bytes / 4,
        };
        let data: Vec<f32> = match name.split_once('+') {
            Some((a, b)) => {
                // Column-concatenation of two [K, N] weights into [K, 2N].
                let half = n / 2;
                let (av, bv) = (synth(a, half), synth(b, half));
                let cols = half / cfg.hidden_size;
                (0..n)
                    .map(|i| {
                        let (row, col) = (i / (2 * cols), i % (2 * cols));
                        if col < cols {
                            av[row * cols + col]
                        } else {
                            bv[row * cols + (col - cols)]
                        }
                    })
                    .collect()
            }
            None => synth(&name, n),
        };
        sess.set_parameter(&name, &data);
    }

    if std::env::var("DECODE_LIST_PARAMS").is_ok() {
        let names: Vec<String> = sess
            .plan()
            .param_buffers
            .iter()
            .map(|(n, _)| n.clone())
            .collect();
        println!(
            "  params ({}): kv_cache entries = {}",
            names.len(),
            names.iter().filter(|n| n.contains("kv_cache")).count()
        );
        for n in &names {
            println!("    {n}");
        }
    }

    let dispatches = sess.plan().dispatches.len();
    println!("SmolLM2-135M decode: {dispatches} dispatches");

    if std::env::var("DECODE_CPU_REF").is_ok() {
        sess.set_input_u32("token_ids", &[42]);
        sess.set_input_u32("kv_pos", &[decode_pos()]);
        sess.step();
        sess.wait();
        let mut gpu = vec![0.0f32; cfg.vocab_size];
        sess.read_output_by_index(0, &mut gpu);
        let cpu = cpu_reference(&cfg, 42, decode_pos());
        let scale = cpu.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let d = gpu
            .iter()
            .zip(&cpu)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!(
            "  vs CPU reference: max diff {d:.4e}, scale {scale:.4e}, rel {:.3e} [{}]",
            d / scale,
            if d / scale < 1e-3 {
                "MATCH"
            } else {
                "MISMATCH"
            }
        );
        return;
    }

    // Dump the logits so the two paths can be compared numerically.
    if let Ok(path) = std::env::var("DECODE_DUMP_LOGITS") {
        sess.set_input_u32("token_ids", &[42]);
        sess.set_input_u32("kv_pos", &[7]);
        sess.step();
        sess.wait();
        let mut out = vec![0.0f32; cfg.vocab_size];
        sess.read_output_by_index(0, &mut out);
        let bytes: Vec<u8> = out.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&path, bytes).expect("write logits");
        println!("  wrote {} logits to {path}", out.len());
    }

    sess.set_input_u32("token_ids", &[42]);
    sess.set_input_u32("kv_pos", &[7]);

    for _ in 0..20 {
        sess.step();
    }
    sess.wait();

    // Split CPU command recording from GPU execution. `step` records and
    // submits; `wait` blocks for completion. If recording dominates, the
    // path is CPU-bound and no shader change can help.
    {
        let mut rec_best = f64::MAX;
        let mut tot_best = f64::MAX;
        for _ in 0..100 {
            let t0 = Instant::now();
            sess.step();
            let rec = t0.elapsed().as_secs_f64();
            sess.wait();
            let tot = t0.elapsed().as_secs_f64();
            rec_best = rec_best.min(rec);
            tot_best = tot_best.min(tot);
        }
        println!(
            "  cpu record+submit {:.3} ms, total {:.3} ms => gpu-visible {:.3} ms ({:.0}% cpu)",
            rec_best * 1e3,
            tot_best * 1e3,
            (tot_best - rec_best) * 1e3,
            100.0 * rec_best / tot_best
        );
    }

    // Wall clock per token, best of many: anything sharing the device only
    // inflates it.
    let mut best = f64::MAX;
    let mut total = 0.0;
    let runs = 200;
    for _ in 0..runs {
        let t = Instant::now();
        sess.step();
        sess.wait();
        let secs = t.elapsed().as_secs_f64();
        best = best.min(secs);
        total += secs;
    }
    println!(
        "  wall clock per token: best {:.3} ms, mean {:.3} ms",
        best * 1e3,
        total * 1e3 / runs as f64
    );

    // Per-dispatch device timestamps. The difference between wall clock and
    // their sum is what the boundaries between dispatches cost.
    sess.set_profiling(true);
    for _ in 0..3 {
        sess.step();
        sess.wait();
    }
    let timings = sess.gpu_timings();
    if timings.is_empty() {
        println!("  (no GPU timings: set MEGANEURA_GPU_TIMING=1 before running)");
        return;
    }
    let dispatch_total: f64 = timings.iter().map(|(_, d)| d.as_secs_f64()).sum::<f64>();
    println!(
        "  device time in {} timed dispatches: {:.3} ms",
        timings.len(),
        dispatch_total * 1e3
    );
    println!(
        "  unaccounted (boundaries, launch, drain): {:.3} ms ({:.0}% of wall clock)",
        (best - dispatch_total) * 1e3,
        100.0 * (best - dispatch_total) / best
    );

    let mut by_kernel: std::collections::HashMap<String, (f64, usize)> =
        std::collections::HashMap::new();
    for (label, d) in &timings {
        let key = label
            .split(['[', '('])
            .next()
            .unwrap_or(label)
            .trim()
            .to_string();
        let e = by_kernel.entry(key).or_insert((0.0, 0));
        e.0 += d.as_secs_f64();
        e.1 += 1;
    }
    let mut rows: Vec<_> = by_kernel.into_iter().collect();
    rows.sort_by(|a, b| b.1.0.partial_cmp(&a.1.0).unwrap());
    println!("  top kernels by device time:");
    for (name, (secs, count)) in rows.iter().take(10) {
        println!(
            "    {name:34} {:7.3} ms  x{count:4}  ({:4.1}% of device time)",
            secs * 1e3,
            100.0 * secs / dispatch_total
        );
    }
}

/// Independent CPU reference for a one-layer SmolLM2 decode step, built
/// from the same synthetic parameters. Comparing both GPU paths against it
/// says which one is wrong, rather than only that they disagree.
#[allow(clippy::too_many_arguments)]
pub fn cpu_reference(cfg: &smollm2::SmolLM2Config, token: u32, pos: u32) -> Vec<f32> {
    let h = cfg.hidden_size;
    let kv = cfg.kv_dim();
    let ffn = cfg.intermediate_size;
    let hd = cfg.head_dim() as usize;
    // When the projections are stored as f16 the GPU sees rounded weights,
    // so the reference has to round too or the comparison measures the
    // precision change rather than the implementation.
    let param = |name: &str, n: usize| -> Vec<f32> {
        let round = false;
        (0..n)
            .map(|i| {
                let x = name.len().wrapping_mul(31).wrapping_add(i);
                let v = ((x % 200) as f32 - 100.0) * 0.002;
                if round {
                    half::f16::from_f32(v).to_f32()
                } else {
                    v
                }
            })
            .collect()
    };
    let embed = param("model.embed_tokens.weight", cfg.vocab_size * h);
    let mut x: Vec<f32> = embed[token as usize * h..(token as usize + 1) * h].to_vec();

    // b is [K, N]: out[n] = sum_k a[k] * b[k*N + n]
    let gemv = |a: &[f32], b: &[f32], k: usize, n: usize| -> Vec<f32> {
        (0..n)
            .map(|j| (0..k).map(|i| a[i] * b[i * n + j]).sum())
            .collect()
    };
    let rms = |v: &[f32], w: &[f32]| -> Vec<f32> {
        let ms: f32 = v.iter().map(|t| t * t).sum::<f32>() / v.len() as f32;
        let r = 1.0 / (ms + cfg.rms_norm_eps).sqrt();
        v.iter().zip(w).map(|(t, wt)| t * r * wt).collect()
    };
    let rope = |v: &mut [f32], dim: usize| {
        let half = hd / 2;
        for i in 0..dim / 2 {
            let head = i / half;
            let pih = i % half;
            let inv = cfg.rope_theta.powf(-2.0 * pih as f32 / hd as f32);
            let (s, c) = (pos as f32 * inv).sin_cos();
            let (i0, i1) = (head * hd + pih, head * hd + pih + half);
            let (a, b) = (v[i0], v[i1]);
            v[i0] = a * c - b * s;
            v[i1] = a * s + b * c;
        }
    };

    for l in 0..cfg.num_hidden_layers {
        let p = format!("model.layers.{l}");
        let xn = rms(&x, &param(&format!("{p}.input_layernorm.weight"), h));
        let mut q = gemv(
            &xn,
            &param(&format!("{p}.self_attn.q_proj.weight"), h * h),
            h,
            h,
        );
        let mut k = gemv(
            &xn,
            &param(&format!("{p}.self_attn.k_proj.weight"), h * kv),
            h,
            kv,
        );
        let v = gemv(
            &xn,
            &param(&format!("{p}.self_attn.v_proj.weight"), h * kv),
            h,
            kv,
        );
        rope(&mut q, h);
        rope(&mut k, kv);

        let max_seq = 512usize;
        let mut kc = param(&format!("kv_cache.layer.{l}.k"), max_seq * kv);
        let mut vc = param(&format!("kv_cache.layer.{l}.v"), max_seq * kv);
        kc[pos as usize * kv..(pos as usize + 1) * kv].copy_from_slice(&k);
        vc[pos as usize * kv..(pos as usize + 1) * kv].copy_from_slice(&v);

        let mut attn = vec![0.0f32; h];
        let n = pos as usize + 1;
        for head in 0..cfg.num_attention_heads as usize {
            let kvh = head / (cfg.num_attention_heads as usize / cfg.num_key_value_heads as usize);
            let scale = 1.0 / (hd as f32).sqrt();
            let sc: Vec<f32> = (0..n)
                .map(|t| {
                    (0..hd)
                        .map(|d| q[head * hd + d] * kc[t * kv + kvh * hd + d])
                        .sum::<f32>()
                        * scale
                })
                .collect();
            let m = sc.iter().cloned().fold(f32::MIN, f32::max);
            let e: Vec<f32> = sc.iter().map(|s| (s - m).exp()).collect();
            let den: f32 = e.iter().sum();
            for d in 0..hd {
                attn[head * hd + d] = (0..n)
                    .map(|t| e[t] * vc[t * kv + kvh * hd + d])
                    .sum::<f32>()
                    / den;
            }
        }
        let o = gemv(
            &attn,
            &param(&format!("{p}.self_attn.o_proj.weight"), h * h),
            h,
            h,
        );
        for i in 0..h {
            x[i] += o[i];
        }
        let xn2 = rms(
            &x,
            &param(&format!("{p}.post_attention_layernorm.weight"), h),
        );
        let g = gemv(
            &xn2,
            &param(&format!("{p}.mlp.gate_proj.weight"), h * ffn),
            h,
            ffn,
        );
        let u = gemv(
            &xn2,
            &param(&format!("{p}.mlp.up_proj.weight"), h * ffn),
            h,
            ffn,
        );
        // Which half the concatenated gate/up buffer treats as the gate is
        // a convention the reference has to match; check both.
        let swap = std::env::var("CPU_REF_SWAP_SWIGLU").is_ok();
        let act: Vec<f32> = g
            .iter()
            .zip(&u)
            .map(|(gv, uv)| {
                if swap {
                    (uv / (1.0 + (-uv).exp())) * gv
                } else {
                    (gv / (1.0 + (-gv).exp())) * uv
                }
            })
            .collect();
        let d = gemv(
            &act,
            &param(&format!("{p}.mlp.down_proj.weight"), ffn * h),
            ffn,
            h,
        );
        for i in 0..h {
            x[i] += d[i];
        }
    }
    let xf = rms(&x, &param("model.norm.weight", h));
    // Tied embeddings: logits = x @ embed^T
    (0..cfg.vocab_size)
        .map(|v| (0..h).map(|i| xf[i] * embed[v * h + i]).sum())
        .collect()
}
