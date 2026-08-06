//! Per-op correctness for the decode path.
//!
//! Builds a one-op graph for each kernel a decode step uses, runs it, and
//! compares against the same arithmetic on the CPU. When a whole-model
//! comparison disagrees this says which op is responsible, and it is cheap
//! enough to run as a gate on kernel changes.
//!
//! It caught the f16 embedding silently returning zeros: `g.embedding`
//! never checked the table's dtype, so an f16 table selected the f32
//! kernel. Keep new ops represented here.
//!
//! Usage: MEGANEURA_DEVICE_ID=<id> cargo run --release --example decode_op_diff

use meganeura::models::smollm2::SmolLM2Config;
use meganeura::{Graph, Mode, SessionConfig};

fn synth(name: &str, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = name.len().wrapping_mul(31).wrapping_add(i);
            ((x % 200) as f32 - 100.0) * 0.002
        })
        .collect()
}

/// Build, run, and read back a single-output graph.
fn run(g: &Graph, out_len: usize, pos: u32) -> Vec<f32> {
    let mut sc = SessionConfig::from_env();
    sc.mode = Mode::Inference;
    let mut sess = meganeura::build(g, sc).0;
    for (name, buf) in sess.plan().param_buffers.clone() {
        let bytes = sess.plan().buffers[buf.0 as usize];
        let n = match sess.plan().weight_buffers.get(&buf).map(|w| w.0) {
            Some(meganeura::compile::WeightFormat::F16) => bytes / 2,
            _ => bytes / 4,
        };
        sess.set_parameter(&name, &synth(&name, n));
    }
    if sess.plan().input_buffers.iter().any(|(n, _)| n == "kv_pos") {
        sess.set_input_u32("kv_pos", &[pos]);
    }
    sess.step();
    sess.wait();
    let mut out = vec![0.0f32; out_len];
    sess.read_output_by_index(0, &mut out);
    out
}

/// Like `run`, but also supplies a token id.
fn run_tok(g: &Graph, out_len: usize, pos: u32, tok: u32) -> Vec<f32> {
    let mut sc = SessionConfig::from_env();
    sc.mode = Mode::Inference;
    let mut sess = meganeura::build(g, sc).0;
    for (name, buf) in sess.plan().param_buffers.clone() {
        let bytes = sess.plan().buffers[buf.0 as usize];
        let n = match sess.plan().weight_buffers.get(&buf).map(|w| w.0) {
            Some(meganeura::compile::WeightFormat::F16) => bytes / 2,
            _ => bytes / 4,
        };
        sess.set_parameter(&name, &synth(&name, n));
    }
    sess.set_input_u32("token_ids", &[tok]);
    if sess.plan().input_buffers.iter().any(|(n, _)| n == "kv_pos") {
        sess.set_input_u32("kv_pos", &[pos]);
    }
    sess.step();
    sess.wait();
    let mut out = vec![0.0f32; out_len];
    sess.read_output_by_index(0, &mut out);
    out
}

fn report(label: &str, gpu: &[f32], cpu: &[f32]) {
    let scale = cpu.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-9);
    let d = gpu
        .iter()
        .zip(cpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!(
        "  {label:22} max diff {d:9.3e}  rel {:9.3e}  [{}]",
        d / scale,
        if d / scale < 1e-4 {
            "MATCH"
        } else {
            "MISMATCH"
        }
    );
}

fn main() {
    env_logger::init();
    let cfg = SmolLM2Config::smollm2_135m();
    let h = cfg.hidden_size;
    let kv = cfg.kv_dim();
    let hd = cfg.head_dim() as usize;
    let pos = 7u32;
    println!("SmolLM2-135M decode ops vs CPU (hidden {h}, kv {kv}, head_dim {hd}, pos {pos}):");

    // --- rms_norm ---
    {
        let mut g = Graph::new();
        let x = g.parameter("t.x", &[1, h]);
        let w = g.parameter("t.w", &[h]);
        let out = g.rms_norm(x, w, cfg.rms_norm_eps);
        g.set_outputs(vec![out]);
        let gpu = run(&g, h, pos);
        let (xv, wv) = (synth("t.x", h), synth("t.w", h));
        let ms: f32 = xv.iter().map(|v| v * v).sum::<f32>() / h as f32;
        let r = 1.0 / (ms + cfg.rms_norm_eps).sqrt();
        let cpu: Vec<f32> = xv.iter().zip(&wv).map(|(v, w)| v * r * w).collect();
        report("rms_norm", &gpu, &cpu);
    }

    // --- matmul (GEMV), b stored [K, N] ---
    {
        let mut g = Graph::new();
        let a = g.parameter("t.a", &[1, h]);
        let b = g.parameter("t.b", &[h, kv]);
        let out = g.matmul(a, b);
        g.set_outputs(vec![out]);
        let gpu = run(&g, kv, pos);
        let (av, bv) = (synth("t.a", h), synth("t.b", h * kv));
        let cpu: Vec<f32> = (0..kv)
            .map(|j| (0..h).map(|i| av[i] * bv[i * kv + j]).sum())
            .collect();
        report("matmul (gemv)", &gpu, &cpu);
    }

    // --- matmul with f16 weights ---
    {
        let mut g = Graph::new();
        let a = g.parameter("t.a16", &[1, h]);
        let b = g.parameter_f16("t.b16", &[h, kv]);
        let out = g.matmul(a, b);
        g.set_outputs(vec![out]);
        let gpu = run(&g, kv, pos);
        let (av, bv) = (synth("t.a16", h), synth("t.b16", h * kv));
        let cpu: Vec<f32> = (0..kv)
            .map(|j| {
                (0..h)
                    .map(|i| {
                        // f16 storage rounds the weight; model that.
                        av[i] * half::f16::from_f32(bv[i * kv + j]).to_f32()
                    })
                    .sum()
            })
            .collect();
        report("matmul f16 weights", &gpu, &cpu);
    }

    // --- embedding with f16 table ---
    {
        let vocab = 512usize;
        let mut g = Graph::new();
        let ids = g.input_u32("token_ids", &[1]);
        let tbl = g.parameter_f16("t.emb16", &[vocab, h]);
        let out = g.embedding(ids, tbl);
        g.set_outputs(vec![out]);
        let gpu = run_tok(&g, h, pos, 42);
        let tv = synth("t.emb16", vocab * h);
        let cpu: Vec<f32> = (0..h)
            .map(|i| half::f16::from_f32(tv[42 * h + i]).to_f32())
            .collect();
        report("embedding f16", &gpu, &cpu);
    }

    // --- rope_dynamic_offset ---
    {
        let mut g = Graph::new();
        let q = g.parameter("t.q", &[1, h]);
        let kp = g.input_u32("kv_pos", &[1]);
        let out = g.rope_dynamic_offset(q, cfg.rope_theta, kp, cfg.head_dim());
        g.set_outputs(vec![out]);
        let gpu = run(&g, h, pos);
        let mut cpu = synth("t.q", h);
        let half = hd / 2;
        for i in 0..h / 2 {
            let (head, pih) = (i / half, i % half);
            let inv = cfg.rope_theta.powf(-2.0 * pih as f32 / hd as f32);
            let (s, c) = (pos as f32 * inv).sin_cos();
            let (i0, i1) = (head * hd + pih, head * hd + pih + half);
            let (a, b) = (cpu[i0], cpu[i1]);
            cpu[i0] = a * c - b * s;
            cpu[i1] = a * s + b * c;
        }
        report("rope_dynamic_offset", &gpu, &cpu);
    }

    // --- swiglu ---
    {
        let n = 1536usize;
        let mut g = Graph::new();
        let gate = g.parameter("t.gate", &[1, n]);
        let up = g.parameter("t.up", &[1, n]);
        let out = g.swiglu(gate, up);
        g.set_outputs(vec![out]);
        let gpu = run(&g, n, pos);
        let (gv, uv) = (synth("t.gate", n), synth("t.up", n));
        let cpu: Vec<f32> = gv
            .iter()
            .zip(&uv)
            .map(|(a, b)| (a / (1.0 + (-a).exp())) * b)
            .collect();
        report("swiglu", &gpu, &cpu);
    }

    // --- cached_attention, over a cache written by cache_write ---
    {
        let max_seq = 512usize;
        let mut g = Graph::new();
        let q = g.parameter("t.q2", &[1, h]);
        let knew = g.parameter("t.knew", &[1, kv]);
        let vnew = g.parameter("t.vnew", &[1, kv]);
        let kc = g.parameter("t.kc", &[max_seq, kv]);
        let vc = g.parameter("t.vc", &[max_seq, kv]);
        let kp = g.input_u32("kv_pos", &[1]);
        let kc2 = g.cache_write(knew, kc, kp);
        let vc2 = g.cache_write(vnew, vc, kp);
        let out = g.cached_attention(
            q,
            kc2,
            vc2,
            kp,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim(),
        );
        g.set_outputs(vec![out]);
        let gpu = run(&g, h, pos);

        let qv = synth("t.q2", h);
        let mut kcv = synth("t.kc", max_seq * kv);
        let mut vcv = synth("t.vc", max_seq * kv);
        kcv[pos as usize * kv..(pos as usize + 1) * kv].copy_from_slice(&synth("t.knew", kv));
        vcv[pos as usize * kv..(pos as usize + 1) * kv].copy_from_slice(&synth("t.vnew", kv));
        let n = pos as usize + 1;
        let mut cpu = vec![0.0f32; h];
        let groups = cfg.num_attention_heads as usize / cfg.num_key_value_heads as usize;
        for head in 0..cfg.num_attention_heads as usize {
            let kvh = head / groups;
            let scale = 1.0 / (hd as f32).sqrt();
            let sc: Vec<f32> = (0..n)
                .map(|t| {
                    (0..hd)
                        .map(|d| qv[head * hd + d] * kcv[t * kv + kvh * hd + d])
                        .sum::<f32>()
                        * scale
                })
                .collect();
            let m = sc.iter().cloned().fold(f32::MIN, f32::max);
            let e: Vec<f32> = sc.iter().map(|s| (s - m).exp()).collect();
            let den: f32 = e.iter().sum();
            for d in 0..hd {
                cpu[head * hd + d] = (0..n)
                    .map(|t| e[t] * vcv[t * kv + kvh * hd + d])
                    .sum::<f32>()
                    / den;
            }
        }
        report("cached_attention", &gpu, &cpu);
    }
}
