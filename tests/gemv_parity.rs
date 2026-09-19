//! Parity tests for the M=1 GEMV matmul path.
//!
//! The compiler routes Op::MatMul with M=1 through `ShaderEntry::MatMulGemv`
//! (a dedicated GEMV kernel, `shaders/matmul_gemv.wgsl`). This test verifies
//! it produces numerically identical output to the tiled matmul for
//! representative decode-sized shapes.

use meganeura::{Graph, compile};

/// Reference CPU matmul for [1,K] × [K,N] → [1,N].
fn cpu_gemv(a: &[f32], b: &[f32], k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; n];
    for col in 0..n {
        let mut acc = 0.0_f32;
        for kk in 0..k {
            acc += a[kk] * b[kk * n + col];
        }
        out[col] = acc;
    }
    out
}

fn run_gpu_gemv(a_data: &[f32], b_data: &[f32], k: usize, n: usize) -> Vec<f32> {
    let mut g = Graph::new();
    let a = g.input("a", &[1, k]);
    let b = g.parameter("b", &[k, n]);
    let c = g.matmul(a, b);
    g.set_outputs(vec![c]);

    let mut session = meganeura::build(&g, meganeura::SessionConfig::inference_from_env()).0;

    // Sanity: when N % 4 == 0 the plan should route through GEMV;
    // otherwise it falls back to the tile matmul.
    let plan = session.plan();
    let gemv = plan.dispatches.iter().find(|d| {
        matches!(
            d.shader,
            compile::ShaderEntry::MatMulGemv | compile::ShaderEntry::MatMulGemvBT
        )
    });
    if n.is_multiple_of(4) {
        let d = gemv.expect("expected a GEMV dispatch for n%4==0");
        assert_eq!(d.workgroups, [n as u32 / 4, 1, 1]);
        assert!(
            !d.gemv_physical_bt && matches!(d.shader, compile::ShaderEntry::MatMulGemv),
            "physical-BT is not selected; expected K-split GEMV, got {:?}",
            (&d.shader, d.gemv_physical_bt)
        );
    } else {
        assert!(
            gemv.is_none(),
            "expected tile-matmul fallback for n%4!=0, got {:?}",
            gemv.map(|d| (&d.shader, d.workgroups, d.gemv_physical_bt))
        );
    }

    session.set_input("a", a_data);
    session.set_parameter("b", b_data);
    session.step();
    session.wait();
    session.read_output(n)
}

fn assert_close(a: &[f32], b: &[f32], rel_tol: f32, abs_tol: f32) {
    assert_close_named("GEMV", a, b, rel_tol, abs_tol);
}

fn assert_close_named(label: &str, a: &[f32], b: &[f32], rel_tol: f32, abs_tol: f32) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let scale = x.abs().max(y.abs()).max(1e-6);
        let rel = (x - y).abs() / scale;
        let abs = (x - y).abs();
        assert!(
            abs <= abs_tol || rel <= rel_tol,
            "{label} mismatch at [{i}]: gpu={x}, cpu={y}, rel={rel:.3e}, abs={abs:.3e}",
        );
    }
}

fn test_shape(k: usize, n: usize, seed: u32) {
    // Deterministic non-uniform data so column structure exercises the kernel.
    let a: Vec<f32> = (0..k)
        .map(|i| ((i as u32 ^ seed) as f32 * 0.003).sin())
        .collect();
    let b: Vec<f32> = (0..k * n)
        .map(|i| ((i as u32 ^ seed.wrapping_mul(31)) as f32 * 0.0007).cos())
        .collect();

    let gpu = run_gpu_gemv(&a, &b, k, n);
    let cpu = cpu_gemv(&a, &b, k, n);
    // fp32 matmul accumulates K products; rel tol scales with sqrt(K) roughly.
    // Allow 1e-4 rel / 1e-5 abs which covers K up to a few thousand.
    assert_close(&gpu, &cpu, 1e-4, 1e-5);
}

#[test]
#[ignore = "GPU timestamp microbench; run with --ignored"]
fn bench_smollm2_gemv_vs_bt() {
    use meganeura::profiler::{CaptureOptions, capture_session_profile};
    use meganeura::{CompileOptions, GemvReduction, GemvShape};
    unsafe { std::env::set_var("MEGANEURA_GPU_TIMING", "1") };
    fn gpu_us(bt: bool, k: usize, n: usize, shape: GemvShape) -> f64 {
        let mut g = Graph::new();
        let a = g.input("a", &[1, k]);
        let c = if bt {
            let b = g.parameter("b", &[n, k]);
            g.matmul_bt(a, b)
        } else {
            let b = g.parameter("b", &[k, n]);
            g.matmul(a, b)
        };
        g.set_outputs(vec![c]);
        let mut cfg = meganeura::SessionConfig::inference_from_env();
        cfg.options = CompileOptions {
            gemv_shape: Some(shape),
            ..CompileOptions::default()
        };
        let mut session = meganeura::build(&g, cfg).0;
        let a_data = vec![0.01_f32; k];
        let b_data = vec![0.001_f32; k * n];
        session.set_input("a", &a_data);
        session.set_parameter("b", &b_data);
        let profile = capture_session_profile(
            &mut session,
            |s| {
                s.set_input("a", &a_data);
            },
            CaptureOptions {
                samples: 8,
                ..CaptureOptions::default()
            },
        )
        .expect("profile");
        profile.dispatches[0].median_ms * 1e3
    }
    let shapes = [32, 64, 256].map(|threads| GemvShape {
        threads,
        reduction: GemvReduction::Tree,
    });
    for (k, n, label) in [(576, 3072, "ffn-up packed"), (1536, 576, "ffn-down")] {
        for shape in shapes {
            let gemv = gpu_us(false, k, n, shape);
            let bt = gpu_us(true, k, n, shape);
            eprintln!("{label} {shape:?}: GEMV {gemv:.1} µs | GEMV-BT {bt:.1} µs");
        }
    }
}

#[test]
fn gemv_square_small() {
    test_shape(64, 64, 1);
}

#[test]
fn gemv_smollm2_qproj() {
    // Real SmolLM2-135M decode Q-projection shape.
    test_shape(576, 576, 2);
}

#[test]
fn gemv_smollm2_mlp_up() {
    test_shape(576, 1536, 3);
}

#[test]
fn gemv_smollm2_mlp_down() {
    test_shape(1536, 576, 4);
}

#[test]
fn gemv_smolvla_shapes() {
    test_shape(720, 720, 5);
    test_shape(720, 2048, 6);
}

#[test]
fn gemv_non_multiple_of_256() {
    // N not a multiple of the workgroup size — exercises the
    // `col < n` bounds check at the tail.
    test_shape(128, 300, 7);
    test_shape(128, 257, 8);
    test_shape(128, 1, 9);
}

// ---- FusedMatMulAdd (GEMV + residual) ----

/// CPU reference: 1×K × K×N + D[1,N].
fn cpu_gemv_add(a: &[f32], b: &[f32], d: &[f32], k: usize, n: usize) -> Vec<f32> {
    let mut out = cpu_gemv(a, b, k, n);
    for (o, r) in out.iter_mut().zip(d.iter()) {
        *o += *r;
    }
    out
}

fn test_gemv_add_shape(k: usize, n: usize, seed: u32) {
    let a: Vec<f32> = (0..k)
        .map(|i| ((i as u32 ^ seed) as f32 * 0.003).sin())
        .collect();
    let b: Vec<f32> = (0..k * n)
        .map(|i| ((i as u32 ^ seed.wrapping_mul(31)) as f32 * 0.0007).cos())
        .collect();
    let d: Vec<f32> = (0..n)
        .map(|i| ((i as u32 ^ seed.wrapping_mul(7)) as f32 * 0.01).sin() * 0.1)
        .collect();

    let mut g = Graph::new();
    let a_n = g.input("a", &[1, k]);
    let b_n = g.parameter("b", &[k, n]);
    let d_n = g.parameter("d", &[1, n]);
    let mm = g.matmul(a_n, b_n);
    let out = g.add(mm, d_n);
    g.set_outputs(vec![out]);

    let mut session = meganeura::build(&g, meganeura::SessionConfig::inference_from_env()).0;
    // Sanity: the optimizer should fuse MatMul+Add to FusedMatMulAdd, which
    // at M=1 with N%4==0 routes through MatMulGemvAdd.
    let plan = session.plan();
    let gemv_add = plan.dispatches.iter().find(|disp| {
        matches!(
            disp.shader,
            compile::ShaderEntry::MatMulGemvAdd | compile::ShaderEntry::MatMulGemvBTAdd
        )
    });
    let disp = gemv_add.expect("expected a fused GEMV-add dispatch");
    assert_eq!(disp.workgroups, [n as u32 / 4, 1, 1]);
    assert!(
        !disp.gemv_physical_bt && matches!(disp.shader, compile::ShaderEntry::MatMulGemvAdd),
        "physical-BT is not selected; expected K-split GEMV-add, got {:?}",
        (&disp.shader, disp.gemv_physical_bt)
    );

    session.set_input("a", &a);
    session.set_parameter("b", &b);
    session.set_parameter("d", &d);
    session.step();
    session.wait();
    let gpu = session.read_output(n);
    let cpu = cpu_gemv_add(&a, &b, &d, k, n);
    assert_close(&gpu, &cpu, 1e-4, 1e-5);
}

#[test]
fn gemv_ntile_rmsnorm_smollm2_ffn_up() {
    // Packed FFN-up: RmsNorm folded into N-tile GEMV, N=3072, K=576.
    const K: usize = 576;
    const N: usize = 3072;
    const EPS: f32 = 1e-5;
    let a: Vec<f32> = (0..K)
        .map(|i| ((i as u32 ^ 9) as f32 * 0.003).sin())
        .collect();
    let w: Vec<f32> = (0..K)
        .map(|i| 0.9 + ((i as u32 ^ 13) as f32 * 0.001).sin() * 0.1)
        .collect();
    let b: Vec<f32> = (0..K * N)
        .map(|i| ((i as u32 ^ 31) as f32 * 0.0007).cos())
        .collect();

    let mut g = Graph::new();
    let x = g.input("x", &[1, K]);
    let nw = g.parameter("norm", &[K]);
    let bw = g.parameter("b", &[K, N]);
    let h = g.rms_norm(x, nw, EPS);
    let c = g.matmul(h, bw);
    g.set_outputs(vec![c]);

    let mut session = meganeura::build(&g, meganeura::SessionConfig::inference_from_env()).0;
    let plan = session.plan();
    let fused = plan
        .dispatches
        .iter()
        .find(|d| d.gemv_rmsnorm.is_some() && matches!(d.shader, compile::ShaderEntry::MatMulGemv));
    let fused = fused.expect("RmsNorm must fold into the FFN-up GEMV");
    assert!(!fused.gemv_physical_bt);
    assert_eq!(fused.workgroups, [N as u32 / 4, 1, 1]);

    session.set_input("x", &a);
    session.set_parameter("norm", &w);
    session.set_parameter("b", &b);
    session.step();
    session.wait();
    let gpu = session.read_output(N);

    let mean_sq: f32 = a.iter().map(|v| v * v).sum::<f32>() / K as f32;
    let inv_rms = (mean_sq + EPS).sqrt().recip();
    let a_hat: Vec<f32> = a.iter().zip(&w).map(|(x, wi)| x * inv_rms * wi).collect();
    let cpu = cpu_gemv(&a_hat, &b, K, N);
    assert_close_named("ntile-rmsnorm", &gpu, &cpu, 1e-4, 1e-5);
}

#[test]
fn gemv_add_smollm2_mlp_down() {
    // MLP down + residual at decode: 1×1536 × 1536×576 + residual
    test_gemv_add_shape(1536, 576, 100);
}

#[test]
fn gemv_add_smollm2_o_proj() {
    // Attention out-proj + residual: 1×576 × 576×576 + residual
    test_gemv_add_shape(576, 576, 101);
}

#[test]
fn gemv_add_smolvla_mlp_down() {
    test_gemv_add_shape(2048, 720, 102);
}

fn cpu_repeat_kv(v: &[f32], n_q: usize, n_kv: usize, head_dim: usize) -> Vec<f32> {
    let group = n_q / n_kv;
    let mut out = vec![0.0_f32; n_q * head_dim];
    for q_head in 0..n_q {
        let kv_head = q_head / group;
        let src = kv_head * head_dim;
        let dst = q_head * head_dim;
        out[dst..dst + head_dim].copy_from_slice(&v[src..src + head_dim]);
    }
    out
}

#[test]
fn seq1_gqa_elide_folds_repeat_into_o_proj() {
    // Stateless seq=1 GQA: softmax of one score is 1, so attention is
    // repeat(V). The shipped session path must fold that into o_proj GEMV-add.
    const HIDDEN: usize = 16;
    const N_Q: usize = 4;
    const N_KV: usize = 2;
    const HD: usize = 4;
    const Q_DIM: usize = N_Q * HD;
    const KV_DIM: usize = N_KV * HD;
    let seed = 77u32;
    let x: Vec<f32> = (0..HIDDEN)
        .map(|i| ((i as u32 ^ seed) as f32 * 0.003).sin())
        .collect();
    let wq: Vec<f32> = (0..HIDDEN * Q_DIM)
        .map(|i| ((i as u32 ^ seed.wrapping_mul(3)) as f32 * 0.0007).cos())
        .collect();
    let wk: Vec<f32> = (0..HIDDEN * KV_DIM)
        .map(|i| ((i as u32 ^ seed.wrapping_mul(5)) as f32 * 0.0007).cos())
        .collect();
    let wv: Vec<f32> = (0..HIDDEN * KV_DIM)
        .map(|i| ((i as u32 ^ seed.wrapping_mul(7)) as f32 * 0.0007).cos())
        .collect();
    let wo: Vec<f32> = (0..Q_DIM * HIDDEN)
        .map(|i| ((i as u32 ^ seed.wrapping_mul(11)) as f32 * 0.0007).cos())
        .collect();
    let residual: Vec<f32> = (0..HIDDEN)
        .map(|i| ((i as u32 ^ seed.wrapping_mul(13)) as f32 * 0.01).sin() * 0.1)
        .collect();

    let mut g = Graph::new();
    let x_n = g.input("x", &[1, HIDDEN]);
    let wq_n = g.parameter("wq", &[HIDDEN, Q_DIM]);
    let wk_n = g.parameter("wk", &[HIDDEN, KV_DIM]);
    let wv_n = g.parameter("wv", &[HIDDEN, KV_DIM]);
    let wo_n = g.parameter("wo", &[Q_DIM, HIDDEN]);
    let res_n = g.parameter("residual", &[1, HIDDEN]);
    let q = g.matmul(x_n, wq_n);
    let k = g.matmul(x_n, wk_n);
    let v = g.matmul(x_n, wv_n);
    let attn = g.causal_attention(q, k, v, N_Q as u32, N_KV as u32, HD as u32);
    let proj = g.matmul(attn, wo_n);
    let out = g.add(proj, res_n);
    g.set_outputs(vec![out]);

    let mut session = meganeura::build(&g, meganeura::SessionConfig::inference_from_env()).0;
    let plan = session.plan();
    assert!(
        !plan.dispatches.iter().any(|d| {
            matches!(
                d.shader,
                compile::ShaderEntry::MultiHeadAttn
                    | compile::ShaderEntry::FlashAttention
                    | compile::ShaderEntry::FlashAttentionCoop
                    | compile::ShaderEntry::RepeatKv
            )
        }),
        "seq=1 GQA must drop attention and fold repeat(V) into o_proj, got {:?}",
        plan.dispatches
            .iter()
            .map(|d| (&d.label, d.shader.clone(), d.gemv_repeat_kv, d.workgroups))
            .collect::<Vec<_>>()
    );
    let o_proj = plan
        .dispatches
        .iter()
        .find(|d| {
            d.gemv_repeat_kv == Some((N_KV as u32, HD as u32))
                && matches!(d.shader, compile::ShaderEntry::MatMulGemvAdd)
        })
        .expect("o_proj GEMV-add must carry gemv_repeat_kv");
    assert_eq!(o_proj.workgroups, [HIDDEN as u32 / 4, 1, 1]);

    session.set_input("x", &x);
    session.set_parameter("wq", &wq);
    session.set_parameter("wk", &wk);
    session.set_parameter("wv", &wv);
    session.set_parameter("wo", &wo);
    session.set_parameter("residual", &residual);
    session.step();
    session.wait();
    let gpu = session.read_output(HIDDEN);

    let v_row = cpu_gemv(&x, &wv, HIDDEN, KV_DIM);
    let attn_row = cpu_repeat_kv(&v_row, N_Q, N_KV, HD);
    let cpu = cpu_gemv_add(&attn_row, &wo, &residual, Q_DIM, HIDDEN);
    assert_close_named("seq1-gqa-repeat-kv", &gpu, &cpu, 1e-4, 1e-5);
}

// ---- MatMulBT GEMV (B stored [N, K]) ----

/// CPU reference for MatMulBT: accumulate in f64 so the reference isn't
/// itself lossy for K ≥ 1024. Narrow to f32 only for the final result.
fn cpu_gemv_bt(a: &[f32], b: &[f32], k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; n];
    for col in 0..n {
        let mut acc = 0.0_f64;
        for kk in 0..k {
            acc += (a[kk] as f64) * (b[col * k + kk] as f64);
        }
        out[col] = acc as f32;
    }
    out
}

fn test_gemv_bt_shape(k: usize, n: usize, seed: u32) {
    let a: Vec<f32> = (0..k)
        .map(|i| ((i as u32 ^ seed) as f32 * 0.003).sin())
        .collect();
    // Note: B is [N, K] layout now (row-major N rows of K columns).
    let b: Vec<f32> = (0..n * k)
        .map(|i| ((i as u32 ^ seed.wrapping_mul(31)) as f32 * 0.0007).cos())
        .collect();

    let mut g = Graph::new();
    let a_n = g.input("a", &[1, k]);
    let b_n = g.parameter("b", &[n, k]);
    let c = g.matmul_bt(a_n, b_n);
    g.set_outputs(vec![c]);

    let mut session = meganeura::build(&g, meganeura::SessionConfig::inference_from_env()).0;

    let plan = session.plan();
    let gemv_bt_count = plan
        .dispatches
        .iter()
        .filter(|d| matches!(d.shader, compile::ShaderEntry::MatMulGemvBT))
        .count();
    if k.is_multiple_of(4) {
        assert_eq!(
            gemv_bt_count, 1,
            "expected one MatMulGemvBT dispatch for k%4==0, got {}",
            gemv_bt_count,
        );
    } else {
        assert_eq!(
            gemv_bt_count, 0,
            "expected tile-MatMulBT fallback for k%4!=0, got {} GemvBT",
            gemv_bt_count,
        );
    }

    session.set_input("a", &a);
    session.set_parameter("b", &b);
    session.step();
    session.wait();
    let gpu = session.read_output(n);
    let cpu = cpu_gemv_bt(&a, &b, k, n);
    // GPU uses fp32 tree-reduce with vec4 FMAs; CPU ref is f64. The
    // remaining error is the GPU's fp32 accumulation over K products —
    // bounded by eps_fp32 × sqrt(K) × max(|A·B|). Tolerance scales with
    // sqrt(K).
    let rel_tol = 1e-4_f32.max(5e-6 * (k as f32).sqrt());
    let abs_tol = 1e-5_f32.max(5e-6 * (k as f32).sqrt());
    assert_close(&gpu, &cpu, rel_tol, abs_tol);
}

#[test]
fn gemv_bt_smollm2_lm_head() {
    // SmolLM2-135M LM head (weight-tied): 1×576 × 49152×576^T → 1×49152.
    test_gemv_bt_shape(576, 49152, 200);
}

#[test]
fn gemv_bt_square() {
    test_gemv_bt_shape(576, 576, 201);
}

#[test]
fn gemv_bt_wide_k() {
    test_gemv_bt_shape(2048, 720, 202);
}

#[test]
fn gemv_bt_non_multiple_k() {
    // K not div by 4 → fallback to tile MatMulBT.
    test_gemv_bt_shape(577, 128, 203);
}

#[test]
fn gemv_non_multiple_k() {
    // K not a multiple of 256 — exercises the shared-memory chunk tail.
    test_shape(100, 256, 10);
    test_shape(511, 256, 11);
    test_shape(513, 256, 12);
}

/// Every GEMV kernel must be correct at every shape the search may install.
/// Plain, fused-add and transposed-B run together so eight sessions cover the
/// 24 combinations while failures retain the shape and kernel name.
#[test]
fn every_gemv_shape_computes_the_same_product() {
    use meganeura::compile::{CompileOptions, ShaderEntry};
    use meganeura::train::{Mode, SessionConfig};
    use meganeura::{GemvReduction, GemvShape};

    let data = |n: usize, seed: u32| -> Vec<f32> {
        let mut state = seed | 1;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(747796405).wrapping_add(2891336453);
                let w = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277803737);
                (((w >> 22) ^ w) % 2001) as f32 * 0.001 - 1.0
            })
            .collect()
    };

    // K and N large enough that a 256-wide workgroup has real work to split,
    // and not multiples of every width, so the loop tails are exercised.
    const K: usize = 320;
    const N: usize = 36;
    let a = data(K, 1);
    let b = data(K * N, 2);
    let addend = data(N, 3);
    // B for the transposed form is [N, K]; transpose the same numbers so all
    // three kernels are compared against the same product.
    let mut b_t = vec![0.0f32; K * N];
    for col in 0..N {
        for row in 0..K {
            b_t[col * K + row] = b[row * N + col];
        }
    }

    let want = cpu_gemv(&a, &b, K, N);
    let want_add = cpu_gemv_add(&a, &b, &addend, K, N);
    let want_bt = cpu_gemv_bt(&a, &b_t, K, N);

    for threads in GemvShape::WIDTHS {
        for reduction in [GemvReduction::Tree, GemvReduction::Subgroup] {
            let shape = GemvShape { threads, reduction };
            let mut g = Graph::new();
            let x = g.input("x", &[1, K]);
            let x_add = g.input("x_add", &[1, K]);
            let x_bt = g.input("x_bt", &[1, K]);
            let w = g.input("w", &[K, N]);
            let w_add = g.input("w_add", &[K, N]);
            let w_t = g.input("w_t", &[N, K]);
            let d = g.input("d", &[1, N]);
            let plain = g.matmul(x, w);
            let product = g.matmul(x_add, w_add);
            let add = g.add(product, d);
            let bt = g.matmul_bt(x_bt, w_t);
            g.set_outputs(vec![plain, add, bt]);

            let config = SessionConfig {
                mode: Mode::Inference,
                options: CompileOptions {
                    gemv_shape: Some(shape),
                    ..CompileOptions::from_env()
                },
                ..SessionConfig::from_env()
            };
            let mut s = meganeura::build(&g, config).0;
            let shaders: Vec<_> = s.plan().dispatches.iter().map(|d| &d.shader).collect();
            for expected in [
                ShaderEntry::MatMulGemv,
                ShaderEntry::MatMulGemvAdd,
                ShaderEntry::MatMulGemvBT,
            ] {
                assert_eq!(
                    shaders
                        .iter()
                        .filter(|&&shader| *shader == expected)
                        .count(),
                    1,
                    "{shape:?}: missing {expected:?}; got {shaders:?}"
                );
            }
            for name in ["x", "x_add", "x_bt"] {
                s.set_input(name, &a);
            }
            s.set_input("w", &b);
            s.set_input("w_add", &b);
            s.set_input("w_t", &b_t);
            s.set_input("d", &addend);
            s.step();
            s.wait();
            for (index, (label, expected)) in
                [("plain", &want), ("add", &want_add), ("bt", &want_bt)]
                    .into_iter()
                    .enumerate()
            {
                let mut got = vec![0.0; N];
                s.read_output_by_index(index, &mut got);
                assert_close_named(&format!("{shape:?} {label}"), &got, expected, 2e-4, 2e-4);
            }
        }
    }
}
