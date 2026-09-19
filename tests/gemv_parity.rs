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
    let gemv_count = plan
        .dispatches
        .iter()
        .filter(|d| matches!(d.shader, compile::ShaderEntry::MatMulGemv))
        .count();
    if n.is_multiple_of(4) {
        assert_eq!(
            gemv_count, 1,
            "expected one MatMulGemv dispatch for n%4==0, found {}",
            gemv_count
        );
    } else {
        assert_eq!(
            gemv_count, 0,
            "expected tile-matmul fallback for n%4!=0, got {} GEMV",
            gemv_count
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
fn q40_rmsnorm_folds_into_gemv() {
    const K: usize = 256;
    const N: usize = 64;
    const BLOCK: usize = 32;
    let x: Vec<f32> = (0..K).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect();
    let nw: Vec<f32> = (0..K).map(|i| 1.0 + ((i % 5) as f32 - 2.0) * 0.1).collect();
    let mut packed = Vec::new();
    let mut reference = vec![0.0f32; K * N];
    for col in 0..N {
        for blk in 0..K / BLOCK {
            let d = 0.02 + (blk as f32) * 0.001;
            let d16 = half::f16::from_f32(d).to_f32();
            packed.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
            let mut nibbles = [0u8; 16];
            for j in 0..16 {
                let q0 = ((blk + j) % 16) as u8;
                let q1 = ((blk + j + 3) % 16) as u8;
                nibbles[j] = q0 | (q1 << 4);
                reference[(blk * BLOCK + j) * N + col] = (f32::from(q0) - 8.0) * d16;
                reference[(blk * BLOCK + j + 16) * N + col] = (f32::from(q1) - 8.0) * d16;
            }
            packed.extend_from_slice(&nibbles);
        }
    }
    packed.resize(packed.len().next_multiple_of(4), 0);

    let mut g = Graph::new();
    let xin = g.input("x", &[1, K]);
    let norm = g.parameter("nw", &[K]);
    let h = g.rms_norm(xin, norm, 1e-5);
    let w = g.parameter_q40("w", &[K, N]);
    let y = g.matmul(h, w);
    g.set_outputs(vec![y]);
    let mut session = meganeura::build(&g, meganeura::SessionConfig::inference_from_env()).0;
    let fused = session
        .plan()
        .dispatches
        .iter()
        .filter(|d| d.gemv_rmsnorm.is_some())
        .count();
    assert_eq!(fused, 1, "Q40 GEMV should fold the RmsNorm");
    assert!(
        session
            .plan()
            .dispatches
            .iter()
            .all(|d| d.shader != compile::ShaderEntry::RmsNorm)
    );
    session.set_input("x", &x);
    session.set_parameter("nw", &nw);
    session.set_parameter_packed("w", &packed);
    session.step();
    session.wait();
    let gpu = session.read_output(N);

    let ms = x.iter().map(|v| v * v).sum::<f32>() / K as f32;
    let inv = (ms + 1e-5).sqrt().recip();
    let mut want = vec![0.0f32; N];
    for col in 0..N {
        for i in 0..K {
            want[col] += x[i] * inv * nw[i] * reference[i * N + col];
        }
    }
    assert_close_named("Q40 RmsNorm GEMV", &gpu, &want, 2e-2, 2e-2);
}

#[test]
fn gemv_non_multiple_of_256() {
    // N not a multiple of the workgroup size — exercises the
    // `col < n` bounds check at the tail.
    test_shape(128, 300, 7);
    test_shape(128, 257, 8);
    test_shape(128, 1, 9);
}

#[test]
fn dense_transposed_glu_matches_reference_after_restage() {
    for (m, k, n) in [(1, 12, 7), (3, 5, 7)] {
        for f16 in [false, true] {
            for gelu in [false, true] {
                let mut g = Graph::new();
                let x = g.input("x", &[m, k]);
                let projected = if m == 1 {
                    let nw = g.parameter("norm", &[k]);
                    g.rms_norm(x, nw, 1e-5)
                } else {
                    x
                };
                let mut project = |name| {
                    let w = if f16 {
                        g.parameter_f16(name, &[n, k])
                    } else {
                        g.parameter(name, &[n, k])
                    };
                    g.matmul_bt(projected, w)
                };
                let gate = project("gate");
                let up = project("up");
                let out = if gelu {
                    g.geglu(gate, up)
                } else {
                    g.swiglu(gate, up)
                };
                g.set_outputs(vec![out]);
                let mut config = meganeura::SessionConfig::inference_from_env();
                config.tune = false;
                let mut session = meganeura::build(&g, config).0;
                assert_eq!(session.plan().derived_params.len(), 1);
                assert!(matches!(
                    session.plan().derived_params[0].2,
                    meganeura::graph::ParamTransform::VerticalConcat
                ));
                let x: Vec<_> = (0..m * k).map(|i| (i as f32 * 0.7).sin()).collect();
                let mut gate: Vec<_> = (0..n * k).map(|i| (i as f32 * 0.3).cos() * 0.25).collect();
                let mut up: Vec<_> = (0..n * k).map(|i| (i as f32 * 0.2).sin() * 0.25).collect();
                session.set_input("x", &x);
                let nw: Vec<_> = (0..k).map(|i| 0.75 + (i % 5) as f32 * 0.125).collect();
                if m == 1 {
                    session.set_parameter("norm", &nw);
                    assert_eq!(
                        session
                            .plan()
                            .dispatches
                            .iter()
                            .filter(|d| d.gemv_rmsnorm.is_some())
                            .count(),
                        1
                    );
                }
                let inv_rms = if m == 1 {
                    (x.iter().map(|v| v * v).sum::<f32>() / k as f32 + 1e-5)
                        .sqrt()
                        .recip()
                } else {
                    1.0
                };
                for pass in 0..3 {
                    if pass == 1 {
                        gate.iter_mut().for_each(|v| *v *= -0.5);
                    }
                    if pass == 2 {
                        up.iter_mut().for_each(|v| *v *= 0.75);
                    }
                    // Reverse source order; mix f32 and exact f16 uploads.
                    if f16 && pass == 2 {
                        let bytes: Vec<_> = up
                            .iter()
                            .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
                            .collect();
                        session.set_parameter_packed("up", &bytes);
                    } else {
                        session.set_parameter("up", &up);
                    }
                    session.set_parameter("gate", &gate);
                    session.step();
                    session.wait();
                    let rounded = |v| {
                        if f16 {
                            half::f16::from_f32(v).to_f32()
                        } else {
                            v
                        }
                    };
                    let mut expected = Vec::new();
                    for row in 0..m {
                        for col in 0..n {
                            let a: f32 = (0..k)
                                .map(|i| {
                                    x[row * k + i]
                                        * inv_rms
                                        * if m == 1 { nw[i] } else { 1.0 }
                                        * rounded(gate[col * k + i])
                                })
                                .sum();
                            let b: f32 = (0..k)
                                .map(|i| {
                                    x[row * k + i]
                                        * inv_rms
                                        * if m == 1 { nw[i] } else { 1.0 }
                                        * rounded(up[col * k + i])
                                })
                                .sum();
                            let activated = if gelu {
                                0.5 * a * (1.0 + (0.7978846 * (a + 0.044715 * a * a * a)).tanh())
                            } else {
                                a / (1.0 + (-a).exp())
                            };
                            expected.push(activated * b);
                        }
                    }
                    assert_close_named(
                        "transposed GLU",
                        &session.read_output(m * n),
                        &expected,
                        1e-4,
                        1e-5,
                    );
                    if m == 1 && !gelu && pass == 0 {
                        let before = session.read_output(m * n);
                        let report = session
                            .tune_with(meganeura::tune::TuneOptions {
                                scope: meganeura::tune::TuneScope::Dense,
                                max_time: std::time::Duration::from_secs(10),
                                sample_pairs: 4,
                                dispatches_per_sample: 1,
                                ..Default::default()
                            })
                            .unwrap();
                        assert_eq!(report.eligible_classes, 1);
                        assert_eq!(report.outcomes.len(), 7);
                        assert!(
                            report
                                .outcomes
                                .iter()
                                .all(|o| o.qualified && o.class.gemv_rmsnorm)
                        );
                        assert_eq!(session.read_output(m * n), before);
                        session.step();
                        session.wait();
                        assert_close_named(
                            "tuned transposed GLU",
                            &session.read_output(m * n),
                            &expected,
                            1e-4,
                            1e-5,
                        );
                    }
                }
            }
        }
    }
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
    let gemv_add_count = plan
        .dispatches
        .iter()
        .filter(|disp| matches!(disp.shader, compile::ShaderEntry::MatMulGemvAdd))
        .count();
    assert_eq!(
        gemv_add_count,
        1,
        "expected one MatMulGemvAdd dispatch, got {}; plan:\n{:?}",
        gemv_add_count,
        plan.dispatches
            .iter()
            .map(|d| format!("{:?}", d.shader))
            .collect::<Vec<_>>(),
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
        assert!(
            plan.dispatches
                .iter()
                .all(|d| d.workgroups.iter().all(|&n| n <= 65_535))
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
    test_gemv_bt_shape(4, 131_071, 204);
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
/// Plain and transposed-B products, with and without addends, share each session.
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
    let want_bt_add: Vec<_> = want_bt.iter().zip(&addend).map(|(a, b)| a + b).collect();

    for threads in GemvShape::WIDTHS {
        for reduction in [GemvReduction::Tree, GemvReduction::Subgroup] {
            let shape = GemvShape { threads, reduction };
            let mut g = Graph::new();
            let x = g.input("x", &[1, K]);
            let x_add = g.input("x_add", &[1, K]);
            let x_bt = g.input("x_bt", &[1, K]);
            let x_bt_add = g.input("x_bt_add", &[1, K]);
            let w = g.input("w", &[K, N]);
            let w_add = g.input("w_add", &[K, N]);
            let w_t = g.input("w_t", &[N, K]);
            let w_t_add = g.input("w_t_add", &[N, K]);
            let d = g.input("d", &[1, N]);
            let plain = g.matmul(x, w);
            let product = g.matmul(x_add, w_add);
            let add = g.add(product, d);
            let bt = g.matmul_bt(x_bt, w_t);
            let bt_product = g.matmul_bt(x_bt_add, w_t_add);
            let bt_add = g.add(bt_product, d);
            g.set_outputs(vec![plain, add, bt, bt_add]);

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
                ShaderEntry::MatMulGemvBTAdd,
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
            for name in ["x", "x_add", "x_bt", "x_bt_add"] {
                s.set_input(name, &a);
            }
            s.set_input("w", &b);
            s.set_input("w_add", &b);
            s.set_input("w_t", &b_t);
            s.set_input("w_t_add", &b_t);
            s.set_input("d", &addend);
            s.step();
            s.wait();
            for (index, (label, expected)) in [
                ("plain", &want),
                ("add", &want_add),
                ("bt", &want_bt),
                ("bt_add", &want_bt_add),
            ]
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
