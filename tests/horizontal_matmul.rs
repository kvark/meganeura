//! Horizontal fusion (D1): independent same-A matmuls in one barrier
//! group pack into a single dispatch. Compare GPU results to a CPU
//! reference and assert the pack actually formed.

use meganeura::{Graph, Mode, SessionConfig, build};

fn pcg_inputs(n: usize, seed: u32) -> Vec<f32> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(747796405).wrapping_add(2891336453);
            let w = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277803737);
            let r = (w >> 22) ^ w;
            (r as f32) * (2.0 / u32::MAX as f32) - 1.0
        })
        .collect()
}

fn cpu_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0_f32;
            for l in 0..k {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

fn assert_close(gpu: &[f32], cpu: &[f32], label: &str) {
    assert_eq!(gpu.len(), cpu.len());
    let max_cpu = cpu
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f32, f32::max)
        .max(1e-6);
    let max_abs = gpu
        .iter()
        .zip(cpu)
        .map(|(g, c)| (g - c).abs())
        .fold(0.0_f32, f32::max);
    let rel = max_abs / max_cpu;
    assert!(
        rel < 0.02,
        "{label}: rel err {:.3}% (max|gpu-cpu|={:.3e}, max|cpu|={:.3e})\n  gpu[..8]={:?}\n  cpu[..8]={:?}",
        rel * 100.0,
        max_abs,
        max_cpu,
        &gpu[..8.min(gpu.len())],
        &cpu[..8.min(cpu.len())],
    );
}

#[test]
fn same_a_qkv_pack_matches_cpu() {
    let m = 32usize;
    let k = 32usize;
    let n = 32usize;
    let mut g = Graph::new();
    let a = g.input("a", &[m, k]);
    let wq = g.input("wq", &[k, n]);
    let wk = g.input("wk", &[k, n]);
    let wv = g.input("wv", &[k, n]);
    let q = g.matmul(a, wq);
    let kk = g.matmul(a, wk);
    let v = g.matmul(a, wv);
    g.set_outputs(vec![q, kk, v]);

    let (mut session, _) = build(
        &g,
        SessionConfig {
            mode: Mode::Inference,
            ..SessionConfig::from_env()
        },
    );
    assert!(
        session
            .plan()
            .dispatches
            .iter()
            .any(|d| d.horizontal_batch == 3),
        "Q/K/V-style siblings should pack into one dispatch; got {:?}",
        session
            .plan()
            .dispatches
            .iter()
            .map(|d| (d.shader.clone(), d.horizontal_batch, d.workgroups))
            .collect::<Vec<_>>()
    );

    let a_data = pcg_inputs(m * k, 1);
    let wq_data = pcg_inputs(k * n, 2);
    let wk_data = pcg_inputs(k * n, 3);
    let wv_data = pcg_inputs(k * n, 4);
    session.set_input("a", &a_data);
    session.set_input("wq", &wq_data);
    session.set_input("wk", &wk_data);
    session.set_input("wv", &wv_data);
    session.step();
    session.wait();

    for (idx, weight) in [(0, &wq_data), (1, &wk_data), (2, &wv_data)] {
        let mut gpu = vec![0.0_f32; m * n];
        session.read_output_by_index(idx, &mut gpu);
        let cpu = cpu_matmul(&a_data, weight, m, k, n);
        assert_close(&gpu, &cpu, &format!("sibling {idx}"));
    }
}

#[test]
fn different_a_does_not_pack() {
    let mut g = Graph::new();
    let a = g.input("a", &[16, 16]);
    let b = g.input("b", &[16, 16]);
    let w0 = g.input("w0", &[16, 16]);
    let w1 = g.input("w1", &[16, 16]);
    let y0 = g.matmul(a, w0);
    let y1 = g.matmul(b, w1);
    g.set_outputs(vec![y0, y1]);

    let (session, _) = build(
        &g,
        SessionConfig {
            mode: Mode::Inference,
            ..SessionConfig::from_env()
        },
    );
    assert!(
        session
            .plan()
            .dispatches
            .iter()
            .all(|d| d.horizontal_batch < 2),
        "independent A operands must not pack"
    );
}

#[test]
fn seq1_kv_gemv_pack_matches_cpu() {
    let m = 1usize;
    let k = 64usize;
    let n = 32usize;
    let mut g = Graph::new();
    let a = g.input("a", &[m, k]);
    let wk = g.input("wk", &[k, n]);
    let wv = g.input("wv", &[k, n]);
    let kk = g.matmul(a, wk);
    let v = g.matmul(a, wv);
    g.set_outputs(vec![kk, v]);

    let (mut session, _) = build(
        &g,
        SessionConfig {
            mode: Mode::Inference,
            ..SessionConfig::from_env()
        },
    );
    assert!(
        session
            .plan()
            .dispatches
            .iter()
            .any(|d| d.horizontal_batch == 2),
        "seq=1 K/V GEMVs should pack; got {:?}",
        session
            .plan()
            .dispatches
            .iter()
            .map(|d| (d.shader.clone(), d.horizontal_batch, d.workgroups))
            .collect::<Vec<_>>()
    );

    let a_data = pcg_inputs(m * k, 11);
    let wk_data = pcg_inputs(k * n, 12);
    let wv_data = pcg_inputs(k * n, 13);
    session.set_input("a", &a_data);
    session.set_input("wk", &wk_data);
    session.set_input("wv", &wv_data);
    session.step();
    session.wait();

    for (idx, weight) in [(0, &wk_data), (1, &wv_data)] {
        let mut gpu = vec![0.0_f32; m * n];
        session.read_output_by_index(idx, &mut gpu);
        let cpu = cpu_matmul(&a_data, weight, m, k, n);
        assert_close(&gpu, &cpu, &format!("gemv sibling {idx}"));
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[test]
fn seq1_swiglu_folds_into_down_gemv() {
    let hidden = 32usize;
    let inter = 64usize;
    let mut g = Graph::new();
    let h = g.input("h", &[1, hidden]);
    let w_pack = g.input("w_pack", &[hidden, 2 * inter]);
    let w_down = g.input("w_down", &[inter, hidden]);
    let residual = g.input("res", &[1, hidden]);
    let wide = g.matmul(h, w_pack);
    let mid = g.swiglu_concat(wide);
    let down = g.matmul(mid, w_down);
    let y = g.add(down, residual);
    g.set_outputs(vec![y]);

    let (mut session, _) = build(
        &g,
        SessionConfig {
            mode: Mode::Inference,
            ..SessionConfig::from_env()
        },
    );
    let h_data = pcg_inputs(hidden, 21);
    let pack_data = pcg_inputs(hidden * 2 * inter, 22);
    let down_data = pcg_inputs(inter * hidden, 23);
    let res_data = pcg_inputs(hidden, 24);
    session.set_input("h", &h_data);
    session.set_input("w_pack", &pack_data);
    session.set_input("w_down", &down_data);
    session.set_input("res", &res_data);
    session.step();
    session.wait();
    let mut gpu = vec![0.0_f32; hidden];
    session.read_output_by_index(0, &mut gpu);

    let wide = cpu_matmul(&h_data, &pack_data, 1, hidden, 2 * inter);
    let mut mid = vec![0.0_f32; inter];
    for i in 0..inter {
        mid[i] = silu(wide[i]) * wide[inter + i];
    }
    let mut cpu = cpu_matmul(&mid, &down_data, 1, inter, hidden);
    for i in 0..hidden {
        cpu[i] += res_data[i];
    }
    assert_close(&gpu, &cpu, "swiglu-gemv");
}
