//! Minimal reproducer for the "many dispatches in one submit produce zeros"
//! behaviour seen on RADV STRIX1. No SpectroStream, no special ops — just a
//! long chain of trivial copies through unary ops.
//!
//! If this fails the same way (zeros from late dispatches), the bug is in
//! Blade or the Vulkan driver, not meganeura's higher-level layers.

use meganeura::{Graph, build_inference_session};

fn run_with_n_ops(n: usize, elem_count: usize) -> (usize, f32) {
    let mut g = Graph::new();
    let mut x = g.input("x", &[elem_count]);
    // n unary ops in series — each writes a fresh buffer. None of these are
    // pointwise-fusion candidates after the first dozen if we vary them.
    for i in 0..n {
        x = match i % 3 {
            0 => g.relu(x),
            1 => {
                let neg_once = g.neg(x);
                g.neg(neg_once)
            }
            _ => {
                let zero = g.input(&format!("zero_{i}"), &[elem_count]);
                g.add(x, zero)
            }
        };
    }
    g.set_outputs(vec![x]);
    let mut s = build_inference_session(&g);
    let input: Vec<f32> = (0..elem_count).map(|i| (i as f32 + 1.0) * 0.001).collect();
    s.set_input("x", &input);
    for i in 0..n {
        if i % 3 == 2 {
            let zeros = vec![0.0f32; elem_count];
            s.set_input(&format!("zero_{i}"), &zeros);
        }
    }
    s.step();
    s.wait();
    let out = s.read_output(elem_count);
    let nz = out.iter().filter(|&&v| v != 0.0 && v.is_finite()).count();
    let max = out.iter().filter(|v| v.is_finite()).fold(0.0_f32, |a, &v| a.max(v.abs()));
    (nz, max)
}

#[test]
#[ignore]
fn many_ops_chain_should_propagate() {
    // Find the threshold (in op count) where the chain starts losing data.
    let elem_count = 64 * 1024;
    for &n in &[10usize, 30, 60, 100, 150, 200, 300] {
        let mut zero_runs = 0;
        for _trial in 0..3 {
            let (nz, max) = run_with_n_ops(n, elem_count);
            if nz == 0 {
                zero_runs += 1;
            }
            println!("n={n:3} trial={_trial} nz={nz:6}/{elem_count} max_abs={max:.4e}");
        }
        if zero_runs > 0 {
            println!("--- n={n} produced zeros in {zero_runs}/3 trials");
        }
    }
}

#[test]
#[ignore]
fn matmul_chain() {
    // Chain of N matmuls, identity-like weights, to test heavy compute under
    // many dispatches. Matrices [M, K] × [K, N] → [M, N].
    let dim = 256usize;
    for &n in &[10usize, 30, 60, 80] {
        let mut g = Graph::new();
        let mut x = g.input("x", &[dim, dim]);
        for i in 0..n {
            let w = g.parameter(&format!("w{i}"), &[dim, dim]);
            x = g.matmul(x, w);
        }
        g.set_outputs(vec![x]);
        let mut s = build_inference_session(&g);
        // Input: identity matrix.
        let mut input = vec![0.0f32; dim * dim];
        for k in 0..dim { input[k * dim + k] = 1.0; }
        s.set_input("x", &input);
        // Weights: identity matrices.
        let mut w_id = vec![0.0f32; dim * dim];
        for k in 0..dim { w_id[k * dim + k] = 1.0; }
        for i in 0..n {
            s.set_parameter(&format!("w{i}"), &w_id);
        }
        s.step();
        s.wait();
        let out = s.read_output(dim * dim);
        // Result should be identity again (I × I × ... × I = I).
        let nz = out.iter().filter(|&&v| v != 0.0 && v.is_finite()).count();
        let diag_sum: f32 = (0..dim).map(|k| out[k * dim + k]).sum();
        println!("[matmul] n={n} nz={nz}/{} expected_nz={dim} diag_sum={diag_sum:.1}", dim * dim);
    }
}

#[test]
#[ignore]
fn conv_transpose_hw_chain() {
    // Chain of N conv_transpose_2d_hw — the heaviest shader in SpectroStream.
    // Identity kernel (1×1 conv-T with 1.0 in kernel) so values should propagate
    // unchanged. Tests whether the heavy shader specifically loses data over
    // a long chain.
    let batch = 1u32; let h = 60u32; let w = 480u32;
    let c = 128u32;
    let in_size = (batch * c * h * w) as usize;
    let k_size = (c * c) as usize; // 1×1 kernel

    for &n in &[10usize, 30, 50] {
        let mut g = Graph::new();
        let mut x = g.input("x", &[in_size]);
        let mut params_to_set = Vec::new();
        for i in 0..n {
            let k = g.parameter(&format!("k{i}"), &[k_size]);
            params_to_set.push(format!("k{i}"));
            // 1×1 conv-T with stride 1: out = in (sum over kernel ci=co diagonal)
            x = g.conv_transpose_2d_hw(x, k, batch, c, h, w, c, 1, 1, 1, 1, 0, 0);
        }
        g.set_outputs(vec![x]);
        let mut s = build_inference_session(&g);
        let input: Vec<f32> = (0..in_size).map(|i| (i as f32 + 1.0) * 0.001).collect();
        s.set_input("x", &input);
        // Identity kernel: 1.0 on diagonal, 0 elsewhere.
        let mut k_id = vec![0.0_f32; k_size];
        for ci in 0..c as usize { k_id[ci * c as usize + ci] = 1.0; }
        for name in &params_to_set { s.set_parameter(name, &k_id); }
        s.step();
        s.wait();
        let out = s.read_output(in_size);
        let nz = out.iter().filter(|&&v| v != 0.0 && v.is_finite()).count();
        let max = out.iter().filter(|v| v.is_finite()).fold(0.0_f32, |a, &v| a.max(v.abs()));
        println!("[convT] n={n} nz={nz}/{in_size} max_abs={max:.4e}");
    }
}

#[test]
#[ignore]
fn branching_dag_chain() {
    // Each "block": main path = conv_transpose_hw → main + sliced shortcut x.
    // This mirrors SpectroStream's residual structure: x → (heavy ops →) +x.
    let h = 60u32; let w = 480u32; let c = 128u32;
    let n_blocks = 50usize;
    let in_size = (c * h * w) as usize;
    let k_size = (c * c) as usize;

    let mut g = Graph::new();
    let mut x = g.input("x", &[in_size]);
    let mut param_names = Vec::new();
    for i in 0..n_blocks {
        // Main path: 1×1 conv_transpose_hw (identity kernel) + ELU
        let k = g.parameter(&format!("k{i}"), &[k_size]);
        param_names.push(format!("k{i}"));
        let main_conv = g.conv_transpose_2d_hw(x, k, 1, c, h, w, c, 1, 1, 1, 1, 0, 0);
        let main = g.elu(main_conv);
        // Add residual back.
        x = g.add(main, x);
    }
    g.set_outputs(vec![x]);
    let mut s = build_inference_session(&g);
    let bufs = s.plan().buffers.len();
    let disps = s.plan().dispatches.len();
    println!("[branched] {n_blocks} blocks → {bufs} buffers, {disps} dispatches");
    let input: Vec<f32> = (0..in_size).map(|i| ((i as f32 + 1.0) * 0.001).sin()).collect();
    s.set_input("x", &input);
    let mut k_id = vec![0.0_f32; k_size];
    for ci in 0..c as usize { k_id[ci * c as usize + ci] = 1.0; }
    for name in &param_names { s.set_parameter(name, &k_id); }
    s.step();
    s.wait();
    let out = s.read_output(in_size);
    let nz = out.iter().filter(|&&v| v != 0.0 && v.is_finite()).count();
    let max = out.iter().filter(|v| v.is_finite()).fold(0.0_f32, |a, &v| a.max(v.abs()));
    println!("[branched] nz={nz}/{in_size} max_abs={max:.4e}");
}

#[test]
#[ignore]
fn many_ops_chain_huge_buffers() {
    // Same chain but with the buffer size SpectroStream sees at late blocks
    // (~7 M elements). Tests whether total memory pressure matters at lower
    // op counts.
    let elem_count = 7_127_040;
    for &n in &[10usize, 30, 60, 80] {
        let mut zero_runs = 0;
        for _trial in 0..3 {
            let (nz, max) = run_with_n_ops(n, elem_count);
            if nz == 0 {
                zero_runs += 1;
            }
            println!("[huge] n={n:3} trial={_trial} nz={nz:9}/{elem_count} max_abs={max:.4e}");
        }
        if zero_runs > 0 {
            println!("--- huge n={n} produced zeros in {zero_runs}/3 trials");
        }
    }
}
