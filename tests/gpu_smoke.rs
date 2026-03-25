/// GPU smoke test: validates that all shaders compile with blade + lavapipe
/// and that a simple forward pass executes without errors.
use meganeura::{Graph, build_inference_session, build_session};

#[test]
fn matmul_non_uniform_values() {
    // Non-uniform matmul: A (16×16) where A[i,j]=i+1, B (16×16) where B[i,j]=j+1.
    // C[i,j] = sum_{k=0}^{15} (i+1) * (k+1) = (i+1) * sum_{k=0}^{15}(k+1) = (i+1) * 136.
    // This catches A@B vs B@A bugs because B@A[i,j] = (j+1)*136 ≠ (i+1)*136 for i≠j.
    let m = 16;
    let k = 16;
    let n = 16;
    let mut g = Graph::new();
    let a = g.input("a", &[m, k]);
    let b = g.parameter("b", &[k, n]);
    let c = g.matmul(a, b);
    g.set_outputs(vec![c]);

    let mut session = build_inference_session(&g);

    let a_data: Vec<f32> = (0..m * k).map(|i| (i / k + 1) as f32).collect(); // A[i,j] = i+1
    let b_data: Vec<f32> = (0..k * n).map(|i| (i % n + 1) as f32).collect(); // B[i,j] = j+1
    session.set_input("a", &a_data);
    session.set_parameter("b", &b_data);

    session.step();
    session.wait();

    let output = session.read_output(m * n);
    // C[i,j] = sum_{k=0}^{15} A[i,k]*B[k,j] = sum_k (i+1)*(j+1) = 16*(i+1)*(j+1)
    // B@A[i,j] = sum_k (k+1)*(j+1) = (j+1)*136  — varies only with j, not i
    // A@B[i,j] = 16*(i+1)*(j+1)               — varies with both i and j
    for row in 0..m {
        for col in 0..n {
            let got = output[row * n + col];
            let expected = k as f32 * (row as f32 + 1.0) * (col as f32 + 1.0);
            assert!(
                (got - expected).abs() < expected.abs() * 0.02 + 1.0,
                "C[{},{}]: got {}, expected {} (f16 precision)",
                row,
                col,
                got,
                expected
            );
        }
    }
}

#[test]
fn shader_compilation_and_forward_pass() {
    // Build a small MLP graph
    let batch = 4;
    let mut g = Graph::new();
    let x = g.input("x", &[batch, 8]);
    let labels = g.input("labels", &[batch, 3]);
    let w1 = g.parameter("w1", &[8, 5]);
    let b1 = g.parameter("b1", &[5]);
    let mm1 = g.matmul(x, w1);
    let h1 = g.bias_add(mm1, b1);
    let a1 = g.relu(h1);
    let w2 = g.parameter("w2", &[5, 3]);
    let mm2 = g.matmul(a1, w2);
    let loss = g.cross_entropy_loss(mm2, labels);
    g.set_outputs(vec![loss]);

    // Build session (this compiles all shaders via blade)
    let mut session = build_session(&g);

    // Initialize with small data
    let w1_data = vec![0.1_f32; 8 * 5];
    let b1_data = vec![0.0_f32; 5];
    let w2_data = vec![0.1_f32; 5 * 3];
    session.set_parameter("w1", &w1_data);
    session.set_parameter("b1", &b1_data);
    session.set_parameter("w2", &w2_data);

    let x_data = vec![1.0_f32; batch * 8];
    let mut labels_data = vec![0.0_f32; batch * 3];
    for b in 0..batch {
        labels_data[b * 3] = 1.0;
    }
    session.set_input("x", &x_data);
    session.set_input("labels", &labels_data);

    // Execute forward + backward
    session.step();
    session.wait();

    // Read back loss - should be a finite number
    let loss_val = session.read_loss();
    assert!(
        loss_val.is_finite(),
        "loss should be finite, got {}",
        loss_val
    );
    assert!(
        loss_val > 0.0,
        "cross-entropy loss should be positive, got {}",
        loss_val
    );
}

#[test]
fn matmul_produces_correct_values() {
    // 16x32 @ 32x16 matmul — tile-aligned for cooperative matmul (16×16 tiles)
    // A: 16×32 filled with 0.1 → each output element = 32 * 0.1 * 0.1 = 0.32
    // B: 32×16 filled with 0.1
    let m = 16;
    let k = 32;
    let n = 16;

    let mut g = Graph::new();
    let a = g.input("a", &[m, k]);
    let b = g.parameter("b", &[k, n]);
    let c = g.matmul(a, b);
    g.set_outputs(vec![c]);

    let mut session = build_inference_session(&g);

    let a_data = vec![0.1_f32; m * k];
    let b_data = vec![0.1_f32; k * n];
    session.set_input("a", &a_data);
    session.set_parameter("b", &b_data);

    session.step();
    session.wait();

    let output = session.read_output(m * n);
    eprintln!("matmul output first 4: {:?}", &output[..4]);
    assert_eq!(output.len(), m * n);
    // Each element = sum_{i=0}^{k-1} 0.1 * 0.1 = k * 0.01 = 0.32
    let expected = k as f32 * 0.01;
    for (i, &got) in output.iter().enumerate() {
        assert!(
            (got - expected).abs() < 0.02, // f16 precision tolerance
            "matmul output[{}]: got {}, expected {}",
            i,
            got,
            expected
        );
    }
}

#[test]
fn fused_matmul_add_correct() {
    // Test FusedMatMulAdd: C = A × B + D
    // A: 16×32 all 0.1, B: 32×16 all 0.1 → A×B = 0.32 per element
    // D: 16×16 all 1.0 → result should be 1.32 per element
    let m = 16;
    let k = 32;
    let n = 16;

    let mut g = Graph::new();
    let a = g.input("a", &[m, k]);
    let b = g.parameter("b", &[k, n]);
    let d = g.input("d", &[m, n]);
    let mm = g.matmul(a, b);
    let out = g.add(mm, d);
    g.set_outputs(vec![out]);

    let mut session = build_inference_session(&g);

    session.set_input("a", &vec![0.1_f32; m * k]);
    session.set_parameter("b", &vec![0.1_f32; k * n]);
    session.set_input("d", &vec![1.0_f32; m * n]);

    session.step();
    session.wait();

    let output = session.read_output(m * n);
    eprintln!("fused matmul+add first 4: {:?}", &output[..4]);
    let expected = k as f32 * 0.01 + 1.0; // 0.32 + 1.0 = 1.32
    for (i, &got) in output.iter().enumerate() {
        assert!(
            (got - expected).abs() < 0.02,
            "fused_matmul_add output[{}]: got {}, expected {}",
            i,
            got,
            expected
        );
    }
}
