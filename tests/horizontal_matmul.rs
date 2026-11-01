//! Horizontal fusion (D1): independent same-A matmuls in one barrier
//! group pack into a single dispatch. Compare GPU results to an f64
//! reference and assert the pack actually formed. `compile` unit tests
//! cover the siblings that must not pack.

use crate::support::data::{assert_close, matmul_reference, pcg_inputs, relative_l2};
use meganeura::{CoopPolicy, Graph, Mode, SessionConfig, build};

/// The default policy (scalar or f16 cooperative) must stay within 2% of the
/// largest output; native f32 cooperative tiles within 1e-5 relative L2.
#[test]
fn same_a_siblings_pack_and_keep_outputs_independent() {
    let f32_shapes = &crate::support::gpu::gpu()
        .capabilities()
        .cooperative_matrix
        .f32_shapes;
    let native_f32 = f32_shapes.contains(&[8, 8, 8]) || f32_shapes.contains(&[16, 16, 16]);
    for (native, [m, n, k], copies) in [
        (false, [32, 32, 32], 3),
        (true, [65, 256, 33], 2),
        (true, [65, 256, 33], 3),
    ] {
        if native && !native_f32 {
            continue;
        }
        let mut g = Graph::new();
        let a = g.input("a", &[m, k]);
        let outputs = (0..copies)
            .map(|copy| {
                let b = g.input(&format!("b{copy}"), &[k, n]);
                g.matmul(a, b)
            })
            .collect();
        g.set_outputs(outputs);
        let config = if native {
            let mut config = crate::support::gpu::inference_config();
            config.runtime.coop = CoopPolicy::NativeF32;
            config.tune = false;
            config
        } else {
            SessionConfig {
                mode: Mode::Inference,
                ..crate::support::gpu::config()
            }
        };
        let (mut session, _) = build(&g, config);
        assert!(
            session
                .plan()
                .dispatches
                .iter()
                .any(|d| d.horizontal_batch == copies && (!native || d.use_coop())),
            "{copies} same-A siblings should pack into one dispatch; got {:?}",
            session
                .plan()
                .dispatches
                .iter()
                .map(|d| (d.shader.clone(), d.horizontal_batch, d.workgroups))
                .collect::<Vec<_>>()
        );
        let a_data = pcg_inputs(m * k, 1);
        let weights: Vec<_> = (0..copies)
            .map(|copy| pcg_inputs(k * n, copy + 2))
            .collect();
        session.set_input("a", &a_data);
        for (copy, b) in weights.iter().enumerate() {
            session.set_input(&format!("b{copy}"), b);
        }
        session.step();
        session.wait();
        for (copy, b) in weights.iter().enumerate() {
            let mut gpu = vec![0.0_f32; m * n];
            session.read_output_by_index(copy, &mut gpu);
            let cpu: Vec<f32> = matmul_reference(0, &a_data, b, [m, n, k])
                .into_iter()
                .map(|v| v as f32)
                .collect();
            let label = format!("native_f32={native} sibling {copy} of {copies}");
            if native {
                let error = relative_l2(&gpu, &cpu);
                assert!(error < 1e-5, "{label}: relative L2 {error}");
            } else {
                assert_close(&label, &gpu, &cpu, 0.02);
            }
        }
    }
}
