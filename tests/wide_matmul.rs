//! Scalar products with enough 128×64 tiles run on the wide kernel. Every
//! layout, a fused addend and ragged edges must match the f64 reference.
use meganeura::{CoopPolicy, DType};

#[test]
fn wide_tiles_match_the_reference() {
    // 8 row and 9 column tiles, 72 workgroups, with no edge aligned to a
    // tile or to the 16-deep K stage.
    let dims = [1000, 520, 77];
    let [m, n, k] = dims;
    let a = crate::support::data::pcg_inputs(m * k, 1);
    let b = crate::support::data::pcg_inputs(k * n, 2);
    let bias = crate::support::data::pcg_inputs(m * n, 3);
    for layout in 0..3 {
        for add in [false, true] {
            let graph = crate::support::data::layout_graph(DType::F32, layout, add, false, dims);
            let mut config = crate::support::gpu::inference_config();
            config.tune = false;
            config.runtime.coop = CoopPolicy::Disabled;
            config.runtime.poison = true;
            let (mut session, _) = meganeura::build(&graph, config);
            let shapes: Vec<_> = session
                .plan()
                .dispatches
                .iter()
                .filter_map(|d| d.scalar_matmul())
                .map(|s| (s.rows(), s.cols()))
                .collect();
            assert_eq!(shapes, [(128, 64)], "layout {layout} add {add}");
            session.set_parameter("a", &a);
            session.set_parameter("b", &b);
            if add {
                session.set_parameter("bias", &bias);
            }
            session.step();
            session.wait();
            let expected: Vec<f32> = crate::support::data::matmul_reference(layout, &a, &b, dims)
                .iter()
                .zip(&bias)
                .map(|(&c, &d)| (c + if add { f64::from(d) } else { 0.0 }) as f32)
                .collect();
            crate::support::data::assert_close(
                &format!("layout {layout} add {add}"),
                &session.read_output(m * n),
                &expected,
                1e-5,
            );
        }
    }
}
