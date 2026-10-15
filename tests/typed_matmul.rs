//! Explicit F16 and BF16 dense products share the checked typed cooperative
//! kernel: the same schedule, F32 accumulation and F32 partials.

use meganeura::{
    DType, Graph,
    codegen::MatrixTypes,
    compile::{Kernel, ShaderEntry},
};

const M: usize = 33;
const N: usize = 65;
const K: usize = 1031;

fn graph(storage: DType, layout: usize) -> Graph {
    let mut g = Graph::new();
    let a_shape = if layout == 1 { [K, M] } else { [M, K] };
    let b_shape = if layout == 2 { [N, K] } else { [K, N] };
    let (a, b) = if storage == DType::F16 {
        (
            g.parameter_f16("a", &a_shape),
            g.parameter_f16("b", &b_shape),
        )
    } else {
        let a = g.parameter("a", &a_shape);
        let b = g.parameter("b", &b_shape);
        (g.to_bf16(a), g.to_bf16(b))
    };
    let y = match layout {
        0 => g.matmul(a, b),
        1 => g.matmul_at(a, b),
        _ => g.matmul_bt(a, b),
    };
    g.set_outputs(vec![y]);
    g
}

#[test]
fn explicit_storage_selects_the_typed_schedule() {
    for storage in [DType::F16, DType::BF16] {
        for layout in 0..3 {
            let plan = meganeura::compile::compile(&graph(storage, layout));
            let product = plan
                .dispatches
                .iter()
                .find(|d| matches!(d.kernel, Kernel::CooperativeTyped { .. }))
                .unwrap();
            assert_eq!(
                product.shader,
                [
                    ShaderEntry::MatMul,
                    ShaderEntry::MatMulAT,
                    ShaderEntry::MatMulBT
                ][layout]
            );
            assert_eq!(product.mnk(), Some((M as u32, N as u32, K as u32)));
            assert_eq!(product.workgroups, [2, 3, 4]);
            assert_eq!(
                product.kernel,
                Kernel::CooperativeTyped {
                    types: MatrixTypes::reduced(storage),
                    splits: 4,
                }
            );
        }
    }
}

/// Values on a 1/64 grid are exact in both half formats, and every product
/// and partial sum is exact in F32, so the result must equal the F64 sum.
fn values(len: usize, multiplier: usize, modulus: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((i * multiplier % modulus) as f32 - (modulus / 2) as f32) / 64.0)
        .collect()
}

#[test]
fn f16_products_match_f64() {
    let gpu = crate::support::gpu::gpu();
    if !meganeura::runtime::typed_matrix_supported(&gpu, DType::F16) {
        eprintln!("native F16 matrices unavailable; skipping device-specific coverage");
        return;
    }
    crate::support::gpu::assert_protocol_headroom(&gpu);
    let a = values(M * K, 17, 61);
    let b = values(N * K, 31, 59);
    for layout in 0..3 {
        let mut config = crate::support::gpu::inference_config();
        config.tune = false;
        config.runtime.poison = true;
        let (mut session, _) = meganeura::build(&graph(DType::F16, layout), config);
        session.set_parameter("a", &a);
        session.set_parameter("b", &b);
        let mut expected = vec![0.0; M * N];
        for row in 0..M {
            for col in 0..N {
                let mut sum = 0.0f64;
                for inner in 0..K {
                    let ai = if layout == 1 {
                        inner * M + row
                    } else {
                        row * K + inner
                    };
                    let bi = if layout == 2 {
                        col * K + inner
                    } else {
                        inner * N + col
                    };
                    sum += f64::from(a[ai]) * f64::from(b[bi]);
                }
                expected[row * N + col] = sum as f32;
            }
        }
        session.step();
        session.wait();
        assert_eq!(session.read_output(M * N), expected, "layout {layout}");
    }
}

/// The F16 storage contract includes subnormal operands. This is the 2^-15
/// case that Mesa 26.0.3 flushes to zero on B570.
#[test]
fn f16_products_keep_subnormal_operands() {
    let gpu = crate::support::gpu::gpu();
    if !meganeura::runtime::typed_matrix_supported(&gpu, DType::F16) {
        eprintln!("native F16 matrices unavailable; skipping device-specific coverage");
        return;
    }
    let mut g = Graph::new();
    let a = g.parameter_f16("a", &[1, 32]);
    let b = g.parameter_f16("b", &[32, 1]);
    let y = g.matmul(a, b);
    g.set_outputs(vec![y]);
    let mut config = crate::support::gpu::inference_config();
    config.tune = false;
    let (mut session, _) = meganeura::build(&g, config);
    session.set_parameter("a", &[2.0f32.powi(-15); 32]);
    session.set_parameter("b", &[1.0; 32]);
    session.step();
    session.wait();
    assert_eq!(session.read_output(1), [2.0f32.powi(-10)]);
}

#[test]
fn unsupported_device_reports_f16_matrix_requirement() {
    let context = crate::support::gpu::gpu();
    if meganeura::runtime::typed_matrix_supported(&context, DType::F16) {
        eprintln!(
            "native F16 storage and matrices available; skipping unsupported-device coverage"
        );
        return;
    }
    // The first missing requirement: storage, a tile shape, then subnormals.
    let caps = context.capabilities();
    let expected = if !caps.shader_float16 {
        "selected plan requires native F16 shader support"
    } else if meganeura::codegen::checked_cooperative_shape(&caps.cooperative_matrix.f16_f32_shapes)
        .is_none()
    {
        "F16 matrix products require native 16x16x16 or 8x16x16 F16/F32 cooperative matrices"
    } else {
        "keep subnormal F16 operands"
    };
    let plan = meganeura::compile::compile(&graph(DType::F16, 0));
    let panic = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        meganeura::Session::with_context(plan, context)
    })) {
        Ok(_) => panic!("unsupported device accepted an F16 matrix plan"),
        Err(panic) => panic,
    };
    let message = panic
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| panic.downcast_ref::<&str>().copied())
        .unwrap_or("non-string panic");
    assert!(message.contains(expected), "unexpected panic: {message}");
}
