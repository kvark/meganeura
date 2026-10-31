use super::{CoopPolicy, ExecutionPlan, SessionOptions, ShaderEntry};
use crate::codegen::MatrixTypes;
use crate::compile::{Kernel, WeightFormat};

pub(super) fn reset(plan: &mut ExecutionPlan) {
    for dispatch in &mut plan.dispatches {
        if let Kernel::CooperativeWorkgroup { types, splits, .. } = dispatch.kernel {
            let (m, n, _) = dispatch.mnk().expect("workgroup product dimensions");
            if let Some(types) = types {
                dispatch.kernel = Kernel::CooperativeTyped { types, splits };
                dispatch.workgroups = [m.div_ceil(32), n.div_ceil(32), splits];
            } else {
                dispatch.kernel = Kernel::Default;
                dispatch.workgroups = [n.div_ceil(64), m.div_ceil(64), 1];
            }
        }
    }
}

pub(super) fn select(
    plan: &mut ExecutionPlan,
    caps: &blade_graphics::Capabilities,
    opts: &SessionOptions,
) {
    let Some(shape) = opts
        .workgroup_matmul
        .filter(|_| opts.coop == CoopPolicy::AllowF16)
    else {
        return;
    };
    for dispatch in &mut plan.dispatches {
        let (types, splits) = match dispatch.kernel {
            Kernel::CooperativeTyped { types, splits } => (Some(types), splits),
            _ => (None, 1),
        };
        let f32 = matches!(
            dispatch.kernel,
            Kernel::Default | Kernel::SmallTile | Kernel::Cooperative
        );
        if !matches!(
            dispatch.shader,
            ShaderEntry::MatMul
                | ShaderEntry::MatMulAT
                | ShaderEntry::MatMulBT
                | ShaderEntry::FusedMatMulAdd
                | ShaderEntry::FusedMatMulATAdd
                | ShaderEntry::FusedMatMulBTAdd
        ) || dispatch.horizontal_batch > 1
            || dispatch.matmul_prologue.is_some()
            || dispatch.matmul_epilogue.is_some()
            || dispatch.requires_full_precision
            || dispatch.weight_format != WeightFormat::F32
            || (types.is_none() && dispatch.schedule_locked)
            || !(types.is_some() || f32)
        {
            continue;
        }
        if let Some((m, n, k)) = dispatch.mnk() {
            if shape.supports(caps, types.unwrap_or(MatrixTypes::F32_AS_F16), m, n, k)
                && splits > 0
                && splits <= k / shape.k_stage
            {
                dispatch.kernel = Kernel::CooperativeWorkgroup {
                    shape,
                    types,
                    splits,
                };
                dispatch.workgroups = [m.div_ceil(shape.rows), n / shape.columns, splits];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Graph, codegen::WorkgroupMatmulShape};

    const TILE: WorkgroupMatmulShape = WorkgroupMatmulShape {
        rows: 64,
        columns: 64,
        k_stage: 32,
        threads: 64,
    };

    fn caps() -> blade_graphics::Capabilities {
        let mut caps = blade_graphics::Capabilities {
            max_compute_shared_memory_size: 32768,
            shader_float16: true,
            shader_bfloat16: true,
            ..Default::default()
        };
        caps.cooperative_matrix.workgroup = blade_graphics::WorkgroupCooperativeMatrix {
            granularities: vec![blade_graphics::CooperativeMatrixGranularity {
                operand: blade_graphics::CooperativeMatrixOperand::F16,
                size: [16, 16, 16],
                invocations: 64,
            }],
            max_workgroup_size: 256,
            max_dimension: 1024,
            reserved_shared_memory: 8192,
        };
        caps
    }

    fn plan(m: usize, n: usize, k: usize) -> ExecutionPlan {
        let mut g = Graph::new();
        let a = g.input("a", &[m, k]);
        let b = g.input("b", &[k, n]);
        let y = g.matmul(a, b);
        g.set_outputs(vec![y]);
        crate::compile::compile(&g)
    }

    #[test]
    fn large_workgroup_tiles_respect_device_memory_and_dimensions() {
        let mut device = caps();
        device.max_compute_shared_memory_size = 49152;
        device.cooperative_matrix.workgroup.granularities = [
            blade_graphics::CooperativeMatrixOperand::F16,
            blade_graphics::CooperativeMatrixOperand::Bf16,
        ]
        .map(|operand| blade_graphics::CooperativeMatrixGranularity {
            operand,
            size: [16, 16, 16],
            invocations: 256,
        })
        .to_vec();
        let tile = WorkgroupMatmulShape {
            rows: 128,
            columns: 256,
            k_stage: 32,
            threads: 256,
        };
        assert!(tile.supports(&device, MatrixTypes::F32_AS_F16, 1536, 512, 384));
        let deeper = WorkgroupMatmulShape {
            k_stage: 64,
            ..tile
        };
        assert!(!deeper.supports(&device, MatrixTypes::F32_AS_F16, 1536, 512, 384));
        assert!(deeper.supports(&device, MatrixTypes::BF16, 1536, 512, 384));
        device.cooperative_matrix.workgroup.max_dimension = 128;
        assert!(!tile.supports(&device, MatrixTypes::BF16, 1536, 512, 384));
        assert!(!tile.supports(&device, MatrixTypes::F32_AS_F16, 1536, 512, 384));

        let overflow = WorkgroupMatmulShape {
            rows: u32::MAX - 3,
            columns: 256,
            ..tile
        };
        device.cooperative_matrix.workgroup.max_dimension = u32::MAX;
        assert_eq!(overflow.shared_bytes(MatrixTypes::F32_AS_F16, false), None);
        assert!(!overflow.supports(&device, MatrixTypes::F32_AS_F16, 1536, 512, 384));
    }

    #[test]
    fn workgroup_selection_respects_precision_device_and_shape() {
        let mut options = SessionOptions {
            coop: CoopPolicy::AllowF16,
            workgroup_matmul: Some(TILE),
            ..Default::default()
        };
        for (dims, policy, device, expected) in [
            ((128, 256, 128), CoopPolicy::AllowF16, caps(), true),
            ((129, 256, 128), CoopPolicy::AllowF16, caps(), true),
            ((128, 257, 128), CoopPolicy::AllowF16, caps(), false),
            ((128, 256, 129), CoopPolicy::AllowF16, caps(), false),
            ((128, 256, 128), CoopPolicy::NativeF32, caps(), false),
            ((128, 256, 128), CoopPolicy::Auto, caps(), false),
            ((128, 256, 128), CoopPolicy::Disabled, caps(), false),
            (
                (128, 256, 128),
                CoopPolicy::AllowF16,
                blade_graphics::Capabilities::default(),
                false,
            ),
        ] {
            let mut plan = plan(dims.0, dims.1, dims.2);
            options.coop = policy;
            select(&mut plan, &device, &options);
            assert_eq!(
                plan.dispatches
                    .iter()
                    .any(|d| matches!(d.kernel, Kernel::CooperativeWorkgroup { .. })),
                expected
            );
        }
        let mut plan = plan(128, 256, 128);
        let matrix = plan
            .dispatches
            .iter_mut()
            .find(|d| d.mnk().is_some())
            .unwrap();
        matrix.requires_full_precision = true;
        select(&mut plan, &caps(), &options);
        assert!(
            !plan
                .dispatches
                .iter()
                .any(|d| matches!(d.kernel, Kernel::CooperativeWorkgroup { .. }))
        );
    }

    #[test]
    fn saved_workgroup_choice_is_reset_before_device_selection() {
        for bf16 in [false, true] {
            let mut plan = plan(128, 256, 128);
            let matrix = plan
                .dispatches
                .iter_mut()
                .find(|d| d.mnk().is_some())
                .unwrap();
            matrix.kernel = Kernel::CooperativeWorkgroup {
                shape: TILE,
                types: bf16.then_some(MatrixTypes::BF16),
                splits: 1,
            };
            matrix.workgroups = [2, 4, 1];
            let encoded = ron::to_string(&plan).unwrap();
            let mut decoded: ExecutionPlan = ron::from_str(&encoded).unwrap();
            reset(&mut decoded);
            let matrix = decoded
                .dispatches
                .iter()
                .find(|d| d.mnk().is_some())
                .unwrap();
            if bf16 {
                assert_eq!(
                    matrix.kernel,
                    Kernel::CooperativeTyped {
                        types: MatrixTypes::BF16,
                        splits: 1
                    }
                );
                assert_eq!(matrix.workgroups, [4, 8, 1]);
            } else {
                assert_eq!(matrix.kernel, Kernel::Default);
                assert_eq!(matrix.workgroups, [4, 2, 1]);
            }
        }
    }
}
