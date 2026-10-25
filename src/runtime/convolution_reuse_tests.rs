use super::*;
use crate::{DType, codegen::ConvolutionReuseShape, compile::Kernel};

fn capabilities() -> blade_graphics::Capabilities {
    let mut caps = blade_graphics::Capabilities {
        shader_bfloat16: true,
        max_compute_shared_memory_size: 49152,
        full_subgroups: Some(blade_graphics::FullSubgroups {
            min_size: 32,
            max_size: 32,
            max_workgroup_subgroups: 32,
        }),
        ..Default::default()
    };
    caps.cooperative_matrix.bf16_f32_shapes.push([16, 16, 16]);
    caps
}

fn forward() -> Dispatch {
    Dispatch {
        shader: ShaderEntry::Conv2dBF16 {
            kernel_h: 5,
            kernel_w: 5,
            stride: 1,
            output_bf16: true,
        },
        params: vec![512, 3, 64, 64, 32, 5, 5, 1, 2, 64, 64, 2],
        workgroups: [1, 128, 512],
        ..Default::default()
    }
}

#[test]
fn convolution_reuse_policy_and_fallback() {
    let original = forward();
    let caps = capabilities();
    let expected = ConvolutionReuseShape {
        rows: 32,
        columns: 128,
    };
    let mut dispatch = original.clone();
    select_convolution_reuse(&mut dispatch, &caps);
    assert_eq!(dispatch.kernel, Kernel::ConvolutionReuse(expected));
    assert_eq!(dispatch.workgroups, [1, 32, 512]);
    assert_eq!(convolution_reuse_choice(&dispatch, &caps), Some(expected));
    let once = dispatch.clone();
    select_convolution_reuse(&mut dispatch, &caps);
    assert_eq!(dispatch, once);
    for failure in 0..5 {
        let mut unsupported = caps.clone();
        match failure {
            0 => unsupported.full_subgroups = None,
            1 => unsupported.full_subgroups.as_mut().unwrap().max_size = 16,
            2 => {
                unsupported.max_compute_shared_memory_size = expected.shared_bytes(DType::BF16) - 1
            }
            3 => unsupported.cooperative_matrix.bf16_f32_shapes.clear(),
            _ => unsupported.shader_bfloat16 = false,
        }
        let mut fresh = original.clone();
        select_convolution_reuse(&mut fresh, &unsupported);
        assert_eq!(fresh, original);
        let mut cached = once.clone();
        select_convolution_reuse(&mut cached, &unsupported);
        assert_eq!(cached, original);
    }
    for failure in 0..3 {
        let mut other = original.clone();
        match failure {
            0 => other.schedule_locked = true,
            1 => other.horizontal_batch = 2,
            _ => {
                other.shader = ShaderEntry::Conv2dBF16 {
                    kernel_h: 5,
                    kernel_w: 5,
                    stride: 1,
                    output_bf16: false,
                }
            }
        }
        let saved = other.clone();
        select_convolution_reuse(&mut other, &caps);
        assert_eq!(other, saved);
    }
    let mut malformed = original.clone();
    malformed.workgroups[0] += 1;
    assert!(convolution_reuse_choice(&malformed, &caps).is_none());
    malformed = original;
    malformed.params[1] = u32::MAX;
    malformed.params[5] = 3;
    assert!(convolution_reuse_choice(&malformed, &caps).is_none());
}

#[test]
fn convolution_reuse_shapes_add_no_padding() {
    for m in 1..=256u32 {
        for n in 1..=512u32 {
            if let Some(shape) = ConvolutionReuseShape::for_dimensions(m, n) {
                assert!(shape.valid());
                assert_eq!(m.div_ceil(shape.rows) * shape.rows, m.div_ceil(32) * 32);
                assert_eq!(
                    n.div_ceil(shape.columns) * shape.columns,
                    n.div_ceil(32) * 32
                );
                assert!(shape.shared_bytes(DType::BF16) <= 15360);
            }
        }
    }
    for (m, n) in [(0, 32), (32, 0), (u32::MAX, 32), (32, u32::MAX)] {
        assert!(ConvolutionReuseShape::for_dimensions(m, n).is_none());
    }
}

#[test]
fn convolution_reuse_cache_identity() {
    let original = forward();
    let mut candidate = original.clone();
    select_convolution_reuse(&mut candidate, &capabilities());
    assert_ne!(Pipelines::key(&original), Pipelines::key(&candidate));
    let a = SharedPipelineKey {
        context: 0,
        entry_point: "main",
        layout: String::new(),
        source: "same shader".into(),
        full_subgroup_size: None,
        convolution_reuse: None,
    };
    let b = SharedPipelineKey {
        full_subgroup_size: Some(32),
        ..SharedPipelineKey {
            context: 0,
            entry_point: "main",
            layout: String::new(),
            source: "same shader".into(),
            full_subgroup_size: None,
            convolution_reuse: None,
        }
    };
    assert!(a != b);
    let c = SharedPipelineKey {
        context: 0,
        entry_point: "main",
        layout: String::new(),
        source: "same shader".into(),
        full_subgroup_size: Some(32),
        convolution_reuse: Some(ConvolutionReuseShape {
            rows: 32,
            columns: 128,
        }),
    };
    assert!(b != c);
    let serialized = serde_json::to_string(&candidate).unwrap();
    let restored: Dispatch = serde_json::from_str(&serialized).unwrap();
    assert_eq!(Pipelines::key(&restored), Pipelines::key(&candidate));
}

#[test]
#[ignore = "requires captured actual-model source and plan paths"]
fn convolution_reuse_actual_plans() {
    use std::{fs, io::BufReader, path::PathBuf};
    let root = PathBuf::from(std::env::var("MEGANEURA_REUSE_AUDIT_ROOT").unwrap());
    let spec: serde_json::Value = serde_json::from_reader(BufReader::new(
        fs::File::open(root.join("fixture.json")).unwrap(),
    ))
    .unwrap();
    let output = root.join("generated");
    fs::create_dir(&output).unwrap();
    let rows: Vec<serde_json::Value> = serde_json::from_reader(BufReader::new(
        fs::File::open(spec["dispatches"].as_str().unwrap()).unwrap(),
    ))
    .unwrap();
    let expected: Vec<serde_json::Value> = serde_json::from_reader(BufReader::new(
        fs::File::open(spec["expected"].as_str().unwrap()).unwrap(),
    ))
    .unwrap();
    assert_eq!(rows.len(), 23);
    let caps = capabilities();
    let mut changed = 0;
    for (row, expected) in rows.iter().zip(&expected) {
        let original: Dispatch = serde_json::from_value(row.clone()).unwrap();
        let mut dispatch = original.clone();
        select_convolution_reuse(&mut dispatch, &caps);
        let selected = matches!(dispatch.kernel, Kernel::ConvolutionReuse(_));
        assert_eq!(selected, expected["changed"].as_bool().unwrap());
        assert_eq!(
            dispatch.workgroups,
            serde_json::from_value::<[u32; 3]>(expected["grid"].clone()).unwrap()
        );
        let generated = if let Kernel::ConvolutionReuse(shape) = dispatch.kernel {
            assert_eq!(
                [shape.rows, shape.columns],
                serde_json::from_value::<[u32; 2]>(expected["tile"].clone()).unwrap()
            );
            changed += 1;
            crate::codegen::generate_convolution_reuse(dispatch.shader.shader_group(), shape)
        } else {
            crate::codegen::generate_module(dispatch.shader.shader_group(), Default::default())
        };
        assert!(!generated.source.contains('$'));
        fs::write(
            output.join(format!("{}.wgsl", row["index"].as_u64().unwrap())),
            generated.source,
        )
        .unwrap();
        reset_convolution_reuse(&mut dispatch);
        assert_eq!(dispatch, original);
    }
    assert_eq!(changed, 7);
    let mut plans = Vec::new();
    for filename in spec["plans"].as_array().unwrap() {
        let path = filename.as_str().unwrap();
        let mut plan: ExecutionPlan =
            serde_json::from_reader(BufReader::new(fs::File::open(path).unwrap())).unwrap();
        let original = plan.clone();
        let mut changed = Vec::new();
        for (index, dispatch) in plan.dispatches.iter_mut().enumerate() {
            select_convolution_reuse(dispatch, &caps);
            if dispatch != &original.dispatches[index] {
                changed.push(serde_json::json!({"index":index,"kernel":dispatch.kernel,"grid":dispatch.workgroups}));
            }
        }
        for dispatch in &mut plan.dispatches {
            reset_convolution_reuse(dispatch);
        }
        assert!(plan == original, "changes beyond kernels/grids: {path}");
        plans.push(serde_json::json!({"path":path,"changes":changed,"dispatches":plan.dispatches.len(),"groups":plan.groups.len()}));
    }
    fs::write(root.join("actual-plan-review.json"),serde_json::to_vec_pretty(&serde_json::json!({
        "status":"passed","changed_products":changed,"actual_products":23,"plans":plans,"gpu_calls":0
    })).unwrap()).unwrap();
}
