use super::*;
use crate::graph::{DerivedParam, Graph};

fn graph() -> Graph {
    let mut g = Graph::new();
    let a = g.parameter("a", &[5, 3]);
    let b = g.parameter("b", &[5, 2]);
    let c = g.parameter("c", &[2, 3]);
    let horizontal = g.parameter("horizontal", &[5, 5]);
    let vertical = g.parameter("vertical", &[7, 3]);
    g.derived_params.extend([
        DerivedParam {
            name: "horizontal".into(),
            sources: vec![("a".into(), 3), ("b".into(), 2)],
            rows: 5,
            transform: ParamTransform::HorizontalConcat,
        },
        DerivedParam {
            name: "vertical".into(),
            sources: vec![("a".into(), 5), ("c".into(), 2)],
            rows: 7,
            transform: ParamTransform::VerticalConcat,
        },
    ]);
    let outputs = [a, b, c, horizontal, vertical].map(|node| g.materialize(node));
    g.set_outputs(outputs.to_vec());
    g
}

fn plan() -> ExecutionPlan {
    crate::compile::compile(&graph())
}

#[test]
fn shader_validates_without_a_gpu() {
    let module =
        naga::front::wgsl::parse_str(include_str!("../../shaders/parameter_blend.wgsl")).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all() ^ naga::valid::ValidationFlags::BINDINGS,
        naga::valid::Capabilities::default(),
    )
    .validate(&module)
    .unwrap();
}

#[test]
fn geometry_maps_partial_concatenations() {
    let plan = plan();
    let (updates, refresh) = plan_updates(&plan, &plan, &["a"]).unwrap();
    assert_eq!(updates.len(), 1);
    assert_eq!(refresh.len(), 2);
    assert_eq!(
        (
            refresh[0].len,
            refresh[0].columns,
            refresh[0].stride,
            refresh[0].offset
        ),
        (15, 3, 5, 0)
    );
    assert_eq!(
        (
            refresh[1].len,
            refresh[1].columns,
            refresh[1].stride,
            refresh[1].offset
        ),
        (15, 3, 3, 0)
    );
    let (_, refresh) = plan_updates(&plan, &plan, &["b", "c"]).unwrap();
    assert_eq!(
        (
            refresh[0].len,
            refresh[0].columns,
            refresh[0].stride,
            refresh[0].offset
        ),
        (10, 2, 5, 3)
    );
    assert_eq!(
        (
            refresh[1].len,
            refresh[1].columns,
            refresh[1].stride,
            refresh[1].offset
        ),
        (6, 3, 3, 15)
    );
}

#[test]
fn rejects_unsupported_or_incompatible_updates() {
    let original = plan();
    assert_eq!(
        plan_updates(&original, &original, &["a", "a"]).unwrap_err(),
        BlendParameterError::DuplicateParameter
    );
    assert!(matches!(
        plan_updates(&original, &original, &["missing"]),
        Err(BlendParameterError::UnknownParameter(_))
    ));
    let buffer = original
        .param_buffers
        .iter()
        .find(|entry| entry.0 == "a")
        .unwrap()
        .1;
    let mut changed = plan();
    changed.param_types.get_mut(&buffer).unwrap().dtype = DType::F16;
    assert_eq!(
        plan_updates(&changed, &original, &["a"]).unwrap_err(),
        BlendParameterError::UnsupportedStorage
    );
    let mut changed = plan();
    changed.param_types.get_mut(&buffer).unwrap().shape = vec![3, 5];
    assert!(matches!(
        plan_updates(&changed, &original, &["a"]),
        Err(BlendParameterError::IncompatibleParameter(_))
    ));
    let mut changed = plan();
    changed.buffers[buffer.0 as usize] = 4;
    assert_eq!(
        plan_updates(&changed, &original, &["a"]).unwrap_err(),
        BlendParameterError::InvalidGeometry
    );
    let mut changed = plan();
    changed.derived_params[0].1[0].1 = usize::MAX;
    assert_eq!(
        plan_updates(&changed, &original, &["a"]).unwrap_err(),
        BlendParameterError::InvalidGeometry
    );
}

#[test]
fn checked_regions_reject_overflow_and_bad_strides() {
    let b = BufferRef(0);
    assert!(region(b, b, 15, 3, 5, 0, 25).is_ok());
    for (len, cols, stride, offset, capacity) in [
        (0, 1, 1, 0, 1),
        (15, 0, 5, 0, 25),
        (15, 3, 2, 0, 25),
        (15, 4, 5, 0, 25),
        (15, 3, 5, 3, 25),
        (15, 3, 5, usize::MAX, 25),
    ] {
        assert!(region(b, b, len, cols, stride, offset, capacity).is_err());
    }
}

fn read(session: &Session, name: &str) -> Vec<f32> {
    session.read_params(&[name]).pop().unwrap()
}

fn check(actual: &[f32], source: &[f32], old: &[f32], rate: f32) -> usize {
    let decay = 1.0f32 - rate;
    let mut host_differences = 0;
    for (i, ((&got, &src), &dst)) in actual.iter().zip(source).zip(old).enumerate() {
        let left = f64::from(rate) * f64::from(src);
        let right = f64::from(decay) * f64::from(dst);
        let expected = left + right;
        let bound = 4.0 * 2.0f64.powi(-24) * (left.abs() + right.abs()) + 2.0f64.powi(-126);
        assert!(
            got.is_finite() && (f64::from(got) - expected).abs() <= bound,
            "element {i}: {got} vs {expected}, bound {bound}"
        );
        host_differences += usize::from(got.to_bits() != (rate * src + decay * dst).to_bits());
    }
    host_differences
}

fn exact(left: &[f32], right: &[f32]) {
    assert_eq!(left.len(), right.len());
    assert!(
        left.iter()
            .zip(right)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
}

#[test]
#[ignore = "guarded GPU; F64 blend, derived refresh, lifetimes and update ordering"]
fn gpu_blend_values_refresh_and_lifetimes() {
    let gpu = crate::reference::gpu::shared_context();
    let device = gpu.device_information();
    assert_eq!(
        device.device_name,
        std::env::var("MEGANEURA_EXPECT_DEVICE_NAME").unwrap()
    );
    assert!(!device.is_software_emulated);
    let memory = gpu.memory_stats();
    assert!(memory.budget.saturating_sub(memory.usage) >= 2 << 30);
    let make =
        |graph: &Graph| Session::with_context(crate::compile::compile(graph), Arc::clone(&gpu));
    let mut checked = 0;
    let mut host_differences = 0;
    for len in [1, 17, 255, 256, 257, 4099, 1_048_579] {
        let mut g = Graph::new();
        let p = g.parameter("weight", &[len]);
        let guard = g.parameter("guard", &[19]);
        let output = g.materialize(p);
        g.set_outputs(vec![output, guard]);
        let mut source = make(&g);
        let mut target = make(&g);
        let src = (0..len)
            .map(|i| ((i * 31 % 997) as f32 - 498.0) * 0.03125)
            .collect::<Vec<_>>();
        let old = src
            .iter()
            .enumerate()
            .map(|(i, &value)| match i % 4 {
                0 => -value,
                1 => 1e-38,
                2 => 5.0,
                _ => value * 7.0,
            })
            .collect::<Vec<_>>();
        let canary = vec![37.25; 19];
        source.set_parameter("weight", &src);
        source.set_parameter("guard", &canary);
        target.set_parameter("guard", &canary);
        for rate in [0.0, 0.02, 0.5, 1.0] {
            target.set_parameter("weight", &old);
            target
                .blend_parameters_from(&mut source, &["weight"], rate)
                .unwrap();
            // No explicit wait before the graph consumes the updated parameter.
            target.step();
            let mut actual = vec![0.0; len];
            target.wait_read_output(0, &mut actual);
            host_differences += check(&actual, &src, &old, rate);
            checked += len;
            exact(&read(&source, "weight"), &src);
            exact(&read(&target, "guard"), &canary);
        }
        target
            .blend_parameters_from(&mut source, &["weight"], 0.02)
            .unwrap();
        // A host write must wait for the preceding GPU blend, not race it.
        target.set_parameter("weight", &old);
        exact(&read(&target, "weight"), &old);
        target
            .blend_parameters_from(&mut source, &["weight"], 0.02)
            .unwrap();
        drop(source);
        host_differences += check(&read(&target, "weight"), &src, &old, 0.02);
        checked += len;
    }
    let mut source = make(&graph());
    let mut target = make(&graph());
    let values = [("a", 15), ("b", 10), ("c", 6)].map(|(name, len)| {
        let src = (0..len).map(|i| i as f32 * 0.125 - 3.0).collect::<Vec<_>>();
        let old = src.iter().map(|x| -7.0 * x).collect::<Vec<_>>();
        source.set_parameter(name, &src);
        target.set_parameter(name, &old);
        (name, src, old)
    });
    for names in [&["a"][..], &["b", "c"][..], &["a", "b", "c"][..]] {
        let before = values
            .iter()
            .map(|(name, _, _)| read(&target, name))
            .collect::<Vec<_>>();
        target
            .blend_parameters_from(&mut source, names, 0.02)
            .unwrap();
        target.step();
        target.wait();
        let updated = values
            .iter()
            .zip(&before)
            .map(|((name, src, _), old)| {
                let actual = read(&target, name);
                if names.contains(name) {
                    host_differences += check(&actual, src, old, 0.02);
                    checked += actual.len();
                } else {
                    exact(&actual, old);
                }
                exact(&read(&source, name), src);
                actual
            })
            .collect::<Vec<_>>();
        let horizontal = (0..5)
            .flat_map(|row| {
                updated[0][row * 3..row * 3 + 3]
                    .iter()
                    .chain(&updated[1][row * 2..row * 2 + 2])
                    .copied()
            })
            .collect::<Vec<_>>();
        let vertical = updated[0]
            .iter()
            .chain(&updated[2])
            .copied()
            .collect::<Vec<_>>();
        exact(&read(&target, "horizontal"), &horizontal);
        exact(&read(&target, "vertical"), &vertical);
        for (output, expected) in [(3, &horizontal), (4, &vertical)] {
            let mut actual = vec![0.0; expected.len()];
            target.wait_read_output(output, &mut actual);
            exact(&actual, expected);
        }
    }
    assert_eq!(
        target.blend_parameters_from(&mut source, &["a", "a"], 0.02),
        Err(BlendParameterError::DuplicateParameter)
    );
    assert_eq!(
        target.blend_parameters_from(&mut source, &["a"], f32::NAN),
        Err(BlendParameterError::InvalidRate)
    );
    target.step_prepared = true;
    assert_eq!(
        target.blend_parameters_from(&mut source, &["a"], 0.02),
        Err(BlendParameterError::PreparedStep)
    );
    target.step_prepared = false;
    target.share_parameter_from(&mut source, "a").unwrap();
    assert_eq!(
        target.blend_parameters_from(&mut source, &["a"], 0.02),
        Err(BlendParameterError::AliasedParameters)
    );
    // A source SGD submission may still be pending when blending starts.
    let mut g = Graph::new();
    let weight = g.parameter("weight", &[257]);
    let zero = g.constant(vec![0.0; 257], &[257]);
    let loss = g.mse_loss(weight, zero);
    g.set_outputs(vec![loss]);
    let mut source = crate::build(
        &g,
        crate::SessionConfig {
            mode: crate::Mode::Training,
            gpu: Some(Arc::clone(&gpu)),
            ..Default::default()
        },
    )
    .0;
    let mut g = Graph::new();
    let weight = g.parameter("weight", &[257]);
    g.set_outputs(vec![weight]);
    let mut target = make(&g);
    source.set_parameter("weight", &vec![3.0; 257]);
    target.set_parameter("weight", &vec![1.0; 257]);
    source.set_learning_rate(0.125);
    source.step();
    target
        .blend_parameters_from(&mut source, &["weight"], 0.02)
        .unwrap();
    let source_values = read(&source, "weight");
    assert!(source_values.iter().all(|&v| v < 3.0));
    host_differences += check(
        &read(&target, "weight"),
        &source_values,
        &vec![1.0; 257],
        0.02,
    );
    checked += 257;
    let memory = gpu.memory_stats();
    assert!(memory.budget.saturating_sub(memory.usage) >= 2 << 30);
    println!(
        "parameter_blend_f64_values={checked}; f32_host_differences={host_differences}; queued_sgd_updates=1"
    );
}
