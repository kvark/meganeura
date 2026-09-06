use std::{
    collections::HashMap,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

use meganeura::{CoopPolicy, Graph, Mode, Session, SessionConfig, SessionOptions};
use safetensors::tensor::{Dtype, TensorView};

struct TestFile(PathBuf);

impl TestFile {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "meganeura-checkpoint-{}-{}.safetensors",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        Self(path)
    }
}

impl Drop for TestFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn graph() -> Graph {
    let mut graph = Graph::new();
    let x = graph.input("x", &[1, 2]);
    let w = graph.parameter("w", &[2, 2]);
    let y = graph.matmul(x, w);
    let loss = graph.mean_all(y);
    graph.set_outputs(vec![loss]);
    graph
}

fn training_session() -> Session {
    let mut session = meganeura::build(
        &graph(),
        SessionConfig {
            runtime: SessionOptions {
                coop: CoopPolicy::Disabled,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .0;
    session.set_parameter("w", &[0.5, 0.6, 0.7, 0.8]);
    session.set_input("x", &[1.0, 0.75]);
    session
}

#[derive(Debug, PartialEq)]
struct Snapshot {
    parameters: Vec<Vec<f32>>,
    moments: Vec<(Vec<f32>, Vec<f32>)>,
    gradient: Vec<f32>,
    step: u32,
    moment_bytes: usize,
}

fn snapshot(session: &Session) -> Snapshot {
    let mut gradient = vec![0.0; 4];
    session.read_param_grad("w", &mut gradient);
    Snapshot {
        parameters: session.read_params(&["w"]),
        moments: session.read_adam_states(&["w"]),
        gradient,
        step: session.adam_step_count(),
        moment_bytes: session.memory_summary().adam_state_bytes,
    }
}

#[test]
fn checkpoint_rejects_oversized_adam_state() {
    let mut session = training_session();
    let before = snapshot(&session);
    let parameter = vec![0_u8; 4 * 4];
    let oversized_moment = vec![0_u8; 8 * 4];
    let moment = vec![0_u8; 4 * 4];
    let views = vec![
        (
            "w".to_string(),
            TensorView::new(Dtype::F32, vec![4], &parameter).unwrap(),
        ),
        (
            "adam_m.w".to_string(),
            TensorView::new(Dtype::F32, vec![8], &oversized_moment).unwrap(),
        ),
        (
            "adam_v.w".to_string(),
            TensorView::new(Dtype::F32, vec![4], &moment).unwrap(),
        ),
    ];
    let metadata = Some(HashMap::from([
        ("meganeura_checkpoint_format".to_string(), "2".to_string()),
        ("adam_step".to_string(), "1".to_string()),
    ]));
    let bytes = safetensors::tensor::serialize(views, &metadata).unwrap();
    let path = TestFile::new();
    std::fs::write(&path.0, bytes).unwrap();

    let error = session.load_checkpoint(&path.0).unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
    assert!(error.to_string().contains("adam_m.w"));
    assert!(error.to_string().contains("expected 16"));
    assert_eq!(snapshot(&session), before);
}

#[test]
fn checkpoint_rejects_invalid_step_metadata() {
    let mut session = training_session();
    let before = snapshot(&session);
    let parameter = vec![0_u8; 4 * 4];
    let views = vec![(
        "w".to_string(),
        TensorView::new(Dtype::F32, vec![4], &parameter).unwrap(),
    )];
    let metadata = Some(HashMap::from([
        ("meganeura_checkpoint_format".to_string(), "2".to_string()),
        ("adam_step".to_string(), "not-a-number".to_string()),
    ]));
    let bytes = safetensors::tensor::serialize(views, &metadata).unwrap();
    let path = TestFile::new();
    std::fs::write(&path.0, bytes).unwrap();

    let error = session.load_checkpoint(&path.0).unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
    assert!(error.to_string().contains("adam_step"));
    assert_eq!(snapshot(&session), before);
}

#[test]
fn logical_checkpoint_rejections_never_write_or_allocate_moments() {
    let mut source = training_session();
    source.set_adam(0.02, 0.9, 0.999, 1e-8);
    source.step();
    let saved = TestFile::new();
    source.save_checkpoint(&saved.0).unwrap();
    let bytes = std::fs::read(&saved.0).unwrap();
    let original = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let (_, header) = safetensors::SafeTensors::read_metadata(&bytes).unwrap();
    for initialized in [false, true] {
        let mut target = training_session();
        target.set_parameter("w", &[3.0; 4]);
        target.set_adam_step_count(11);
        if initialized {
            target.set_adam(0.005, 0.9, 0.999, 1e-8);
            target.step();
            target.wait();
        }
        let before = snapshot(&target);
        for case in 0..5 {
            let mut tensors: Vec<_> = original
                .tensors()
                .into_iter()
                .map(|(name, t)| (name, t.dtype(), t.shape().to_vec(), t.data().to_vec()))
                .collect();
            let mut metadata = header.metadata().clone().unwrap();
            match case {
                0 => {
                    metadata.insert("adam_step".into(), "invalid".into());
                }
                1 => {
                    let t = tensors.iter_mut().find(|t| t.0 == "adam_v.w").unwrap();
                    t.2 = vec![5];
                    t.3.resize(20, 0);
                }
                2 => {
                    tensors.iter_mut().find(|t| t.0 == "w").unwrap().2 = vec![1, 4];
                }
                3 => {
                    tensors.retain(|t| t.0 != "adam_v.w");
                }
                _ => {
                    metadata.insert("meganeura_checkpoint_format".into(), "99".into());
                }
            }
            let views: Vec<_> = tensors
                .iter()
                .map(|t| {
                    (
                        t.0.clone(),
                        TensorView::new(t.1, t.2.clone(), &t.3).unwrap(),
                    )
                })
                .collect();
            let path = TestFile::new();
            std::fs::write(
                &path.0,
                safetensors::tensor::serialize(views, &Some(metadata)).unwrap(),
            )
            .unwrap();
            assert_eq!(
                target.load_checkpoint(&path.0).unwrap_err().kind(),
                std::io::ErrorKind::InvalidData
            );
            assert_eq!(
                snapshot(&target),
                before,
                "case {case}, initialized={initialized}"
            );
        }
    }
}

#[test]
fn checkpoint_without_moments_stays_lazy_or_resets_existing_moments() {
    let mut source = training_session();
    let path = TestFile::new();
    source.save_checkpoint(&path.0).unwrap();
    let mut fresh = training_session();
    fresh.load_checkpoint(&path.0).unwrap();
    assert_eq!(fresh.memory_summary().adam_state_bytes, 0);
    let mut reused = training_session();
    reused.set_adam(0.001, 0.9, 0.999, 1e-8);
    reused.step();
    reused.load_checkpoint(&path.0).unwrap();
    assert_eq!(reused.adam_step_count(), 0);
    assert_eq!(
        reused.read_adam_states(&["w"]),
        vec![(vec![0.0; 4], vec![0.0; 4])]
    );
    fresh.set_adam(0.001, 0.9, 0.999, 1e-8);
    for s in [&mut fresh, &mut reused] {
        s.step();
        s.wait();
    }
    assert_eq!(snapshot(&fresh), snapshot(&reused));
}

#[test]
fn training_checkpoint_loads_into_inference_without_optimizer_allocation() {
    let mut source = training_session();
    source.set_adam(0.001, 0.9, 0.999, 1e-8);
    source.step();
    let file = TestFile::new();
    source.save_checkpoint(&file.0).unwrap();
    let (mut target, _) = meganeura::build(
        &graph(),
        SessionConfig {
            mode: Mode::Inference,
            runtime: SessionOptions {
                coop: CoopPolicy::Disabled,
                ..Default::default()
            },
            ..Default::default()
        },
    );
    target.load_checkpoint(&file.0).unwrap();
    assert_eq!(target.read_params(&["w"]), source.read_params(&["w"]));
    assert_eq!(target.memory_summary().adam_state_bytes, 0);
}

fn padded_session(padding: usize, debug: bool) -> Session {
    let mut graph = Graph::new();
    let weight = graph.parameter("w", &[2, 3]);
    let loss = graph.mean_all(weight);
    graph.set_outputs(vec![loss]);
    let mut plan = meganeura::compile::compile(&meganeura::autodiff::differentiate(&graph));
    let (parameter, gradient) = plan.param_grad_pairs[0];
    plan.buffers[parameter.0 as usize] += padding;
    plan.buffers[gradient.0 as usize] += padding;
    Session::with_context_opts(
        plan,
        std::sync::Arc::new(meganeura::init_gpu_context().unwrap()),
        SessionOptions {
            debug,
            coop: CoopPolicy::Disabled,
            ..Default::default()
        },
    )
}

#[test]
fn logical_checkpoint_round_trips_across_padding_and_preserves_next_update() {
    for (from, to) in [(64, 0), (0, 128), (64, 128)] {
        let mut source = padded_session(from, true);
        source.set_parameter("w", &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        source.write_adam_m("w", &[0.125; 6]);
        source.write_adam_v("w", &[0.25; 6]);
        source.set_adam_step_count(7);
        let file = TestFile::new();
        source.save_checkpoint(&file.0).unwrap();
        let bytes = std::fs::read(&file.0).unwrap();
        let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
        for name in ["w", "adam_m.w", "adam_v.w"] {
            let tensor = tensors.tensor(name).unwrap();
            assert_eq!(tensor.shape(), &[2, 3]);
            assert_eq!(tensor.data().len(), 24);
        }
        let mut target = padded_session(to, false);
        target.set_parameter("w", &vec![-99.0; (24 + to) / 4]);
        target.load_checkpoint(&file.0).unwrap();
        assert_eq!(target.param_size("w"), Some(6));
        assert_eq!(target.read_params(&["w"]), source.read_params(&["w"]));
        assert_eq!(
            target.read_adam_states(&["w"]),
            source.read_adam_states(&["w"])
        );
        let mut physical = vec![1.0; (24 + to) / 4];
        target.read_buffer(target.param_buffer("w").unwrap(), &mut physical);
        assert!(physical[6..].iter().all(|v| *v == 0.0));
        for s in [&mut source, &mut target] {
            s.set_adam(0.001, 0.9, 0.999, 1e-8);
            s.step();
            s.wait();
        }
        assert_eq!(target.adam_step_count(), 8);
        assert_eq!(target.read_params(&["w"]), source.read_params(&["w"]));
        assert_eq!(
            target.read_adam_states(&["w"]),
            source.read_adam_states(&["w"])
        );
    }
}

#[test]
fn quantized_checkpoint_preserves_storage_identity_and_payload() {
    for q8 in [false, true] {
        let mut graph = Graph::new();
        let x = graph.input("x", &[2, 32]);
        let w = if q8 {
            graph.parameter_q8("w", &[32, 3])
        } else {
            graph.parameter_q4("w", &[32, 3])
        };
        let y = graph.matmul(x, w);
        graph.set_outputs(vec![y]);
        let make = || {
            meganeura::build(
                &graph,
                SessionConfig {
                    mode: Mode::Inference,
                    ..Default::default()
                },
            )
            .0
        };
        let mut source = make();
        let weights: Vec<_> = (0..96).map(|i| ((i % 17) as f32 - 8.0) * 0.1).collect();
        source.set_parameter("w", &weights);
        source.set_input("x", &[0.25; 64]);
        source.step();
        source.wait();
        let expected = source.read_output(6);
        let file = TestFile::new();
        source.save_checkpoint(&file.0).unwrap();
        let bytes = std::fs::read(&file.0).unwrap();
        let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
        assert_eq!(tensors.tensor("w").unwrap().dtype(), Dtype::U8);
        assert_eq!(
            tensors.tensor("w").unwrap().data().len(),
            if q8 { 108 } else { 60 }
        );
        let mut target = make();
        target.load_checkpoint(&file.0).unwrap();
        assert_eq!(target.param_size("w"), Some(96));
        target.set_input("x", &[0.25; 64]);
        target.step();
        target.wait();
        assert_eq!(target.read_output(6), expected);
    }
}

#[test]
fn checkpoint_refreshes_derived_f16_staging_before_later_source_updates() {
    let mut graph = Graph::new();
    graph.parameter("a", &[2, 1]);
    graph.parameter("b", &[2, 1]);
    let w = graph.parameter_f16("joined", &[2, 2]);
    let x = graph.input("x", &[1, 2]);
    let y = graph.matmul(x, w);
    graph.set_outputs(vec![y]);
    let mut plan = meganeura::compile::compile(&graph);
    let joined = plan
        .param_buffers
        .iter()
        .find(|p| p.0 == "joined")
        .unwrap()
        .1;
    plan.derived_params.push((
        joined,
        vec![("a".into(), 1), ("b".into(), 1)],
        meganeura::graph::ParamTransform::HorizontalConcat,
    ));
    let mut source = Session::new(plan.clone());
    source.set_parameter("a", &[1.0, 2.0]);
    source.set_parameter("b", &[3.0, 4.0]);
    let file = TestFile::new();
    source.save_checkpoint(&file.0).unwrap();
    let mut target = Session::new(plan);
    target.set_parameter("a", &[10.0, 20.0]);
    target.set_parameter("b", &[30.0, 40.0]);
    target.load_checkpoint(&file.0).unwrap();
    target.set_parameter("a", &[5.0, 6.0]);
    target.set_input("x", &[1.0; 2]);
    target.step();
    target.wait();
    assert_eq!(target.read_output(2), vec![11.0, 7.0]);
}
