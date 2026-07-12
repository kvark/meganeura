use std::collections::HashMap;

use meganeura::{Graph, build_session};
use safetensors::tensor::{Dtype, TensorView};

fn training_session() -> meganeura::Session {
    let mut graph = Graph::new();
    let x = graph.input("x", &[1, 2]);
    let w = graph.parameter("w", &[2, 2]);
    let y = graph.matmul(x, w);
    let loss = graph.mean_all(y);
    graph.set_outputs(vec![loss]);
    build_session(&graph)
}

#[test]
fn checkpoint_rejects_oversized_adam_state() {
    let mut session = training_session();
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
    let path = std::env::temp_dir().join("meganeura_oversized_adam.safetensors");
    std::fs::write(&path, bytes).unwrap();

    let error = session.load_checkpoint(&path).unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
    assert!(error.to_string().contains("adam_m.w"));
    assert!(error.to_string().contains("expected 16"));
    let _ = std::fs::remove_file(path);
}

#[test]
fn checkpoint_rejects_invalid_step_metadata() {
    let mut session = training_session();
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
    let path = std::env::temp_dir().join("meganeura_bad_adam_step.safetensors");
    std::fs::write(&path, bytes).unwrap();

    let error = session.load_checkpoint(&path).unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
    assert!(error.to_string().contains("adam_step"));
    let _ = std::fs::remove_file(path);
}
