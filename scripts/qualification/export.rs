//! Full tensors from the unmodified benchmark graphs, outside timed regions.
use meganeura::{Graph, Session};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{cell::RefCell, path::PathBuf};

thread_local! {
    static OUTPUT_SHAPES: RefCell<Vec<Vec<usize>>> = const { RefCell::new(Vec::new()) };
}

pub fn set_graph(graph: &Graph) {
    OUTPUT_SHAPES.with_borrow_mut(|shapes| {
        *shapes = graph
            .outputs()
            .iter()
            .map(|&id| graph.node(id).ty.shape.clone())
            .collect();
    });
}

fn root() -> PathBuf {
    PathBuf::from(std::env::var_os("QUALIFICATION_DIR").expect("QUALIFICATION_DIR"))
}

fn tensor(values: &[f32], shape: &[usize], dtype: &str) -> Value {
    assert_eq!(values.len(), shape.iter().product::<usize>());
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let hash = hex::encode(Sha256::digest(&bytes));
    let file = format!("blobs/{hash}.bin");
    let path = root().join(&file);
    if !path.exists() {
        std::fs::write(path, &bytes).unwrap();
    }
    json!({"file": file, "sha256": hash, "shape": shape, "dtype": dtype, "elements": values.len()})
}

pub fn before(session: &Session, phase: &str) -> Value {
    assert_eq!(std::env::var("INFERENA_STRICT").as_deref(), Ok("1"));
    assert!(!meganeura::config::TUNE.bool_or(true));
    std::fs::create_dir_all(root().join("blobs")).unwrap();
    let plan = session.plan();
    let mut parameters = serde_json::Map::new();
    for (name, len) in crate::qualification::source_parameters(session) {
        let buffer = plan.param_buffers.iter().find(|p| p.0 == name).unwrap().1;
        let ty = &plan.param_types[&buffer];
        assert_eq!(ty.num_elements(), len);
        let mut values = vec![0.0; len];
        session.read_param(&name, &mut values);
        parameters.insert(name, tensor(&values, &ty.shape, "F32"));
    }
    let mut inputs = serde_json::Map::new();
    for (name, buffer) in &plan.input_buffers {
        let ty = &plan.input_types[buffer];
        let mut values = vec![0.0; ty.num_elements()];
        session.read_buffer(*buffer, &mut values);
        inputs.insert(
            name.clone(),
            tensor(&values, &ty.shape, &format!("{:?}", ty.dtype)),
        );
    }
    json!({"phase": phase, "parameters": parameters, "inputs": inputs,
        "device": session.device_information().device_name,
        "precision": "strict-f32", "tuning": false})
}

pub fn after(session: &Session, phase: &str, manifest: &mut Value) {
    if phase == "training" {
        manifest["loss"] = json!(session.read_loss());
        let mut gradients = serde_json::Map::new();
        for (name, values) in crate::qualification::gradients(session).unwrap() {
            let shape: Vec<usize> =
                serde_json::from_value(manifest["parameters"][&name]["shape"].clone()).unwrap();
            gradients.insert(name, tensor(&values, &shape, "F32"));
        }
        manifest["gradients"] = json!(gradients);
    } else {
        let shapes = OUTPUT_SHAPES.with_borrow(Clone::clone);
        let mut outputs = Vec::new();
        for (i, shape) in shapes.iter().enumerate() {
            let mut values = vec![0.0; shape.iter().product()];
            session.read_output_by_index(i, &mut values);
            outputs.push(tensor(&values, shape, "F32"));
        }
        manifest["outputs"] = json!(outputs);
    }
    let path = root().join(format!("{phase}.json"));
    std::fs::write(path, serde_json::to_vec_pretty(manifest).unwrap()).unwrap();
    eprintln!("[qualification] exported {phase}");
}
