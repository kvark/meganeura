/// Gradient check: run one meganeura forward+backward pass and emit parameter
/// gradients as JSON for comparison with a reference implementation (PyTorch).
///
/// By default uses SmolVLAConfig::smolvla_base() (full production config).
/// Use --small for the tiny smoke-test config.
///
/// Output (stdout): JSON with loss, and per-parameter gradient norms + first-N
/// flat elements (in meganeura's storage order, i.e. [in, out] for matmuls).
///
/// The companion script bench/grad_check_pytorch.py reads this JSON and runs
/// the equivalent PyTorch computation, reporting per-parameter cosine similarity
/// and relative error.
///
/// Usage:
///   cargo run --release --example grad_check [-- --small] [--vlm-seq 4] [--sample 32]
use std::collections::HashMap;

use meganeura::{
    build_session,
    graph::Op,
    models::smolvla::{self, SmolVLAConfig},
};

fn main() {
    env_logger::init();

    let mut args = std::env::args().skip(1);
    let mut use_small = false;
    let mut vlm_seq_len: usize = 4;
    let mut sample_n: usize = 32;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--small" => use_small = true,
            "--vlm-seq" => vlm_seq_len = args.next().expect("--vlm-seq value").parse().unwrap(),
            "--sample" => sample_n = args.next().expect("--sample value").parse().unwrap(),
            other => {
                eprintln!("unknown arg: {}", other);
                std::process::exit(1);
            }
        }
    }

    let config = if use_small {
        SmolVLAConfig::small_test()
    } else {
        SmolVLAConfig::smolvla_base()
    };
    let action_seq_len = config.chunk_size;
    let expert_hidden = config.expert.hidden_size;
    let kv_dim = config.expert.kv_dim();

    eprintln!(
        "config: hidden={} layers={} heads={}/{} head_dim={} chunk={} vlm_seq={}",
        config.expert.hidden_size,
        config.expert.num_layers,
        config.expert.num_attention_heads,
        config.expert.num_key_value_heads,
        config.expert.head_dim,
        action_seq_len,
        vlm_seq_len,
    );

    // Build graph and collect parameter shapes before session compilation
    eprintln!("building training graph...");
    let g = smolvla::build_action_expert_training(&config, action_seq_len, vlm_seq_len);

    // Extract param name → shape from graph nodes
    let mut param_shapes: HashMap<String, Vec<usize>> = HashMap::new();
    for node in g.nodes() {
        if let Op::Parameter { name } = &node.op {
            param_shapes.insert(name.clone(), node.ty.shape.clone());
        }
    }

    eprintln!("compiling session...");
    let mut session = build_session(&g);

    // Fill shapes for derived params (e.g. SwiGLU concat "gate+up") from buffer sizes.
    // These aren't in the original graph, so param_shapes doesn't have them.
    for (name, buf_ref) in session.plan().param_buffers.iter() {
        if !param_shapes.contains_key(name) {
            let n = session.plan().buffers[buf_ref.0 as usize] / 4;
            // Derived concat weights are 2D [in_dim, concat_out_dim]; infer shape
            // from the first component's shape.
            if name.contains('+') {
                let first = name.split('+').next().unwrap();
                if let Some(first_shape) = param_shapes.get(first) {
                    if first_shape.len() == 2 {
                        let in_dim = first_shape[0];
                        let out_dim = n / in_dim;
                        param_shapes.insert(name.clone(), vec![in_dim, out_dim]);
                        continue;
                    }
                }
            }
            param_shapes.insert(name.clone(), vec![n]);
        }
    }

    // Initialize: sin(element_idx * 0.01 + 1.0) * 0.1, same as bench_smolvla_train
    for (name, buf_ref) in session.plan().param_buffers.clone() {
        let n = session.plan().buffers[buf_ref.0 as usize] / 4;
        let data: Vec<f32> = (0..n)
            .map(|i| (i as f32 * 0.01 + 1.0).sin() * 0.1)
            .collect();
        session.set_parameter(&name, &data);
    }

    // Inputs: same sin/cos patterns as bench_smolvla_train.rs
    let noisy_actions: Vec<f32> = (0..action_seq_len * config.max_action_dim)
        .map(|i| (i as f32 * 0.01).sin() * 0.1)
        .collect();
    let timestep: Vec<f32> = (0..expert_hidden * 2)
        .map(|i| (i as f32 * 0.005).cos() * 0.1)
        .collect();
    let vlm_kv: Vec<f32> = (0..vlm_seq_len * kv_dim)
        .map(|i| (i as f32 * 0.002).sin() * 0.05)
        .collect();
    let target_actions = vec![0.0_f32; action_seq_len * config.max_action_dim];

    session.set_input("noisy_actions", &noisy_actions);
    session.set_input("timestep", &timestep);
    for i in 0..config.expert.num_layers {
        if i % config.expert.self_attn_every_n_layers != 0 {
            session.set_input(&format!("vlm_kv_layer_{}", i), &vlm_kv);
        }
    }
    session.set_input("target_actions", &target_actions);

    eprintln!("running forward + backward...");
    session.step();
    session.wait();

    let loss = session.read_loss();
    eprintln!("loss = {:.8}", loss);

    // Collect gradients
    let param_buffers: HashMap<String, meganeura::compile::BufferRef> =
        session.plan().param_buffers.iter().cloned().collect();
    let grad_map: HashMap<meganeura::compile::BufferRef, meganeura::compile::BufferRef> =
        session.plan().param_grad_pairs.iter().cloned().collect();

    // Ordered parameter list (same as param_buffers order, which matches graph node order)
    let param_names: Vec<String> = session
        .plan()
        .param_buffers
        .iter()
        .map(|(n, _)| n.clone())
        .collect();

    // ---- JSON output ----
    println!("{{");
    println!("  \"config\": {{");
    println!("    \"hidden_size\": {},", config.expert.hidden_size);
    println!("    \"num_layers\": {},", config.expert.num_layers);
    println!("    \"num_heads\": {},", config.expert.num_attention_heads);
    println!(
        "    \"num_kv_heads\": {},",
        config.expert.num_key_value_heads
    );
    println!("    \"head_dim\": {},", config.expert.head_dim);
    println!(
        "    \"intermediate_size\": {},",
        config.expert.intermediate_size
    );
    println!("    \"action_dim\": {},", config.max_action_dim);
    println!("    \"chunk_size\": {},", action_seq_len);
    println!("    \"vlm_seq_len\": {},", vlm_seq_len);
    println!(
        "    \"self_attn_every_n\": {},",
        config.expert.self_attn_every_n_layers
    );
    println!("    \"rms_norm_eps\": {}", config.expert.rms_norm_eps);
    println!("  }},");
    println!("  \"loss\": {:.8},", loss);
    println!("  \"param_grads\": {{");

    for (pi, name) in param_names.iter().enumerate() {
        let shape = param_shapes.get(name).cloned().unwrap_or_default();
        let param_buf = param_buffers[name];
        let n = session.plan().buffers[param_buf.0 as usize] / 4;

        let (norm, sample) = if let Some(&grad_buf) = grad_map.get(&param_buf) {
            let mut grad = vec![0.0f32; n];
            session.read_buffer(grad_buf, &mut grad);
            let norm = grad.iter().map(|v| v * v).sum::<f32>().sqrt();
            let sample: Vec<f32> = grad.iter().copied().take(sample_n).collect();
            (norm, sample)
        } else {
            (0.0, vec![])
        };

        let shape_str: Vec<String> = shape.iter().map(|s| s.to_string()).collect();
        let sample_str: Vec<String> = sample.iter().map(|v| format!("{:.8e}", v)).collect();
        let comma = if pi < param_names.len() - 1 { "," } else { "" };

        println!("    \"{}\": {{", name);
        println!("      \"shape\": [{}],", shape_str.join(", "));
        println!("      \"norm\": {:.8e},", norm);
        println!("      \"sample\": [{}]", sample_str.join(", "));
        print!("    }}{}", comma);
        println!();
    }
    println!("  }}");
    println!("}}");
}
