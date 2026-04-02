#![allow(dead_code, clippy::too_many_arguments)]
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

#[allow(dead_code, clippy::too_many_arguments)]
use meganeura::{build_session, graph::Op};

// ---------------------------------------------------------------------------
// Inlined SmolVLA model (with SmolVLM2 types)
// ---------------------------------------------------------------------------
mod smolvla {
    use meganeura::graph::Graph;

    pub struct VisionConfig {
        pub image_size: usize,
        pub patch_size: usize,
        pub hidden_size: usize,
        pub num_attention_heads: u32,
        pub num_hidden_layers: usize,
        pub intermediate_size: usize,
        pub layer_norm_eps: f32,
    }

    pub struct TextConfig {
        pub vocab_size: usize,
        pub hidden_size: usize,
        pub num_hidden_layers: usize,
        pub num_attention_heads: u32,
        pub num_key_value_heads: u32,
        pub intermediate_size: usize,
        pub rms_norm_eps: f32,
        pub rope_theta: f32,
    }

    pub struct SmolVLM2Config {
        pub vision: VisionConfig,
        pub text: TextConfig,
        pub scale_factor: usize,
    }

    pub struct ExpertConfig {
        pub hidden_size: usize,
        pub num_layers: usize,
        pub num_attention_heads: u32,
        pub num_key_value_heads: u32,
        pub head_dim: u32,
        pub intermediate_size: usize,
        pub rms_norm_eps: f32,
        pub self_attn_every_n_layers: usize,
    }

    impl ExpertConfig {
        pub fn kv_dim(&self) -> usize {
            self.num_key_value_heads as usize * self.head_dim as usize
        }
    }

    pub struct SmolVLAConfig {
        pub vlm: SmolVLM2Config,
        pub expert: ExpertConfig,
        pub max_action_dim: usize,
        pub max_state_dim: usize,
        pub chunk_size: usize,
        pub num_steps: usize,
        pub num_vlm_layers: usize,
    }

    impl SmolVLAConfig {
        pub fn small_test() -> Self {
            Self {
                vlm: SmolVLM2Config {
                    vision: VisionConfig {
                        image_size: 32,
                        patch_size: 16,
                        hidden_size: 64,
                        num_attention_heads: 2,
                        num_hidden_layers: 2,
                        intermediate_size: 128,
                        layer_norm_eps: 1e-6,
                    },
                    text: TextConfig {
                        vocab_size: 256,
                        hidden_size: 64,
                        num_hidden_layers: 2,
                        num_attention_heads: 2,
                        num_key_value_heads: 2,
                        intermediate_size: 128,
                        rms_norm_eps: 1e-5,
                        rope_theta: 10000.0,
                    },
                    scale_factor: 1,
                },
                expert: ExpertConfig {
                    hidden_size: 64,
                    num_layers: 2,
                    num_attention_heads: 2,
                    num_key_value_heads: 2,
                    head_dim: 32,
                    intermediate_size: 128,
                    rms_norm_eps: 1e-5,
                    self_attn_every_n_layers: 2,
                },
                max_action_dim: 8,
                max_state_dim: 8,
                chunk_size: 4,
                num_steps: 2,
                num_vlm_layers: 2,
            }
        }

        pub fn smolvla_base() -> Self {
            Self {
                vlm: SmolVLM2Config {
                    vision: VisionConfig {
                        image_size: 512,
                        patch_size: 16,
                        hidden_size: 768,
                        num_attention_heads: 12,
                        num_hidden_layers: 12,
                        intermediate_size: 3072,
                        layer_norm_eps: 1e-6,
                    },
                    text: TextConfig {
                        vocab_size: 49280,
                        hidden_size: 960,
                        num_hidden_layers: 32,
                        num_attention_heads: 15,
                        num_key_value_heads: 5,
                        intermediate_size: 2560,
                        rms_norm_eps: 1e-5,
                        rope_theta: 100000.0,
                    },
                    scale_factor: 4,
                },
                expert: ExpertConfig {
                    hidden_size: 720,
                    num_layers: 16,
                    num_attention_heads: 15,
                    num_key_value_heads: 5,
                    head_dim: 64,
                    intermediate_size: 2048,
                    rms_norm_eps: 1e-5,
                    self_attn_every_n_layers: 2,
                },
                max_action_dim: 32,
                max_state_dim: 32,
                chunk_size: 50,
                num_steps: 10,
                num_vlm_layers: 16,
            }
        }
    }

    pub fn build_action_expert_training(
        config: &SmolVLAConfig,
        action_seq_len: usize,
        vlm_seq_len: usize,
    ) -> Graph {
        let mut g = Graph::new();
        let expert = &config.expert;
        let expert_hidden = expert.hidden_size;
        let kv_dim = expert.kv_dim();
        let eps = expert.rms_norm_eps;

        let num_heads = expert.num_attention_heads;
        let num_kv_heads = expert.num_key_value_heads;
        let hd = expert.head_dim;
        let q_dim = (num_heads * hd) as usize;
        let kv_dim_full = (num_kv_heads * hd) as usize;

        let noisy_actions = g.input("noisy_actions", &[action_seq_len, config.max_action_dim]);
        let timestep = g.input("timestep", &[1, expert_hidden * 2]);

        let action_in_w = g.parameter(
            "model.action_in_proj.weight",
            &[config.max_action_dim, expert_hidden],
        );
        let action_in_b = g.parameter("model.action_in_proj.bias", &[expert_hidden]);
        let mut x = g.matmul(noisy_actions, action_in_w);
        x = g.bias_add(x, action_in_b);

        let time_in_w = g.parameter(
            "model.action_time_mlp_in.weight",
            &[expert_hidden * 2, expert_hidden],
        );
        let time_in_b = g.parameter("model.action_time_mlp_in.bias", &[expert_hidden]);
        let time_out_w = g.parameter(
            "model.action_time_mlp_out.weight",
            &[expert_hidden, expert_hidden],
        );
        let time_out_b = g.parameter("model.action_time_mlp_out.bias", &[expert_hidden]);
        let time_h = g.matmul(timestep, time_in_w);
        let time_h = g.bias_add(time_h, time_in_b);
        let time_h = g.silu(time_h);
        let time_h = g.matmul(time_h, time_out_w);
        let time_embed = g.bias_add(time_h, time_out_b);
        x = g.broadcast_add(x, time_embed);

        for i in 0..expert.num_layers {
            let prefix = format!("model.vlm_with_expert.lm_expert.layers.{}", i);
            let is_cross_attn = i % expert.self_attn_every_n_layers != 0;

            let ln1_w = g.parameter(
                &format!("{}.input_layernorm.weight", prefix),
                &[expert_hidden],
            );
            let h = g.rms_norm(x, ln1_w, eps);

            if is_cross_attn {
                let wq = g.parameter(
                    &format!("{}.self_attn.q_proj.weight", prefix),
                    &[expert_hidden, q_dim],
                );
                let q = g.matmul(h, wq);

                let vlm_kv = g.input(&format!("vlm_kv_layer_{}", i), &[vlm_seq_len, kv_dim]);
                let wk = g.parameter(
                    &format!("{}.self_attn.k_proj.weight", prefix),
                    &[kv_dim, kv_dim_full],
                );
                let wv = g.parameter(
                    &format!("{}.self_attn.v_proj.weight", prefix),
                    &[kv_dim, kv_dim_full],
                );
                let k = g.matmul(vlm_kv, wk);
                let v = g.matmul(vlm_kv, wv);

                let attn = g.multi_head_attn(q, k, v, num_heads, num_kv_heads, hd, true);

                let wo = g.parameter(
                    &format!("{}.self_attn.o_proj.weight", prefix),
                    &[q_dim, expert_hidden],
                );
                let attn_out = g.matmul(attn, wo);
                x = g.add(x, attn_out);
            } else {
                let wq = g.parameter(
                    &format!("{}.self_attn.q_proj.weight", prefix),
                    &[expert_hidden, q_dim],
                );
                let wk = g.parameter(
                    &format!("{}.self_attn.k_proj.weight", prefix),
                    &[expert_hidden, kv_dim_full],
                );
                let wv = g.parameter(
                    &format!("{}.self_attn.v_proj.weight", prefix),
                    &[expert_hidden, kv_dim_full],
                );
                let q = g.matmul(h, wq);
                let k = g.matmul(h, wk);
                let v = g.matmul(h, wv);

                let attn = g.multi_head_attn(q, k, v, num_heads, num_kv_heads, hd, false);

                let wo = g.parameter(
                    &format!("{}.self_attn.o_proj.weight", prefix),
                    &[q_dim, expert_hidden],
                );
                let attn_out = g.matmul(attn, wo);
                x = g.add(x, attn_out);
            }

            let ln2_w = g.parameter(
                &format!("{}.post_attention_layernorm.weight", prefix),
                &[expert_hidden],
            );
            let h = g.rms_norm(x, ln2_w, eps);

            let w_gate = g.parameter(
                &format!("{}.mlp.gate_proj.weight", prefix),
                &[expert_hidden, expert.intermediate_size],
            );
            let w_up = g.parameter(
                &format!("{}.mlp.up_proj.weight", prefix),
                &[expert_hidden, expert.intermediate_size],
            );
            let w_down = g.parameter(
                &format!("{}.mlp.down_proj.weight", prefix),
                &[expert.intermediate_size, expert_hidden],
            );
            let gate = g.matmul(h, w_gate);
            let up = g.matmul(h, w_up);
            let gate_up = g.swiglu(gate, up);
            let ffn_out = g.matmul(gate_up, w_down);
            x = g.add(x, ffn_out);
        }

        let action_out_w = g.parameter(
            "model.action_out_proj.weight",
            &[expert_hidden, config.max_action_dim],
        );
        let action_out_b = g.parameter("model.action_out_proj.bias", &[config.max_action_dim]);
        let out = g.matmul(x, action_out_w);
        let out = g.bias_add(out, action_out_b);

        let target = g.input("target_actions", &[action_seq_len, config.max_action_dim]);
        let neg_target = g.neg(target);
        let diff = g.add(out, neg_target);
        let sq_diff = g.mul(diff, diff);
        let loss = g.mean_all(sq_diff);
        g.set_outputs(vec![loss]);
        g
    }
}
use smolvla::SmolVLAConfig;

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
