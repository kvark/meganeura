//! Gemma 4 text decoder: GGUF weights, hybrid SWA/global attention, GeGLU,
//! per-layer embeddings, and KV sharing.
//!
//! Tensor names follow llama.cpp's GGUF layout (`blk.%d.attn_q.weight`, …).
//! The graph is the text path only — vision/audio encoders stay out.

use crate::graph::{DType, Graph, NodeId};
use crate::load::gguf::{GgufError, GgufModel, GgufTensor};

/// Hyperparameters read from a Gemma 4 GGUF, plus the per-layer extras
/// that GGUF stores as arrays rather than a single value.
#[derive(Clone, Debug)]
pub struct Gemma4Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub rms_norm_eps: f32,
    pub sliding_window: u32,
    pub n_embd_per_layer: usize,
    pub n_kv_from_start: usize,
    pub logit_softcap: f32,
    pub rope_theta_global: f32,
    pub rope_theta_swa: f32,
    pub head_dim_global: u32,
    pub head_dim_swa: u32,
    /// `true` = sliding-window layer. Length = `num_hidden_layers`.
    pub is_swa: Vec<bool>,
    pub ffn_size: Vec<usize>,
}

impl Gemma4Config {
    pub fn from_gguf(model: &GgufModel) -> Result<Self, String> {
        let arch = model.architecture().unwrap_or("?");
        if arch != "gemma4" {
            return Err(format!("expected architecture gemma4, got {arch}"));
        }
        let u = |key: &str| {
            model
                .arch_key(key)
                .and_then(|v| v.as_u64())
                .ok_or_else(|| format!("missing gemma4.{key}"))
        };
        let f = |key: &str| {
            model
                .arch_key(key)
                .and_then(|v| v.as_f64())
                .ok_or_else(|| format!("missing gemma4.{key}"))
        };
        let layers = u("block_count")? as usize;
        let pattern = model
            .arch_key("attention.sliding_window_pattern")
            .and_then(|v| v.as_array())
            .ok_or("missing sliding_window_pattern")?;
        if pattern.len() != layers {
            return Err(format!(
                "sliding_window_pattern has {} entries, block_count is {layers}",
                pattern.len()
            ));
        }
        let is_swa: Vec<bool> = pattern
            .iter()
            .map(|v| match v {
                crate::load::gguf::GgufValue::Bool(b) => Ok(*b),
                _ => v.as_u64().map(|n| n != 0).ok_or("pattern entry"),
            })
            .collect::<Result<_, _>>()
            .map_err(|_| "sliding_window_pattern entries are not booleans".to_string())?;
        let ffn_size = match model.arch_key("feed_forward_length") {
            Some(v) if v.as_array().is_some() => v
                .as_array()
                .unwrap()
                .iter()
                .map(|e| e.as_u64().map(|n| n as usize))
                .collect::<Option<Vec<_>>>()
                .ok_or("feed_forward_length array")?,
            Some(v) => {
                let n = v.as_u64().ok_or("feed_forward_length")? as usize;
                vec![n; layers]
            }
            None => return Err("missing feed_forward_length".into()),
        };
        if ffn_size.len() != layers {
            return Err("feed_forward_length length mismatch".into());
        }
        let shared = u("attention.shared_kv_layers").unwrap_or(0) as usize;
        Ok(Self {
            vocab_size: model
                .tensors
                .get("token_embd.weight")
                .and_then(|t| t.dims.get(1).copied())
                .unwrap_or(262_144),
            hidden_size: u("embedding_length")? as usize,
            num_hidden_layers: layers,
            num_attention_heads: u("attention.head_count")? as u32,
            num_key_value_heads: u("attention.head_count_kv")? as u32,
            rms_norm_eps: f("attention.layer_norm_rms_epsilon")? as f32,
            sliding_window: u("attention.sliding_window")? as u32,
            n_embd_per_layer: u("embedding_length_per_layer_input").unwrap_or(0) as usize,
            n_kv_from_start: layers.saturating_sub(shared).max(1),
            logit_softcap: f("final_logit_softcapping").unwrap_or(0.0) as f32,
            rope_theta_global: f("rope.freq_base").unwrap_or(1_000_000.0) as f32,
            rope_theta_swa: f("rope.freq_base_swa").unwrap_or(10_000.0) as f32,
            head_dim_global: u("attention.key_length").unwrap_or(256) as u32,
            head_dim_swa: u("attention.key_length_swa").unwrap_or(256) as u32,
            is_swa,
            ffn_size,
        })
    }

    pub fn has_kv(&self, layer: usize) -> bool {
        layer < self.n_kv_from_start
    }

    pub fn kv_source(&self, layer: usize) -> usize {
        if self.has_kv(layer) {
            layer
        } else {
            layer % self.n_kv_from_start
        }
    }

    pub fn head_dim(&self, layer: usize) -> u32 {
        if self.is_swa[layer] {
            self.head_dim_swa
        } else {
            self.head_dim_global
        }
    }

    pub fn q_dim(&self, layer: usize) -> usize {
        self.num_attention_heads as usize * self.head_dim(layer) as usize
    }

    pub fn kv_dim(&self, layer: usize) -> usize {
        self.num_key_value_heads as usize * self.head_dim(layer) as usize
    }
}

/// Declare a 2-D (or 1-D) parameter whose storage matches the GGUF tensor.
pub fn declare_parameter(
    g: &mut Graph,
    tensor: &GgufTensor,
    name: &str,
) -> Result<NodeId, GgufError> {
    let (k, n) = match tensor.dims.len() {
        1 => (tensor.dims[0], 1),
        2 => (tensor.dims[0], tensor.dims[1]),
        n => {
            return Err(GgufError::BadShape(format!(
                "{name} has {n} dims, expected 1 or 2"
            )));
        }
    };
    let shape: Vec<usize> = if n == 1 && tensor.dims.len() == 1 {
        vec![k]
    } else {
        vec![k, n]
    };
    match tensor.packed_dtype() {
        Ok(DType::Q40) => Ok(g.parameter_q40(name, &shape)),
        Ok(DType::Q8_0) => Ok(g.parameter_q8(name, &shape)),
        Ok(DType::Q4_0) => Ok(g.parameter_q4(name, &shape)),
        Ok(DType::Q4K) => Ok(g.parameter_q4k(name, &shape)),
        Ok(DType::Q6K) => Ok(g.parameter_q6k(name, &shape)),
        Ok(DType::Q5K) => Ok(g.parameter_q5k(name, &shape)),
        Ok(DType::Q3K) => Ok(g.parameter_q3k(name, &shape)),
        Ok(DType::F16) => Ok(g.parameter_f16(name, &shape)),
        Ok(_) | Err(GgufError::UnsupportedPack(_)) => Ok(g.parameter(name, &shape)),
        Err(e) => Err(e),
    }
}

fn require<'a>(model: &'a GgufModel, name: &str) -> Result<&'a GgufTensor, String> {
    model
        .tensors
        .get(name)
        .ok_or_else(|| format!("missing tensor {name}"))
}

/// Single-token decode graph with KV cache.
///
/// Inputs:
/// - `x`: `[1, hidden]` token embedding (CPU-gathered when the table is packed)
/// - `ple`: `[n_layers, n_embd_per_layer]` per-layer token embedding, or absent
///   when the model has no PLE
/// - `kv_pos`: u32 scalar write index
/// - `valid_len`: u32 scalar, `1` for single-token decode
///
/// Returns (logits `[1, vocab]`, k_cache params, v_cache params).
pub fn build_decode_graph(
    g: &mut Graph,
    model: &GgufModel,
    config: &Gemma4Config,
    max_seq_len: usize,
) -> Result<(NodeId, Vec<NodeId>, Vec<NodeId>), String> {
    let hidden = config.hidden_size;
    let eps = config.rms_norm_eps;
    let heads = config.num_attention_heads;
    let kv_heads = config.num_key_value_heads;

    let kv_pos = g.input_u32("kv_pos", &[1]);
    let valid_len = g.input_u32("valid_len", &[1]);
    let mut x = g.input("x", &[1, hidden]);
    x = g.scale(x, (hidden as f32).sqrt());

    let mut ple_all = None;
    if config.n_embd_per_layer > 0 {
        let proj = declare_parameter(
            g,
            require(model, "per_layer_model_proj.weight")?,
            "per_layer_model_proj.weight",
        )
        .map_err(|e| e.to_string())?;
        let proj_norm = g.parameter("per_layer_proj_norm.weight", &[config.n_embd_per_layer]);
        let mut layered = g.matmul(x, proj);
        layered = g.scale(layered, 1.0 / (hidden as f32).sqrt());
        layered = g.reshape(
            layered,
            &[config.num_hidden_layers, config.n_embd_per_layer],
        );
        layered = g.rms_norm(layered, proj_norm, eps);
        let ple_tok = g.input("ple", &[config.num_hidden_layers, config.n_embd_per_layer]);
        layered = g.add(layered, ple_tok);
        layered = g.scale(layered, 1.0 / 2.0f32.sqrt());
        ple_all = Some(layered);
    }

    let mut k_caches = Vec::new();
    let mut v_caches = Vec::new();
    // Indexed by source layer so shared-KV layers can reuse the written cache.
    let mut k_updated = vec![None; config.num_hidden_layers];
    let mut v_updated = vec![None; config.num_hidden_layers];

    for i in 0..config.num_hidden_layers {
        let p = format!("blk.{i}");
        let head_dim = config.head_dim(i);
        let q_dim = config.q_dim(i);
        let kv_dim = config.kv_dim(i);
        let theta = if config.is_swa[i] {
            config.rope_theta_swa
        } else {
            config.rope_theta_global
        };
        let window = if config.is_swa[i] {
            config.sliding_window
        } else {
            0
        };

        let attn_norm = g.parameter(&format!("{p}.attn_norm.weight"), &[hidden]);
        let h = g.rms_norm(x, attn_norm, eps);

        let wq = declare_parameter(
            g,
            require(model, &format!("{p}.attn_q.weight"))?,
            &format!("{p}.attn_q.weight"),
        )
        .map_err(|e| e.to_string())?;
        let mut q = g.matmul(h, wq);
        q = g.reshape(q, &[heads as usize, head_dim as usize]);
        let q_norm = g.parameter(&format!("{p}.attn_q_norm.weight"), &[head_dim as usize]);
        q = g.rms_norm(q, q_norm, eps);
        q = g.reshape(q, &[1, q_dim]);
        q = g.rope_dynamic_offset(q, theta, kv_pos, head_dim);

        let src = config.kv_source(i);
        if config.has_kv(i) {
            let wk = declare_parameter(
                g,
                require(model, &format!("{p}.attn_k.weight"))?,
                &format!("{p}.attn_k.weight"),
            )
            .map_err(|e| e.to_string())?;
            let wv = declare_parameter(
                g,
                require(model, &format!("{p}.attn_v.weight"))?,
                &format!("{p}.attn_v.weight"),
            )
            .map_err(|e| e.to_string())?;
            let mut k = g.matmul(h, wk);
            let mut v = g.matmul(h, wv);
            k = g.reshape(k, &[kv_heads as usize, head_dim as usize]);
            let k_norm = g.parameter(&format!("{p}.attn_k_norm.weight"), &[head_dim as usize]);
            k = g.rms_norm(k, k_norm, eps);
            k = g.reshape(k, &[1, kv_dim]);
            let ones = g.constant(vec![1.0; head_dim as usize], &[head_dim as usize]);
            v = g.reshape(v, &[kv_heads as usize, head_dim as usize]);
            v = g.rms_norm(v, ones, eps);
            v = g.reshape(v, &[1, kv_dim]);
            k = g.rope_dynamic_offset(k, theta, kv_pos, head_dim);

            let k_cache = g.parameter(&format!("kv_cache.layer.{i}.k"), &[max_seq_len, kv_dim]);
            let v_cache = g.parameter(&format!("kv_cache.layer.{i}.v"), &[max_seq_len, kv_dim]);
            k_caches.push(k_cache);
            v_caches.push(v_cache);
            let k_w = g.cache_write(k, k_cache, kv_pos);
            let v_w = g.cache_write(v, v_cache, kv_pos);
            k_updated[i] = Some(k_w);
            v_updated[i] = Some(v_w);
        }

        let k_w = k_updated[src].expect("KV source layer must have a cache");
        let v_w = v_updated[src].expect("KV source layer must have a cache");
        let attn = g.cached_block_attention(
            q, k_w, v_w, kv_pos, valid_len, heads, kv_heads, head_dim, window,
        );
        let wo = declare_parameter(
            g,
            require(model, &format!("{p}.attn_output.weight"))?,
            &format!("{p}.attn_output.weight"),
        )
        .map_err(|e| e.to_string())?;
        let mut attn_out = g.matmul(attn, wo);
        let post_a = g.parameter(&format!("{p}.post_attention_norm.weight"), &[hidden]);
        attn_out = g.rms_norm(attn_out, post_a, eps);
        let attn_res = g.add(attn_out, x);

        let ffn_norm = g.parameter(&format!("{p}.ffn_norm.weight"), &[hidden]);
        let h = g.rms_norm(attn_res, ffn_norm, eps);
        let w_gate = declare_parameter(
            g,
            require(model, &format!("{p}.ffn_gate.weight"))?,
            &format!("{p}.ffn_gate.weight"),
        )
        .map_err(|e| e.to_string())?;
        let w_up = declare_parameter(
            g,
            require(model, &format!("{p}.ffn_up.weight"))?,
            &format!("{p}.ffn_up.weight"),
        )
        .map_err(|e| e.to_string())?;
        let w_down = declare_parameter(
            g,
            require(model, &format!("{p}.ffn_down.weight"))?,
            &format!("{p}.ffn_down.weight"),
        )
        .map_err(|e| e.to_string())?;
        let gate = g.matmul(h, w_gate);
        let up = g.matmul(h, w_up);
        let mut ffn_out = g.geglu(gate, up);
        ffn_out = g.matmul(ffn_out, w_down);
        let post_f = g.parameter(&format!("{p}.post_ffw_norm.weight"), &[hidden]);
        ffn_out = g.rms_norm(ffn_out, post_f, eps);
        let mut cur = g.add(ffn_out, attn_res);

        if let Some(ple) = ple_all {
            let gate_w = declare_parameter(
                g,
                require(model, &format!("{p}.inp_gate.weight"))?,
                &format!("{p}.inp_gate.weight"),
            )
            .map_err(|e| e.to_string())?;
            let proj_w = declare_parameter(
                g,
                require(model, &format!("{p}.proj.weight"))?,
                &format!("{p}.proj.weight"),
            )
            .map_err(|e| e.to_string())?;
            let post = g.parameter(&format!("{p}.post_norm.weight"), &[hidden]);
            let mut sel = vec![0.0f32; config.num_hidden_layers];
            sel[i] = 1.0;
            let sel = g.constant(sel, &[1, config.num_hidden_layers]);
            let ple_i = g.matmul(sel, ple);
            let gated = g.matmul(cur, gate_w);
            let gated = g.gelu(gated);
            let mixed = g.mul(gated, ple_i);
            let mixed = g.matmul(mixed, proj_w);
            let mixed = g.rms_norm(mixed, post, eps);
            cur = g.add(cur, mixed);
        }

        if model
            .tensors
            .contains_key(&format!("{p}.layer_output_scale.weight"))
        {
            // A single f32 in the file; bake it as a constant scale.
            let s = require(model, &format!("{p}.layer_output_scale.weight"))?
                .to_f32()
                .map_err(|e| e.to_string())?;
            if let Some(&s) = s.first() {
                cur = g.scale(cur, s);
            }
        }
        x = cur;
    }

    let out_norm = g.parameter("output_norm.weight", &[hidden]);
    x = g.rms_norm(x, out_norm, eps);

    let logits_w = if model.tensors.contains_key("output.weight") {
        declare_parameter(g, require(model, "output.weight")?, "output.weight")
            .map_err(|e| e.to_string())?
    } else {
        declare_parameter(g, require(model, "token_embd.weight")?, "token_embd.weight")
            .map_err(|e| e.to_string())?
    };
    let mut logits = g.matmul(x, logits_w);
    if config.logit_softcap > 0.0 {
        logits = g.scale(logits, 1.0 / config.logit_softcap);
        logits = g.tanh(logits);
        logits = g.scale(logits, config.logit_softcap);
    }
    Ok((logits, k_caches, v_caches))
}

/// Upload every graph parameter that exists in the GGUF file.
///
/// Derived HorizontalConcat buffers are filled by the runtime from their
/// sources. KV caches are left at zero.
pub fn load_parameters(session: &mut crate::Session, model: &GgufModel) -> Result<(), String> {
    let names: Vec<String> = session
        .plan()
        .param_buffers
        .iter()
        .map(|(n, _)| n.clone())
        .collect();
    for name in names {
        if name.contains('+') || name.starts_with("kv_cache.") {
            continue;
        }
        let Some(tensor) = model.tensors.get(&name) else {
            continue;
        };
        match tensor.packed_dtype() {
            Ok(_) => {
                let (_, bytes) = tensor.to_packed().map_err(|e| format!("{name}: {e}"))?;
                session.set_parameter_packed(&name, &bytes);
            }
            Err(GgufError::UnsupportedPack(_)) => {
                let data = tensor.to_f32().map_err(|e| format!("{name}: {e}"))?;
                session.set_parameter(&name, &data);
            }
            Err(e) => return Err(format!("{name}: {e}")),
        }
    }
    Ok(())
}

/// CPU gather of one token's embedding and (if present) PLE rows.
pub fn gather_token(
    model: &GgufModel,
    config: &Gemma4Config,
    token: u32,
) -> Result<(Vec<f32>, Option<Vec<f32>>), GgufError> {
    let embed = model
        .tensors
        .get("token_embd.weight")
        .ok_or(GgufError::BadShape("missing token_embd.weight".into()))?;
    let x = embed.column_f32(token as usize)?;
    let ple = if config.n_embd_per_layer > 0 {
        let table = model
            .tensors
            .get("per_layer_token_embd.weight")
            .ok_or(GgufError::BadShape(
                "missing per_layer_token_embd.weight".into(),
            ))?;
        let mut row = table.column_f32(token as usize)?;
        let scale = (config.n_embd_per_layer as f32).sqrt();
        for v in &mut row {
            *v *= scale;
        }
        Some(row)
    } else {
        None
    };
    Ok((x, ple))
}
