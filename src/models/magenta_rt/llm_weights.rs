//! Real-weight loader for the Magenta-RT LLM (`llm_base_x4286_c1860k`).
//!
//! Maps the T5X/flaxformer checkpoint tensor names (`target.*`) onto the graph
//! parameter names used by [`super::llm`]'s builders, and loads them from a
//! dumped safetensors file. `tools/magenta_rt/dump_llm.py` produces that file
//! (`weights_llm_base.safetensors`) keyed by exactly these `target.*` names. The
//! mapping is verified against the committed
//! `tools/magenta_rt/llm_base_manifest.json` (name + element-count coverage) in
//! `tests/llm_weight_map.rs` — no weight download needed for that check.
//!
//! **No transposes / reshapes are needed.** T5X `DenseGeneral` kernels are
//! stored `[in, out]` row-major, which is exactly the `matmul(x, W)` convention
//! the builders use (`query`/`out` `[embed, embed]`, `wi_0` `[embed, mlp]`, `wo`
//! `[mlp, embed]`, `logits_dense` `[embed, vocab]`, `embedding` `[vocab,
//! embed]`). The rel-pos `rel_embedding [heads, buckets]` flattens row-major to
//! the `[heads*buckets]` table the attention kernels index as
//! `table[head*buckets + bucket]`. So every parameter is a flat copy.
//!
//! **Encoder gap.** The checkpoint has no encoder positional tensor (neither a
//! learned PE nor a rel-pos table), so the encoder's `encoder.pos_embed` and
//! per-layer `attn.rel_pos_bias_table` graph params are intentionally *not*
//! mapped — they are left unset and reported as skipped. Resolving the encoder
//! position scheme (see `LLM_FINDINGS.md`) is required before an encoder forward
//! on real weights; the decoder + embeddings + heads load fully.

use super::llm::LlmConfig;

/// One parameter mapping: graph parameter name → checkpoint tensor name, with
/// the expected element count (for a size sanity-check on load).
#[derive(Clone, Debug)]
pub struct ParamMap {
    pub graph: String,
    pub ckpt: String,
    pub numel: usize,
}

fn push(out: &mut Vec<ParamMap>, graph: String, ckpt: String, numel: usize) {
    out.push(ParamMap { graph, ckpt, numel });
}

/// Full graph-param → checkpoint-tensor mapping for the given config. Covers
/// every `target.*` tensor in the checkpoint (token embedder, the shared
/// `decoder_norm` + `logits_dense` head, the temporal/depth rel-pos tables, and
/// all encoder / temporal / depth layers). Does **not** include the encoder
/// positional params (no checkpoint tensor — see the module docs) or the
/// runtime KV caches.
pub fn checkpoint_param_map(cfg: &LlmConfig) -> Vec<ParamMap> {
    let embed = cfg.embed_dim as usize;
    let attn = (cfg.num_heads * cfg.head_dim) as usize;
    let mlp = cfg.mlp_dim as usize;
    let vocab = cfg.vocab_size as usize;
    let heads = cfg.num_heads as usize;

    let mut m = Vec::new();

    // --- Shared / global ---
    push(
        &mut m,
        "shared_token_embedder".into(),
        "target.token_embedder.embedding".into(),
        vocab * embed,
    );
    push(
        &mut m,
        "decoder.decoder_norm".into(),
        "target.decoder.decoder_norm.scale".into(),
        embed,
    );
    push(
        &mut m,
        "decoder.logits_dense".into(),
        "target.decoder.logits_dense.kernel".into(),
        embed * vocab,
    );
    push(
        &mut m,
        "decoder.temporal_decoder.rel_pos_bias_table".into(),
        "target.decoder.decoder.temporal_decoder.relpos_bias.rel_embedding".into(),
        heads * cfg.rel_pos_num_buckets as usize,
    );
    push(
        &mut m,
        "decoder.depth_decoder.rel_pos_bias_table".into(),
        "target.decoder.decoder.depth_decoder.relpos_bias_depth.rel_embedding".into(),
        heads * cfg.depth_rel_pos_num_buckets as usize,
    );
    push(
        &mut m,
        "encoder.final_norm".into(),
        "target.encoder.encoder_norm.scale".into(),
        embed,
    );

    // T5X DenseGeneral attention kernels: q/k/v are [embed, attn], out is [attn, embed].
    let attn_kernels = |m: &mut Vec<ParamMap>, gp: &str, cp: &str| {
        push(
            m,
            format!("{gp}.q"),
            format!("{cp}.query.kernel"),
            embed * attn,
        );
        push(
            m,
            format!("{gp}.k"),
            format!("{cp}.key.kernel"),
            embed * attn,
        );
        push(
            m,
            format!("{gp}.v"),
            format!("{cp}.value.kernel"),
            embed * attn,
        );
        push(
            m,
            format!("{gp}.o"),
            format!("{cp}.out.kernel"),
            attn * embed,
        );
    };
    let mlp_kernels = |m: &mut Vec<ParamMap>, gp: &str, cp: &str| {
        push(
            m,
            format!("{gp}.w_gate"),
            format!("{cp}.wi_0.kernel"),
            embed * mlp,
        );
        push(
            m,
            format!("{gp}.w_up"),
            format!("{cp}.wi_1.kernel"),
            embed * mlp,
        );
        push(
            m,
            format!("{gp}.w_down"),
            format!("{cp}.wo.kernel"),
            mlp * embed,
        );
    };

    // --- Encoder layers ---
    for i in 0..cfg.num_encoder_layers as usize {
        let gp = format!("encoder.layers.{i}");
        let cp = format!("target.encoder.layers_{i}");
        push(
            &mut m,
            format!("{gp}.pre_attn_norm"),
            format!("{cp}.pre_attention_layer_norm.scale"),
            embed,
        );
        attn_kernels(&mut m, &format!("{gp}.attn"), &format!("{cp}.attention"));
        push(
            &mut m,
            format!("{gp}.pre_mlp_norm"),
            format!("{cp}.pre_mlp_layer_norm.scale"),
            embed,
        );
        mlp_kernels(&mut m, &format!("{gp}.mlp"), &format!("{cp}.mlp"));
    }

    // --- Temporal decoder layers (self-attn + cross-attn + mlp) ---
    for i in 0..cfg.num_temporal_decoder_layers as usize {
        let gp = format!("decoder.temporal_layers.{i}");
        let cp = format!("target.decoder.decoder.temporal_decoder.layers_{i}");
        push(
            &mut m,
            format!("{gp}.pre_self_attn_norm"),
            format!("{cp}.pre_self_attention_layer_norm.scale"),
            embed,
        );
        attn_kernels(
            &mut m,
            &format!("{gp}.self_attn"),
            &format!("{cp}.self_attention"),
        );
        push(
            &mut m,
            format!("{gp}.pre_cross_attn_norm"),
            format!("{cp}.pre_cross_attention_layer_norm.scale"),
            embed,
        );
        attn_kernels(
            &mut m,
            &format!("{gp}.cross_attn"),
            &format!("{cp}.encoder_decoder_attention"),
        );
        push(
            &mut m,
            format!("{gp}.pre_mlp_norm"),
            format!("{cp}.pre_mlp_layer_norm.scale"),
            embed,
        );
        mlp_kernels(&mut m, &format!("{gp}.mlp"), &format!("{cp}.mlp"));
    }

    // --- Depth decoder layers (self-attn + mlp, no cross-attn) ---
    for i in 0..cfg.num_depth_decoder_layers as usize {
        let gp = format!("decoder.depth_layers.{i}");
        let cp = format!("target.decoder.decoder.depth_decoder.depth_layers_{i}");
        push(
            &mut m,
            format!("{gp}.pre_self_attn_norm"),
            format!("{cp}.pre_self_attention_layer_norm.scale"),
            embed,
        );
        attn_kernels(
            &mut m,
            &format!("{gp}.self_attn"),
            &format!("{cp}.self_attention"),
        );
        push(
            &mut m,
            format!("{gp}.pre_mlp_norm"),
            format!("{cp}.pre_mlp_layer_norm.scale"),
            embed,
        );
        mlp_kernels(&mut m, &format!("{gp}.mlp"), &format!("{cp}.mlp"));
    }

    m
}

/// Load real LLM weights from a dumped safetensors model (keyed by the
/// flaxformer `target.*` names) into `session`. Only the parameters the session
/// actually declares are set; any session param with no checkpoint tensor (KV
/// caches, and the encoder positional params — see the module docs) is left
/// untouched and returned in the skipped list.
///
/// Errors on a missing checkpoint tensor or an element-count mismatch — a
/// tripwire against a stale/wrong dump.
pub fn load_llm_weights(
    session: &mut crate::Session,
    model: &crate::data::safetensors::SafeTensorsModel,
    cfg: &LlmConfig,
) -> Result<Vec<String>, String> {
    use std::collections::HashMap;
    let map: HashMap<String, ParamMap> = checkpoint_param_map(cfg)
        .into_iter()
        .map(|p| (p.graph.clone(), p))
        .collect();

    let mut skipped = Vec::new();
    for (name, _) in session.plan().param_buffers.clone() {
        match map.get(&name) {
            Some(p) => {
                let data = model
                    .tensor_f32_auto(&p.ckpt)
                    .map_err(|e| format!("{}: {e}", p.ckpt))?;
                if data.len() != p.numel {
                    return Err(format!(
                        "{}: expected {} elements, checkpoint has {}",
                        p.ckpt,
                        p.numel,
                        data.len()
                    ));
                }
                session.set_parameter(&name, &data);
            }
            None => skipped.push(name),
        }
    }
    Ok(skipped)
}
