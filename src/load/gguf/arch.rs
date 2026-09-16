//! Architecture metadata: what the file says the model *is*.
//!
//! GGUF namespaces its hyperparameters under the architecture name, so a
//! llama file writes `llama.block_count` where a gemma2 one writes
//! `gemma2.block_count`. [`GgufModel::arch_key`](super::GgufModel::arch_key)
//! resolves that prefix, and everything here is read through it.
//!
//! The point of this module is that no dimension is hard-coded. A
//! [`ModelConfig`] is entirely a reading of the file, which is what lets one
//! builder serve models it has never been compiled against.
//!
//! # What the [`Architecture`] discriminates
//!
//! Every architecture here is a pre-norm decoder stack, so the enum is not a
//! choice of model so much as a record of where each family departs from the
//! llama shape. The departures are few, and each is a predicate below:
//! Qwen3's per-head Q/K norms, Gemma's scaled embeddings and its second pair
//! of norms, Gemma3's five-to-one window pattern and its separate local RoPE
//! base, Phi's parallel residual, LayerNorm and packed tensors — and, least
//! visibly, whether the file's Q and K were permuted for GGML's interleaved
//! RoPE ([`Architecture::rope_is_interleaved`]).
//!
//! A name is mapped onto a variant only when its graph is that variant's
//! graph, not merely close to it. The alias list is short on purpose: a
//! family that differs anywhere would load without complaint and decode
//! wrongly, which is the failure mode hardest to notice.

use super::{GgufError, GgufModel, GgufValue};

/// The model family a GGUF file declares, as far as graph shape is
/// concerned.
///
/// Families that build the same graph share a variant: `general.architecture`
/// of `mistral` or `smollm2` is [`Architecture::Llama`], because nothing
/// about the graph differs. Two variants exist only where the builder must
/// actually branch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Architecture {
    /// RMSNorm, GQA, RoPE, SwiGLU, no biases. Also Mistral, SmolLM2,
    /// TinyLlama, Vicuna — anything that kept the llama shape.
    Llama,
    /// Llama plus biases on Q, K and V.
    Qwen2,
    /// Llama plus RMSNorm on Q and K *per head*, before RoPE. Qwen3 dropped
    /// Qwen2's QKV biases when it added these.
    Qwen3,
    /// Embeddings scaled by `sqrt(n_embd)`, GELU rather than SiLU in the
    /// feed-forward, and a norm weight that is stored as `w` but applied as
    /// `1 + w`.
    Gemma,
    /// Gemma plus a norm after each of the attention and feed-forward
    /// blocks, alternating sliding-window and full attention, and logit
    /// softcapping.
    Gemma2,
    /// Gemma2's block shape, but with per-head Q/K norms, five local
    /// layers to each global one rather than alternating, and a separate
    /// RoPE base for the local layers. It dropped the softcapping.
    Gemma3,
    /// LayerNorm rather than RMSNorm, a single norm per block feeding
    /// attention and feed-forward *in parallel*, GELU, and biases
    /// throughout.
    Phi2,
    /// Phi's successor, back to the llama shape: RMSNorm, SwiGLU, no
    /// biases, two norms per block. It differs from llama only in packing
    /// QKV and gate/up as single tensors.
    Phi3,
}

impl Architecture {
    /// Map `general.architecture` onto a graph shape.
    ///
    /// Unrecognized names are rejected rather than approximated: a decoder
    /// that is *almost* right produces fluent, wrong text, which is far
    /// worse to debug than a refusal naming the architecture.
    pub fn from_name(name: &str) -> Result<Self, GgufError> {
        Ok(match name {
            // Families that are the llama graph, not merely close to it.
            // A name is listed here only if its GGUF converter is
            // llama.cpp's `LlamaModel` — the list is deliberately short,
            // because aliasing a family that differs anywhere would load
            // cleanly and decode wrongly.
            "llama" | "mistral" | "smollm" | "smollm2" | "tinyllama" | "vicuna" => Self::Llama,
            "qwen2" => Self::Qwen2,
            "qwen3" => Self::Qwen3,
            "gemma" => Self::Gemma,
            "gemma2" => Self::Gemma2,
            "gemma3" => Self::Gemma3,
            "phi2" => Self::Phi2,
            "phi3" => Self::Phi3,
            other => return Err(GgufError::UnsupportedArchitecture(other.to_string())),
        })
    }

    /// Whether RoPE rotates *adjacent* pairs rather than halves of a head.
    ///
    /// GGML has both conventions. Meganeura's RoPE is the half-split one,
    /// and llama.cpp's converter permutes Q and K on the way into a llama
    /// GGUF so that GGML's interleaved rope reproduces the same rotation.
    /// Those permuted weights therefore have to be permuted *back* on load
    /// — see [`super::weights`]. Every other family here converts without
    /// the permutation and needs nothing.
    pub fn rope_is_interleaved(self) -> bool {
        matches!(self, Self::Llama)
    }

    /// Whether the file packs Q, K and V into one `attn_qkv.weight`.
    pub fn packs_qkv(self) -> bool {
        matches!(self, Self::Phi2 | Self::Phi3)
    }

    /// Whether the file packs the feed-forward gate and up projections
    /// into one double-width `ffn_up.weight`.
    pub fn packs_gate_up(self) -> bool {
        matches!(self, Self::Phi3)
    }

    /// Whether blocks normalize with LayerNorm instead of RMSNorm.
    pub fn uses_layer_norm(self) -> bool {
        matches!(self, Self::Phi2)
    }

    /// Whether the feed-forward is a plain GELU MLP rather than SwiGLU.
    ///
    /// Gemma's feed-forward is gated like SwiGLU but activates with GELU;
    /// that is [`Self::gated_ffn`], not this. This is Phi2's genuinely
    /// ungated `down(gelu(up(x)))`.
    pub fn uses_gelu_mlp(self) -> bool {
        matches!(self, Self::Phi2)
    }

    /// Whether the feed-forward gates — `down(act(gate(x)) * up(x))`.
    pub fn gated_ffn(self) -> bool {
        !self.uses_gelu_mlp()
    }

    /// Whether attention and feed-forward read the same normed input and
    /// add into the residual together, rather than in sequence.
    pub fn parallel_residual(self) -> bool {
        matches!(self, Self::Phi2)
    }

    /// Whether Q, K and V projections carry biases.
    pub fn qkv_bias(self) -> bool {
        matches!(self, Self::Qwen2 | Self::Phi2)
    }

    /// Whether Q and K are normed per head before RoPE.
    pub fn qk_norm(self) -> bool {
        matches!(self, Self::Qwen3 | Self::Gemma3)
    }

    /// Whether the token embedding is scaled by `sqrt(n_embd)` on the way
    /// in.
    pub fn scales_embeddings(self) -> bool {
        matches!(self, Self::Gemma | Self::Gemma2 | Self::Gemma3)
    }

    /// Whether norm weights are stored as `w` but applied as `1 + w`.
    ///
    /// Gemma trains its norm weights centred on zero. Folding the `+1` in
    /// at load time keeps the shaders unaware of it.
    pub fn norm_weight_offset_by_one(self) -> bool {
        matches!(self, Self::Gemma | Self::Gemma2 | Self::Gemma3)
    }

    /// Whether each block carries a second pair of norms, applied to the
    /// attention and feed-forward outputs before they rejoin the residual.
    pub fn post_block_norms(self) -> bool {
        matches!(self, Self::Gemma2 | Self::Gemma3)
    }
}

impl std::fmt::Display for Architecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match *self {
            Self::Llama => "llama",
            Self::Qwen2 => "qwen2",
            Self::Qwen3 => "qwen3",
            Self::Gemma => "gemma",
            Self::Gemma2 => "gemma2",
            Self::Gemma3 => "gemma3",
            Self::Phi2 => "phi2",
            Self::Phi3 => "phi3",
        };
        f.write_str(name)
    }
}

/// Everything the graph builder needs, read out of the file.
///
/// Construct with [`ModelConfig::from_gguf`]; the fields are public so a
/// caller can override one (a shorter context than the file advertises, say)
/// before building.
#[derive(Clone, Debug)]
pub struct ModelConfig {
    /// Which family's graph to build.
    pub architecture: Architecture,
    /// Rows in the token embedding table.
    pub vocab_size: usize,
    /// Residual stream width, `{arch}.embedding_length`.
    pub hidden_size: usize,
    /// Transformer blocks, `{arch}.block_count`.
    pub num_layers: usize,
    /// Query heads, `{arch}.attention.head_count`.
    pub num_heads: u32,
    /// Key/value heads, `{arch}.attention.head_count_kv`. Equal to
    /// [`Self::num_heads`] for multi-head attention, fewer for grouped-query,
    /// one for multi-query.
    pub num_kv_heads: u32,
    /// Width of one attention head.
    ///
    /// Usually `hidden_size / num_heads`, but Gemma sets
    /// `{arch}.attention.key_length` independently, so this is read rather
    /// than derived when the file says so.
    pub head_dim: u32,
    /// Feed-forward width, `{arch}.feed_forward_length`.
    pub intermediate_size: usize,
    /// Epsilon for whichever norm [`Architecture::uses_layer_norm`] selects.
    pub norm_eps: f32,
    /// RoPE base frequency, `{arch}.rope.freq_base`.
    pub rope_theta: f32,
    /// RoPE base for the sliding-window layers, where the file gives them
    /// their own — Gemma3's `{arch}.rope.local_freq_base`. `None` means
    /// every layer rotates with [`Self::rope_theta`].
    pub rope_theta_local: Option<f32>,
    /// How many of each head's dimensions RoPE rotates.
    ///
    /// Phi2 rotates a fraction of the head and leaves the rest untouched;
    /// everything else here rotates the whole head.
    pub rope_dim: u32,
    /// Longest sequence the file claims to support,
    /// `{arch}.context_length`.
    pub context_length: usize,
    /// Sliding-window width for the layers that use one, or `None` when
    /// every layer attends to the whole prefix.
    pub sliding_window: Option<usize>,
    /// Ceiling applied to attention logits, Gemma2's
    /// `{arch}.attn_logit_softcapping`.
    pub attn_logit_softcap: Option<f32>,
    /// Ceiling applied to output logits, Gemma2's
    /// `{arch}.final_logit_softcapping`.
    pub final_logit_softcap: Option<f32>,
    /// Whether the output head reuses the embedding table.
    ///
    /// Not a metadata key: GGUF records weight tying by *omitting*
    /// `output.weight`, so this is read off the tensor inventory.
    pub tie_word_embeddings: bool,
}

impl ModelConfig {
    /// Read a config out of a parsed file.
    ///
    /// Every field comes from metadata or the tensor inventory. Keys that
    /// producers write inconsistently are defaulted where a default is
    /// unambiguous — `head_count_kv` falls back to `head_count`, which is
    /// multi-head attention — and required where guessing would silently
    /// change the model.
    pub fn from_gguf(model: &GgufModel) -> Result<Self, GgufError> {
        let name = model
            .architecture()
            .ok_or_else(|| GgufError::MissingKey("general.architecture".to_string()))?;
        let architecture = Architecture::from_name(name)?;

        let hidden_size = require_usize(model, "embedding_length")?;
        let num_layers = require_usize(model, "block_count")?;
        let num_heads = require_u32(model, "attention.head_count")?;
        let num_kv_heads = opt_u32(model, "attention.head_count_kv")?.unwrap_or(num_heads);
        let intermediate_size = require_usize(model, "feed_forward_length")?;

        if num_heads == 0 {
            return Err(GgufError::BadMetadata(
                "attention.head_count is zero".to_string(),
            ));
        }
        if num_kv_heads == 0 || !num_heads.is_multiple_of(num_kv_heads) {
            return Err(GgufError::BadMetadata(format!(
                "attention.head_count {num_heads} is not a whole multiple of \
                 head_count_kv {num_kv_heads}"
            )));
        }

        // Gemma's head is wider than hidden/heads, so the file's own
        // key_length wins wherever it is written.
        let head_dim = match opt_u32(model, "attention.key_length")? {
            Some(k) => k,
            None => {
                if !hidden_size.is_multiple_of(num_heads as usize) {
                    return Err(GgufError::BadMetadata(format!(
                        "embedding_length {hidden_size} is not divisible by head_count \
                         {num_heads}, and attention.key_length is absent"
                    )));
                }
                (hidden_size / num_heads as usize) as u32
            }
        };

        // The two norms are alternatives, not fallbacks for one another:
        // reading an RMS epsilon into a LayerNorm would be a quiet change
        // of model, so each architecture asks for its own.
        let norm_eps = if architecture.uses_layer_norm() {
            opt_f32(model, "attention.layer_norm_epsilon")?.unwrap_or(1e-5)
        } else {
            opt_f32(model, "attention.layer_norm_rms_epsilon")?.unwrap_or(1e-5)
        };

        let rope_theta = opt_f32(model, "rope.freq_base")?.unwrap_or(10_000.0);
        let rope_theta_local = opt_f32(model, "rope.local_freq_base")?.filter(|v| *v > 0.0);
        let rope_dim = opt_u32(model, "rope.dimension_count")?.unwrap_or(head_dim);
        let context_length = opt_usize(model, "context_length")?.unwrap_or(2048);
        let sliding_window = opt_usize(model, "attention.sliding_window")?.filter(|&w| w > 0);
        let attn_logit_softcap = opt_f32(model, "attn_logit_softcapping")?.filter(|v| *v > 0.0);
        let final_logit_softcap = opt_f32(model, "final_logit_softcapping")?.filter(|v| *v > 0.0);

        let vocab_size = vocab_size(model)?;
        // GGUF ties weights by leaving the head out of the file, so the
        // inventory is the only statement of it.
        let tie_word_embeddings = !model.tensors.contains_key("output.weight");

        Ok(Self {
            architecture,
            vocab_size,
            hidden_size,
            num_layers,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            norm_eps,
            rope_theta,
            rope_theta_local,
            rope_dim,
            context_length,
            sliding_window,
            attn_logit_softcap,
            final_logit_softcap,
            tie_word_embeddings,
        })
    }

    /// Combined width of the key (or value) projection across all KV heads.
    pub fn kv_dim(&self) -> usize {
        self.num_kv_heads as usize * self.head_dim as usize
    }

    /// Combined width of the query projection across all heads.
    ///
    /// Equal to [`Self::hidden_size`] except where the file sets an
    /// independent `attention.key_length`.
    pub fn q_dim(&self) -> usize {
        self.num_heads as usize * self.head_dim as usize
    }

    /// Whether layer `index` attends to a sliding window rather than the
    /// whole prefix.
    ///
    /// The two Gemmas interleave differently — Gemma2 alternates one for
    /// one, Gemma3 takes five local layers to each global one, the global
    /// being every sixth. A model with no window attends fully everywhere,
    /// and one with a window and no interleaving pattern uses it in every
    /// layer.
    pub fn layer_is_windowed(&self, index: usize) -> bool {
        match self.sliding_window {
            None => false,
            Some(_) => match self.architecture {
                Architecture::Gemma2 => index.is_multiple_of(2),
                Architecture::Gemma3 => !(index + 1).is_multiple_of(6),
                _ => true,
            },
        }
    }

    /// The RoPE base layer `index` rotates with.
    ///
    /// Gemma3 gives its local layers a much smaller base than its global
    /// ones, so the base is a property of the layer rather than of the
    /// model. Everything else uses one base throughout.
    pub fn layer_rope_theta(&self, index: usize) -> f32 {
        match self.rope_theta_local {
            Some(local) if self.layer_is_windowed(index) => local,
            _ => self.rope_theta,
        }
    }
}

/// The embedding table's row count.
///
/// `{arch}.vocab_size` is written by some producers and not others, and the
/// tokenizer's token list is written by nearly all — but the table itself is
/// the only unarguable answer, since it is what the graph must be shaped
/// against. Prefer it, and fall back only if the file has no embedding.
fn vocab_size(model: &GgufModel) -> Result<usize, GgufError> {
    if let Some(embd) = model.tensors.get("token_embd.weight") {
        // GGUF names dimensions fastest-first, so the table is
        // [n_embd, n_vocab] and the row count is the second.
        if let Some(&rows) = embd.dims.get(1) {
            return Ok(rows);
        }
    }
    if let Some(tokens) = model
        .metadata
        .get("tokenizer.ggml.tokens")
        .and_then(GgufValue::as_array)
    {
        return Ok(tokens.len());
    }
    opt_usize(model, "vocab_size")?.ok_or_else(|| {
        GgufError::MissingKey("token_embd.weight, tokenizer.ggml.tokens or vocab_size".to_string())
    })
}

// ---------------------------------------------------------------------------
// Typed metadata reads
//
// Each returns a named error rather than `None` on a type mismatch, so a
// file that writes `block_count` as a string says so instead of looking
// absent.
// ---------------------------------------------------------------------------

fn typed<'a, T>(
    model: &'a GgufModel,
    suffix: &str,
    want: &str,
    read: impl Fn(&'a GgufValue) -> Option<T>,
) -> Result<Option<T>, GgufError> {
    let Some(value) = model.arch_key(suffix) else {
        return Ok(None);
    };
    read(value)
        .map(Some)
        .ok_or_else(|| GgufError::BadMetadata(format!("{suffix} is not {want}: {value:?}")))
}

fn opt_usize(model: &GgufModel, suffix: &str) -> Result<Option<usize>, GgufError> {
    let raw = typed(model, suffix, "an unsigned integer", GgufValue::as_u64)?;
    raw.map(|v| {
        usize::try_from(v).map_err(|_| {
            GgufError::BadMetadata(format!("{suffix} = {v} does not fit this platform"))
        })
    })
    .transpose()
}

fn opt_u32(model: &GgufModel, suffix: &str) -> Result<Option<u32>, GgufError> {
    let raw = typed(model, suffix, "an unsigned integer", GgufValue::as_u64)?;
    raw.map(|v| {
        u32::try_from(v).map_err(|_| GgufError::BadMetadata(format!("{suffix} = {v} exceeds u32")))
    })
    .transpose()
}

fn opt_f32(model: &GgufModel, suffix: &str) -> Result<Option<f32>, GgufError> {
    Ok(typed(model, suffix, "a number", GgufValue::as_f64)?.map(|v| v as f32))
}

fn require_usize(model: &GgufModel, suffix: &str) -> Result<usize, GgufError> {
    opt_usize(model, suffix)?.ok_or_else(|| missing(model, suffix))
}

fn require_u32(model: &GgufModel, suffix: &str) -> Result<u32, GgufError> {
    opt_u32(model, suffix)?.ok_or_else(|| missing(model, suffix))
}

/// Name the key the way the file would have written it, so the error is
/// something the reader can grep the file for.
fn missing(model: &GgufModel, suffix: &str) -> GgufError {
    match model.architecture() {
        Some(arch) => GgufError::MissingKey(format!("{arch}.{suffix}")),
        None => GgufError::MissingKey(suffix.to_string()),
    }
}

/// Metadata for a synthetic model, for tests that need a config without a
/// file. Not public: fixtures belong to the tests that use them.
#[cfg(test)]
pub(super) fn test_metadata(arch: &str) -> std::collections::HashMap<String, GgufValue> {
    let mut m = std::collections::HashMap::new();
    let mut set = |k: &str, v: GgufValue| {
        m.insert(k.to_string(), v);
    };
    set("general.architecture", GgufValue::String(arch.to_string()));
    set(&format!("{arch}.embedding_length"), GgufValue::U32(64));
    set(&format!("{arch}.block_count"), GgufValue::U32(2));
    set(&format!("{arch}.attention.head_count"), GgufValue::U32(4));
    set(
        &format!("{arch}.attention.head_count_kv"),
        GgufValue::U32(2),
    );
    set(&format!("{arch}.feed_forward_length"), GgufValue::U32(128));
    set(
        &format!("{arch}.attention.layer_norm_rms_epsilon"),
        GgufValue::F32(1e-6),
    );
    set(&format!("{arch}.rope.freq_base"), GgufValue::F32(10_000.0));
    set(&format!("{arch}.context_length"), GgufValue::U32(512));
    set(&format!("{arch}.vocab_size"), GgufValue::U32(32));
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load::gguf::{GgmlType, GgufTensor};

    fn model(arch: &str) -> GgufModel {
        GgufModel {
            metadata: test_metadata(arch),
            tensors: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn reads_every_dimension_from_the_file() {
        let cfg = ModelConfig::from_gguf(&model("llama")).unwrap();
        assert_eq!(cfg.architecture, Architecture::Llama);
        assert_eq!(cfg.hidden_size, 64);
        assert_eq!(cfg.num_layers, 2);
        assert_eq!(cfg.num_heads, 4);
        assert_eq!(cfg.num_kv_heads, 2);
        assert_eq!(cfg.intermediate_size, 128);
        assert_eq!(cfg.head_dim, 16, "derived from hidden/heads");
        assert_eq!(cfg.kv_dim(), 32);
        assert_eq!(cfg.context_length, 512);
    }

    #[test]
    fn families_that_share_the_llama_graph_share_the_variant() {
        for name in ["llama", "mistral", "smollm2", "tinyllama"] {
            let cfg = ModelConfig::from_gguf(&model(name)).unwrap();
            assert_eq!(cfg.architecture, Architecture::Llama, "{name}");
        }
    }

    #[test]
    fn an_unknown_architecture_is_named_rather_than_approximated() {
        let mut m = model("llama");
        m.metadata.insert(
            "general.architecture".to_string(),
            GgufValue::String("mamba".to_string()),
        );
        let err = ModelConfig::from_gguf(&m).unwrap_err();
        assert!(
            matches!(&err, GgufError::UnsupportedArchitecture(a) if a == "mamba"),
            "{err:?}"
        );
    }

    #[test]
    fn head_count_kv_defaults_to_multi_head() {
        let mut m = model("llama");
        m.metadata.remove("llama.attention.head_count_kv");
        let cfg = ModelConfig::from_gguf(&m).unwrap();
        assert_eq!(cfg.num_kv_heads, cfg.num_heads);
    }

    #[test]
    fn an_explicit_key_length_beats_the_derived_head_dim() {
        let mut m = model("gemma");
        m.metadata.insert(
            "gemma.attention.key_length".to_string(),
            GgufValue::U32(256),
        );
        let cfg = ModelConfig::from_gguf(&m).unwrap();
        assert_eq!(cfg.head_dim, 256);
        // hidden/heads would have said 16, and q_dim follows the real head.
        assert_eq!(cfg.q_dim(), 4 * 256);
    }

    #[test]
    fn a_head_count_that_does_not_divide_the_width_is_rejected() {
        let mut m = model("llama");
        m.metadata
            .insert("llama.attention.head_count".to_string(), GgufValue::U32(5));
        assert!(matches!(
            ModelConfig::from_gguf(&m),
            Err(GgufError::BadMetadata(_))
        ));
    }

    #[test]
    fn grouped_query_heads_must_divide_evenly() {
        let mut m = model("llama");
        m.metadata.insert(
            "llama.attention.head_count_kv".to_string(),
            GgufValue::U32(3),
        );
        assert!(matches!(
            ModelConfig::from_gguf(&m),
            Err(GgufError::BadMetadata(_))
        ));
    }

    #[test]
    fn the_embedding_table_settles_the_vocabulary_size() {
        let mut m = model("llama");
        // Deliberately disagreeing with the metadata key, which says 32.
        m.tensors.insert(
            "token_embd.weight".to_string(),
            GgufTensor::new(vec![64, 99], GgmlType::F32, vec![0; 64 * 99 * 4]),
        );
        let cfg = ModelConfig::from_gguf(&m).unwrap();
        assert_eq!(cfg.vocab_size, 99);
    }

    #[test]
    fn a_missing_output_weight_means_tied_embeddings() {
        let mut m = model("llama");
        assert!(ModelConfig::from_gguf(&m).unwrap().tie_word_embeddings);
        m.tensors.insert(
            "output.weight".to_string(),
            GgufTensor::new(vec![64, 32], GgmlType::F32, vec![0; 64 * 32 * 4]),
        );
        assert!(!ModelConfig::from_gguf(&m).unwrap().tie_word_embeddings);
    }

    #[test]
    fn a_required_key_names_itself_with_the_architecture_prefix() {
        let mut m = model("llama");
        m.metadata.remove("llama.block_count");
        let err = ModelConfig::from_gguf(&m).unwrap_err();
        assert!(
            matches!(&err, GgufError::MissingKey(k) if k == "llama.block_count"),
            "{err:?}"
        );
    }

    #[test]
    fn a_key_of_the_wrong_type_is_not_silently_absent() {
        let mut m = model("llama");
        m.metadata.insert(
            "llama.block_count".to_string(),
            GgufValue::String("many".to_string()),
        );
        assert!(matches!(
            ModelConfig::from_gguf(&m),
            Err(GgufError::BadMetadata(_))
        ));
    }

    #[test]
    fn phi2_reads_the_layer_norm_epsilon_not_the_rms_one() {
        let mut m = model("phi2");
        m.metadata.insert(
            "phi2.attention.layer_norm_rms_epsilon".to_string(),
            GgufValue::F32(0.125),
        );
        m.metadata.insert(
            "phi2.attention.layer_norm_epsilon".to_string(),
            GgufValue::F32(0.25),
        );
        let cfg = ModelConfig::from_gguf(&m).unwrap();
        assert_eq!(cfg.norm_eps, 0.25);
        assert!(cfg.architecture.uses_layer_norm());
        assert!(cfg.architecture.parallel_residual());
    }

    #[test]
    fn gemma2_alternates_windowed_and_full_attention() {
        let mut m = model("gemma2");
        m.metadata.insert(
            "gemma2.attention.sliding_window".to_string(),
            GgufValue::U32(128),
        );
        let cfg = ModelConfig::from_gguf(&m).unwrap();
        assert_eq!(cfg.sliding_window, Some(128));
        assert!(cfg.layer_is_windowed(0));
        assert!(!cfg.layer_is_windowed(1));
        assert!(cfg.layer_is_windowed(2));
    }

    #[test]
    fn a_model_without_a_window_attends_fully_everywhere() {
        let cfg = ModelConfig::from_gguf(&model("llama")).unwrap();
        assert_eq!(cfg.sliding_window, None);
        assert!(!cfg.layer_is_windowed(0));
        assert!(!cfg.layer_is_windowed(1));
    }

    #[test]
    fn a_zero_sliding_window_reads_as_no_window() {
        let mut m = model("gemma2");
        m.metadata.insert(
            "gemma2.attention.sliding_window".to_string(),
            GgufValue::U32(0),
        );
        assert_eq!(ModelConfig::from_gguf(&m).unwrap().sliding_window, None);
    }

    #[test]
    fn rope_dimensions_default_to_the_whole_head() {
        let cfg = ModelConfig::from_gguf(&model("llama")).unwrap();
        assert_eq!(cfg.rope_dim, cfg.head_dim);
    }

    #[test]
    fn a_partial_rope_is_read_as_written() {
        let mut m = model("phi2");
        m.metadata
            .insert("phi2.rope.dimension_count".to_string(), GgufValue::U32(8));
        let cfg = ModelConfig::from_gguf(&m).unwrap();
        assert_eq!(cfg.rope_dim, 8);
        assert_eq!(cfg.head_dim, 16);
    }

    #[test]
    fn architecture_traits_match_the_families() {
        assert!(Architecture::Qwen2.qkv_bias());
        assert!(!Architecture::Qwen3.qkv_bias());
        assert!(Architecture::Qwen3.qk_norm());
        assert!(Architecture::Gemma.scales_embeddings());
        assert!(Architecture::Gemma.norm_weight_offset_by_one());
        assert!(!Architecture::Gemma.post_block_norms());
        assert!(Architecture::Gemma2.post_block_norms());
        assert!(Architecture::Phi2.uses_gelu_mlp());
        assert!(!Architecture::Phi2.gated_ffn());
        assert!(Architecture::Phi3.gated_ffn());
        assert!(!Architecture::Phi3.uses_layer_norm());
    }
}
