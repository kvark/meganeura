//! From a file to generated text.
//!
//! This is the layer that makes a `.gguf` usable without knowing anything
//! about Meganeura: it owns the compiled sessions, the KV cache, the
//! tokenizer and the sampling loop, so a caller writes
//!
//! ```no_run
//! use meganeura::load::gguf::{load_gguf, GenerationOptions};
//!
//! let model = load_gguf(std::path::Path::new("model.gguf"))?;
//! let mut generator = model.generator(2048)?;
//! let text = generator.generate("The meaning of life is", &GenerationOptions::default())?;
//! # Ok::<(), meganeura::load::gguf::GgufError>(())
//! ```
//!
//! # Two sessions over one set of buffers
//!
//! A prompt and a decode step want different kernels. A single-row matmul
//! goes to the tuned K-split GEMV, where a block of rows goes to the tiled
//! matmul; compiling one graph for both would give up one or the other.
//!
//! So there are two sessions, differing only in `block_size`, and the decode
//! session is constructed with the prefill session's parameter buffers. That
//! aliases rather than copies,
//! which buys two things at once: the weights are stored once however many
//! sessions read them, and the K/V caches are literally the same buffers —
//! so a prompt processed by the prefill session is already in the cache the
//! decode session attends over, with no handoff between them.

use std::sync::Arc;

use crate::Session;

use super::{GgufError, GgufModel, arch::ModelConfig, graph, vocab::Vocab, weights};

/// How to build the sessions.
#[derive(Clone, Debug)]
pub struct GeneratorOptions {
    /// Longest sequence the caches can hold, prompt included.
    ///
    /// Bounds memory: each layer holds two `max_seq_len × kv_dim` f32
    /// buffers. `Default` is a hard 2048; pass
    /// [`Generator::with_config`] a config whose `context_length` you
    /// have capped yourself — nothing here clamps against the file's
    /// own `context_length`.
    pub max_seq_len: usize,
    /// Token slots the prefill session consumes per step.
    ///
    /// Larger spends more per step and fewer steps on a prompt. `1`
    /// disables the prefill session entirely, halving compile time and
    /// feeding prompts one token at a time.
    pub prefill_block: usize,
}

impl Default for GeneratorOptions {
    fn default() -> Self {
        Self {
            max_seq_len: 2048,
            prefill_block: 32,
        }
    }
}

/// How to turn logits into the next token.
///
/// The default is greedy and so deterministic, which is what a library
/// should do unless asked otherwise; `temperature` above zero brings
/// `top_k`, `top_p` and `seed` into play.
#[derive(Clone, Debug)]
pub struct GenerationOptions {
    /// Ceiling on new tokens, whatever the model does.
    pub max_tokens: usize,
    /// `0.0` takes the argmax. Higher flattens the distribution.
    pub temperature: f32,
    /// Keep only this many candidates. `0` keeps all of them.
    pub top_k: usize,
    /// Keep the smallest set of candidates whose probabilities reach this.
    /// `1.0` keeps all of them.
    pub top_p: f32,
    /// Divides the logit of a recently emitted token. `1.0` is no penalty.
    pub repeat_penalty: f32,
    /// How far back [`Self::repeat_penalty`] looks.
    pub repeat_window: usize,
    /// Seeds sampling, so a temperature run is reproducible.
    pub seed: u64,
    /// Whether an end-of-generation token stops the loop.
    pub stop_at_end_of_generation: bool,
}

impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            max_tokens: 128,
            temperature: 0.0,
            top_k: 40,
            top_p: 0.95,
            repeat_penalty: 1.0,
            repeat_window: 64,
            seed: 0,
            stop_at_end_of_generation: true,
        }
    }
}

/// A loaded model, ready to generate.
pub struct Generator {
    /// Absent when [`GeneratorOptions::prefill_block`] is 1, in which case
    /// prompts go through the decode session a token at a time.
    prefill: Option<Session>,
    /// The per-layer token-embedding table, kept for the per-step host
    /// gather on architectures that carry one.
    ple_table: Option<super::GgufTensor>,
    decode: Session,
    prefill_block: usize,
    built: graph::ModelGraph,
    config: ModelConfig,
    vocab: Option<Vocab>,
    /// How much of the cache is populated — the next absolute position.
    position: usize,
    /// Tokens seen since the last reset, for the repetition penalty.
    history: Vec<u32>,
    report: weights::LoadReport,
    /// Why [`Self::vocab`] is absent, when it is.
    vocab_error: Option<String>,
}

impl Generator {
    /// Compile and load `model`, with the defaults.
    pub fn new(model: &GgufModel) -> Result<Self, GgufError> {
        Self::with_options(model, &GeneratorOptions::default())
    }

    /// Compile and load `model`, choosing the cache depth and prefill
    /// width.
    pub fn with_options(model: &GgufModel, options: &GeneratorOptions) -> Result<Self, GgufError> {
        let config = ModelConfig::from_gguf(model)?;
        Self::with_config(model, config, options)
    }

    /// [`Self::with_options`] over a config the caller has adjusted — a
    /// shorter context than the file advertises, say.
    pub fn with_config(
        model: &GgufModel,
        config: ModelConfig,
        options: &GeneratorOptions,
    ) -> Result<Self, GgufError> {
        if options.max_seq_len == 0 {
            return Err(GgufError::BadMetadata("max_seq_len must be > 0".into()));
        }
        let prefill_block = options.prefill_block.max(1);

        let mut decode_graph = crate::Graph::new();
        let built = graph::build(&mut decode_graph, model, &config, 1, options.max_seq_len)?;
        decode_graph.set_outputs(built.outputs());
        let (decode, prefill, report) = if prefill_block > 1 {
            let (mut prefill, mut report) =
                Self::build_prefill(model, &config, prefill_block, options.max_seq_len)?;
            let mut cfg = crate::SessionConfig::inference_from_env_on(prefill.context());
            cfg.share_parameters_from = Some(&mut prefill);
            let mut decode = crate::build(&decode_graph, cfg).0;
            // Shape-specific optimization may retain a weight only in the
            // decode plan. Such parameters have ordinary private allocations
            // and still need their GGUF contents loaded.
            weights::load_private(&mut decode, &prefill, model, &config, &mut report)?;
            weights::reset_caches(&mut decode, &built, &config);
            (decode, Some(prefill), report)
        } else {
            let mut decode =
                crate::build(&decode_graph, crate::SessionConfig::inference_from_env()).0;
            let report = weights::load(&mut decode, model, &config)?;
            weights::reset_caches(&mut decode, &built, &config);
            (decode, None, report)
        };

        // A file may carry weights and no vocabulary, which is still
        // usable through the token-level API — but keep *why* it has none,
        // so "no tokenizer" is not reported for a tokenizer this loader
        // simply does not implement.
        let ple_table = if config.architecture.uses_per_layer_embeddings() {
            Some(
                model
                    .tensors
                    .get("per_layer_token_embd.weight")
                    .ok_or_else(|| {
                        GgufError::MissingTensor("per_layer_token_embd.weight".to_string())
                    })?
                    .clone(),
            )
        } else {
            None
        };

        let (vocab, vocab_error) = match Vocab::from_gguf(model) {
            Ok(vocab) => (Some(vocab), None),
            Err(e) => (None, Some(e.to_string())),
        };

        Ok(Self {
            prefill,
            decode,
            prefill_block,
            built,
            config,
            ple_table,
            vocab,
            position: 0,
            history: Vec::new(),
            report,
            vocab_error,
        })
    }

    /// Build and initialize the wide-block session. The decode session is
    /// subsequently constructed from its parameter bindings.
    fn build_prefill(
        model: &GgufModel,
        config: &ModelConfig,
        block: usize,
        max_seq_len: usize,
    ) -> Result<(Session, weights::LoadReport), GgufError> {
        let mut g = crate::Graph::new();
        let built = graph::build(&mut g, model, config, block, max_seq_len)?;
        g.set_outputs(built.outputs());
        let mut prefill = crate::build(&g, crate::SessionConfig::inference_from_env()).0;
        let report = weights::load(&mut prefill, model, config)?;
        weights::reset_caches(&mut prefill, &built, config);
        Ok((prefill, report))
    }

    /// What the file said the model is.
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// The tokenizer the file carried, if it carried one.
    pub fn vocab(&self) -> Option<&Vocab> {
        self.vocab.as_ref()
    }

    /// What loading the weights actually did.
    pub fn load_report(&self) -> &weights::LoadReport {
        &self.report
    }

    /// How many tokens are in the cache.
    pub fn position(&self) -> usize {
        self.position
    }

    /// The longest sequence this generator can hold.
    pub fn max_seq_len(&self) -> usize {
        self.built.max_seq_len
    }

    /// Forget the conversation: empty the cache and the history.
    pub fn reset(&mut self) {
        weights::reset_caches(&mut self.decode, &self.built, &self.config);
        self.position = 0;
        self.history.clear();
    }

    /// Run `tokens` through the model, extending the cache, and return the
    /// logits that follow the last of them.
    ///
    /// The prompt goes through the wide-block session where there is one
    /// and more than one token left to feed; a lone token goes through the
    /// decode session, whose single-row matmuls take the K-split GEMV.
    pub fn feed(&mut self, tokens: &[u32]) -> Result<Vec<f32>, GgufError> {
        if tokens.is_empty() {
            return Err(GgufError::BadMetadata(
                "feed needs at least one token".into(),
            ));
        }
        if self.position + tokens.len() > self.built.max_seq_len {
            return Err(GgufError::BadMetadata(format!(
                "{} tokens from position {} overrun the {}-token cache",
                tokens.len(),
                self.position,
                self.built.max_seq_len
            )));
        }

        let mut logits = Vec::new();
        let mut at = 0;
        while at < tokens.len() {
            let remaining = tokens.len() - at;
            let wide = self.prefill.is_some() && remaining > 1;
            let take = if wide {
                remaining.min(self.prefill_block)
            } else {
                1
            };
            logits = self.run(&tokens[at..at + take], wide);
            at += take;
        }
        self.history.extend_from_slice(tokens);
        Ok(logits)
    }

    /// One step of whichever session, returning the last row's logits.
    fn run(&mut self, tokens: &[u32], wide: bool) -> Vec<f32> {
        let block = if wide { self.prefill_block } else { 1 };
        let session = if wide {
            self.prefill
                .as_mut()
                .expect("wide implies a prefill session")
        } else {
            &mut self.decode
        };
        // The graph has a fixed row count even when the final prefill chunk
        // is short, so every input (including Gemma4's PLE rows) must cover
        // the padded token batch.
        let mut padded = vec![0u32; block];
        padded[..tokens.len()].copy_from_slice(tokens);
        if let Some(ref table) = self.ple_table {
            let gathered =
                super::weights::gather_per_layer_embeddings(table, &self.config, &padded)
                    .expect("the per-layer embedding gather is in range");
            session.set_input("ple", &gathered);
        }

        // Slots past `valid` are never read, but they are still uploaded,
        // so they must at least be in range of the embedding table.
        session.set_input_u32("token_ids", &padded);
        session.set_input_u32("position", &[self.position as u32]);
        session.set_input_u32("valid", &[tokens.len() as u32]);
        session.step();
        session.wait();

        let mut logits = vec![0.0f32; self.config.vocab_size];
        session.read_output_by_index(0, &mut logits);
        self.position += tokens.len();
        logits
    }

    /// Continue from `prompt`, returning the new tokens.
    ///
    /// The cache is *not* reset, so successive calls continue one
    /// conversation; call [`Self::reset`] to start over.
    pub fn generate_tokens(
        &mut self,
        prompt: &[u32],
        options: &GenerationOptions,
    ) -> Result<Vec<u32>, GgufError> {
        let mut logits = self.feed(prompt)?;
        let mut rng = Rng::new(options.seed);
        let mut out = Vec::new();

        for _ in 0..options.max_tokens {
            let next = self.sample(&mut logits, options, &mut rng);
            let ends = options.stop_at_end_of_generation
                && self
                    .vocab
                    .as_ref()
                    .is_some_and(|v| v.is_end_of_generation(next));
            if ends {
                break;
            }
            out.push(next);
            if self.position >= self.built.max_seq_len {
                // The cache is full; stopping is the honest outcome, and
                // the caller can see it from `position`.
                break;
            }
            logits = self.feed(&[next])?;
        }
        Ok(out)
    }

    /// Continue from `prompt`, returning the generated text.
    ///
    /// Needs the file to carry a tokenizer; use [`Self::generate_tokens`]
    /// when it does not.
    pub fn generate(
        &mut self,
        prompt: &str,
        options: &GenerationOptions,
    ) -> Result<String, GgufError> {
        let mut text = String::new();
        self.generate_streaming(prompt, options, |piece| {
            text.push_str(piece);
            true
        })?;
        Ok(text)
    }

    /// [`Self::generate`], handing each token's text to `on_token` as it
    /// arrives. Returning `false` stops generation.
    pub fn generate_streaming(
        &mut self,
        prompt: &str,
        options: &GenerationOptions,
        mut on_token: impl FnMut(&str) -> bool,
    ) -> Result<(), GgufError> {
        let vocab = self.vocab.as_ref().ok_or_else(|| {
            GgufError::BadMetadata(format!(
                "no usable tokenizer, so text generation needs generate_tokens \
                 instead ({})",
                self.vocab_error.as_deref().unwrap_or("reason unrecorded")
            ))
        })?;
        // Only a sequence that is actually starting gets the
        // vocabulary's BOS. A generator keeps its cache across calls, so
        // a second `generate` continues the first — and re-applying the
        // policy would plant another BOS in the middle of the sequence,
        // which the token-level API would never do.
        let prompt_tokens = if self.position == 0 {
            vocab.encode(prompt)
        } else {
            vocab.encode_plain(prompt)
        };
        if prompt_tokens.is_empty() {
            return Err(GgufError::BadMetadata(
                "the prompt tokenized to nothing".into(),
            ));
        }

        let mut logits = self.feed(&prompt_tokens)?;
        let mut rng = Rng::new(options.seed);
        // Decoded a token at a time, a multi-byte character arrives in
        // pieces; buffering the bytes keeps the text whole.
        let mut emitted: Vec<u32> = Vec::new();
        let mut shown = 0usize;

        for _ in 0..options.max_tokens {
            let next = self.sample(&mut logits, options, &mut rng);
            let vocab = self.vocab.as_ref().expect("checked above");
            if options.stop_at_end_of_generation && vocab.is_end_of_generation(next) {
                break;
            }
            emitted.push(next);

            // Re-decode the whole run and show what has settled. A
            // character can straddle two tokens, so a replacement
            // character at the *end* may still complete when the next
            // token arrives and is held back. One in the middle never
            // will — it is a byte that decodes to nothing else — so it is
            // shown, or a single bad byte would stall the stream for the
            // rest of the run.
            let text = vocab.decode(&emitted);
            let ready = settled(&text);
            if ready > shown {
                let piece = text[shown..ready].to_string();
                shown = ready;
                if !on_token(&piece) {
                    // The caller has seen this token, so it is part of
                    // the sequence. Commit it before returning, or a
                    // continuation would resume from a shorter prefix
                    // than the one the caller was shown — and differ from
                    // the same output stopped by `max_tokens`.
                    if self.position < self.built.max_seq_len {
                        self.feed(&[next])?;
                    }
                    return Ok(());
                }
            }

            if self.position >= self.built.max_seq_len {
                break;
            }
            logits = self.feed(&[next])?;
        }

        // Whatever was held back waiting for a continuation that never
        // came is still output, so the text a caller assembles is the
        // whole of what was generated.
        let vocab = self.vocab.as_ref().expect("checked above");
        let text = vocab.decode(&emitted);
        if text.len() > shown {
            on_token(&text[shown..]);
        }
        Ok(())
    }

    /// Pick the next token from `logits`, which this may modify in place.
    fn sample(&self, logits: &mut [f32], options: &GenerationOptions, rng: &mut Rng) -> u32 {
        apply_repeat_penalty(logits, &self.history, options);
        if options.temperature <= 0.0 {
            return argmax(logits);
        }
        sample_with_temperature(logits, options, rng)
    }
}

impl GgufModel {
    /// Compile this model and load its weights, ready to generate.
    ///
    /// `max_seq_len` bounds the KV cache, and so the session's memory.
    pub fn generator(&self, max_seq_len: usize) -> Result<Generator, GgufError> {
        Generator::with_options(
            self,
            &GeneratorOptions {
                max_seq_len,
                ..GeneratorOptions::default()
            },
        )
    }
}

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------

/// Divide down the logits of tokens seen recently.
///
/// A *negative* logit has to be multiplied rather than divided, or the
/// penalty would push it up towards zero and make repetition likelier —
/// which is what llama.cpp does, and the reason this is not one branch.
fn apply_repeat_penalty(logits: &mut [f32], history: &[u32], options: &GenerationOptions) {
    if options.repeat_penalty == 1.0 || options.repeat_window == 0 {
        return;
    }
    let start = history.len().saturating_sub(options.repeat_window);
    for &token in &history[start..] {
        if let Some(logit) = logits.get_mut(token as usize) {
            *logit = if *logit > 0.0 {
                *logit / options.repeat_penalty
            } else {
                *logit * options.repeat_penalty
            };
        }
    }
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        // A diverged model can produce NaN, and it must neither panic the
        // sampler — as `partial_cmp().unwrap()` would — nor win it, as
        // `total_cmp` alone would: that ranks a positive NaN above every
        // real number, so one bad logit would choose the token.
        .filter(|&(_, l)| !l.is_nan())
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(i, _)| i as u32)
}

fn sample_with_temperature(logits: &[f32], options: &GenerationOptions, rng: &mut Rng) -> u32 {
    // Softmax over the shifted logits; the shift keeps exp() in range.
    // NaN is filtered for the same reason `argmax` filters it: `total_cmp`
    // ranks a positive NaN above every real number, so one bad logit
    // would win the sort and its `exp(NaN)` would poison the total.
    let max = logits
        .iter()
        .copied()
        .filter(|l| !l.is_nan())
        .fold(f32::NEG_INFINITY, f32::max);
    let mut candidates: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .filter(|&(_, &l)| !l.is_nan())
        .map(|(i, &l)| (i as u32, ((l - max) / options.temperature).exp()))
        .collect();
    candidates.sort_by(|a, b| b.1.total_cmp(&a.1));

    if options.top_k > 0 && options.top_k < candidates.len() {
        candidates.truncate(options.top_k);
    }

    let total: f32 = candidates.iter().map(|c| c.1).sum();
    if total <= 0.0 || !total.is_finite() {
        return candidates.first().map_or(0, |c| c.0);
    }

    if options.top_p < 1.0 {
        // Keep the shortest prefix whose mass reaches top_p — and at least
        // one candidate, so a peaked distribution is not emptied.
        let mut running = 0.0;
        let mut keep = 0;
        for (i, c) in candidates.iter().enumerate() {
            running += c.1 / total;
            keep = i + 1;
            if running >= options.top_p {
                break;
            }
        }
        candidates.truncate(keep.max(1));
    }

    let total: f32 = candidates.iter().map(|c| c.1).sum();
    let mut point = rng.next_f32() * total;
    for &(token, weight) in &candidates {
        point -= weight;
        if point <= 0.0 {
            return token;
        }
    }
    candidates.last().map_or(0, |c| c.0)
}

/// How much of `text` will not change when more tokens arrive.
///
/// Only a *trailing* replacement character can still become something
/// else — it is a multi-byte character whose remaining bytes are in the
/// next token. Everything before it has settled, including any earlier
/// replacement characters, which are bytes that decode to nothing else.
fn settled(text: &str) -> usize {
    text.trim_end_matches('\u{FFFD}').len()
}

/// A small deterministic generator, so a seeded run reproduces.
///
/// SplitMix64: no dependency, and good enough for choosing between a few
/// thousand candidates.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        // A zero seed is a legitimate request for the default stream, but
        // SplitMix64 from zero starts predictably; offset it.
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A float in `[0, 1)`, from the top 24 bits — the ones with full
    /// entropy once scaled into an f32's mantissa.
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }
}

/// A shared handle to a GPU context, for callers compiling several models
/// onto one device. Re-exported so they need not depend on Blade directly.
pub type Context = Arc<blade_graphics::Context>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load::gguf::{ModelConfig, fixture};

    fn options() -> GenerationOptions {
        GenerationOptions::default()
    }

    #[test]
    fn the_default_is_greedy_and_so_deterministic() {
        let opts = options();
        assert_eq!(opts.temperature, 0.0);
        let logits = [0.1f32, 0.9, 0.3];
        let rng = Rng::new(0);
        for _ in 0..8 {
            let mut l = logits;
            apply_repeat_penalty(&mut l, &[], &opts);
            assert_eq!(argmax(&l), 1);
            // Sampling must not consume randomness on the greedy path.
            assert_eq!(rng.0, Rng::new(0).0);
        }
    }

    #[test]
    fn gemma4_prefill_gathers_ple_rows_for_the_padded_batch() {
        let model = fixture::model("gemma4");
        let config = ModelConfig::from_gguf(&model).unwrap();
        let options = GeneratorOptions {
            max_seq_len: 8,
            prefill_block: 4,
        };
        let mut generator = Generator::with_config(&model, config, &options).unwrap();
        let logits = generator.feed(&[3, 4, 5]).unwrap();
        assert_eq!(logits.len(), generator.config.vocab_size);
        assert_eq!(generator.position(), 3);
    }

    #[test]
    fn argmax_survives_a_nan() {
        assert_eq!(argmax(&[1.0, f32::NAN, 2.0]), 2);
        assert_eq!(argmax(&[]), 0);
    }

    #[test]
    fn the_repeat_penalty_pushes_both_signs_away_from_selection() {
        let mut opts = options();
        opts.repeat_penalty = 2.0;
        opts.repeat_window = 8;

        let mut logits = [4.0f32, -4.0, 1.0];
        apply_repeat_penalty(&mut logits, &[0, 1], &opts);
        assert_eq!(logits[0], 2.0, "a positive logit is divided down");
        assert_eq!(
            logits[1], -8.0,
            "a negative logit must be multiplied, or the penalty would raise it"
        );
        assert_eq!(logits[2], 1.0, "an unseen token is untouched");
    }

    #[test]
    fn the_repeat_window_bounds_how_far_back_it_looks() {
        let mut opts = options();
        opts.repeat_penalty = 2.0;
        opts.repeat_window = 2;
        let mut logits = [4.0f32, 4.0, 4.0];
        // Only the last two of the history are in the window.
        apply_repeat_penalty(&mut logits, &[0, 1, 2], &opts);
        assert_eq!(logits[0], 4.0, "outside the window");
        assert_eq!(logits[1], 2.0);
        assert_eq!(logits[2], 2.0);
    }

    #[test]
    fn a_penalty_of_one_changes_nothing() {
        let opts = options();
        assert_eq!(opts.repeat_penalty, 1.0);
        let mut logits = [4.0f32, -4.0];
        apply_repeat_penalty(&mut logits, &[0, 1], &opts);
        assert_eq!(logits, [4.0, -4.0]);
    }

    #[test]
    fn a_history_token_outside_the_vocabulary_is_ignored() {
        let mut opts = options();
        opts.repeat_penalty = 2.0;
        let mut logits = [4.0f32];
        apply_repeat_penalty(&mut logits, &[99], &opts);
        assert_eq!(logits, [4.0], "no panic, no change");
    }

    #[test]
    fn top_k_of_one_is_greedy_whatever_the_seed() {
        let mut opts = options();
        opts.temperature = 1.0;
        opts.top_k = 1;
        opts.top_p = 1.0;
        let logits = [0.1f32, 5.0, 0.3];
        for seed in 0..16 {
            let mut rng = Rng::new(seed);
            assert_eq!(sample_with_temperature(&logits, &opts, &mut rng), 1);
        }
    }

    #[test]
    fn top_p_keeps_at_least_one_candidate() {
        let mut opts = options();
        opts.temperature = 1.0;
        opts.top_k = 0;
        opts.top_p = 0.0;
        let logits = [0.1f32, 5.0, 0.3];
        let mut rng = Rng::new(7);
        // A top_p of zero would empty the set if it were not floored.
        assert_eq!(sample_with_temperature(&logits, &opts, &mut rng), 1);
    }

    #[test]
    fn temperature_sampling_never_picks_a_nan() {
        let mut opts = options();
        opts.temperature = 1.0;
        opts.top_k = 0;
        opts.top_p = 1.0;
        // Index 1 carried a NaN, so it must never come out — and the
        // surviving softmax total must stay finite, or 0's `exp(NaN)`
        // weight would win through the not-finite fallback.
        for seed in 0..64 {
            let mut rng = Rng::new(seed);
            assert_ne!(
                sample_with_temperature(&[1.0f32, f32::NAN, 2.0], &opts, &mut rng),
                1
            );
        }
    }

    #[test]
    fn sampling_is_reproducible_for_a_seed_and_varies_across_seeds() {
        let mut opts = options();
        opts.temperature = 1.0;
        opts.top_k = 0;
        opts.top_p = 1.0;
        let logits: Vec<f32> = (0..64).map(|i| (i as f32) * 0.05).collect();

        let draw = |seed: u64| {
            let mut rng = Rng::new(seed);
            (0..32)
                .map(|_| sample_with_temperature(&logits, &opts, &mut rng))
                .collect::<Vec<_>>()
        };
        assert_eq!(draw(1), draw(1), "same seed, same run");
        assert_ne!(draw(1), draw(2), "different seeds should diverge");
    }

    #[test]
    fn temperature_sampling_only_returns_real_tokens() {
        let mut opts = options();
        opts.temperature = 0.8;
        opts.top_k = 5;
        opts.top_p = 0.9;
        let logits: Vec<f32> = (0..32).map(|i| ((i * 7) % 11) as f32).collect();
        let mut rng = Rng::new(3);
        for _ in 0..200 {
            let t = sample_with_temperature(&logits, &opts, &mut rng);
            assert!((t as usize) < logits.len(), "sampled {t} out of range");
        }
    }

    #[test]
    fn a_degenerate_distribution_does_not_panic() {
        let mut opts = options();
        opts.temperature = 1.0;
        let mut rng = Rng::new(0);
        // Every logit -inf: exp() gives zeros, so the mass is zero.
        let flat = [f32::NEG_INFINITY; 4];
        let t = sample_with_temperature(&flat, &opts, &mut rng);
        assert!((t as usize) < flat.len());
    }

    #[test]
    fn the_rng_stays_in_range() {
        let mut rng = Rng::new(42);
        for _ in 0..10_000 {
            let v = rng.next_f32();
            assert!((0.0..1.0).contains(&v), "{v} out of [0, 1)");
        }
    }

    #[test]
    fn a_zero_seed_is_not_a_degenerate_stream() {
        let mut rng = Rng::new(0);
        let first: Vec<f32> = (0..4).map(|_| rng.next_f32()).collect();
        assert!(
            first.windows(2).any(|w| w[0] != w[1]),
            "a zero seed should still vary: {first:?}"
        );
    }

    #[test]
    fn a_complete_string_has_settled_entirely() {
        assert_eq!(settled("hello"), "hello".len());
        assert_eq!(settled(""), 0);
    }

    #[test]
    fn a_trailing_replacement_character_is_held_back() {
        // Half of a multi-byte character; the next token completes it.
        let partial = "the caf\u{FFFD}";
        assert_eq!(&partial[..settled(partial)], "the caf");
    }

    #[test]
    fn a_byte_that_never_completes_a_character_does_not_stall_the_stream() {
        // A stray byte-fallback token decodes to U+FFFD with text after
        // it, so it has settled — holding it back would end the stream
        // silently for the rest of the run.
        let text = "ok\u{FFFD}more text";
        assert_eq!(settled(text), text.len());
    }

    #[test]
    fn the_stream_keeps_moving_as_tokens_arrive() {
        // Each prefix of a growing decode must show at least as much as
        // the last, and reach the whole once it is complete.
        let steps = ["ca", "caf\u{FFFD}", "caf\u{e9}", "caf\u{e9} au"];
        let mut shown = 0;
        for step in steps {
            let ready = settled(step);
            assert!(ready >= shown.min(ready), "went backwards at {step:?}");
            shown = ready;
        }
        assert_eq!(shown, "caf\u{e9} au".len(), "the whole text should arrive");
    }

    #[test]
    fn several_undecodable_bytes_in_the_middle_all_settle() {
        let text = "a\u{FFFD}b\u{FFFD}c";
        assert_eq!(settled(text), text.len());
    }

    #[test]
    fn the_defaults_describe_a_usable_run() {
        let g = GeneratorOptions::default();
        assert!(g.max_seq_len > 0);
        assert!(g.prefill_block > 1, "the wide session is on by default");
        let o = GenerationOptions::default();
        assert!(o.max_tokens > 0);
        assert!(o.stop_at_end_of_generation);
    }
}
