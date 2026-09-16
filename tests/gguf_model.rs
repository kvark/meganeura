//! End-to-end GGUF model loading: metadata to graph to generated tokens.
//!
//! The model is synthesized rather than read from disk — a real `.gguf`
//! would mean shipping weights — but it goes through exactly the public
//! path a file does, since [`GgufModel`] is what `load_gguf` returns.
//!
//! The load-bearing test here is
//! [`a_wide_prefill_agrees_with_one_token_at_a_time`]: the prefill and
//! decode sessions share their K/V cache buffers, so feeding a prompt as
//! one wide block and feeding it a token at a time must reach the same
//! logits. That single equality covers the cache aliasing, the
//! `cache_write_prefix` bounds, the per-row RoPE offsets and `prefix_last`
//! all at once — if any of them is off by one, the two paths diverge.

use std::collections::HashMap;

use meganeura::load::gguf::{
    GenerationOptions, Generator, GeneratorOptions, GgmlType, GgufModel, GgufTensor, GgufValue,
    ModelConfig,
};

const HIDDEN: usize = 64;
const LAYERS: usize = 2;
const HEADS: u32 = 4;
const KV_HEADS: u32 = 2;
const FFN: usize = 128;
const VOCAB: usize = 32;
const HEAD_DIM: usize = HIDDEN / HEADS as usize;
/// The vocabulary covers `VOCAB` consecutive bytes from here. Printable
/// ASCII, so a test prompt is readable and every byte has a token.
const FIRST_BYTE: u8 = b'a';
const KV_DIM: usize = KV_HEADS as usize * HEAD_DIM;

/// Small, bounded, and different at every index, so a transposed or
/// misindexed weight changes the answer rather than hiding in symmetry.
fn ramp(count: usize, salt: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(count * 4);
    for i in 0..count {
        let x = ((i * 37 + salt * 101) % 211) as f32 / 211.0 - 0.5;
        bytes.extend_from_slice(&(x * 0.25).to_le_bytes());
    }
    bytes
}

fn tensor(dims: Vec<usize>, salt: usize) -> GgufTensor {
    let count: usize = dims.iter().product();
    GgufTensor::new(dims, GgmlType::F32, ramp(count, salt))
}

/// A complete llama-shaped model: metadata, weights, and a tiny
/// byte-level BPE vocabulary.
fn tiny_llama() -> GgufModel {
    let mut metadata: HashMap<String, GgufValue> = HashMap::new();
    let mut set = |k: &str, v: GgufValue| {
        metadata.insert(k.to_string(), v);
    };
    set("general.architecture", GgufValue::String("llama".into()));
    set("llama.embedding_length", GgufValue::U32(HIDDEN as u32));
    set("llama.block_count", GgufValue::U32(LAYERS as u32));
    set("llama.attention.head_count", GgufValue::U32(HEADS));
    set("llama.attention.head_count_kv", GgufValue::U32(KV_HEADS));
    set("llama.feed_forward_length", GgufValue::U32(FFN as u32));
    set(
        "llama.attention.layer_norm_rms_epsilon",
        GgufValue::F32(1.0e-5),
    );
    set("llama.rope.freq_base", GgufValue::F32(10_000.0));
    set("llama.context_length", GgufValue::U32(512));

    // A byte-level vocabulary over a printable run, so ASCII text in this
    // range encodes to one token per byte and round-trips exactly.
    let tokens: Vec<GgufValue> = (0..VOCAB)
        .map(|i| GgufValue::String(byte_alphabet_char(FIRST_BYTE + i as u8).to_string()))
        .collect();
    set("tokenizer.ggml.model", GgufValue::String("gpt2".into()));
    set("tokenizer.ggml.tokens", GgufValue::Array(tokens));
    set("tokenizer.ggml.merges", GgufValue::Array(Vec::new()));
    set("tokenizer.ggml.add_bos_token", GgufValue::Bool(false));

    let mut tensors = HashMap::new();
    let mut salt = 0;
    let mut add = |tensors: &mut HashMap<String, GgufTensor>, name: &str, dims: Vec<usize>| {
        salt += 1;
        tensors.insert(name.to_string(), tensor(dims, salt));
    };

    // GGUF names dimensions fastest-first, so a weight is [K, N] and the
    // embedding table is [n_embd, n_vocab].
    add(&mut tensors, "token_embd.weight", vec![HIDDEN, VOCAB]);
    add(&mut tensors, "output_norm.weight", vec![HIDDEN]);
    for layer in 0..LAYERS {
        let p = format!("blk.{layer}");
        add(&mut tensors, &format!("{p}.attn_norm.weight"), vec![HIDDEN]);
        add(
            &mut tensors,
            &format!("{p}.attn_q.weight"),
            vec![HIDDEN, HIDDEN],
        );
        add(
            &mut tensors,
            &format!("{p}.attn_k.weight"),
            vec![HIDDEN, KV_DIM],
        );
        add(
            &mut tensors,
            &format!("{p}.attn_v.weight"),
            vec![HIDDEN, KV_DIM],
        );
        add(
            &mut tensors,
            &format!("{p}.attn_output.weight"),
            vec![HIDDEN, HIDDEN],
        );
        add(&mut tensors, &format!("{p}.ffn_norm.weight"), vec![HIDDEN]);
        add(
            &mut tensors,
            &format!("{p}.ffn_gate.weight"),
            vec![HIDDEN, FFN],
        );
        add(
            &mut tensors,
            &format!("{p}.ffn_up.weight"),
            vec![HIDDEN, FFN],
        );
        add(
            &mut tensors,
            &format!("{p}.ffn_down.weight"),
            vec![FFN, HIDDEN],
        );
    }

    GgufModel { metadata, tensors }
}

/// GPT-2's printable stand-in for a byte, matching the loader's own table.
fn byte_alphabet_char(b: u8) -> char {
    let printable = |b: u8| (b'!'..=b'~').contains(&b) || (0xA1..=0xAC).contains(&b) || b >= 0xAE;
    if printable(b) {
        return b as char;
    }
    let rank = (0..b).filter(|&x| !printable(x)).count();
    char::from_u32(0x100 + rank as u32).unwrap()
}

fn generator(prefill_block: usize) -> Generator {
    Generator::with_options(
        &tiny_llama(),
        &GeneratorOptions {
            max_seq_len: 32,
            prefill_block,
        },
    )
    .expect("the synthetic model describes a complete llama")
}

fn close(a: &[f32], b: &[f32], tolerance: f32) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| (x - y).abs() <= tolerance * (1.0 + x.abs().max(y.abs())))
}

#[test]
fn the_config_is_read_entirely_from_the_file() {
    let config = ModelConfig::from_gguf(&tiny_llama()).unwrap();
    assert_eq!(config.hidden_size, HIDDEN);
    assert_eq!(config.num_layers, LAYERS);
    assert_eq!(config.num_heads, HEADS);
    assert_eq!(config.num_kv_heads, KV_HEADS);
    assert_eq!(config.head_dim, HEAD_DIM as u32);
    assert_eq!(config.vocab_size, VOCAB);
    assert!(
        config.tie_word_embeddings,
        "no output.weight means a tied head"
    );
}

#[test]
fn a_loaded_model_produces_logits_for_the_whole_vocabulary() {
    let mut model_gen = generator(1);
    let logits = model_gen.feed(&[1, 2, 3]).unwrap();
    assert_eq!(logits.len(), VOCAB);
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "every logit should be finite: {logits:?}"
    );
    assert!(
        logits.windows(2).any(|w| w[0] != w[1]),
        "a uniform output would mean the weights never reached the GPU"
    );
    assert_eq!(model_gen.position(), 3);
}

/// The test this file exists for.
#[test]
fn a_wide_prefill_agrees_with_one_token_at_a_time() {
    let prompt: Vec<u32> = vec![3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5];

    // One wide block through the prefill session...
    let mut wide = generator(8);
    let wide_logits = wide.feed(&prompt).unwrap();

    // ...and the same prompt a token at a time through the decode session.
    let mut narrow = generator(1);
    let mut narrow_logits = Vec::new();
    for &token in &prompt {
        narrow_logits = narrow.feed(&[token]).unwrap();
    }

    assert_eq!(wide.position(), narrow.position());
    assert!(
        close(&wide_logits, &narrow_logits, 2.0e-3),
        "prefill and decode disagree.\n wide: {:?}\n narrow: {:?}",
        &wide_logits[..8.min(wide_logits.len())],
        &narrow_logits[..8.min(narrow_logits.len())]
    );
}

#[test]
fn a_prompt_longer_than_one_block_spans_several() {
    // 11 tokens through a 4-wide session is three steps, the last partial.
    let prompt: Vec<u32> = vec![3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5];
    let mut chunked = generator(4);
    let chunked_logits = chunked.feed(&prompt).unwrap();

    let mut narrow = generator(1);
    let mut narrow_logits = Vec::new();
    for &token in &prompt {
        narrow_logits = narrow.feed(&[token]).unwrap();
    }
    assert!(
        close(&chunked_logits, &narrow_logits, 2.0e-3),
        "a prompt spanning blocks should reach the same state"
    );
}

#[test]
fn continuing_after_a_prefill_matches_continuing_without_one() {
    let prompt: Vec<u32> = vec![7, 7, 1, 2];
    let options = GenerationOptions {
        max_tokens: 6,
        stop_at_end_of_generation: false,
        ..GenerationOptions::default()
    };

    let mut wide = generator(8);
    let from_wide = wide.generate_tokens(&prompt, &options).unwrap();

    let mut narrow = generator(1);
    let from_narrow = narrow.generate_tokens(&prompt, &options).unwrap();

    assert_eq!(
        from_wide, from_narrow,
        "the decode loop should not depend on how the prompt was fed"
    );
    assert_eq!(from_wide.len(), 6);
}

#[test]
fn greedy_generation_is_deterministic() {
    let options = GenerationOptions {
        max_tokens: 5,
        stop_at_end_of_generation: false,
        ..GenerationOptions::default()
    };
    let mut model_gen = generator(4);
    let first = model_gen.generate_tokens(&[2, 3, 4], &options).unwrap();
    model_gen.reset();
    let second = model_gen.generate_tokens(&[2, 3, 4], &options).unwrap();
    assert_eq!(first, second);
}

#[test]
fn resetting_returns_the_generator_to_an_empty_prefix() {
    let mut model_gen = generator(4);
    let before = model_gen.feed(&[5, 6, 7]).unwrap();
    assert_eq!(model_gen.position(), 3);

    model_gen.reset();
    assert_eq!(model_gen.position(), 0);
    let after = model_gen.feed(&[5, 6, 7]).unwrap();
    assert!(
        close(&before, &after, 1.0e-5),
        "a reset generator should see the same prompt the same way"
    );
}

#[test]
fn a_prompt_that_overruns_the_cache_is_an_error_not_a_corruption() {
    let mut model_gen = generator(4);
    let too_long: Vec<u32> = (0..64).map(|i| (i % VOCAB) as u32).collect();
    let err = model_gen.feed(&too_long).unwrap_err();
    let message = err.to_string();
    assert!(
        message.contains("overrun") || message.contains("cache"),
        "unhelpful message: {message}"
    );
}

#[test]
fn generation_stops_when_the_cache_fills() {
    let mut model_gen = generator(4);
    let options = GenerationOptions {
        max_tokens: 1000,
        stop_at_end_of_generation: false,
        ..GenerationOptions::default()
    };
    let out = model_gen.generate_tokens(&[1, 2], &options).unwrap();
    assert!(
        out.len() < 1000,
        "the loop should stop at the cache, not run to max_tokens"
    );
    assert!(model_gen.position() <= model_gen.max_seq_len());
}

#[test]
fn the_vocabulary_comes_from_the_file_and_round_trips() {
    let model_gen = generator(1);
    let vocab = model_gen.vocab().expect("the fixture carries a vocabulary");
    let text = "hello";
    let ids = vocab.encode(text);
    assert_eq!(ids.len(), 5, "no merges, so one token per byte");
    assert_eq!(vocab.decode(&ids), text);
}

#[test]
fn text_generation_goes_from_a_prompt_to_a_string() {
    let mut model_gen = generator(4);
    let options = GenerationOptions {
        max_tokens: 4,
        stop_at_end_of_generation: false,
        ..GenerationOptions::default()
    };
    let text = model_gen.generate("cab", &options).unwrap();
    // Every token in this vocabulary is one printable byte, so four
    // tokens is four characters.
    assert_eq!(text.chars().count(), 4, "got {text:?}");
    assert!(
        text.bytes()
            .all(|b| (FIRST_BYTE..FIRST_BYTE + VOCAB as u8).contains(&b)),
        "generation left the vocabulary's byte range: {text:?}"
    );
}

#[test]
fn every_weight_in_the_file_reaches_the_session() {
    let model_gen = generator(1);
    let report = model_gen.load_report();
    // Nine tensors per layer, plus the embedding table and the final norm.
    assert_eq!(report.packed + report.dequantized, LAYERS * 9 + 2);
    assert!(
        report.skipped.is_empty(),
        "nothing should be left unloaded: {:?}",
        report.skipped
    );
}

#[test]
fn an_architecture_with_no_builder_is_refused_by_name() {
    let mut model = tiny_llama();
    model.metadata.insert(
        "general.architecture".to_string(),
        GgufValue::String("mamba".to_string()),
    );
    let Err(err) = Generator::new(&model) else {
        panic!("an unknown architecture should not compile a graph");
    };
    assert!(
        err.to_string().contains("mamba"),
        "the error should name the architecture: {err}"
    );
}
