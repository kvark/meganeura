//! MusicCoCa: text/audio joint embedding model.
//!
//! Inputs: a text prompt (e.g., "synthwave") or a 16 kHz mono audio clip.
//! Output: a 768-dim style embedding (joint space), then RVQ-quantized into
//! 12 codes of 1024 entries each. Magenta-RT's LLM only consumes the first 6
//! of those 12 RVQ levels as `encoder_style_rvq_depth` tokens.
//!
//! For meganeura's first cut we only need the **text** branch (no audio
//! conditioning yet) plus the RVQ quantizer.
//!
//! Text path (from the official python `MusicCoCaV212F._embed_batch_text`):
//! - SentencePiece tokenizer (`musiccoca_vocab.model`) → up to 128 ids
//! - Prepend SOS=1, right-pad with 0; 128-element padding mask
//! - TF SavedModel `embed_text(inputs_0=ids, inputs_0_1=paddings) → contrastive_txt_embed[768]`
//!
//! The text encoder is a transformer of unknown depth/dim (until we see the
//! manifest). It's small enough (~1.16 GB SavedModel including audio branch)
//! that running it on CPU once per chunk is acceptable for now.
//!
//! TODO(blocked-on-manifest): reverse-engineer the text encoder structure from
//! the dumped variable list. SentencePiece integration via the `tokenizers`
//! crate already in meganeura's deps.

// Placeholder — see module doc.
