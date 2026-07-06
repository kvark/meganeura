//! End-to-end Magenta-RT driver: the deterministic host-side glue that ties the
//! verified components together into the text+context → 2 s audio pipeline.
//!
//! ```text
//! text prompt ─► MusicCoCa ─► 6 style tokens ┐
//! 10 s audio context ─► SpectroStream encode ─┴► assemble_encoder_input (1006)
//!                                                  │  LLM encoder (pos + neg)
//!                                                  ▼  LLM decode (CFG, 800 tok)
//!                                            SpectroStream decode ─► 2 s audio
//!                                                  │
//!                                            crossfade_chunks (40 ms overlap-add)
//! ```
//!
//! This module owns the parts that are pure, deterministic, and verifiable
//! *without* the real weights:
//!
//! - [`assemble_encoder_input`] — packs the 250 context frames × 4 codec RVQ
//!   levels + 6 style RVQ levels into the LLM's unified vocabulary (the exact
//!   per-level offset arithmetic and the context-then-style ordering), including
//!   the masked-style negative branch used for classifier-free guidance.
//! - [`llm_grid_to_rvq`] — the inverse map for the decoder's output: unified-
//!   vocab grid tokens → raw RVQ codebook indices (what the codec and the next
//!   chunk's context consume).
//! - [`StreamState`] — the rolling 10 s context window for **continuous**
//!   generation: each chunk conditions on the previous chunk's generated tokens
//!   (token-domain continuation), sliding the window after each step.
//! - [`crossfade_ramp`] / [`crossfade_chunks`] — the 40 ms (1 codec frame, 1920
//!   sample) overlap-add that stitches consecutive 2 s chunks into a continuous
//!   stream.
//! - [`run_encoder`] — the thin "run the LLM encoder once" helper that turns an
//!   assembled token sequence into the `[seq, embed]` encoder output that the
//!   temporal decoder cross-attends to.
//!
//! The heavy model stages it orchestrates — MusicCoCa text→style tokens, the
//! SpectroStream encoder (audio→codec tokens) and decoder (tokens→audio) — live
//! in their own modules, each with its own real-weight gate. [`generate_token_
//! grid`] is the weight-independent LLM orchestration (encoder + CFG decode) that
//! `tests/llm_end_to_end_driver.rs` drives on lavapipe. The full assembled
//! pipeline runs on real weights in two examples: `magenta_rt_generate.rs` (one
//! 2 s chunk from a text prompt) and `magenta_rt_stream.rs` (continuous multi-
//! chunk generation via [`StreamState`]).

use crate::runtime::Session;

use super::llm::{decode, DecodeOptions, LlmConfig};
use super::MagentaRtConfig;

/// "No history" sentinel for context codec tokens (the reference's `-1`):
/// [`assemble_encoder_input`] maps context positions holding this value to
/// [`MagentaRtConfig::vocab_mask_token`], mirroring magenta-realtime v1's
/// `np.where(context_tokens >= 0, rvq_to_llm(...), vocab_mask_token)`.
pub const NO_HISTORY: u32 = u32::MAX;

/// Rolling state for continuous (streaming) generation.
///
/// Magenta-RT generates audio 2 s at a time, each chunk conditioned on the
/// previous 10 s of **generated codec tokens** — token-domain continuation, with
/// no audio re-encoding in the hot loop (mirrors `MagentaRTState` /
/// `MagentaRTT5X.generate_chunk` in magenta-realtime v1). This holds that rolling
/// 10 s window so the driver can feed each chunk's context into
/// [`assemble_encoder_input`] and slide the window after each generation.
///
/// The buffer keeps the full [`MagentaRtConfig::decoder_codec_rvq_depth`] (16)
/// levels per frame (what the LLM produces and the decoder consumes); the encoder
/// context conditioning takes only the first
/// [`MagentaRtConfig::encoder_codec_rvq_depth`] (4) of them — see
/// [`StreamState::context_codec`].
#[derive(Clone, Debug)]
pub struct StreamState {
    /// `context_length_frames` (250) × `decoder_codec_rvq_depth` (16) raw codebook
    /// indices, frame-major (frame 0's 16 levels, then frame 1's, …).
    context_tokens: Vec<u32>,
    ctx_frames: usize,
    full_depth: usize,
    enc_depth: usize,
}

impl StreamState {
    /// A cold start: no audio history. Every context frame is seeded with
    /// [`NO_HISTORY`], which [`assemble_encoder_input`] maps to the mask token —
    /// exactly the reference's `np.full(context_tokens_shape, -1)` seeding.
    ///
    /// The cold-start frames wash out of the 250-frame window gradually: with
    /// 50-frame chunks they survive the first `250/50 = 5` chunks (the first
    /// 10 s condition partly on masked "no history" context), after which the
    /// window is entirely generated tokens.
    pub fn cold_start(cfg: &MagentaRtConfig) -> Self {
        let ctx_frames = cfg.context_length_frames() as usize;
        let full_depth = cfg.decoder_codec_rvq_depth as usize;
        StreamState {
            context_tokens: vec![NO_HISTORY; ctx_frames * full_depth],
            ctx_frames,
            full_depth,
            enc_depth: cfg.encoder_codec_rvq_depth as usize,
        }
    }

    /// The `context_length_frames * encoder_codec_rvq_depth` (250 × 4 = 1000)
    /// context tokens [`assemble_encoder_input`] consumes: the first `enc_depth`
    /// levels of every context frame, frame-major.
    pub fn context_codec(&self) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.ctx_frames * self.enc_depth);
        for frame in self.context_tokens.chunks_exact(self.full_depth) {
            out.extend_from_slice(&frame[..self.enc_depth]);
        }
        out
    }

    /// The last context frame (all `full_depth` = 16 levels): the boundary frame a
    /// streaming decoder prepends to the next chunk's grid so consecutive decoded
    /// audio chunks share exactly one overlap frame to crossfade over.
    ///
    /// On a fresh [`StreamState::cold_start`] (before any [`Self::push_chunk`])
    /// this is all [`NO_HISTORY`] — there is no previous chunk to seam with, so a
    /// streaming decoder must decode chunk 0's grid *without* a boundary prefix
    /// (check [`Self::has_history`]).
    pub fn boundary_frame(&self) -> &[u32] {
        &self.context_tokens[(self.ctx_frames - 1) * self.full_depth..]
    }

    /// Whether any chunk has been pushed (i.e. [`Self::boundary_frame`] holds
    /// real generated tokens rather than the cold-start [`NO_HISTORY`] fill).
    pub fn has_history(&self) -> bool {
        self.boundary_frame().iter().all(|&t| t != NO_HISTORY)
    }

    /// Slide the window after generating a chunk: drop the oldest `grid_frames`
    /// and append the freshly generated grid (`grid_frames * full_depth` raw
    /// indices, frame-major; 50 × 16 = 800 in production). The chunk size is
    /// inferred from the grid, so a shorter grid (e.g. a reduced-frame smoke run)
    /// just slides the window by fewer frames.
    pub fn push_chunk(&mut self, grid: &[u32]) {
        assert_eq!(
            grid.len() % self.full_depth,
            0,
            "grid length {} must be a multiple of {} levels",
            grid.len(),
            self.full_depth,
        );
        let grid_frames = grid.len() / self.full_depth;
        assert!(
            grid_frames <= self.ctx_frames,
            "chunk ({grid_frames} frames) larger than the {} -frame context window",
            self.ctx_frames,
        );
        self.context_tokens.drain(..grid_frames * self.full_depth);
        self.context_tokens.extend_from_slice(grid);
        debug_assert_eq!(self.context_tokens.len(), self.ctx_frames * self.full_depth);
    }
}

/// Assemble the LLM encoder input sequence (length
/// [`MagentaRtConfig::encoder_input_length`] = 1006) from raw codec + style RVQ
/// tokens, mapping each into the model's unified vocabulary.
///
/// Layout (mirrors magenta-realtime v1: "(Context, 1000 tokens) then (Style, 6
/// tokens)"):
///
/// - **Context** — `context_length_frames` (250) frames, each `encoder_codec_
///   rvq_depth` (4) codec tokens, **frame-major** (frame 0's 4 levels, then frame
///   1's, …). `context_codec` must therefore be exactly `250 * 4 = 1000` raw
///   codebook indices in `[0, codec_rvq_codebook_size)`. Level `q`'s index `v`
///   maps to `vocab_codec_offset + q * codec_rvq_codebook_size + v` — the same
///   per-level slotting the 16-level decoder vocabulary uses (the encoder takes
///   the first 4 of those 16 level ranges).
/// - **Style** — `encoder_style_rvq_depth` (6) MusicCoCa RVQ tokens. Level `s`'s
///   index `v` maps to `vocab_style_offset + s * style_rvq_codebook_size + v`.
///   `style = None` fills all 6 slots with `vocab_mask_token` — the masked
///   ("unconditional") style used for the classifier-free-guidance negative pass.
///
/// The per-level offsets are exactly why `vocab_codec_size == 16 *
/// codec_rvq_codebook_size` and `vocab_style_size == 6 * style_rvq_codebook_size`
/// in [`MagentaRtConfig`]: each RVQ level occupies its own contiguous codebook-
/// sized range, so the model can tell levels apart by value alone.
///
/// Panics if the input lengths or token ranges are wrong (keeps the caller
/// honest — these are exact, fixed-size packings).
pub fn assemble_encoder_input(
    context_codec: &[u32],
    style: Option<&[u32]>,
    cfg: &MagentaRtConfig,
) -> Vec<u32> {
    let ctx_frames = cfg.context_length_frames() as usize;
    let ctx_depth = cfg.encoder_codec_rvq_depth as usize;
    let style_depth = cfg.encoder_style_rvq_depth as usize;
    let codec_cb = cfg.codec_rvq_codebook_size;
    let style_cb = cfg.style_rvq_codebook_size;

    assert_eq!(
        context_codec.len(),
        ctx_frames * ctx_depth,
        "context_codec must be {ctx_frames} frames × {ctx_depth} levels",
    );
    if let Some(s) = style {
        assert_eq!(s.len(), style_depth, "style must be {style_depth} tokens");
    }

    let mut out = Vec::with_capacity(cfg.encoder_input_length() as usize);

    // --- Context: frame-major, per-level codec offsets. NO_HISTORY positions
    // (the reference's -1 "no audio yet" sentinel) become the mask token. ---
    let codec_base = cfg.vocab_codec_offset();
    for frame in context_codec.chunks_exact(ctx_depth) {
        for (level, &raw) in frame.iter().enumerate() {
            if raw == NO_HISTORY {
                out.push(cfg.vocab_mask_token());
                continue;
            }
            assert!(raw < codec_cb, "codec token {raw} ≥ codebook {codec_cb}");
            out.push(codec_base + level as u32 * codec_cb + raw);
        }
    }

    // --- Style: per-level style offsets, or all-masked for the CFG negative ---
    let style_base = cfg.vocab_style_offset();
    match style {
        Some(s) => {
            for (level, &raw) in s.iter().enumerate() {
                assert!(raw < style_cb, "style token {raw} ≥ codebook {style_cb}");
                out.push(style_base + level as u32 * style_cb + raw);
            }
        }
        None => out.extend(std::iter::repeat_n(cfg.vocab_mask_token(), style_depth)),
    }

    debug_assert_eq!(out.len(), cfg.encoder_input_length() as usize);
    out
}

/// Convert the LLM decoder's grid of **unified-vocab** tokens into raw RVQ
/// codebook indices (mirrors `utils.llm_to_rvq` in magenta-realtime).
///
/// The decoder samples over the full model vocabulary; a codec token for level
/// `q` is laid out as `vocab_codec_offset + q * codec_rvq_codebook_size + raw`
/// (the same per-level slotting [`assemble_encoder_input`] re-applies). This
/// recovers `raw ∈ [0, codec_rvq_codebook_size)` for each of the
/// `decoder_codec_rvq_depth` (16) levels per frame, so the grid can drive
/// [`super::spectrostream::dequantize_tokens`] (raw codebook indices) and feed
/// the next chunk's context via [`StreamState::push_chunk`].
///
/// `grid.len()` must be a multiple of `decoder_codec_rvq_depth`. The recovery is
/// **value-based**, exactly the reference's `llm_to_rvq(safe=False)`:
/// `max(t - offset, 0) % codebook_size` — a token the model sampled into the
/// *wrong level's* range (rare, but the decoder samples the full vocabulary)
/// still recovers its in-codebook index rather than being clamped to an extreme
/// entry. `out_of_range` counts tokens whose value-derived level disagreed with
/// their grid position (the condition the reference warns on), so callers can
/// surface it.
pub fn llm_grid_to_rvq(grid: &[u32], cfg: &MagentaRtConfig) -> (Vec<u32>, usize) {
    let depth = cfg.decoder_codec_rvq_depth as usize;
    let cb = cfg.codec_rvq_codebook_size;
    let off = cfg.vocab_codec_offset();
    assert_eq!(
        grid.len() % depth,
        0,
        "grid length {} must be a multiple of {depth} levels",
        grid.len(),
    );
    let mut out_of_range = 0usize;
    let raw = grid
        .iter()
        .enumerate()
        .map(|(i, &t)| {
            let q = (i % depth) as u32;
            let r = t.saturating_sub(off); // np.maximum(t - offset, 0)
            if t < off || r / cb != q {
                out_of_range += 1;
            }
            r % cb
        })
        .collect();
    (raw, out_of_range)
}

/// Crossfade style for [`crossfade_ramp`] (mirrors `audio.crossfade_ramp` in the
/// magenta-realtime reference).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CrossfadeStyle {
    /// Linear ramp `0 → 1`; `fade_in + fade_out == 1` (preserves a constant /
    /// DC-coherent signal across the seam).
    Linear,
    /// Equal-power ramp `sin(0 → π/2)`; `fade_in² + fade_out² == 1` (preserves
    /// signal power across an incoherent seam). This is the reference default.
    EqPower,
}

/// The fade-in ramp over `num_samples`, matching `np.linspace(..., endpoint=
/// False)` in the reference: sample `i` is the weight applied to the *incoming*
/// chunk at overlap position `i` (`t = i / num_samples`, so it starts at 0 and
/// reaches full weight only at the first post-overlap sample). The matching
/// fade-out (outgoing) weight is the style's complement — `1 - t` for `Linear`
/// (so `fade_in + fade_out == 1`, DC-preserving) and `cos(t·π/2)` for `EqPower`
/// (so `fade_in² + fade_out² == 1`, power-preserving) — see [`fade_out`].
pub fn crossfade_ramp(num_samples: usize, style: CrossfadeStyle) -> Vec<f32> {
    (0..num_samples)
        .map(|i| {
            let t = i as f32 / num_samples as f32; // endpoint=False
            match style {
                CrossfadeStyle::Linear => t,
                CrossfadeStyle::EqPower => (t * std::f32::consts::FRAC_PI_2).sin(),
            }
        })
        .collect()
}

/// The outgoing (fade-out) weight complementing a [`crossfade_ramp`] fade-in:
/// `1 - fade_in` for `Linear`, `sqrt(1 - fade_in²)` (= `cos` of the same angle)
/// for `EqPower`.
fn fade_out(fade_in: f32, style: CrossfadeStyle) -> f32 {
    match style {
        CrossfadeStyle::Linear => 1.0 - fade_in,
        CrossfadeStyle::EqPower => (1.0 - fade_in * fade_in).max(0.0).sqrt(),
    }
}

/// Overlap-add a sequence of equal-rate audio chunks, crossfading each chunk's
/// first `overlap` samples (per channel) into the previous chunk's last `overlap`
/// samples. Samples are interleaved by channel (`[s0c0, s0c1, s1c0, …]`).
///
/// For Magenta-RT this stitches the 2 s (96000-sample) chunks with a 40 ms
/// (1920-sample = 1 codec frame) overlap, so `N` chunks yield
/// `N * chunk_len - (N - 1) * overlap` samples. The blended seam is
/// `prev_tail * fade_out + next_head * fade_in` with [`crossfade_ramp`].
///
/// Each chunk must be at least `overlap` samples long; the first chunk is copied
/// whole, and every later chunk contributes its `overlap..` remainder after the
/// blended seam.
pub fn crossfade_chunks(
    chunks: &[&[f32]],
    overlap: usize,
    channels: usize,
    style: CrossfadeStyle,
) -> Vec<f32> {
    if chunks.is_empty() {
        return Vec::new();
    }
    let ramp = crossfade_ramp(overlap, style);
    let mut out: Vec<f32> = chunks[0].to_vec();

    for chunk in &chunks[1..] {
        assert!(
            chunk.len() >= overlap * channels,
            "chunk shorter than the crossfade overlap",
        );
        let seam_start = out.len() - overlap * channels;
        // Blend the overlap region in place: out_tail ← out_tail*fade_out + head*fade_in.
        for (i, &fi) in ramp.iter().enumerate() {
            let fo = fade_out(fi, style);
            for c in 0..channels {
                let o = seam_start + i * channels + c;
                let h = i * channels + c;
                out[o] = out[o] * fo + chunk[h] * fi;
            }
        }
        // Append the remainder past the overlap.
        out.extend_from_slice(&chunk[overlap * channels..]);
    }
    out
}

/// Run the LLM encoder once on an assembled token sequence, returning its
/// `[seq, embed]` output (the key/value source the temporal decoder cross-
/// attends to).
///
/// `session` must wrap a [`super::llm::build_encoder_graph`] graph (input
/// `encoder_input_tokens`, weights already loaded) sized for `tokens.len()`.
pub fn run_encoder(session: &mut Session, tokens: &[u32], cfg: &LlmConfig) -> Vec<f32> {
    session.set_input_u32("encoder_input_tokens", tokens);
    session.step();
    session.wait();
    session.read_output(tokens.len() * cfg.embed_dim as usize)
}

/// Orchestrate one LLM chunk: encode the positive (and, for CFG, the masked
/// negative) token sequences, then run the CFG decode loop to produce the
/// `[num_frames * num_levels]` RVQ token grid.
///
/// This is the weight-independent half of the pipeline — given the assembled
/// encoder inputs and the built/loaded sessions, it is exactly what the deployed
/// system runs (the only stubbed inputs are the *raw* codec/style tokens, which
/// come from the weight-gated SpectroStream/MusicCoCa front-ends).
///
/// Sessions (built once by the caller, weights loaded, caches zeroed):
/// - `encoder` — runs both passes (re-`set_input` + re-run per pass).
/// - `temporal_pos` / `temporal_neg` — [`super::llm::build_temporal_decode_step`]
///   graphs; `enc_out` is set here from the corresponding encoder pass.
///   `temporal_neg = None` ⇒ no guidance (single pass).
/// - `depth` — the shared [`super::llm::build_depth_decoder_stack`] graph.
#[allow(clippy::too_many_arguments)]
pub fn generate_token_grid(
    cfg: &LlmConfig,
    encoder: &mut Session,
    pos_tokens: &[u32],
    neg_tokens: Option<&[u32]>,
    temporal_pos: &mut Session,
    mut temporal_neg: Option<&mut Session>,
    depth: &mut Session,
    embed_table: &[f32],
    opts: &DecodeOptions,
) -> Vec<u32> {
    let enc_pos = run_encoder(encoder, pos_tokens, cfg);
    temporal_pos.set_input("enc_out", &enc_pos);

    if let (Some(neg), Some(tneg)) = (neg_tokens, temporal_neg.as_deref_mut()) {
        let enc_neg = run_encoder(encoder, neg, cfg);
        tneg.set_input("enc_out", &enc_neg);
    }

    decode(cfg, temporal_pos, temporal_neg, depth, embed_table, opts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoder_input_layout_and_offsets() {
        let cfg = MagentaRtConfig::default();
        let frames = cfg.context_length_frames() as usize;
        let depth = cfg.encoder_codec_rvq_depth as usize;

        // Context: every token = codec_offset + level*codebook + raw.
        let context: Vec<u32> = (0..frames * depth).map(|i| (i % 1024) as u32).collect();
        let style = [0u32, 1, 2, 3, 4, 5];
        let seq = assemble_encoder_input(&context, Some(&style), &cfg);

        assert_eq!(seq.len(), cfg.encoder_input_length() as usize);
        // Spot-check the first frame's 4 levels.
        for level in 0..depth {
            let raw = context[level];
            let want = cfg.vocab_codec_offset() + level as u32 * cfg.codec_rvq_codebook_size + raw;
            assert_eq!(seq[level], want, "context level {level}");
        }
        // Style is the last 6 tokens, per-level offset.
        let style_start = frames * depth;
        for (s, &raw) in style.iter().enumerate() {
            let want = cfg.vocab_style_offset() + s as u32 * cfg.style_rvq_codebook_size + raw;
            assert_eq!(seq[style_start + s], want, "style level {s}");
        }
        // Every token lands inside the model vocab.
        assert!(seq.iter().all(|&t| t < cfg.vocab_size()));
    }

    #[test]
    fn masked_style_is_all_mask_token() {
        let cfg = MagentaRtConfig::default();
        let frames = cfg.context_length_frames() as usize;
        let depth = cfg.encoder_codec_rvq_depth as usize;
        let context = vec![0u32; frames * depth];
        let pos = assemble_encoder_input(&context, Some(&[0, 0, 0, 0, 0, 0]), &cfg);
        let neg = assemble_encoder_input(&context, None, &cfg);

        // The context prefix is identical; only the 6 style slots differ.
        let style_start = frames * depth;
        assert_eq!(pos[..style_start], neg[..style_start]);
        for &t in &neg[style_start..] {
            assert_eq!(t, cfg.vocab_mask_token());
        }
    }

    #[test]
    fn llm_grid_to_rvq_roundtrips_assembled_layout() {
        let cfg = MagentaRtConfig::default();
        let depth = cfg.decoder_codec_rvq_depth as usize; // 16
        let cb = cfg.codec_rvq_codebook_size;
        // Build a 2-frame grid of known raws, encode to vocab tokens exactly the
        // way the model's vocab is laid out, then recover the raws.
        let raws: Vec<u32> = (0..2 * depth).map(|i| (i as u32 * 37) % cb).collect();
        let grid: Vec<u32> = raws
            .iter()
            .enumerate()
            .map(|(i, &r)| cfg.vocab_codec_offset() + (i % depth) as u32 * cb + r)
            .collect();
        let (got, oor) = llm_grid_to_rvq(&grid, &cfg);
        assert_eq!(oor, 0);
        assert_eq!(got, raws);
        assert!(got.iter().all(|&t| t < cb));
    }

    #[test]
    fn llm_grid_to_rvq_recovers_out_of_range_by_value() {
        let cfg = MagentaRtConfig::default();
        let depth = cfg.decoder_codec_rvq_depth as usize;
        let cb = cfg.codec_rvq_codebook_size;
        // A valid one-frame grid (level q's raw 0), then corrupt two entries:
        // level 0 gets a stray pad token (below the codec range); level 1 gets a
        // token from level 2's range. Both are counted, and the value-based
        // `max(t-off, 0) % cb` recovery (reference `llm_to_rvq(safe=False)`)
        // still yields the encoded in-codebook index.
        let mut grid: Vec<u32> =
            (0..depth).map(|q| cfg.vocab_codec_offset() + q as u32 * cb).collect();
        grid[0] = 0; // pad, below offset → max(t-off,0) = 0
        grid[1] = cfg.vocab_codec_offset() + 2 * cb + 5; // level 2's raw 5, in level 1's slot
        let (got, oor) = llm_grid_to_rvq(&grid, &cfg);
        assert_eq!(oor, 2);
        assert_eq!(got[0], 0);
        assert_eq!(got[1], 5); // modulo recovers raw 5, not a clamped extreme
        assert!(got.iter().all(|&t| t < cb));
    }

    #[test]
    fn streamstate_cold_start_is_masked_no_history() {
        let cfg = MagentaRtConfig::default();
        let st = StreamState::cold_start(&cfg);
        // context_codec is exactly the 1000-token context assemble_encoder_input wants.
        let ctx = st.context_codec();
        assert_eq!(
            ctx.len(),
            (cfg.context_length_frames() * cfg.encoder_codec_rvq_depth) as usize
        );
        assert!(ctx.iter().all(|&t| t == NO_HISTORY));
        assert!(!st.has_history());
        // Assembling a cold-start context maps every context slot to the mask
        // token (the reference's np.where(context >= 0, ..., mask)).
        let seq = assemble_encoder_input(&ctx, Some(&[0, 0, 0, 0, 0, 0]), &cfg);
        assert_eq!(seq.len(), cfg.encoder_input_length() as usize);
        let style_start = seq.len() - cfg.encoder_style_rvq_depth as usize;
        assert!(seq[..style_start].iter().all(|&t| t == cfg.vocab_mask_token()));
        // boundary frame is one full-depth (16-level) frame.
        assert_eq!(st.boundary_frame().len(), cfg.decoder_codec_rvq_depth as usize);
    }

    #[test]
    fn streamstate_push_slides_window_and_keeps_first_levels() {
        let cfg = MagentaRtConfig::default();
        let frames = cfg.chunk_length_frames() as usize; // 50
        let depth = cfg.decoder_codec_rvq_depth as usize; // 16
        let enc = cfg.encoder_codec_rvq_depth as usize; // 4
        let mut st = StreamState::cold_start(&cfg);

        // A grid whose every level value encodes its frame index (mod codebook).
        let grid: Vec<u32> = (0..frames * depth)
            .map(|i| ((i / depth) as u32 + 1) % cfg.codec_rvq_codebook_size)
            .collect();
        st.push_chunk(&grid);

        // The last 50 context frames are now the pushed grid; the boundary frame
        // is the grid's final frame.
        let ctx_frames = cfg.context_length_frames() as usize; // 250
        let bf = st.boundary_frame();
        assert_eq!(bf, &grid[(frames - 1) * depth..]);

        // context_codec keeps the first `enc` levels of every frame, frame-major,
        // and the newest 50 frames mirror the pushed grid's first `enc` levels.
        let ctx = st.context_codec();
        assert_eq!(ctx.len(), ctx_frames * enc);
        let tail_start = (ctx_frames - frames) * enc;
        for f in 0..frames {
            for l in 0..enc {
                assert_eq!(ctx[tail_start + f * enc + l], grid[f * depth + l]);
            }
        }
        // The oldest frames slid out: with a single push onto a cold start, the
        // pre-tail context is still the NO_HISTORY fill (but the boundary frame
        // is now real, so has_history is true).
        assert!(ctx[..tail_start].iter().all(|&t| t == NO_HISTORY));
        assert!(st.has_history());
    }

    #[test]
    fn streamstate_two_pushes_drop_oldest() {
        let cfg = MagentaRtConfig::default();
        let frames = cfg.chunk_length_frames() as usize;
        let depth = cfg.decoder_codec_rvq_depth as usize;
        let mut st = StreamState::cold_start(&cfg);
        let grid_a = vec![7u32; frames * depth];
        let grid_b = vec![9u32; frames * depth];
        st.push_chunk(&grid_a);
        st.push_chunk(&grid_b);
        // After two pushes the 250-frame window holds 150 cold-start NO_HISTORY
        // frames, then grid_a's 50 frames, then grid_b's 50 frames.
        let ctx_frames = cfg.context_length_frames() as usize;
        let enc = cfg.encoder_codec_rvq_depth as usize;
        let ctx = st.context_codec();
        let a_start = (ctx_frames - 2 * frames) * enc;
        let b_start = (ctx_frames - frames) * enc;
        assert!(ctx[..a_start].iter().all(|&t| t == NO_HISTORY));
        assert!(ctx[a_start..b_start].iter().all(|&t| t == 7));
        assert!(ctx[b_start..].iter().all(|&t| t == 9));
        assert_eq!(st.boundary_frame(), &grid_b[(frames - 1) * depth..]);
    }

    #[test]
    fn linear_crossfade_reconstructs_a_constant() {
        // Linear fade_in + fade_out == 1, so a DC-coherent seam is lossless.
        let overlap = 8;
        let chunk_a = vec![1.0f32; 20]; // mono
        let chunk_b = vec![1.0f32; 20];
        let out = crossfade_chunks(&[&chunk_a, &chunk_b], overlap, 1, CrossfadeStyle::Linear);
        assert_eq!(out.len(), 20 + 20 - overlap);
        for (i, &v) in out.iter().enumerate() {
            assert!((v - 1.0).abs() < 1e-6, "sample {i} = {v}");
        }
    }

    #[test]
    fn eqpower_crossfade_preserves_power_and_endpoints() {
        let overlap = 16;
        let ramp = crossfade_ramp(overlap, CrossfadeStyle::EqPower);
        // fade_in² + fade_out² == 1 across the seam (equal power).
        for i in 0..overlap {
            let fi = ramp[i];
            let fo = fade_out(fi, CrossfadeStyle::EqPower);
            assert!((fi * fi + fo * fo - 1.0).abs() < 1e-5, "power at {i}");
        }
        // endpoint=False ⇒ ramp starts at exactly 0 (incoming fully faded out).
        assert!(ramp[0].abs() < 1e-6);
    }

    #[test]
    fn crossfade_stitch_lengths_and_seam() {
        let overlap = 4;
        let channels = 2;
        // Two stereo chunks of 6 samples each.
        let a: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let b: Vec<f32> = (0..12).map(|x| 100.0 + x as f32).collect();
        let out = crossfade_chunks(&[&a, &b], overlap, channels, CrossfadeStyle::Linear);
        // N*chunk - (N-1)*overlap, in samples × channels.
        assert_eq!(out.len(), (6 + 6 - overlap) * channels);
        // The pre-seam prefix is a's first (6-overlap)=2 samples untouched.
        assert_eq!(
            &out[..(6 - overlap) * channels],
            &a[..(6 - overlap) * channels]
        );
    }
}
