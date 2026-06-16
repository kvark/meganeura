# Magenta-RT LLM (Depthformer) — architecture notes & open questions

Status of the encoder-decoder generation model (`llm_base_x4286_c1860k`,
magenta-realtime **v1**). This is the analogue of `MUSICCOCA_FINDINGS.md`:
what's resolved, what's verified, and what still blocks an end-to-end
text→audio-tokens path. (`MODEL.md` on the public repo describes the **v2**
decoder-only 2.4B model — *not* this checkpoint.)

## What we know

- **Encoder-decoder T5X model.** Decoder output = 50 frames × 16 RVQ = 800
  tokens (2 s of audio). Encoder input = 1006 tokens (250 context frames × 4
  codec RVQ + 6 style tokens).
- **Decoder is a "Depthformer"** — two stacked modules:
  - a **temporal** module over the 50 frames (cross-attends to the encoder),
  - a **depth** module over the 16 RVQ levels within each frame (no
    cross-attention; conditioned on the temporal output — the RQ-Transformer
    pattern).
- **T5 1.1 components**: RMSNorm (T5LayerNorm, no centering/bias), GeGLU MLP
  (`wi_0` gate × `wi_1` up → `wo`), `DenseGeneral(use_bias=False)` everywhere.
- Config (`base`), confirmed against `weights_llm_base.safetensors` dims:
  embed=768, heads=12, head_dim=64, mlp=2048, vocab=29824, encoder=12 layers,
  temporal decoder=20 layers, depth decoder=4 layers.

## What's implemented + verified (Rust)

- `build_encoder_layer` / `build_encoder_graph` — T5 encoder (pre-existing).
- `build_decoder_layer` (**new**) — one temporal decoder layer: pre-norm
  causal self-attention (rel-pos bias) → pre-norm cross-attention to the
  encoder → pre-norm GeGLU MLP, all residual. This is a standard T5 1.1
  decoder layer.
  - **Verified** by `tests/llm_decoder_correctness.rs`: built with random
    weights and a *zero* rel-pos table (so self-attn is plain causal), it
    matches an independent CPU reference within 1e-3 on the GPU. This pins
    down the layer wiring (causal mask, cross-attention to a different-length
    encoder sequence, GeGLU) and meganeura's op composition — independent of
    the real weights.
- `sampling.rs` — CFG + temperature + top-k, host-side, complete (unwired).

## UNRESOLVED: position encoding (blocks encoder *and* decoder correctness)

The two in-repo references disagree, and neither is checked against the
checkpoint:

| source                | self-attn rel-pos bias | absolute position embedding |
|-----------------------|------------------------|-----------------------------|
| `llm.rs` (Rust)       | **yes** (T5 buckets)   | **yes** (`encoder.pos_embed`, sinusoidal) |
| `llm_numpy_ref.py`    | **no**                 | **yes** (sinusoidal, interleaved) |

- Standard T5 1.1 / flaxformer uses **rel-pos bias and NO absolute PE**. The
  numpy ref author noted the PE table is "not in checkpoint, computed" — which
  is exactly what you'd expect if the model uses rel-pos bias instead of a PE
  table. So the most likely truth is **rel-pos bias, no absolute PE** (the
  `llm.rs` `encoder.pos_embed` add is probably spurious, and the numpy ref's
  "no rel-pos bias" is probably wrong).
- **To settle**: dump the checkpoint tensor names (`dump_llm.py` writes
  `llm_base_manifest.json`) and check for a `relpos_bias`/`relative_attention`
  tensor vs a `position_embed` tensor. `build_decoder_layer` already uses
  rel-pos bias; if the manifest confirms it, drop the `pos_embed` add from
  `build_encoder_graph` to match.

## Remaining to reach text→tokens (in dependency order)

1. **Settle position encoding** (above) and add an encoder GPU-vs-reference
   check once the weight dump is present (the harness mirrors the MusicCoCa
   one).
2. **Depth decoder** (`build_depth_decoder_layer` + the temporal→depth
   wiring). Open question — needs the checkpoint to confirm: how the
   temporal module's per-frame output conditions the depth module (input
   prefix vs added embedding), and the exact level-embedding / logits layout.
   The depth module itself is a small causal transformer over 16 positions
   with its own rel-pos buckets (16/16 per the gin config) and **no**
   cross-attention.
3. **Weight-tied logits head** (shared token embedder), and the
   **temporal-stack graph** (token embed + position encoding + 20 layers +
   final norm), analogous to `build_encoder_graph`.
4. **KV cache + autoregressive generation loop**: 800-step decode that runs
   the encoder once, decodes frame-by-frame (temporal) and level-by-level
   (depth), reads `[2, vocab]` logits per step, and calls
   `super::sampling::{cfg_combine, top_k_sample}` (already implemented).
   Needs the CFG batch=2 broadcast-add the encoder currently asserts against.
5. **Weight loader**: map the T5X `target.*` tensor names onto the graph
   params (flaxformer naming → local names), like `musiccoca` and
   `spectrostream`.

## Why not just build it all now

Unlike MusicCoCa (whose architecture a prior session fully reverse-engineered
to 0.9993 cosine before the Rust port), the decoder's depth-module topology
and the position-encoding scheme are **not pinned down**, and nothing here can
be verified end-to-end without the weight dump (not in-tree). The temporal
decoder layer is the part that *is* standard-T5 and verifiable, so it's built
and GPU-checked; the rest is documented above rather than guessed.
