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
- `build_temporal_decoder_stack` (**new**) — the full parallel temporal
  forward: token embed → N temporal layers → final RMSNorm → transpose-
  weight-tied logits `[seq, vocab]`. No absolute PE (standard T5; see the
  open question below).
  - **Verified** by `tests/llm_decoder_stack_correctness.rs` (2-layer stack,
    GPU vs CPU within 1e-3): layer chaining, the embedding lookup, the final
    norm, and the weight-tied logits projection.
- `sampling.rs` — CFG + temperature + top-k, host-side, complete (unwired).

## RESOLVED: full tensor manifest (`llm_base_manifest.json`)

The HF egress block is lifted; `tools/magenta_rt/fetch_llm_manifest.py` now
lists all **430** `target.*` weights of `checkpoints/llm_base_x4286_c1860k`
(T5X/flaxformer tensorstore) with shapes, into
`tools/magenta_rt/llm_base_manifest.json` (metadata only, no weight download).
Confirmed dims: embed=768, heads=12, head_dim=64, mlp=2048, vocab=29824,
encoder=12 layers, temporal=20, depth=4. Flaxformer naming → graph remap for the
weight loader (`self_attention.{query,key,value,out}.kernel`,
`mlp.{wi_0,wi_1,wo}.kernel`, `pre_*_layer_norm.scale`, `relpos_bias.rel_embedding`).

## RESOLVED (mostly): position encoding

Settled from the manifest + the reference `magenta_rt/jax/depthformer.py`
(pip `magenta-rt`):

- **No absolute PE anywhere.** The checkpoint has no `pos_embed`/`position_embed`
  tensor, and `depthformer.py`'s encoder embedder leaves the position-embedding
  slot a **no-op** by default. The `encoder.pos_embed` add in `build_encoder_graph`
  and the numpy ref's sinusoidal PE are both **wrong** — drop them.
- **Decoders use learned T5 rel-pos bias**, shared per sub-decoder:
  `temporal_decoder.relpos_bias.rel_embedding` `[12, 128]` and
  `depth_decoder.relpos_bias_depth.rel_embedding` `[12, 16]`. So `build_decoder_layer`'s
  rel-pos bias is right; the per-layer registration should become per-sub-decoder
  (one shared table), and the temporal bucket count is **128**, not 32.
- **Encoder embedder applies `Scale(sqrt(embed))`** after the lookup
  (`scale_sqrt_depth`, PAX-compat) — `build_encoder_graph` is currently missing
  this multiply.
- **Open sub-question (encoder position):** the encoder carries **no** rel-pos
  tensor in the checkpoint (unlike both decoders) **and** no PE. Either the
  encoder uses a non-parametric scheme (the v2 `transformer.py` is RoPE-based,
  which leaves no checkpoint trace) or genuinely no position signal. This is the
  one thing the manifest can't settle (non-parametric ⇒ invisible). To pin down:
  read the v1 `mrt_merged_*.gin` body config, or a real-weight encoder check.
  **Until settled, leave the encoder's position handling untouched** beyond
  removing the clearly-spurious learned `pos_embed`.

## RESOLVED: logits are NOT weight-tied

The checkpoint carries `decoder.logits_dense.kernel` `[768, 29824]` **and**
`token_embedder.embedding` `[29824, 768]` as two distinct tensors. The final
`decoder_norm` `[768]` (shared, at `target.decoder.*`, above both sub-decoders)
is applied to the **depth** output, then `logits_dense` projects to vocab.
`build_temporal_decoder_stack`'s transpose-weight-tied logits is a placeholder
(its self-consistent CPU-ref test still passes); the real head lives in
`build_depth_decoder_stack` (new).

## RESOLVED: depth-module topology + temporal→depth wiring

From `MultivariateDecoder.layer_with_emits` in `magenta_rt/jax/depthformer.py`
(the RQ-Transformer pattern), with `x` the target token grid `(B, T, Q=16)`:

1. **Embed** every level with the *shared* token embedder → `(B, T+1, Q, D)`
   (input is left-padded one SOS frame).
2. **Temporal input** = mean over the 16 level-embeddings per frame, dropping
   the last frame: `embedded.mean(axis=levels)[:, :-1]` → `(B, T, D)`.
3. **Temporal body** (20 layers, causal self-attn + cross-attn to encoder) →
   `(B, T, D)`.
4. **Depth input** = `concat([temporal_out[:, :, None, :],
   embedded[:, 1:, :Q-1, :]], axis=levels)` → `(B, T, Q, D)`: position 0 is the
   temporal per-frame vector, positions 1..15 are the embeddings of the already-
   decoded levels 0..14 (teacher-forcing prefix). **This is how the temporal
   output conditions the depth module — as the level-0 input, not cross-attn
   and not an added embedding.**
5. **Depth body** (4 layers, causal self-attn over the 16 levels, no
   cross-attn), run per frame (flatten `B*T`) → `(B, T, Q, D)`.
6. **decoder_norm → logits_dense** → `(B, T, Q, vocab)`.

There is **no** separate depth level-embedding table and **no** depth-specific
logits head — both are shared with the temporal path (the manifest shows only
`token_embedder.embedding`, one `decoder_norm`, one `logits_dense`).

## What's now implemented (this session)

- `build_depth_decoder_layer` + `build_depth_decoder_stack` (**new**) — the
  per-frame depth core: 4 causal-self-attn-only T5 layers (depth rel-pos buckets
  16/16) → shared `decoder_norm` → non-tied `logits_dense` → `[16, vocab]`.
  Smoke-tested (`depth_decoder_stack_builds`); a GPU-vs-CPU correctness test
  mirroring `llm_decoder_stack_correctness.rs` is TODO (no Vulkan device in this
  container — verification is environment-gated, not knowledge-gated now).

## Remaining to reach text→tokens (in dependency order)

1. **Encoder fixes**: drop the spurious learned `pos_embed`; add
   `Scale(sqrt(embed))`; make the temporal rel-pos table shared per sub-decoder
   with bucket count 128. Settle the encoder position sub-question (above), then
   a real-weight encoder GPU-vs-reference check (mirrors MusicCoCa's).
2. **Full-decoder wiring**: embed levels → mean-pool temporal input → temporal
   stack → assemble depth inputs (concat) → depth stack, per the 6 steps above.
   Needs an axis-mean and a level-axis concat in the graph.
3. **KV cache + autoregressive loop**: 50 frames × 16 levels; run the encoder
   once; read `[2, vocab]` logits per step; call
   `super::sampling::{cfg_combine, top_k_sample}`. Needs the CFG batch=2
   broadcast-add the encoder currently asserts against.
4. **Weight loader**: map `target.*` (flaxformer) → graph params using
   `llm_base_manifest.json`, like `musiccoca`/`spectrostream`. Real-weight gate:
   greedy-decode logits match the Colab reference for a fixed seed.

## Status

The depth-module topology and the absolute-PE / untied-logits questions are now
**settled** from the checkpoint manifest + the reference `depthformer.py`. The
remaining unknown is narrow (the encoder's *position* scheme, which is
non-parametric and so invisible in the checkpoint) and is flagged, not guessed.
Everything correctness-real is now gated only on a Vulkan device + the weight
download, not on missing architecture knowledge.
