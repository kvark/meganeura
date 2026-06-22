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

## Position encoding — RESOLVED from the recovered v1 source (gin config)

> 🔑 **The v1 reference code is available after all** — in the **git history** of
> `magenta/magenta-realtime`. The current `main` is the **v2** model; the **v1**
> encoder-decoder lives at the *initial commit* `b35a850` (`magenta_rt/
> depthformer/{model,modules,decode}.py`, `system.py`, and the gin configs under
> `depthformer/configs/`). A shallow depth-1 clone of `main` (what we had) hides
> it; `git clone` (full) exposes it. This supersedes the earlier "no v1 source /
> can't settle PE from the weights" caveat — we can read the architecture
> directly. (An even earlier draft claiming "drop the PE" cited the v2
> `depthformer.py` and was wrong.)

The model is a **standard flaxformer T5 1.1 encoder-decoder** wired by
`configs/mrt_merged_base.gin` (+ `posembed.gin`, `t5_1_1_flaxformer.gin`):

- **Encoder = `t5_architecture.Encoder`** (NOT the `DepthformerEncoder` in the
  same file — `encoder_factory = @t5_architecture.Encoder`), so it's a plain
  bidirectional stack over the 1006-token input (no mean-pool).
- **Absolute position = `embedding.FixedEmbed` (fixed sinusoidal), on BOTH the
  encoder and the decoder** (`posembed.gin`:
  `POSITION_EMBEDDER_FACTORY = @embedding.FixedEmbed`, set on
  `t5_architecture.{Encoder,Decoder}.position_embedder_factory`). The PE is
  **added** to the token embeddings. There is **no embedding `Scale(sqrt(embed))`**
  (the token `embedding.Embed` sets no scaling; `SCALE=1.0` in the gin is only the
  kernel-initializer scale — irrelevant at inference). So the earlier
  `Scale(sqrt(embed))` worry is **resolved: no scale**.
- **Encoder rel-pos = None** (`SHARED_RELATIVE_POSITION_BIAS_FACTORY = None`),
  matching the manifest (no `target.encoder.relpos_bias`). The two **decoders**
  use learned rel-pos: `temporal_decoder.relpos_bias.rel_embedding` `[12, 128]`
  and `depth_decoder.relpos_bias_depth.rel_embedding` `[12, 16]` — so the
  temporal bucket count is **128**, not 32 (already fixed).
- **`FixedEmbed`'s exact formula** = flaxformer `initializers.sinusoidal()`
  (`min_scale=1.0`, `max_scale=10000`): **split-half** layout — first `embed/2`
  columns are sines, next `embed/2` cosines — with
  `div_term[i] = exp(i * -ln(10000)/(embed/2 - 1))`. **NOT** interleaved.

- ✅ **Encoder PE — FIXED + real-weight gated.** `sinusoidal_pos_embed` (and the
  numpy ref + the op-composition test) now use the exact flaxformer split-half
  formula. `tests/llm_encoder_real_weight.rs` loads the real 1.3 GB weights and
  matches the NumPy reference (both flaxformer-formula now) to **9.6e-7** on
  lavapipe — cross-validating the flat-copy weight mapping (no transpose; a
  transpose bug diverges by O(1)) and the full encoder op composition. *Prior to
  this, the Rust + numpy ref shared an **interleaved** formula with a slightly
  different frequency base — wrong vs flaxformer; the old gate couldn't catch it
  because both sides shared the bug. The recovered gin/flaxformer source caught
  it.*

- ⚠️ **OPEN — temporal/depth decoder is missing the FixedEmbed PE.** `posembed.gin`
  wires `FixedEmbed` onto the **decoder** too, and `DepthformerDecoder` does
  **not** override the base `t5_architecture.Decoder` embed path (confirmed: it
  has `setup`/`_setup_layer_sequence` but no `__call__`), so the base Decoder
  adds the sinusoidal PE to the **per-level** token embeddings over the `T*Q`
  grid *before* `DepthformerDecoderStack` mean-pools the levels into the temporal
  input — i.e. temporal frame `t` effectively gains `mean_q PE[t*Q + q]`, and the
  depth-module level prefix embeddings (`embedded[:, 1:, :Q-1]`) carry their
  `PE[t*Q+q]` too. Our `build_decoder` / `build_temporal_decode_step` add **no**
  absolute PE (they rely on the rel-pos bias alone). This is a likely real
  discrepancy; it needs the decode-time position semantics worked out and a
  decoder reference (extend `llm_numpy_ref.py` to the decoder using the v1 code,
  or run t5x) to verify a fix. Not yet fixed.

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

## What's now implemented + GPU-verified (lavapipe)

All three new builders are checked GPU-vs-CPU on Mesa **lavapipe** (software
Vulkan; `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json`), random weights,
zero rel-pos tables, 1e-3 tolerance — the same harness as the prior temporal
tests:

- `build_depth_decoder_layer` + `build_depth_decoder_stack` (**new**) — the
  per-frame depth core: 4 causal-self-attn-only T5 layers (one *shared* depth
  rel-pos table, matching the checkpoint's single `relpos_bias_depth`) →
  shared `decoder_norm` → non-tied `logits_dense` → `[16, vocab]`.
  Verified by `tests/llm_depth_decoder_correctness.rs`.
- `build_temporal_decoder` (**new**) — the *full* temporal forward: per-frame
  RVQ token grid → shared token embed → **mean-pool over the levels** (the real
  temporal input, via a constant pooling-matrix matmul) → 20 temporal layers
  (one shared rel-pos table) → `[num_frames, embed]` states, **no** final norm /
  logits (those belong after the depth module). Verified by
  `tests/llm_temporal_decoder_correctness.rs`.
- `build_decoder` (**new**) — the *full* parallel (teacher-forcing) decoder:
  SOS-padded grid → embed → mean-pool → temporal layers (cross-attn) →
  per-frame `concat([temporal_state, embed(prev levels)])` → depth stack (run
  per frame, since its attention is causal *within* a frame's levels; depth
  params registered once and shared) → `[num_frames*num_levels, vocab]`. The
  slice/concat use `slice_2d`/`concat`. Verified end-to-end by
  `tests/llm_full_decoder_correctness.rs`. This parallel form is for
  training/verification; production inference is the autoregressive step loop.
- `build_temporal_decode_step` (**new**) — the incremental (autoregressive)
  temporal decode: one frame per `step()`, KV cache persisting across steps as
  mutable `parameter` buffers (`cache_write` + `cached_attention`, the smollm2
  pattern), cross-attending to the fixed encoder output. Verified frame-by-frame
  against the CPU reference (and the parallel forward) by
  `tests/llm_temporal_decode_step_correctness.rs`. This is the temporal half of
  the generation loop; the depth half reuses `build_depth_decoder_stack` per
  frame (16 positions, cheap, rel-pos-correct).

## Two head_dim=64 attention kernel bugs found + fixed

Building the incremental decode surfaced two real GPU bugs, both the same race:
the online-softmax loops read the reduced score from shared memory, then the
next iteration overwrote that shared buffer with **no `workgroupBarrier`
between**. This is benign when the workgroup is a single subgroup but corrupts
results across subgroups.

- `cached_attention.wgsl` (workgroup = 64 lanes ⇒ always multi-subgroup): wrong
  for any kv_len≥2. Only prior coverage was a finiteness smoke test.
- generated attention (`generate_attention_module` +
  `generate_attention_with_rel_pos_module`, workgroup = `head_dim`): correct at
  head_dim≤8 (one subgroup) but wrong at **head_dim=64 — the real model's
  value**. Every prior LLM/MusicCoCa test used head_dim≤8, so this was invisible;
  on real weights the whole port would have been broken.

Fixed by adding the missing barriers. New regression tests pin head_dim=64:
`tests/full_attention_probe.rs`, `tests/cached_attention_probe.rs` (both exact
to <1e-6 vs CPU SDPA). **`cached_attention` requires head_dim==64** (it reduces
the dot product over its full 64-lane workgroup with no `tid<head_dim` mask) —
fine for Magenta-RT, but a constraint to remember.

## Remaining to reach text→tokens (in dependency order)

1. ✅ **Encoder fixes** — done: dropped the spurious learned `pos_embed` (now a
   computed sinusoidal PE), dropped the encoder rel-pos (plain bidirectional
   SDPA), temporal bucket count → 128. Op-composition GPU-verified
   (`tests/llm_encoder_correctness.rs`). Open: `Scale(sqrt(embed))?` and the
   real-weight encoder numeric check (both need real weights / a JAX reference).
2. ✅ **Full-decoder wiring** — done (`build_decoder`, GPU-verified). The
   per-frame depth-input assembly (slice temporal state + level-prefix embeds,
   concat) and the block-per-frame depth pass are in place.
3. ✅ **KV cache + autoregressive loop, CFG, and sampling** — done. The temporal
   step (`build_temporal_decode_step`) is KV-cached and rel-pos-correct via the
   new `cached_attention_rel_pos` op (verified vs CPU SDPA in
   `tests/cached_attention_probe.rs`). The host driver `decode` (with
   `decode_greedy` as a thin wrapper) loops frames × levels: temporal step →
   per-frame depth decode → sample → feed back. **CFG is host-side** — two
   batch=1 passes (positive/negative encoder output) combined per level with
   `cfg_combine` before sampling, so **no graph batch=2 / encoder broadcast-add
   is needed**. Sampling is `argmax` (greedy, `temperature<=0`) or
   `top_k_sample` (temperature + top-k, seeded). Verified on lavapipe:
   `tests/llm_decode_loop.rs` (greedy = parallel argmax) and
   `tests/llm_decode_cfg.rs` (CFG-greedy = parallel combined argmax; top-k
   samples within the parallel top-k; reproducible from a seed).
   **Remaining:** a depth KV-cache to drop the per-level O(levels²) recompute
   (optimisation only).
4. ✅ **Weight loader** — `llm_weights::{checkpoint_param_map, load_llm_weights}`
   maps the flaxformer `target.*` names → graph params. **No transposes**: T5X
   `DenseGeneral` stores `[in, out]` row-major (matches `matmul(x, W)`), and
   `rel_embedding [heads, buckets]` flattens to the `[heads*buckets]` table — so
   every param is a flat copy. Verified against the committed manifest by
   `tests/llm_weight_map.rs`: **100% of the 430 checkpoint tensors** map to a
   graph param with matching element count. `dump_llm.py` writes the
   safetensors keyed by these `target.*` names — **and the dump now runs here**:
   the `google/magenta-realtime` checkpoint is ungated and the HF file CDN serves
   the bytes, so `dump_llm.py` produced a 1.3 GB `weights_llm_base.safetensors`
   (430 tensors; its emitted manifest == the committed one). The `#[ignore]`d
   `load_real_weights_into_decoder` test (`MEGANEURA_LLM_WEIGHTS=…`) loads all 430
   into a `build_decoder` session (**0 skips**), then runs a forward on lavapipe
   and asserts the 1,908,736 logits are finite (max|logit|≈81). This is a
   load+run **smoke** test — it pins that the dump, the flat-copy mapping, the
   param shapes, and the GPU forward are mutually consistent — **not** a numeric
   gate. (Manifest fix: temporal rel-pos is **128 buckets**, not 32 —
   `LlmConfig::base/large` updated.)
   **Encoder gap stands**: the checkpoint has no encoder PE/rel-pos tensor, so
   `encoder.pos_embed` + the encoder rel-pos tables are left unmapped (skipped) —
   the decoder + embeddings + heads load fully; the encoder needs its position
   scheme resolved + the `Scale(√d)` fix before a real-weight forward.
   **Real-weight numeric gate** (greedy logits vs the Colab reference) still
   needs JAX + the reference outputs; the loader + mapping are in place for it.

## Status

The depth-module topology, the temporal→depth wiring, and the absolute-PE /
untied-logits questions are now **settled** from the checkpoint manifest + the
reference `depthformer.py`, and the parallel decoder forward is GPU-verified on
lavapipe. The remaining unknown is narrow (the encoder's *position* scheme,
which is non-parametric and so invisible in the checkpoint) and is flagged, not
guessed. The autoregressive KV-cache loop and the real-weight loader are done,
and the **real 1.3 GB checkpoint now dumps, loads (0 skips), and runs a finite
forward on lavapipe**. What's left to a trustworthy text→tokens path: (1) settle
the encoder position scheme + `Scale(√d)` and (2) a real-weight **numeric** gate
— both need a trusted reference forward (JAX/T5X) to diff against; the dump gives
the weights, not the expected activations.
