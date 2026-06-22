# Magenta-RT in meganeura — completion plan

A single forward-looking plan for finishing the Magenta-RT port, consolidating
the per-component findings (`ARCH_FINDINGS.md` for SpectroStream,
`MUSICCOCA_FINDINGS.md`, `LLM_FINDINGS.md`). The pipeline is:

```
text prompt ─► MusicCoCa ─► 6 style tokens ┐
10 s audio context ─► SpectroStream encode ─┴► LLM encoder (1006 tok)
                                                  │
                                            LLM decoder (Depthformer)
                                            50 frames × 16 RVQ = 800 tok
                                                  │
                                            SpectroStream decode ─► 48 kHz stereo
```

## Where each component stands (verified on Lavapipe unless noted)

| component | graph builder | weight loader | verified | gaps |
|-----------|---------------|---------------|----------|------|
| **MusicCoCa** text encoder | ✅ `build_text_encoder_graph` | ✅ `load_text_encoder_weights` | ✅ GPU-vs-CPU (random weights, 1e-3) | real-weight check vs the 26 testdata embeddings |
| **SpectroStream** decoder body | ✅ `build_decoder_graph` | ✅ `load_decoder_weights` | ⚠️ only structurally; demo runs tokens→WAV | 3 unverified "free reinterpret" claims; RADV NaN bug |
| **LLM** encoder | ✅ `build_encoder_graph` (FixedEmbed sinusoidal PE, bidirectional, no rel-pos) | ✅ `llm_weights` (fully loadable, 0 skips) | ✅ GPU-vs-CPU op-composition + real-weight numeric gate vs NumPy ref (**9.6e-7**, lavapipe) | PE scheme RESOLVED from recovered v1 gin (FixedEmbed, no scale); formula fixed (split-half) |
| **LLM** temporal decoder | ✅ `build_temporal_decoder` + `build_decoder_layer` | ✅ `llm_weights` (shared) | ✅ GPU-vs-CPU on lavapipe (random weights, 1e-3) | — |
| **LLM** depth decoder | ✅ `build_depth_decoder_layer` + `build_depth_decoder_stack` | ✅ `llm_weights` (shared) | ✅ GPU-vs-CPU on lavapipe (random weights, 1e-3) | — |
| **LLM** full decoder | ✅ `build_decoder` (temporal→depth) | ✅ `llm_weights::load_llm_weights` (real 1.3 GB dump loads, 0 skips) | ✅ map = 100% of checkpoint (`llm_weight_map`); GPU-vs-CPU random-weight; **real-weight forward runs finite on lavapipe** | real-weight numeric gate needs JAX ref |
| **LLM** generation | ✅ `decode` (CFG + temperature/top-k) / `decode_greedy`: temporal KV-cache step + per-frame depth + sampler | — | ✅ greedy + CFG match parallel argmax; top-k within parallel top-k; reproducible (lavapipe) | depth recomputed per level (KV-cache later); encoder still needed for real enc output |

The verification pattern that works without real weights: build the graph with
small random weights, run on Lavapipe, compare to an independent CPU reference
forward pass (`tests/musiccoca_correctness.rs`,
`tests/llm_decoder_correctness.rs`, `tests/llm_decoder_stack_correctness.rs`).
This validates op composition; it does **not** validate that param layouts match
the real checkpoints — that needs the weight dumps.

> **Caveat the head_dim:** these random-weight tests historically used
> head_dim≤8, which hid two attention-kernel subgroup-race bugs that only bite at
> **head_dim=64 (the real value)** — now fixed, with regression tests at
> head_dim=64 (`tests/{full,cached}_attention_probe.rs`). Keep new attention
> verification at head_dim=64. See `LLM_FINDINGS.md`.

## The blocker: weight/manifest access — LIFTED (LLM weights now dumped)

`huggingface.co` egress works — and not just the metadata API: the **file CDN**
serves real checkpoint bytes (verified `200` on a `target.*` `.zarray`+chunk
through CloudFront), the model is **ungated** (`gated: false`), and PyPI egress
installs `huggingface_hub`/`safetensors`. So the LLM weight dump runs here:
`python3 tools/magenta_rt/dump_llm.py` pulled the 1.3 GB `target.*` checkpoint
and wrote `weights_llm_base.safetensors` (430 tensors; emitted manifest ==
committed manifest). Loading all 430 into a `build_decoder` session and running a
forward on lavapipe gives 1,908,736 **finite** logits (`load_real_weights_into_decoder`).

The LLM **architecture was already settled without any weight download**:

- `tools/magenta_rt/fetch_llm_manifest.py` lists all 430 `target.*` tensors of
  `llm_base_x4286_c1860k` with shapes → committed `llm_base_manifest.json`.
- The reference model code (pip `magenta-rt`, `magenta_rt/jax/depthformer.py`)
  settles the depth-module wiring, the no-absolute-PE question, and the untied
  `logits_dense` head. See `LLM_FINDINGS.md` (RESOLVED sections).

GPU-vs-CPU tests run locally on Mesa **lavapipe** (software Vulkan):
`apt-get install -y mesa-vulkan-drivers`, then
`VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json cargo test`. What the dump
does **not** give you is a numeric *correctness* gate — that needs a trusted
reference forward (JAX/T5X) to diff the real-weight logits against; the dump
provides the weights, not the expected activations. Still to dump: the
MusicCoCa + SpectroStream-encoder weights (their dumpers exist).

## Plan

### Phase 0 — get the data in (unblocks everything below)
0.1 ✅ **DONE** — HF egress works; `llm_base_manifest.json` committed (430 tensors).
0.2 ✅ **LLM weights dumped + loaded + forward-run.** `dump_llm.py` → 1.3 GB
    `weights_llm_base.safetensors` (430 tensors; emitted manifest == committed).
    `load_real_weights_into_decoder` loads all 430 and runs a forward on lavapipe
    (1,908,736 finite logits) — a load+run smoke test, not a numeric gate (2.5).
    Still TODO: the MusicCoCa + SpectroStream-encoder weight dumps.

### Phase 1 — verify what's already built against real weights
1.1 **MusicCoCa**: load `weights_musiccoca.safetensors`, run the encoder on the 26
    testdata prompts, assert ≥0.999 cosine vs `musiccoca_testdata.safetensors`
    embeddings and ≥90% RVQ token match (the numpy ref hits 0.9993 / 93.3%). This
    is the first real-weight gate; the loader + `transpose_2d` O-projection
    handling are the likely failure points.
1.2 **SpectroStream**: settle the three unverified reinterprets against a TF
    `body_out` dump (`tests/spectrostream_vs_tf_body.rs`, already scaffolded):
    the decoder_0→1 batch-fold pixel-shuffle, the final batch/channel merge, and
    the `base_conv_last` W-padding (code keeps SAME; `ARCH_FINDINGS.md` expects
    VALID). Use the `DecoderStage` bisection to localise the first divergent block.
1.3 ✅ **LLM encoder — RESOLVED + real-weight gated.** The v1 source recovered
    from git history (`magenta/magenta-realtime` initial commit `b35a850`) settles
    the position scheme from its gin config: encoder = `t5_architecture.Encoder`
    with `embedding.FixedEmbed` (fixed sinusoidal PE), no encoder rel-pos, no
    embedding scale. Fixed our PE formula to flaxformer's exact split-half
    `sinusoidal()` (was interleaved — a real bug the prior gate couldn't see, as
    Rust + numpy ref shared it). `tests/llm_encoder_real_weight.rs` now matches the
    NumPy reference to **9.6e-7** on lavapipe (no-transpose mapping + op
    composition + correct PE). **NEW open item** surfaced from the same source: the
    *decoder* is also wired with `FixedEmbed` PE, but `build_decoder` adds none —
    see Phase 2.6.

### Phase 2 — finish the LLM decoder
2.1 ✅ **DONE** — depth-module topology resolved (manifest + `depthformer.py`):
    temporal output is the depth module's **level-0 input** (concat prefix, not
    added embedding / not cross-attn); shared token embedder + shared
    `decoder_norm` + non-tied `logits_dense`. Documented in `LLM_FINDINGS.md`.
2.2 ✅ **`build_depth_decoder_layer`/`build_depth_decoder_stack`**,
    **`build_temporal_decoder`** (mean-pooled input), and **`build_decoder`**
    (the full parallel temporal→depth forward: per-frame depth-input assembly via
    slice+concat, block-per-frame depth pass, shared head) — all **GPU-verified
    on lavapipe** (random weights, 1e-3): `tests/llm_{depth,temporal,full}_decoder_correctness.rs`.
2.3 **CFG batch=2**: add a broadcast-add op (or batched attention) so the encoder
    and decoder can run positive+negative style rows; lift the `batch==1` assert.
2.4 **KV cache + autoregressive decode loop**: encoder runs once; decode 50 frames
    × 16 levels, reading `[2, vocab]` logits per step, calling
    `sampling::{cfg_combine, top_k_sample}` (already implemented), feeding tokens
    back. Reuse the temporal layers with cached K/V (see `runtime` KV-cache ops:
    `CacheWrite`/`CachedAttention`).
2.5 ✅ **LLM weight loader** — `llm_weights::{checkpoint_param_map,
    load_llm_weights}` maps `target.*` → graph params (flat copy, no transpose);
    `tests/llm_weight_map.rs` verifies 100% checkpoint coverage vs the manifest;
    `dump_llm.py` writes the compatible safetensors. The real 1.3 GB dump now
    **loads with 0 skips and runs a finite forward on lavapipe**
    (`load_real_weights_into_decoder`, `#[ignore]`d). Real-weight *numeric* gate
    (greedy logits vs a Colab/JAX reference) still needs the reference outputs —
    the loader + a runnable real-weight session are in place for it.
2.6 ✅ **Decoder absolute PE — RESOLVED + real-weight gated.** The recovered v1
    gin wires `FixedEmbed` (sinusoidal) onto the decoder too, and
    `DepthformerDecoder` inherits the base `t5_architecture.Decoder` embed path —
    so the model adds absolute PE to the per-level token embeddings over the `T*Q`
    grid before the level mean-pool. `build_decoder_faithful` (new) reproduces
    this exactly (flat `[T*Q]` input, FixedEmbed PE, the edge-pad/mean +
    edge-pad/concat helpers). `tests/llm_decoder_real_weight.rs` gates it vs the
    faithful NumPy decoder reference on the real weights: **1.1e-4** max abs diff,
    0/48 argmax mismatches — the decoder's **first real-weight numeric gate**
    (also the first check of nonzero T5 rel-pos bucketing + cross-attn at real
    magnitudes). The PE matters: toggling it flips 7/48 greedy tokens.
    **Remaining:** migrate the generation path (`build_decoder` + the incremental
    `build_temporal_decode_step` KV-cache loop) onto this verified behavior.

### Phase 3 — end-to-end wiring
3.1 Tokenizer + MusicCoCa text→6 style tokens. ✅ **Tokenizer done** —
    `tokenizer::SpmModel` is a pure-Rust SentencePiece Unigram tokenizer
    (`.model` protobuf parse + whitespace normalization + Viterbi + unknown-run
    merging), verified bit-exact vs the reference `sentencepiece` library
    (`tests/spm_tokenizer.rs`). `musiccoca_token_ids` applies MusicCoCa's recipe
    (lowercase, truncate 127, prepend SOS=1). Remaining: feed the ids through the
    (existing) MusicCoCa text encoder + RVQ quantizer → 6 style tokens — gated on
    the MusicCoCa weights + the real `spm.model` (not in the public HF release).
3.2 SpectroStream **encoder** (audio context → 4-RVQ tokens). ⚙️ **STFT
    front-end done** — `spectrostream_encoder::stft_features` (audio → `[frames,
    480, 4]` features, 960/480/960 Hann, keep_dc, `[re,im]`-per-channel layout),
    verified vs a naive DFT (`tests/spectrostream_stft.rs`, ~5e-7). Remaining:
    the strided conv-residual stack (`ratios=((1,2),(1,2),(1,3),(1,2),(1,2),
    (2,2),(2,1))`) + RVQ quantizer — **blocked on the encoder weight manifest**
    (per-block channel counts), like the decoder was. Also: confirm the STFT
    `semicausal` alignment against real audio.
3.3 Driver: text+context → LLM generate 800 tokens → SpectroStream decode → 2 s
    audio, with the 40 ms crossfade between chunks (`MagentaRtConfig`).
    ✅ **Deterministic glue done** — `magenta_rt::driver`:
    - `assemble_encoder_input` packs 250 context frames × 4 codec RVQ + 6 style
      RVQ into the unified vocab (per-level offsets, context-then-style order,
      masked-style negative for CFG); unit-tested against the `MagentaRtConfig`
      layout (`driver::tests`).
    - `crossfade_ramp` / `crossfade_chunks` — 40 ms (1920-sample) overlap-add,
      linear (DC-preserving) and eqpower (power-preserving); verified by
      constant-reconstruction, power, and seam-length tests.
    - `generate_token_grid` — the weight-independent LLM orchestration (encoder
      pos+neg → CFG decode), **GPU-verified on lavapipe**
      (`tests/llm_end_to_end_driver.rs`): CFG(w=1) ≡ greedy(positive).
    Remaining (weight-gated): feed real SpectroStream codec tokens + MusicCoCa
    style tokens in, and run the SpectroStream decoder on the 800-token grid.

### Phase 4 — correctness & performance
4.1 **RADV NaN bug** (`tools/blade_radv_repro/`): a cross-pass cache-visibility
    issue on AMD RDNA3.5 under single-submit. Either upstream a finer-grained
    barrier in blade or gate AMD onto multi-submit; today the demo clamps NaN→0.
4.2 Move host-side glue (input_layer conv1×1, iSTFT) onto GPU / into the library.
4.3 GPU perf pass once correct (this is meganeura's wedge): the decoder body is
    conv-heavy — the `emit_int_div_checks` elision (already in) and the conv coop
    paths matter here.

## Sequencing

```
Phase 0 (allowlist HF / commit manifests)   ← single unblock for all of the below
  ├─ 1.1 MusicCoCa real-weight verify        ← smallest, highest-confidence first
  ├─ 1.3 LLM encoder PE resolution ─► 2.1 depth topology ─► 2.2 depth layer
  │        └─► 2.3 CFG batch ─► 2.4 KV-cache + decode loop ─► 2.5 loader
  ├─ 1.2 SpectroStream bit-exactness
  └─ Phase 3 end-to-end ─► Phase 4 RADV + perf
```

The work splits cleanly into "needs the data" (everything correctness-real) and
"doesn't" (more graph builders + CPU-reference tests, which can proceed for the
depth layer once 2.1's topology is known). The temporal decoder is done; the
depth module is the main remaining graph-building task, and it's blocked on the
manifest only for its *topology*, not its verification.
