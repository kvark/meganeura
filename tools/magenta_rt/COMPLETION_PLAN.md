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
| **LLM** encoder | ✅ `build_encoder_graph` | ❌ | ❌ | drop spurious `pos_embed`, add `Scale(√d)`; encoder position scheme is the one open question |
| **LLM** temporal decoder | ✅ `build_temporal_decoder` (mean-pool input, shared rel-pos) + `build_decoder_layer` | ❌ | ✅ GPU-vs-CPU on lavapipe (random weights, 1e-3) | buckets=128 for real weights |
| **LLM** depth decoder | ✅ `build_depth_decoder_layer` + `build_depth_decoder_stack` (shared rel-pos) | ❌ | ✅ GPU-vs-CPU on lavapipe (random weights, 1e-3) | topology resolved |
| **LLM** full decoder | ✅ `build_decoder` (temporal→depth, parallel/teacher-forcing) | ❌ | ✅ GPU-vs-CPU on lavapipe (random weights, 1e-3) | autoregressive KV-cache form next |
| **LLM** generation | host sampler ✅ (`sampling.rs`) | — | sampler unit-tested | KV-cache + decode loop unwired |

The verification pattern that works without real weights: build the graph with
small random weights, run on Lavapipe, compare to an independent CPU reference
forward pass (`tests/musiccoca_correctness.rs`,
`tests/llm_decoder_correctness.rs`, `tests/llm_decoder_stack_correctness.rs`).
This validates op composition; it does **not** validate that param layouts match
the real checkpoints — that needs the weight dumps.

## The blocker: weight/manifest access — LIFTED

`huggingface.co` egress now works (verified `200` from the model API). The LLM
**architecture is fully settled without any weight download**:

- `tools/magenta_rt/fetch_llm_manifest.py` lists all 430 `target.*` tensors of
  `llm_base_x4286_c1860k` with shapes → committed `llm_base_manifest.json`.
- The reference model code (pip `magenta-rt`, `magenta_rt/jax/depthformer.py`)
  settles the depth-module wiring, the no-absolute-PE question, and the untied
  `logits_dense` head. See `LLM_FINDINGS.md` (RESOLVED sections).

GPU-vs-CPU tests run locally on Mesa **lavapipe** (software Vulkan):
`apt-get install -y mesa-vulkan-drivers`, then
`VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json cargo test`. What still
needs the **weights** (not architecture, not a HW GPU): real-weight correctness
checks. For the full weight dump: `python3 tools/magenta_rt/dump_llm.py` (after
`cargo clean` for the ~3 GB), plus the MusicCoCa/SpectroStream dumpers.

## Plan

### Phase 0 — get the data in (unblocks everything below)
0.1 ✅ **DONE** — HF egress works; `llm_base_manifest.json` committed (430 tensors).
0.2 Dump LLM weights (`dump_llm.py`), confirm MusicCoCa + SpectroStream dumps are
    present (`magenta_rt_codec_dump/`). *Needs ~3 GB + a Vulkan device to verify.*

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
1.3 **LLM encoder**: settle the position-encoding discrepancy from the manifest
    (is there a `relpos_bias`/`relative_attention` tensor and/or a
    `position_embed`?). Standard T5 1.1 ⇒ rel-pos bias, no absolute PE; if
    confirmed, drop the `encoder.pos_embed` add from `build_encoder_graph`. Then
    a real-weight encoder check against a reference forward (numpy ref exists for
    the encoder; fix its rel-pos/PE assumptions first).

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
2.5 **LLM weight loader**: map T5X `target.*` flaxformer names → graph params
    (like `musiccoca`/`spectrostream`). Real-weight gate: greedy-decode logits
    match the Colab reference for a fixed seed.

### Phase 3 — end-to-end wiring
3.1 Tokenizer + MusicCoCa text→6 style tokens (SentencePiece via the `tokenizers`
    crate; `musiccoca_vocab.model`).
3.2 SpectroStream **encoder** (audio context → 4-RVQ tokens) — currently only the
    decoder exists; the encoder is needed for audio-conditioned continuation.
3.3 Driver: text+context → LLM generate 800 tokens → SpectroStream decode → 2 s
    audio, with the 40 ms crossfade between chunks (`MagentaRtConfig`).

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
