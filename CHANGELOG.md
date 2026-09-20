# Unreleased

- Use one egglog rewrite engine and calibrated whole-program search instead of live attention/submission retuning.
- Broadcast scalar gradients directly and eliminate single-row RoPE at static position zero.
- Fix split-attention partial indexing and clear empty splits when reusing a KV cache.
- Measure mapped versus staged readback per allocation and size; reuse bounded staging for faster CPU reads.
- Include reduced-storage dense matmuls in scalar-tile tuning, keeping weight decoding and validation unchanged.
- The codegen debug hooks are parameters, not process state. WGSL dumping
  resolved `MEGANEURA_DUMP_WGSL` into a `SessionOptions::wgsl_dump_dir`
  that each session's pipeline layer owns; every module it compiles (the
  standard, coop, weighted, epilogue-fused and scheduled forms, plus the
  tuner's candidates) is written there, each file naming the shader and a
  content hash. The matmul knobs (`MEGANEURA_MATMUL_K_STAGE`,
  `MEGANEURA_INTERLEAVE_COLUMNS`) ride `TuningKnobs` into the plan the
  same way the flash caps do. The profiler state is armed once — by
  `init` or by GPU-context initialization — instead of being lazily
  conjured on first use, and `default_gpu_context` returns a fresh
  context per call rather than a hidden process-global first-device
  cache.
- The environment is resolved once: `src/config.rs` is the only place that
  reads `MEGANEURA_*` variables, and its product is configuration — it no
  longer modifies the process environment. The one boolean decoding rule
  is a pure function under test, so the config tests don't flip env vars.
  The six stragglers that still read the environment inline are typed
  options now: `MEGANEURA_OPTIMIZER` neighbors `GREEDY_PACK_SWIGLU`
  (`OptimizeConfig::pack_swiglu`), `MATMUL_K_STAGE` and
  `MEGANEURA_INTERLEAVE_COLUMNS` ride `TuningKnobs` into the matmul
  codegen, and `MEGANEURA_DEVICE_PARAMETERS` / `MEGANEURA_REUSE_UPLOAD`
  are `SessionOptions` fields. All are registered and documented in the
  README; an unknown `MEGANEURA_*` name warns instead of panicking.
- First-party model configs drop their prefixed names — `SmolLM2Config`,
  `SmolVLAConfig`, `SmolVLM2Config`, `SDUNetConfig` and `WhisperConfig`
  are simply `Config` in their modules, so `use
  meganeura::models::smollm2::Config` reads on its own.
- The optional weight-download feature is named `hf-hub` after its
  dependency rather than the generic `hub`.
- GGUF is now a model format, not just a weight container. `load::gguf`
  reads the architecture description into a `ModelConfig`, builds the graph
  it implies, fills it from the file's own tensors, and tokenizes with the
  vocabulary the file embeds — so `load_gguf(path)?.generator(2048)?` is
  everything between a `.gguf` and generated text, with no `tokenizer.json`
  and no hard-coded dimensions. See `examples/gguf_generate.rs`. This lives
  behind the new `gguf` cargo feature; nothing is on by default, so a
  pinning embedder chooses `--features gguf` (and optionally `models`)
  and the loader pulls no external dependency of its own. First-party
  model definitions moved behind a new `models` feature the same way.

  The llama family (also Mistral, SmolLM2, TinyLlama), Qwen2, Qwen3, Gemma,
  Gemma2, Gemma3 and Phi3 build from one parameterised decoder; the enum
  records only where a family departs from the llama shape — Qwen3's
  per-head Q/K norms, Gemma's scaled embeddings and `1 + w` norm weights and
  second pair of norms, Gemma3's five-local-to-one-global window pattern and
  its separate RoPE base for the local layers. An architecture whose graph
  cannot be expressed *exactly* is refused by name rather than approximated,
  since a subtly wrong decoder still emits fluent text: partial RoPE
  (`rope.dimension_count < head_dim`, as Phi2 uses) needs a strided split
  with no op behind it, and Gemma2's `attn_logit_softcapping` has no
  parameter on the cached attention ops. Final logit softcapping *is*
  applied, being expressible after the head. The alias list is deliberately
  short for the same reason — a family mapped onto a graph that is merely
  close to its own would load without complaint and decode wrongly.

  Llama's Q and K are un-permuted on load. GGML has two RoPE conventions,
  and llama.cpp's converter permutes those two weights into a llama GGUF so
  that GGML's *interleaved* rope reproduces what HuggingFace's *half-split*
  rope would have done. Meganeura's RoPE is the half-split one, so the
  permutation has to come back out; leaving it in is not a crash but fluent,
  wrong text. Every other family here converts unpermuted.

  Phi's packed tensors are sliced rather than refused: `attn_qkv.weight`
  backs three projections and Phi3's double-width `ffn_up.weight` backs two.
  Rows are output features and GGUF blocks along the other axis, so a row
  range is a contiguous run of bytes whatever the encoding — and both the
  slice and the un-permute happen in GGUF's own layout, where one rule
  covers every format, rather than after packing into Meganeura's, where
  Q4 keeps block headers in a region of their own.

  Biases are optional. Qwen2 biases Q, K and V but leaves the attention
  output unbiased, and requiring all four rejected every real Qwen2 file; an
  absent bias is a zero bias, which is the graph without the add.

  Gemma norm weights are loaded exactly as written. They are trained centred
  on zero and applied as `1 + w`, but llama.cpp's converter already folds
  the one in, so the file holds the applied scale; shifting again would turn
  a trained zero into two rather than one.

  RoPE position scaling is refused rather than dropped. A file declaring
  `rope.scaling.*`, the legacy `rope.scale_linear`, or carrying a correction
  tensor (Llama 3.1's `rope_freqs.weight`, Phi3's long-context factors) is
  not a plain-RoPE model however ordinary its architecture name looks, and
  reading the base while ignoring the scheme rotates every position wrongly
  while still producing fluent text.

  `tokenizer.ggml.pre` selects the pre-tokenizer, because
  `tokenizer.ggml.model = gpt2` names the merge algorithm and not the rule
  that decides where merges may apply. Llama 3, Qwen and SmolLM all declare
  `gpt2` and split differently — `1234` is one pre-token under GPT-2,
  `123|4` under Llama 3 and `1|2|3|4` under Qwen2 — and merges cannot cross
  a boundary, so the merge list cannot repair the wrong rule. Four rule sets
  are implemented against llama.cpp's own `regex_exprs`; any other
  identifier is refused by name, `default` included, since it is its own
  cascade rather than a synonym for `gpt-2`.

  SentencePiece score ties merge the leftmost pair, matching GGML's
  `llm_bigram_spm` comparator — `Iterator::max_by` keeps the *last* maximum,
  so with `ab` and `bc` scored alike, `abc` would otherwise tokenize as
  `[a, bc]` instead of `[ab, c]`.

  Text generation continues rather than restarts. A generator keeps its KV
  cache between calls, so a second `generate` encodes without the
  vocabulary's BOS — re-applying the policy planted another one mid-sequence
  — and a token the streaming callback has already been shown is committed
  before returning, so cancelling leaves the same state as stopping at
  `max_tokens`.

  One graph shape serves prompt and decode: a block of `block_size` token
  slots of which `valid` are real, starting at `position`, which is what
  `rope_dynamic_offset` and `cached_block_attention` already assume.
  `prefix_last` narrows to the one row that can predict before the output
  head rather than after, so the widest matmul in the model runs once per
  step instead of `block_size` times.

  Two sessions are compiled from it, differing only in `block_size`,
  because a single-row matmul takes the tuned K-split GEMV where a block of
  rows takes the tiled matmul. They are not two copies of the model:
  `share_parameter_from` aliases buffers rather than copying, so the
  weights are stored once and the K/V caches are literally the same
  buffers — a prompt the prefill session processes is already in the cache
  the decode session attends over, with no handoff. `tests/gguf_model.rs`
  pins that: feeding a prompt as one wide block and a token at a time must
  reach the same logits.

  The embedding table is dequantized to f16 whatever the file stores,
  because the gather has no block-quantized variant and a tied head reads
  the same tensor through `matmul_bt`, which block formats cannot serve
  either. It is also the one tensor read in GGUF's own row order rather
  than transposed, hence `GgufTensor::to_f32_rows` alongside `to_f32`: a
  lookup table is a list of rows and is already oriented, where a
  projection weight is a matrix and is not. Every other quantized weight
  keeps the file's own block encoding.

  The tokenizer is implemented here rather than delegated, because
  `tokenizers` is a dev-dependency on purpose — it pulls `onig`'s C library
  through every cross-compile. Both GGUF models are covered: BPE (`gpt2`)
  recodes into GPT-2's printable byte alphabet and merges by merge-list
  rank, and SentencePiece (`llama`) merges by score with `<0x..>` byte
  fallback. The GPT-2 pre-tokenizer pattern is a hand-written scan; std's
  `char::is_alphabetic` and `is_numeric` carry the real Unicode tables, so
  the character classes are not ASCII approximations. End-of-generation is
  a set rather than one id, since an instruction-tuned model ends its turn
  with `<|im_end|>` or `<|eot_id|>` rather than the declared EOS.

- Flash-decoding split-K for cached-block attention (`max_seq > 64`):
  the KV range splits across workgroups per head, each running its own
  online softmax over 32-token chunks, and a combine kernel merges the
  partials into the output. The single kernel's reduction rounds grow
  linearly with kv_len — at a 512-token context that was 85% of decode
  time — while the split form stays flat, matching llama.cpp's
  context-insensitive decode. Gemma 4 tg512 goes from 35 to 125 tok/s.
  Short contexts keep the fused single dispatch, whose per-token loop
  already fits one reduction round.
- RmsNorm with its consumer's residual add fused into one dispatch
  (`ShaderEntry::RmsNormAdd`, selected automatically by
  `fuse_rmsnorm_into_add`). The post-norm of every transformer block feeds
  a residual add; the fused kernel computes the same values without a
  dispatch and a round trip between them. Gemma 4 decode drops from 740 to
  635 dispatches.
- Cached-block attention processes its KV range in 16-token masked
  reduction rounds instead of full tiles plus a per-token remainder loop.
  A ragged tail costs one round instead of six barriers, and long
  contexts halve the round count outright. A subgroup-add form of the
  score reduction was measured and rejected: the wave's cross-lane adds
  cost more than the barriers they remove at a 64-thread workgroup.
- Gemma 4 GGUF text decode (`examples/gemma4.rs`). Pull a GGUF from
  HuggingFace, load it into Meganeura and llama.cpp, and compare decode
  throughput. GeGLU packing uses the same HorizontalConcat as SwiGLU.
  RmsNorm folds into packed GEMVs; `tune_with` searches that kernel,
  including vocab-width Q8 GEMV whose N/4 workgroups exceed the portable
  65535 minimum. Default comparison uses `set_submission_chunks(1)`.
- Quantized activations for packed GEMVs, following llama.cpp's
  `vec_dot_*_q8_1` kernels. A model that ships quantized weights now
  decodes with quantized (Q8_1) activations by default — for GGML Q4_0,
  Meganeura Q8 and the K-quants Q4_K/Q5_K/Q6_K/Q3_K, the formats with a
  kernel layout for it. `CompileOptions::quantized_activations` stays as
  an explicit opt-out because this changes results; tuning may reshape the
  selected kernel but never flips it. Where the device reports
  `shader_integer_dot_product` the four-byte dots are the hardware
  `dot4I8Packed` (DP4A) instruction; other devices run an exact scalar
  expansion of the same arithmetic. An RmsNorm before the GEMV folds into
  the same kernel, keeping one fewer dispatch than the unfused form.
- Native, load-only GGML Q4_0 storage (`DType::Q40` and
  `Graph::parameter_q40`) removes the host repack and stores 4.5 rather than
  5 bits per weight. Q4_1 continues to use Meganeura's existing Q4 layout.
- K-split GEMV now supports 32/64/128/256-thread tree and subgroup reductions
  across plain, fused-add, transposed, f16 and packed-weight kernels.
  `Session::tune_with` searches the shape for GEMV, including reduced-storage
  weights; `CompileOptions::gemv_shape` pins one for reproduction.
- Structured profiles split plans larger than Blade's timestamp budget into
  complete replay windows and stitch per-dispatch samples. The schema is now
  version 2, failed captures restore unprofiled execution, and Blade is pinned
  to resolve after waiting and tolerate Vulkan calibration skew.
- GGUF weight import (`load::gguf`). Reads the container's metadata and tensor
  inventory, and resolves GGML's block encodings into Meganeura's at load time.
  `Q4_1` and `Q8_0` repack losslessly for `set_parameter_packed` — GGML
  splits a block's nibbles across halves and interleaves each block's header
  with its payload, where Meganeura pairs adjacent nibbles and keeps headers
  in their own region. (`Q4_0` repacked here too until it gained native
  storage, above.) The block *order* already agreed, since packing
  performs the `[K, N]` transpose that GGUF's layout implies. See
  `examples/gguf_info.rs`.
- Native GGML K-quant storage: `Q4K`, `Q6K`, `Q5K` and `Q3K`, each with a
  `DType` and a `Graph::parameter_q*k` constructor. Superblocks are stored
  byte-for-byte as GGUF writes them, so loading copies nothing and the tiled
  matmul and K-split GEMV shaders decode them directly. The weights of a
  `Q3_K_M`, `Q4_K_M` or `Q5_K_M` file now load without a requantize. This is
  weight-format support, not a GGUF model builder: `Q2_K` and `Q8_K` are
  still unread, and quantized embedding tables have no gather variant.

  Q4_K reads 10% less weight data than Meganeura Q4 (4.5 bits/weight against
  5.0), because it quantizes its own sub-block scales to 6 bits instead of
  storing an f16 pair per 32 elements. Q6_K replaces what would otherwise be
  Q8_0 for high-precision layers, at 6.56 bits/weight against 9.0 — 27% less.
  Both remove a dequantize/requantize round trip that measured ~3% peak error
  on Q4_K: quantizing an already-quantized weight roughly doubles the error,
  since each stage contributes its own half-step.

  Q5_K is Q4_K plus one bit: the 5-bit quant is the nibble with `qh`'s bit
  for that sub-block contributing 16, at 5.5 bits/weight. Q3_K is the
  smallest at 3.44, and the only one whose high bit is *inverted* — a clear
  `hmask` bit subtracts 4 — with its own 6-bit scale shuffle rather than
  `get_scale_min_k4`.

  Q6_K's 210-byte superblocks and Q3_K's 110-byte ones are not whole numbers
  of words, so they alternate word alignment and the shader reads every field
  byte-addressed; buffers get a zero-padded tail while the superblocks stay
  verbatim. Q4_K (144) and Q5_K (176) need no tail.

  Block scales decode with `unpack2x16float`, so subnormals survive — an
  imported `0x0100` is 2^-16, an ordinary f32, and the hand-assembled
  decoder this replaces folded it to zero and erased the block. Needs
  Blade's `SHADER_FLOAT16_IN_FLOAT32`, hence the dependency bump.

  A GGUF file is read once and shared: tensors index ranges of one buffer
  rather than owning copies, and `to_packed` borrows it for the K-quants,
  so loading costs roughly the file rather than the file plus a copy of
  every payload.

  These are the first load-only formats: no K-quant encoder is implemented
  here, so they only ever arrive already packed. `set_parameter` rejects
  such parameters and points at `set_parameter_packed`, and
  `generate_module_weighted` asserts rather than falling through to an f32
  shader for a group with no K-quant variant. Packed SwiGLU `gate+up`
  fusion restages the derived concat from those uploads: Q4 merges its
  split header/nibble regions, and the native K-quants concatenate
  unpadded superblocks so a per-source word-alignment tail never lands
  between two sources' blocks — two Q3_K `[256, 1]` sources pad to 112
  bytes each but the combined parameter is 220, not 224.
- `matmul_bt` now rejects block-quantized weights. Their blocks run along
  the parameter's first dimension, which is N for a transposed B, while
  every packed decoder indexes along K — so the kernels that served this
  returned plausible but wrong numbers. The same refusal covers
  `FusedMatMulBTAdd` (greedy `Add(MatMulBT, ?)`) and store-side epilogue
  codegen, which previously compiled a quantized BT kernel after Relu
  fusion. f16 is unaffected, being an elementwise cast rather than a
  block layout.
- Autotuning searches shape-specialized scalar convolutions and K-stage sizes
  for forward and both gradients; unused candidates are released after search.
- Store-side unary epilogues (Relu/Sigmoid/Silu/Neg) now fuse into F16/Q4/Q8
  tiled matmuls instead of running as a separate dispatch. The epilogue does
  not inspect B, so the packed-weight kernels reuse the same `$STORE_BODY`
  hook. Cooperative matmuls stay out.
- Q4 tiled staging unpacks eight nibbles from one data word and reuses the
  block `(d, m)` header, replacing eight independent `dequant_q4` calls. GEMV
  keeps the scalar helper; `MatMulGemvBT` stays off Q4.
- Fixed a 4× over-dispatch: a matmul demoted to 32×32 tiles by the occupancy
  pass kept a 64×64 epilogue shader, so three quarters of its workgroups
  bounds-checked their way to writing nothing. The epilogue pipeline is now
  generated for the dispatch's own tile geometry, staging maps included, and
  the tile is part of the pipeline key.
- A dispatch with a fused epilogue no longer pulls the small-tile or
  weighted kernels into the compile set. Variant selection resolves the
  epilogue first, so those modules could never be selected and were built
  and kept for nothing; groups that also hold an unfused dispatch still get
  them from it.

# v0.3 (8 Sep 2026)

## Inference & models

- Qwen3 example, F16/Q4/Q8 weight storage, and sharded SafeTensors loading
- Q4 projections for SmolLM2 prefill/decode; wider K-split GEMV and RmsNorm fusion
- EfficientNetV2-S feature extractor, depthwise convolution and per-channel ops
- Shared Blade contexts, GPU input/output buffers, external buffer import and
  chunked submissions for renderer integration

## Training

- LaProp, AdamW, adaptive/global gradient clipping and per-parameter learning rates
- Temporal gradient accumulation and multi-input data loaders
- Lazy device-local optimizer moments; batched, host-cached state readback
- Logical-shape checkpoints omit padding and derived Winograd caches;
  formats 1/2 remain readable
- Differentiable row scans, exponential, softplus, normalization and pairwise ops

## Optimizations

- Unified matmul, convolution and attention variants; generated pointwise and
  multi-accumulator reduction kernels with producer/epilogue fusion
- E-graph extraction rewrites the graph, including repeated regions and
  packed SwiGLU; horizontal fusion packs independent same-input matmuls
- Device-local intermediates and lifetime-based buffer aliasing
- Opt-in, state-isolated f32 matmul/convolution tile search with numerical
  qualification, paired measurements and read-optimized private staging
- Parallel GroupNorm, packed narrow reductions and source-parallel scatter-add

## Correctness fixes

- Loss/softmax backward broadcasts use linear storage instead of a dense
  vocabulary-square matrix; fix BCE gradient buffer overrun
- Protect derivative exponent range in automatic cooperative selection
- Fix rewritten graph ordering, partial/batched cooperative convolution,
  mixed-width attention pipelines and scratch synchronization
- Validate checkpoint metadata before writes; preserve logical parameter sizes
  and share derived Winograd caches
- Stage Metal private-buffer access, synchronize uploads and release GPU state

## Infrastructure

- Unified `build(graph, SessionConfig)` and typed compiler/runtime/GPU options;
  environment overrides now require explicit `from_env` opt-in
- Eager graph inspection, named intermediate reads, nonfinite attribution,
  dispatch provenance and structured GPU profiles
- Empty default features; Hub downloads and CPU profiling are opt-in
- Rust 1.92 minimum, published Blade 0.9 and Naga 30 dependencies
- Remove unused Mistral/Phi-3/Gemma-4 builders; replace family-wide
  `TuneOutcome` fields with class/candidate evidence; invalidate old plan caches
- Consolidated regression executables, Rust host coverage and full package
  verification in CI; keep paper artifacts out of the published crate

# v0.2 (14 Apr 2026)

## Inference & models
- Conv2d forward/backward via implicit GEMM (im2col fused into matmul)
- MaxPool2d, GlobalAvgPool ops; GroupNormSilu fused op
- KV cache infrastructure for autoregressive decode
- U-Net, ResNet-50, and Whisper example models
- ONNX and NNEF model loaders
- macOS / Metal support improvements
- Sliding-window attention op for local attention patterns
- Gemma-4 model configs (1B, 4B, 12B, 27B)
- Mistral model configs (7B, Nemo 12B)
- Phi-3 model configs (Mini, Small)

## Training
- Differentiable MultiHeadAttn with GQA and CausalAttention backward
- nn module with Adam optimizer and SGD
- Abs/Log/Recip ops, ScatterAdd, MSE/L1 losses, Embedding backward
- GELU backward, SumRows op for bias/RmsNorm weight gradients
- Weight sharing and checkpointing support
- Metrics callbacks and MemorySummary
- LayerNorm backward (GradW, GradB, GradX)
- FullAttention backward for Whisper training
- Tanh op with backprop support
- Identity op for zero-cost reshape in training graphs
- Whisper encoder training graph helper

## Optimizations
- 4×4 register-tiled matmul: 1.5× faster forward, beats PyTorch inference
- 4×4 register-tiled backward matmuls with fused grad accumulation
- Generalize cooperative matrix for any tile size and precision
- 32×32 small-tile matmul/conv shader variants for low-occupancy layers
- Fuse SGD into step() submission (130ms → 99ms training)
- SwiGLUConcat: merge gate+up into single matmul
- Fused SwiGLU/Silu backward ops
- Fused RmsNorm+MatMul kernel with two-phase dispatch and rsqrt prologue
- Parallelize Conv2dGradWeight and GroupNormGradW shaders
- K-aware coop threshold for high-K backward matmuls
- e-graph: encode full graph with SwiGLU fusion, optimize before autodiff
- Pre-compute barrier group pass names at session creation
- Epilogue fusion infrastructure for matmul dispatch
- CausalAttentionRoPE: fuse RoPE into attention kernel
- BKV=8 tiled attention KV loop and dQ backward kernel
- Parallel prefill for KV-cache SmolLM2 benchmark
- Lower coop workgroup threshold from 128 to 32

## Correctness fixes
- Fix O(rows×cols²) complexity in RmsNormGradW shader
- Fix attention backward precision: store scores, add weight tying
- Fix derived_params lost during autodiff
- Fix GroupNorm grad race condition
- Fix RoPE convention and dispatch ordering
- Fix Adam buffer cleanup
- Fix coop RmsNorm shader: use workgroup reduction, not subgroups
- Eliminate O(N²) score buffer — recompute scores in backward
- Remove coop edge safety check — buffer padding handles all edges
- Various Metal execution fixes

## Infrastructure
- Switch codegen to WGSL templating
- CI latency benchmark with regression detection
- Conv2d split padding into h/w dimensions
- Windows compatibility, automated venv setup
- SmolVLA and SmolLM2 training benchmarks
- Subgroup reference cleanup, link NVIDIA driver bug tracker
- KV-cache decode mode for SmolLM2 benchmark
- Chunk-size flag for SmolVLA training benchmark

# v0.1 (26 Mar 2026)

## Inference & models
- SmolLM2-135M and SmolVLA action expert inference via blade-graphics (Vulkan)
- Vision ops: RoPE, causal/full/cross attention, RMSNorm, LayerNorm, SwiGLU, GELU, Embedding
- Single-pass causal attention (KV computed and consumed in one dispatch)
- HuggingFace SafeTensors model loading

## Optimizations
- Cooperative-matrix 2×2-tile matmul (16×16×16 WMMA, 32×32 output per workgroup)
- FusedMatMulAdd: merges `MatMul + Add` into one dispatch
- SwiGLU elementwise fusion: `silu(gate) * up` in a single kernel
- e-graph (egglog) optimization pass for pattern-driven fusion and canonicalization
- Parallel attention: 64 threads per workgroup (one lane per head dimension)
- Occupancy gate for coop matmul: falls back to scalar tiled path when parallelism is too low (e.g. SmolVLA chunk=50)

## Correctness fixes
- Coop matmul edge-tile corruption: secondary accumulators (acc_01/acc_10/acc_11) now guarded against writing to valid-but-wrong buffer positions when the tile extends past matrix bounds
- Coop self-test fixed (N=16→32) to avoid false negatives that disabled WMMA on working hardware
- Fixed OOB storage buffer reads in tiled matmul shader
- Fixed split-K shader binding crash

## Infrastructure
- Execution plan cache (RON serialization) to skip recompilation on repeated runs
- Perfetto binary trace support (`MEGANEURA_TRACE=path`) with blade GPU timestamps
- Benchmarks: SmolVLA meganeura vs PyTorch ROCm comparison script
- System precondition checks (AC power, GPU busy%, clock speed) before benchmarking
- DataLoader with MNIST IDX parser and mini-batch iteration
- Trainer struct with epoch/batch SGD loop
