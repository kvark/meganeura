# Unreleased

- The K-split GEMV family gains a measured shape: workgroup width (32, 64,
  128 or 256) and cross-lane reduction (a workgroup-memory halving tree, or
  `subgroupAdd` within a wave and one partial per wave through workgroup
  memory). Every GEMV kernel — plain, fused-add, transposed-B, f16,
  block-packed, RmsNorm-folded — is derived from one source and shaped in one
  place, so the choice is a single axis across all of them rather than a knob
  per kernel.

  The subgroup reduction spends one `workgroupBarrier` whatever the width,
  against one per halving level for the tree: six fewer on AMD's 64-wide
  wave, five on a 32-wide one. It is written against the `subgroup_size`
  builtin rather than a compiled-in width, so one kernel is correct on wave64,
  wave32 and the 8-wide waves of a software rasterizer.

  `Session::tune_with` now searches this axis, which required admitting the
  GEMV entries and reduced-storage weights it had excluded outright — between
  them most of what a decode step runs. A packed B is qualified by agreement
  with the kernel the plan already runs rather than against an f32 reference
  dot, since forming one would mean writing a second block decoder and
  trusting it. Which shape wins is a device property and is not predicted.

  `MEGANEURA_GEMV_REDUCTION=tree|subgroup` sets the starting point, alongside
  the existing `MEGANEURA_GEMV_THREADS`.
- Structured profiling of plans larger than Blade's timestamp budget.
  `capture_session_profile` used to refuse outright any plan with more
  dispatches than the 1,000 timestamps Blade writes per submission, which a
  decode or training step of a real model exceeds. It now splits the plan into
  windows of dispatch indices and replays the session once per window,
  stitching the per-dispatch results back together. Every dispatch is still
  measured `samples` times; only the replay count grows, and a plan that fits
  in one window is captured exactly as before.

  The dispatches outside a window still execute — each replay is a complete,
  correct run — but batch into one grouped pass per side, keeping the plan's
  barriers while costing two timestamp slots rather than one per barrier
  group. `Session::set_profiling_window` exposes the same mechanism directly.

  `measurement` gains `window_count` and `max_window_dispatches`.
  `profiled_wall_samples_ms` now holds one entry per replay, and
  `instrumentation_wall_ratio` understates full-instrumentation overhead once
  windowing kicks in, since each replay runs most of the plan at nearly normal
  pass counts; `gpu_total_samples_ms` still sums to whole-plan GPU time.
  `schema_version` is 2 and `ProfileError::TooManyDispatches` is gone.
- GGUF weight import (`load::gguf`). Reads the container's metadata and tensor
  inventory, and resolves GGML's block encodings into Meganeura's at load time.
  `Q4_0`, `Q4_1` and `Q8_0` repack losslessly for `set_parameter_packed` —
  GGML splits a block's nibbles across halves and interleaves each block's
  header with its payload, where Meganeura pairs adjacent nibbles and keeps
  headers in their own region. The block *order* already agreed, since packing
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
