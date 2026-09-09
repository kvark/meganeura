# Unreleased

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
