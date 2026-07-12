# Unreleased

- E-graph extraction is now the rewrite mechanism: extracted terms are
  stamped back into the graph IR (roots and interior nodes rewritten in
  place so node ids, id-carrying attributes, and output bindings stay
  valid; fused nodes appended; dead nodes swept). The hand-written
  pattern appliers, the kind-level applier gating, the legacy text
  `(extract …)` path, and `MEGANEURA_NO_TRAFFIC_COST` are gone —
  egglog's cost-model decision is what runs, per site.
- Generic egglog encoding: only ops that rewrite rules mention keep
  named constructors; everything else encodes through arity-generic
  `Op1..Op6` constructors tagged with the node id. Ops with different
  attributes can no longer be wrongly unified, any op (present or
  future, e.g. from ONNX import) encodes without prelude changes, and
  parameter/input names no longer appear in program text (no escaping
  hazards). `StopGradient` is no longer conflated with `Identity`.
- Unified segmentation: repeated regions saturate one instance and
  stamp every instance (per-instance parameter substitution included —
  packed-SwiGLU derived params are created per layer at stamp time);
  all remaining nodes are chunked into under-cutoff windows and
  saturated too. Every node now passes through the e-graph exactly
  once — previously >300-node graphs bypassed it for everything
  outside detected regions, and graphs with encoding gaps silently
  fell back to pattern matching.
- The packed-SwiGLU rewrite (`SwiGLU(MatMul(h,wg), MatMul(h,wu))` →
  one wide matmul over a concatenated derived weight) is now an egglog
  rule (`SwiGLUPacked`) chosen by the cost model, with a stamp-time
  fallback to the unpacked form when weights aren't plain parameters.
  The `FusedRmsNormMatMul` rewrite rule is removed (its applier was
  disabled for a documented ~25% regression; the kernels remain).
- `compile::topological_order` rebuilt on an adjacency list: nodes
  listing the same input twice (`Mul(x,x)`, grad-sum `Add(g,g)`) used
  to under-decrement and fall into an append-in-id-order fallback,
  which silently mis-ordered dispatches once the optimizer created
  nodes referenced by earlier ids — reads of never-written buffers,
  i.e. zero gradients. Also removes two O(n²) scans.
- autodiff: CrossEntropyLoss/Softmax/LogSoftmax backward now broadcast
  per-row sums via `SumInner` + `ones[1, F]` instead of a dense
  `ones[F, F]` matmul — the old form materialized a 9.66 GB constant
  for a 49k vocab and was 76% of the SmolLM2-135M train step
  (RTX 5070: 106.8 → 22.0 ms/step; Radeon 610M: 1774 → 325 ms/step;
  session buffers 11.3 GB → 1.69 GB).
- Audit hardening: execution-plan cache format v2 now fingerprints the complete
  typed graph, constant data, compilation options, runtime mode, optimization
  switches, and cooperative-matrix target. Stale, partial, or cross-device
  plans are rejected instead of being executed.
- Session construction now probes the selected GPU before compilation and
  compiles against that context's capabilities. This removes process-global
  first-device coupling and lets f16 cooperative attention be selected safely
  on both f16-only and scalar-only adapters.
- Attention pipelines are keyed by `(entry point, head dimension)`, fixing
  mixed-width attention graphs that previously reused the last width seen.
- Checkpoint format v2 records truthful tensor dtypes and shapes and validates
  every parameter and Adam-state byte length before upload. Invalid metadata is
  reported as `InvalidData` instead of silently resetting state; the old
  unbounded Adam copy could overwrite mapped memory.
- Session teardown now releases weight-specialized and attention pipelines,
  gradient-clipping storage, and accumulation buffers. Host uploads wait for
  prior GPU submissions before modifying shared allocations.
- Gradient clipping now uses an f32 accumulator with explicit barriers between
  parameter dispatches instead of a device-scope storage atomic. This removes
  Vulkan VUID 06265 on cooperative-matrix contexts where Blade enables the
  Vulkan memory model without enabling device scope.
- `DataLoader` rejects zero batch and sample sizes; cooperative 16×16 paths now
  require an exact advertised tile size; decimal and hexadecimal device IDs are
  accepted.
- Portability documentation now matches the pinned Blade implementation:
  Vulkan on Linux/Windows/Android and Metal on Apple platforms; DX12 is not
  currently a backend.
- CI now enforces formatting, all-target Clippy, strict rustdoc, MSRV, package
  assembly, security, and Windows compile checks. Offline SPIR-V coverage
  includes the f16 cooperative attention path used by NVIDIA and AMD, while
  hardware-driver coverage remains an explicit release requirement.
- Bump `naga` to the wgpu git rev carrying
  `spv::Options::emit_int_div_checks`, pinned to the same rev as
  `blade-graphics` so the `naga::Module` we build unifies with blade's
  SPIR-V backend. Blade (rev `ba0fb5a`) sets the flag to `false`,
  eliding the divide-by-zero / `INT_MIN/-1` guard wrappers naga
  otherwise emits around integer division and modulo — a win for the
  index-heavy conv2d im2col and reduction shaders. Adds the new
  `CommandEncoderDesc::manual_barriers` field (kept `false`).
- Device-local intermediates (default on, kill switch
  `MEGANEURA_NO_DEVICE_LOCAL=1`): allocations holding only step-local
  intermediates live in `Memory::Device` (GPU-zeroed at session build),
  keeping user-visible buffers host-visible. Avoids routing
  intermediate traffic through the host-visible (ReBAR) heap on
  discrete GPUs.
- Repeated-region outlining: graphs over the egglog saturation cutoff
  (300 nodes — every real training graph) no longer skip the e-graph.
  Structurally repeated blocks (transformer layers, forward and
  backward) are detected by signature periodicity + edge-isomorphism
  verification, and equality saturation runs on one instance per
  region. `OptimizeReport` gains `outlined_regions`.
- Traffic-aware extraction cost: when shapes are known, an e-node's
  cost is the HBM bytes it moves (inputs read + output written), so a
  fusion wins by exactly the intermediate traffic it eliminates — no
  hand-tuned constants. `MEGANEURA_NO_TRAFFIC_COST=1` reverts to the
  constant per-constructor scheme.
- On graphs under the cutoff, the extractor's term choices now gate
  which fusion appliers run (previously extraction results were only
  logged); egglog saturation deepened from 1 to 3 iterations so
  chained rewrites (Silu → SwiGLU) reach fixpoint inside the e-graph.
- Lifetime-based buffer aliasing: step-local intermediates with
  disjoint live ranges (at barrier-group granularity) now share one
  physical GPU allocation. Parameters, inputs, outputs, gradients,
  KV caches, and other persistent buffers are never aliased. Opt out
  with `MEGANEURA_NO_ALIAS=1`. `MemorySummary` reports allocated vs
  logical bytes.
- `docs/roadmap.md`: strategic plan across compiler, memory,
  precision, latency, conv, autotuning, and quantized fine-tuning
  tracks.
- `Session::with_context(plan, Arc<Context>)` lets a host application
  (renderer, game) share a single `blade_graphics::Context` with
  meganeura's training and inference sessions instead of each side
  opening its own device.
- New unified entry point `build(graph, SessionConfig)` plus `Mode`
  enum. Replaces the `build_session` / `build_session_with` /
  `build_session_with_report` / `build_session_with_report_and_options`
  / `build_session_cached` / `build_inference_session_with` family
  with a single struct-parameterised call. Sugar functions
  `build_session`, `build_inference_session` and
  `build_session_unoptimized` remain for the common cases.

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
