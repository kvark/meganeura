# Unreleased

- Compensated f16 cooperative matmul (C1): on devices that only expose
  f16 tiles, `requires_full_precision` work stages each operand as
  `hi = f16(x)` plus residual `lo` and accumulates `hi·hi + hi·lo +
  lo·hi` in f32. Inference keeps plain f16 staging. `AllowF16` still
  forces the uncompensated path.
- Horizontal fusion (D1): independent same-A matmuls that already share
  a barrier group (Q/K/V projections) pack into one dispatch with
  `workgroups.z` selecting the sibling.

- Parameter gradients, Adam m/v, the clip accumulator, and temporal
  grad-accumulators default to device-local memory. Host read/write
  (checkpoints, `read_param_grad`, `read_adam_*`) stages through a
  transient buffer — including on Metal, where `Memory::Device` is
  `MTLStorageModePrivate` and `contents()` is not a valid host
  pointer. `MEGANEURA_NO_DEVICE_LOCAL=1` and debug sessions keep the
  previous host-visible layout.

- Cross-entropy training reuses the forward kernel's fused logits gradient
  instead of rebuilding softmax and a ones-row broadcast matmul. Inference
  skips the unused gradient write.
- Binary cross-entropy no longer writes a per-element gradient into the
  per-workgroup loss buffer (that overran the allocation for `n > 1`).
- Large scatter-add uses float CAS with source-parallel work mapping. Blade
  enables Vulkan memory-model device scope when required, so serializing every
  source row behind one invocation per column is unnecessary. Narrow
  row-scaled scatters map one invocation to each source row.
- `egglog` and `naga` disable crate default features. Default crate
  features are now empty: `hub` (HuggingFace downloads) and `profiler`
  (Perfetto CPU-span subscriber) are opt-in. `SafeTensorsModel::from_bytes`
  loads in-memory assets.
- Apple flash EPT defaults key off `target_vendor = "apple"` (covers iOS).
  The 8×8 f32 cooperative veto keys off the advertised tile, not macOS.
- Unused in-tree Mistral / Phi-3 / Gemma-4 builders removed. Compiler
  internals (`compile`, `codegen`, `schedule`, …) are `#[doc(hidden)]`.

- External GPU composition: `Session::output_buffer` exposes a pinned graph
  output as a Blade `BufferPiece`, so a renderer sharing the context can feed
  a prediction directly into its next compute pass without a host readback.
- GroupNorm inference splits large image groups into parallel statistics and
  apply passes. Statistics use the generated reduction path and application is
  an entry point in the existing GroupNorm module, rather than two new shader
  groups. Small tensors retain the original single-pass kernel, avoiding an
  extra tensor traversal and barrier when the split would contain one chunk.

- The library core is now environment-free: `compile`, `runtime`,
  `codegen`, and `optimize` accept strongly typed options and never read
  `MEGANEURA_*` variables. New typed surface: `CoopPolicy`
  (Auto/Disabled/AllowF16) and the diagnostic switches on
  `SessionOptions`, flash coop toggles on `CompileOptions`,
  `SessionConfig::{optimize, runtime}`, and `GpuOptions` +
  `init_gpu_context_with` for adapter selection/timestamps.
  `TuningKnobs::default()` is pure platform defaults. Env-driven behavior
  is an explicit opt-in via the `from_env` constructors in
  `meganeura::config` (`SessionConfig::from_env()` resolves everything,
  including WGSL dump-dir installation and env-selected GPU contexts);
  the repo's examples, benches, and tests opt in, so external
  `MEGANEURA_*` workflows keep working — embedders that never call
  `from_env` get a fully hermetic library.

- Central environment-variable registry (`meganeura::config`): every
  `MEGANEURA_*` variable is declared once with type, class, and docs.
  Session build logs active overrides and warns on unrecognized
  `MEGANEURA_*` names (typos no longer fail silently); a test pins the
  README table against the registry. Boolean semantics are now uniform —
  unset = default, `0` = off, anything else = on. Two behavior
  normalizations: `MEGANEURA_DISABLE_COOP=0` (and other diagnostic
  flags set to `0`) no longer *enable* the switch, and
  `MEGANEURA_FLASH_BWD_COOP` accepts any non-zero value instead of
  exactly `1`. Precedence is per class: diagnostic switches override
  code configuration; tuning variables only feed defaults, so
  explicitly set `TuningKnobs`/`SessionConfig` fields win.

- Eager evaluation (`meganeura::eager::Eager`): inspect any node of a graph
  while building it — `e.eval(&g, node)` executes the same builder-produced
  graph through the same generated kernels (no rewrites, no dispatch
  fusion, NodeIds stay valid as the graph grows) and returns a printable
  `Tensor`. The same graph then compiles unchanged via `build_session` for
  the fast path; eager-vs-compiled parity is tested. A development mode:
  each growth step re-executes the prefix.
- Shader consolidation (75 → 55 WGSL files, −2.3k lines): deleted the four
  small-tile twins (`matmul_small`, `conv2d_gemm_small`,
  `conv2d_grad_input_gemm_small`, `conv2d_grad_weight_gemm_small`) — tile
  size is now a template parameter with the unrolled register-tile bodies
  generated by `codegen::tiled_gemm_body`; retired the template-based conv
  coop kernels (`conv2d_gemm_coop.wgsl`, `conv2d_grad_input_gemm_coop.wgsl`
  and the legacy non-generated coop entries) in favor of the per-(kernel,
  stride) generated modules the runtime already preferred.
- Reduction archetype: multi-accumulator support — `extra_prologues` reduce
  additional integrands over the same inputs in one pass, and the epilogue
  sees all reduced scalars. LayerNorm forward now compiles to a single
  two-accumulator archetype kernel (sum + sum-of-squares), GPU-parity-tested
  against the hand-written shader. Gelu gained a pointwise-DAG mapping
  (tanh-approx, parity-tested), closing the last unary gap. The hand-written
  pointwise/reduction shaders stay as the `use_schedule_* = false` parity
  oracles rather than being deleted. Still hand-written by design for now:
  GroupNorm (epilogue needs indexed per-channel loads), the norm backwards
  (column-axis reductions), and LogSoftmax.
- Empirical variant selection (roadmap Track F, first cut):
  `SessionConfig { tune: true }` / `MEGANEURA_TUNE=1` measures real `step()`
  wall-clock with each flippable kernel family (plain coop matmuls,
  generated coop conv kernels) on its cooperative variant vs its scalar
  fallback, and keeps whichever is faster on this device — replacing static
  promotion thresholds with measurement. Selection now records a
  `scalar_fallback` on every promoted dispatch and compiles both variants'
  pipelines, so flips need no recompilation. Off by default until the bench
  fleet validates it; needs a coop-capable adapter to do anything.
- Kernel-variant selection has a single owner (roadmap A2, scoped):
  cooperative promotion (incl. generated conv kernels and output padding),
  the RmsNorm→matmul prologue fusion, and small-tile demotion moved from
  inline session-construction code into one `select_variants` pass.
- Tuning knobs are data, not globals: the flash-attention EPT caps moved
  into `CompileOptions::knobs` (`TuningKnobs`) and are stamped into the
  compiled plan, which pipeline generation reads back — geometry and
  generated WGSL can no longer disagree, and the plan cache fingerprints
  them via the options instead of re-reading env vars. `MEGANEURA_FLASH_*`
  env overrides still work as `TuningKnobs::from_env` defaults.
- Debug sessions: `build(&g, SessionConfig::debug())` (or
  `SessionOptions { debug: true }` on `Session::with_context_opts`) disables
  lifetime aliasing, keeps every buffer host-visible, and skips the
  numerics-neutral dispatch-level fusions so every graph node stays
  materialized. `Session::read_node` / `read_node_by_name` return any
  value after a step (with structured `ReadNodeError`s — fused-away and
  aliased values are reported as such instead of returning garbage), and
  `Session::step_debug()` scans dispatch outputs in execution order and
  attributes the first NaN/Inf to a named dispatch and graph node.
- Value identity and dispatch provenance: `Graph::named(id, "blk3.mlp.gate")`
  attaches names that survive autodiff, rewrites, and toposort (`nn` layers
  name their outputs automatically); every `Dispatch` now carries the graph
  node ids it implements (`origin`), merged through dispatch-level fusion.
  Labels, `MEGANEURA_DUMP_PLAN`, profiler rows, and NaN traces show
  `"blk3.mlp.gate: MatMul[50x960x720]"` instead of anonymous shader names,
  and the plan records a full node→buffer map (`ExecutionPlan::node_buffers`)
  plus node names for debug readback. Graph builder methods are
  `#[track_caller]`, so shape-assert panics report the model-builder line
  that created the bad node. Plan-cache format bumped to v3.
- Dead kernel purge: removed 10 unreachable WGSL files (~1.1k lines) and
  their plumbing — the direct `conv2d`/`conv2d_grad_input`/
  `conv2d_grad_weight` shaders (superseded by the GEMM path), the
  never-constructed `FusedRmsNormMatMul` op and both its shaders
  (superseded by `RmsNormRsqrt` + matmul prologue), the never-selected
  `Conv2dGradInputGemmCoop3x3` (superseded by the generated per-shape
  conv coop kernels; its documented `MEGANEURA_CONV_COOP` switch had no
  reader), the `WinogradBatchedMatMulSmall` alias, and four files no
  code included (`mha_forward`, `winograd_matmul_at`, both winograd
  grad transforms).
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
