# Unreleased

- `KernelMemo` is public. `BuildSearchOptions::kernel_memo` lets searches
  share private kernel decisions, and `KernelMemo::save`/`load` carry them
  between processes. Decisions are resumed only on the same device, driver
  and decision policy; budgets do not affect them. A resumed winner is still
  covered by whole-program qualification. Without a memo, each search keeps
  its decisions private, as before.
- Measured-search challengers are constructed on the incumbent's identically
  stored parameters that neither program writes, including fused weights,
  instead of allocating and zeroing private copies. The new
  `Session::inherits_parameter` tells an initializer which uploads it can
  skip. Writing an inherited parameter writes the incumbent's storage, so
  initializers must keep writing the same values for every candidate.
- Sessions on one GPU context share compiled pipelines with identical
  generated WGSL, entry point and binding layout, and the last user destroys
  each pipeline. A measured-search challenger now compiles only the kernels
  that differ from the live incumbent's. The cooperative-matrix self-test
  runs once per context and configuration, not once per session.
- Measured builds qualify a program once before timing it, instead of up to
  four times per trial. The check before kernel tuning runs only when kernel
  classes remain to be probed. The check after measurement runs only for a
  winner. The incumbent is requalified once before construction returns, not
  after every trial. `BuildSearchTrial::qualifications` and
  `BuildSearchReport::final_qualification_time` record the checks.
- Native16 f32 GEMM keeps K-split boundaries independent of staging width.
  A wider staging tile masks its final half tile instead of moving terms
  between partial sums. This preserves reduction order when tuning staging
  and prefetch, avoiding amplified rounding changes in later f16 attention.
- Measured builds retain native16 f32 attention forward, dQ and dK/dV as
  independent egglog implementation alternatives on supporting devices.
  Selection uses the existing numeric qualification and whole-program timing;
  ordinary builds and precision policies keep their previous choices. Forward
  supports power-of-two heads from 16; backward initially covers 64-wide heads
  with at least 128 rows on each side. Native8 and f16 kernels remain available.
  Plan-cache format 23 rejects reuse by readers that always generate native8
  for f32 attention gradients.
- Devices with native 16x16 f32 cooperative matrices (MI300X on RADV) run
  aligned f32 products on 64-row tiles shared by four subgroups, split
  along K when the output alone cannot occupy the device. Tuning retunes a
  split's tile shape and partition count together with its SumRows. Dense
  f32 GEMVs from 512 columns up group adjacent output columns on these
  devices by default, and `GemvShape::column_groups` also accepts 2.
  Inputs whose allocation a cooperative tile pads accept their logical
  size. Plan-cache format 22.
- `Graph::gelu_erf`: the exact GELU, `x·(1 + erf(x/√2))/2`; `Graph::gelu`
  stays the tanh approximation. The Whisper encoder and the `sd_unet`
  feed-forward now use the exact form, so their outputs and benchmark
  workloads change.
- `normalize_inner_sum` computes as its decomposition does, forward and
  gradient, so the fused and explicit graphs agree bit for bit on lavapipe
  as well.
- Tuning treats disjoint slices of one allocation as separate bindings,
  and the cooperative probe at session start no longer reads mapped
  memory back on the CPU.
- Blade is pinned to `56f0565`. Vulkan devices with `shaderFloat16` but no
  f16 cooperative matrix (MI300X on RADV, lavapipe) now get the 16-bit
  storage features that f16 shaders declare, and descriptor sets are reused
  across recordings instead of being reallocated. A Vulkan device must now
  offer both 16-bit storage features for f16 cooperative shapes to be
  reported.
- Kernel families (`kernels`): interchangeable implementations of one
  computation, each declaring what it admits on a target and why it
  declines, with a fallback that admits every supported problem. Attention
  backward is the first: `ShaderEntry::AttentionGrad` carries the part
  (dQ or dK/dV) and path (f32 or f16 cooperative, flash or rowwise) in
  place of eight entries, and one selector replaces the per-part
  promotion logic. Profile and pipeline names read
  `AttentionGrad(dQ-flash)`, `AttentionGrad(dKV-cooperative-f32)` and so
  on. Plan-cache format 21. `CompileOptions::prefer_attention_grad` tries
  one path first wherever it admits the problem, and the oracle suite runs
  every path on shapes spanning each one's admission edges.
- Dense matrix products are the second kernel family: session
  construction picks cooperative, 32-wide or compiled tiles through one
  selector with stated reasons, and tuning asks the same family whether a
  cooperative kernel is legal, where the two kept separate copies of the
  alignment, grid and padding checks.
- Convolutions are the third: the compiler's register-tile choice and
  session construction's cooperative promotion live in one module, and
  promotion states why it declines. Selection is unchanged.
- Measured training builds explore independent scalar dQ and dK/dV layouts
  through egglog, interleaved with graph alternatives under the existing
  search bounds. Each dispatch and pipeline key retains its extracted EPT
  cap. Plan-cache format 19 records these layouts.
- Batch scalar attention-gradient reductions across query/key tiles, reducing
  workgroup synchronization while preserving f32 arithmetic and mask bounds.
  [Repeated Intel measurements](docs/gpu-gap-2026-10.md) reduce Whisper-tiny
  training time by 35.5% and SmolLM2-135M training time by 12.5%.
- `Graph::biased_attention` and `Graph::biased_cached_attention`: softmax
  attention with an additive bias per head, query row and key (T5 relative
  positions, ALiBi), a configurable logit scale, and full, causal or cache
  masking, lowered to one fused kernel for heads up to 512 wide.
- Building blocks in `graph::helpers`, spelled in primitives, for audio and
  sequence models such as Lyria: `elu`, `crop_2d`, `pixel_shuffle_w`,
  `dilate_w`/`dilate_h`, `upsample_nearest`, `conv_transpose_2d` (PyTorch
  kernel layout, separate strides, through the forward convolution),
  `t5_relative_bias` and `t5_relative_bias_cached` (with `t5_bucket`).
  New primitives `to_u32` and `constant_u32` drive gathers by computed or
  constant indices.
- Ops are classified as primitive, composite or private (`Op::class`).
  Every composite has a decomposition into primitives (`Graph::decompose`),
  and optimized builds recognize decompositions again (`Graph::recompose`),
  so a model spelled in primitives builds the plan of the model written
  with composites. Composites a graph names stay as written. Inference
  builds recognize every composite; training builds only a list of those
  whose gradient is verified to be exactly their decomposition's
  (`Graph::recompose_for`), leaving the losses, the inference-only
  attention forms and RoPE with dynamic positions in primitives
  (`Graph::decompose_for` shows a graph as a build in a mode treats it);
  builds with the optimizer off recognize nothing, so they run the
  primitives.
  Recognition matches whole templates exactly, comparing attributes by
  their bits. Tests hold every composite, every model builder, every GGUF
  architecture and the ONNX fixtures to the same dispatches
  (`ExecutionPlan::dispatch_inventory`) and the same values flowing between
  them (`ExecutionPlan::dataflow_digest`), compare each recognized
  composite's gradient with its decomposition's, and compile plans without
  a GPU (`compile_plan`).
- Equivalent spellings build one way: full, cross and multi-head attention
  share a lowering, a sliding window spanning the sequence is causal
  attention, and upsample and per-channel gate attributes that do not
  change the result take one canonical value.
- `MulPerChannel` has a gradient. Its lowering, and `AddPerChannel`'s,
  process every element of a tensor of any rank.
- `CausalAttentionRoPE`, which nothing builds, is a private fused op.
- A primitive op set (`Op::is_primitive`) that model builders can rely on,
  with new primitives `max_inner`, `sqrt`, `rsqrt`, `add_scalar`,
  `broadcast_to` (NumPy broadcasting at equal rank, up to 4) and
  `exclusive_cumsum`, and a gradient for `clamp`.
  `Graph::decomposed_softmax` and `Graph::decomposed_rms_norm` and
  `Graph::decomposed_layer_norm` are written in primitives only; builds
  recognize them as the fused `Softmax`, `RmsNorm` and `LayerNorm` kernels,
  so they reach the same plan-level fusions as the named ops. Norms written
  with `x / sqrt(variance + eps)` are recognized as well.
- The ONNX importer accepts decomposed exports instead of rejecting them:
  `Sqrt`, `Exp`, `Tanh`, `Pow` with a constant exponent, and last-axis
  `ReduceMean`, `ReduceSum` and `ReduceMax` map onto primitives, and binary
  ops follow NumPy broadcasting along each axis and fold scalar constants.
  RMSNorm, LayerNorm and softmax written this way build as the fused
  kernels. Incompatible shapes are load errors rather than panics or wrong
  results.
- New primitives `erf`, `batch_matmul` (with `_at`/`_bt` forms for its
  gradient) and `permute` (rank at most 4). The ONNX importer uses them to
  load transformer layers in the form PyTorch's exporter writes: N-D
  `MatMul`, any `Transpose`, `Slice` and `Concat` on any axis, `Erf`, and
  masks, rotary tables and biases broadcast along trailing axes. Reshape
  honors `0` dimensions and shape constants. `tests/fixtures/onnx` holds
  BERT- and Llama-style layers authored node by node in that form (not
  produced by the exporter itself), with outputs from ONNX's reference
  evaluator; they run on the GPU and build their norms, softmax and SwiGLU
  as fused kernels.
- The ONNX importer folds shape arithmetic (`Shape`, `Gather`, `Concat`,
  `Unsqueeze` on constants), so exports with dynamic axes load; reads
  `Squeeze`/`Unsqueeze` axes from inputs as opset 13 writes them, inserting
  several unsqueezed axes in output order and squeezing only the named
  ones; and expands broadcast axes with `Expand`, as grouped-query
  attention's `repeat_kv` does, in one broadcast. A `Gather` that is neither foldable nor a
  table lookup by U32 indices is a load error rather than a panic.
- Shaders keep every read inside its buffer themselves, since they compile
  without bounds checks: tile loads of the scalar, quantized, cooperative,
  convolution and Winograd matmuls clamp the index of lanes outside the
  matrix (reading past the end crashed software Vulkan); token ids, gather
  indices and cache positions past their table or cache stay inside it,
  bounded by row counts passed in the kernels' parameters; and buffers are
  allocated in whole 16-byte vec4s.
- The new elementwise primitives (`sqrt`, `rsqrt`, `erf`, `sin`, `cos`,
  `add_scalar`) and `max_inner` refuse storage other than F32 at
  construction instead of relabeling it.
- Concatenation, split, convolution, group norm and upsample gradients
  size their operands by element count, so operands of any rank
  differentiate; permute and broadcast gradients accept a gradient that
  arrives through a view.
- Keep scalar consumers of cooperative attention staging in f32. Check
  forward, dQ and dK/dV workgroup storage against the selected device's limit
  and include that limit in plan-cache compatibility.
- Reject malformed ONNX field lengths and reversed NNEF body delimiters;
  retain importer fuzz regressions and check RoPE dimensions before narrowing.
- Fail when a requested GPU cannot be opened. Share GPU contexts across
  test sessions and model generators to avoid exhausting driver resources.
- Compare windowed profiling results with a finite-value tolerance.
- Centralize contraction dimensions, quantized block geometry, attention
  uniforms and mask ranges. Move scheduling into the compiler and separate
  emission, binding, transfers and kernel generation into modules.
- Restore the flat dispatch representation and exhaustive shader binding.
  Cache format 18 rejects older schemas before decoding the execution plan.

- `optimizer_memory::optimizer_clipping_and_diagnostics_ignore_poisoned_allocation_padding`
  compared `f32` results for bit equality across allocation paddings, which
  failed on rounding rather than on a defect. Padding cannot reach the
  arithmetic — every optimizer, clip and accumulation pass bounds its loops by
  the segment table's logical length — but it can permute the order in which
  f32 values are summed, and the LaProp plus adaptive-clip path reduces a
  workgroup-sized tree whose lane occupancy follows the tile layout. Both
  observed values sit within one ULP of the exact f64 result. The test now
  compares with a `1e-5` relative tolerance, which still separates rounding
  from a real poison leak by about five orders of magnitude.
- Adam's bias correction is computed on the host. `adam.wgsl` used to raise
  `pow(beta, step)` once per parameter element — four transcendental ops per
  element, per step, for a value that is uniform across the whole dispatch
  and constant for the step. The uniform now carries `1 / (1 - beta1^step)`
  and `1 / (1 - beta2 ^ step)`, which also keeps one rounding for the whole
  step instead of one per thread.
- Check the test harness itself. `smoke::harness_manifest` walks the `mod`
  graph from the `[[test]]` targets and fails when a file under `tests/` is
  unreachable or a `mod` names a file that is not there. `outline_optimize`,
  `profile_windows` and `resnet_correctness` each carried real assertions
  and none of them was compiled. The two PyTorch-parity tests that compare
  against a gitignored fixture are now `#[ignore]`d instead of returning
  early, so a run without the fixture reports them as not-run rather than
  green.
- Move the lint contract from `lib.rs` to `[lints]` in `Cargo.toml`, add a
  `clippy.toml` with the two thresholds the codebase overrides, and drop 24
  per-function `#[allow(clippy::too_many_arguments)]` attributes that the
  crate-level allow already covered.
- `Session::record` puts a step — inference, or training with its optimizer
  update — into a caller's command encoder instead of a submission of its
  own; the encoder must use automatic barriers.
  `Session::track_submission` lets `wait`, uploads, readbacks and drop wait
  for the caller's submission.
- Fix wrong training gradients: scalar attention backward dropped every
  dimension past 63 for heads wider than 64 on sequences shorter than the
  flash block; LayerNorm weight gradients for two or three rows kept only
  the first row and wrote past the output; MaxPool2d backward spread
  gradients evenly over the wrong elements instead of routing them to each
  window's maximum (ResNet's stem); BCE produced NaN once a sigmoid
  saturated; GroupNorm backward derived its variance by cancellation.
  The plan cache format is bumped so stale plans are rebuilt.
- Fold uniform constant tensors into pointwise kernels, take the ReLU mask
  from its output, and read biases through broadcast loads, so bias adds
  fuse with their activations and backward passes stop reading
  activation-sized constants.
- Spread global and adaptive gradient clipping, and large whole-tensor sums,
  across many workgroups instead of one per tensor.
- Reduce per-channel bias gradients in place instead of transposing them,
  compute GroupNorm backward statistics once, share attention's dot(dO, O)
  between dQ and dK/dV to preserve parallel execution, fold row blocks in
  norm weight gradients, pool wide planes with a workgroup each, tile
  transposes, and read the logits twice rather than three times in cross-entropy.

- Search matrix tiles, K staging, unrolling and split-K as e-graph alternatives,
  preserving logical fusion choices and qualifying complete implementations.
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
  content hash. Matmul schedules are now measured e-graph alternatives,
  not environment switches. The profiler state is armed once — by
  `init` or by GPU-context initialization — instead of being lazily
  conjured on first use, and `default_gpu_context` returns a fresh
  context per call rather than a hidden process-global first-device
  cache.
- The environment is resolved once: `src/config.rs` is the only place that
  reads `MEGANEURA_*` variables, and its product is configuration — it no
  longer modifies the process environment. The one boolean decoding rule
  is a pure function under test, so the config tests don't flip env vars.
  `MEGANEURA_OPTIMIZER` configures graph rewriting;
  `MEGANEURA_DEVICE_PARAMETERS` / `MEGANEURA_REUSE_UPLOAD` are
  `SessionOptions` fields. Current controls are registered and documented in the
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
