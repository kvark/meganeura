# Plan: Generic kernel fusion via schedule templates

## Thesis

Triton isn't winning because it has thousands of kernels — it wins because it
has ~4 **kernel schedules** parameterized by a DSL, plus autotuning. Luminal
has essentially one schedule (elementwise) and falls off a cliff the moment
matmul or reduction is involved.

For a small project, we can replace ~35 hand-written WGSL files with **4
schedule-template generators** sharing one lowerer to Naga IR, driven by
e-graph cut selection and on-device autotuning. This closes most of the
perceived Triton gap without carrying a kernel zoo.

## Non-goals

- A full Triton-style tile DSL users can write. We expose templates only
  through the existing ML graph.
- A new IR. Naga `Module` is our IR; we add a thin **schedule-level builder**
  layer that targets it.
- Eliminating every bespoke shader. Conv, scatter, embedding, optimizer
  updates, concat/split/transpose stay hand-written for now.

## The four archetypes

All four lower to `naga::Module` through one shared lowerer. All share one
`PointwiseDAG` type for fused elementwise chains in prologues/epilogues.

### 1. Pointwise (elementwise DAG)

One generator accepting any DAG of pure elementwise ops over a common
broadcast shape. Arbitrary fan-in, fan-out, chain length. One thread per
output element (or a small vector).

**Replaces**: `unary.wgsl`, `binary.wgsl`, `bias_add.wgsl`, the individual
unary/binary ops in `graph::Op`, `swiglu_concat.wgsl`, `swiglu_grad.wgsl`,
elementwise halves of norm-grad / cross-entropy / bce. **~12 shader files.**

### 2. Reduction

One generator parameterized by: reduction op, identity, axes, pointwise
prologue (applied to inputs), pointwise epilogue (applied to output).
Two-stage tree reduction; subgroup shuffle for small axes, shared memory
for large.

**Replaces**: `reduce.wgsl`, `sum_rows.wgsl`, reduction halves of
`softmax.wgsl` / `rms_norm.wgsl` / `layer_norm.wgsl` / `group_norm.wgsl`.
**~6 shader files.** Softmax and RMSNorm *emerge* from this template rather
than being hand-coded.

### 3. Matmul with prologue + epilogue

The biggest win. Parameters: A/B pointwise prologues, tile config
(BM, BN, BK, coop-matrix on/off), pointwise epilogue DAG (arbitrary —
replaces the closed `EpilogueOp` enum).

- **Prologue** fuses RMSNorm scale into K loads, RoPE into Q/K loads,
  future dequant into weight loads.
- **Epilogue** fuses arbitrary pointwise chains, not just the current 6
  hardcoded ops.

**Replaces**: `matmul.wgsl`, `matmul_add.wgsl`, `matmul_{at,bt}*`,
`matmul_small*`, `matmul_coop*`, `matmul_rms_norm*`, `FusedMatMul*` variants
in `Op`. **~10 shader files.**

### 4. Attention (FlashAttention-style)

Parameters: mask type (none/causal/sliding/block), KV layout, GQA grouping,
Q/K prologue (for RoPE fusion), optional LSE output for backward.
Shared-memory tiling over K/V sequence; online softmax in registers.

**Replaces**: `attention.wgsl`, `cached_attention.wgsl`, `mha_forward.wgsl`,
`sliding_window_attention.wgsl`, the three `mha_grad_*.wgsl`. **~6 shader
files.**

## Architecture

### Schedule layer, not a new IR

```rust
// src/schedule.rs (new)
enum KernelTemplate {
    Pointwise { dag: PointwiseDAG, grid: GridShape },
    Reduction { op: ReduceOp, prologue: PointwiseDAG, epilogue: PointwiseDAG, ... },
    Matmul    { tile: TileConfig, prologue_a: PointwiseDAG,
                prologue_b: PointwiseDAG, epilogue: PointwiseDAG, ... },
    Attention { ... },
}

fn lower(t: &KernelTemplate) -> naga::Module { ... }
```

`PointwiseDAG` is a flat list of scalar ops (Add, Mul, Silu, Sigmoid, …)
with a small expression tree. Reused across all four archetypes.

The lowerer emits WGSL source text and hands it to Naga's WGSL frontend.
This is the same path `codegen::epilogue_to_wgsl` already uses; parsing is
~100µs per shader so a round-trip through text is acceptable and keeps
kernels dumpable for debugging. Direct `naga::Module` construction is a
later optimization, not a requirement.

### E-graph picks topology, autotune picks tiles

Fusion is a cut-selection problem, not a rewrite. The two stages:

**Stage 1 — cost-model extraction.** E-graph saturates with fusion
rewrites:

- `pointwise(matmul(A,B))       ↔ matmul_epilogue(A, B, pointwise)`
- `matmul(pointwise(A), B)       ↔ matmul_prologue_a(A, pointwise, B)`
- `reduce(op, pointwise(x))      ↔ reduce_prologue(op, x, pointwise)`
- `pointwise_a(pointwise_b(x))   ↔ pointwise_fused(...)`

Extraction minimizes modeled HBM traffic (bytes in + bytes out per
dispatch; matmul inputs scaled by 1/tile-reuse). Flops as a tiebreaker.
Deterministic, fast, no GPU needed. Picks the fusion topology.

**Stage 2 — autotune.** For the chosen topology, enumerate tile configs
for heavy archetypes (matmul, attention). Matmul: BM,BN ∈ {32,64,128},
BK ∈ {16,32}, coop on/off (~36 candidates). Attention: BQ, BK_tile,
head-parallelism (~12). Run each 3× on device, take median, cache winner
by `(kernel_hash, shape, device_name)` in `cache.rs`.

Pointwise and reduction skip autotune — their tile size is a direct
function of shape.

**Escalation.** When cost-model extraction has top-K topologies within
~10% of each other, autotune each; pick overall winner. K usually 1.

### Naga is fast → specialize aggressively

Naga at ~100µs per shader means compile-time caching isn't load-bearing.
Two implications:

1. **Autotune is cheap.** 36 matmul candidates × 100µs compile = 3.6ms.
   GPU run dominates; total autotune per shape ≈ 2–5ms.
2. **Specialize on values we'd otherwise pass as uniforms** — constant
   tile dims, small-loop unroll counts, stride constants. Turns loops
   into unrolled const-propagated code. 10–20% win on elementwise /
   reduction. Triton does this heavily.

Cache shifts role: no longer cache Naga `Module` (no point), instead
cache **autotune winners** (shape → tile config) and **driver-level
pipelines** (SPIR-V/DXIL first-touch compile can be tens of ms).

## Sequencing

Each step is a shippable improvement that keeps the benchmark suite at
parity (correctness ≤1e-3, latency within 5%). Flip new path behind an
env var (`MEGANEURA_SCHEDULE=1`); remove old path only after 2–3 benches
show parity.

1. **Schedule scaffold + pointwise lowerer.** New `src/schedule.rs` with
   `PointwiseDAG`, `KernelTemplate::Pointwise`, WGSL emitter. Unit test:
   generate the equivalent of `binary.wgsl` entries, parse via Naga, run
   on-device, compare output. No integration with compile.rs yet.
2. **Pointwise archetype wired into compile.** Collapse `Op::{Relu,
   Sigmoid, Tanh, Neg, Abs, Log, Recip, Silu, Gelu, SwiGLU, Greater, Add,
   Mul, BiasAdd}` into `Op::Pointwise(DagId)`. Fusion pass walks maximal
   elementwise subgraphs. Delete ~10 shader files.
3. **Epilogue generalization.** Replace `EpilogueOp` enum with a
   `PointwiseDAG` on `Dispatch`. Delete `FusedMatMul*` variants.
4. **Reduction archetype.** Collapse softmax/rmsnorm/layernorm/mean/sum
   into the parameterized template with prologue/epilogue.
5. **Matmul prologue support.** Absorb `matmul_rms_norm*` and standalone
   RoPE fusion sites.
6. **E-graph cost-driven cut selection.** Add fusion rewrites to egglog;
   switch fusion pass from heuristic to extraction-based. Before this,
   fusion is rule-based; after, it's search.
7. **Attention archetype.** FlashAttention generator. Hardest, biggest
   single perf win — last so shared infra is battle-tested.
8. **Autotuning.** Tile-config enumeration + on-device measurement +
   cache. Matmul first, then attention.

## Risks

- **Blast radius.** Steps 2–4 touch the hottest test-covered code path.
  Each step gated by env var; both paths live side-by-side until parity
  verified on `bench_smolvla_train`, `bench_smolvla_meganeura`,
  `bench_sd_unet_train`.
- **No warp intrinsics in vanilla WGSL.** FlashAttention wants warp-shuffle
  softmax. Blade exposes subgroup ops (already used in `matmul_coop.wgsl`);
  attention generator gates on subgroup availability with shared-memory
  fallback.
- **Debuggability.** Generated kernels are harder to read than hand-written
  ones. Mitigation: always dump generated WGSL to `target/kernels/<hash>.wgsl`
  when `MEGANEURA_DUMP_KERNELS=1`; keep human-readable dispatch labels.

## First milestone

Step 1 from the sequencing: `src/schedule.rs` with `PointwiseDAG` and a
WGSL emitter that reproduces `binary.wgsl`'s `add` entry. Unit test asserts
Naga parse succeeds and the emitted source computes the right thing.
Deliberately disconnected from the rest of the compiler — this proves the
codegen pattern works in isolation before we rewire anything.
