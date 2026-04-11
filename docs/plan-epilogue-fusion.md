# Plan: Epilogue Fusion (Triton-style composable codegen)

## Goal

Eliminate separate dispatches for elementwise ops (BiasAdd, Silu, Add, ReLU,
RmsNorm scaling) that follow a matmul by injecting them into the matmul's
store loop. Each fused op saves one dispatch + one barrier (~33μs on NVIDIA).

## Why not hand-written shaders?

Today, every fusion combination needs a pre-written `ShaderGroup` variant
(`MatMulAdd`, `MatMulATAdd`, `MatMulBTAdd`, etc.) — 6+ entries for one
epilogue. Adding Silu, ReLU, BiasAdd, RmsNorm would explode to hundreds.

Instead, we generate the WGSL epilogue **dynamically** from the computation
graph, similar to how Triton composes tile-level programs.

## Design

### 1. Epilogue IR

Define a small enum representing composable post-matmul operations:

```rust
// In compile.rs or a new epilogue.rs
enum EpilogueOp {
    Add(BufferRef),              // + src[idx]
    BiasAdd(BufferRef),          // + bias[col]
    Mul(BufferRef),              // * src[idx]
    Silu,                        // x * sigmoid(x)
    Relu,                        // max(x, 0)
    Sigmoid,                     // 1 / (1 + exp(-x))
    Tanh,                        // tanh(x)
    RmsNormScale(BufferRef),     // * rsqrt_cache[row] * weight[col]
}
```

A `Vec<EpilogueOp>` on each matmul `Dispatch` describes the chain.

### 2. WGSL codegen

Extend `matmul_vars()` in `codegen.rs` to accept an epilogue chain and
generate two things:

- **Declarations** (`$EPILOGUE_DECL`): `var<storage> src: array<f32>;` etc.
- **Expression** (`$EPILOGUE_EXPR`): the composed WGSL inline expression

Example for `MatMul + BiasAdd + Silu`:
```wgsl
// $EPILOGUE_DECL
var<storage> bias_buf: array<f32>;

// Store loop becomes:
let val = s[i][j] + bias_buf[col];
matrix_c[idx] = val / (1.0 + exp(-val));  // silu
```

The key: epilogue codegen is **string composition**, not a new shader file.
Each unique epilogue chain produces a unique `naga::Module` (keyed by the
chain hash), compiled once and cached.

### 3. E-graph discovery

Add egglog rules for common patterns:
```lisp
(rewrite (Silu (Add (MatMul ?a ?b) ?d))
         (FusedMatMulEpilogue ?a ?b (epilogue-add-silu ?d)))
(rewrite (Relu (Add (MatMul ?a ?b) ?d))
         (FusedMatMulEpilogue ?a ?b (epilogue-add-relu ?d)))
```

Or, simpler: in the graph reconstruction pass, walk backward from each
elementwise op, checking if its input chain leads to a single-use MatMul.
If so, absorb the chain into the matmul's epilogue.

### 4. Cooperative matrix compatibility

The scalar matmul store loop (lines 100-106 of matmul.wgsl) can fuse any
epilogue since each thread stores one element at a time.

The cooperative matrix path uses `coopStoreT` which stores a full tile —
no per-element epilogue is possible. Options:
- Use the scalar store path for the last tile (where epilogue applies)
- Only fuse epilogues on the scalar matmul path
- Add a post-coop epilogue pass using shared memory (load coopStore result,
  apply epilogue, write back) — overhead may negate the benefit

**Recommendation**: Start with scalar-path-only epilogue fusion. The coop
path is already fast; the scalar path is where the barrier savings matter
most (it's used for shapes that don't meet the coop threshold).

### 5. Implementation steps

1. Define `EpilogueOp` enum and add `epilogue: Vec<EpilogueOp>` to `Dispatch`
2. Write `fn epilogue_to_wgsl(chain: &[EpilogueOp]) -> (String, String)` in codegen
3. Modify `matmul_vars()` to inject epilogue WGSL
4. Add pattern matching in `optimize.rs` to detect epilogue-fusible chains
5. Modify `compile.rs` to build `Dispatch` with epilogue when the graph has fused ops
6. Add pipeline caching keyed by `(ShaderGroup, epilogue_hash)`
7. Handle buffer bindings for epilogue operands

### Expected impact

SmolVLA forward has ~7 elementwise dispatches (BiasAdd, Silu, Add, Neg, Mul,
MeanAll). Fusing the BiasAdd + Silu chains saves ~4 dispatches × 33μs =
**~0.13ms**. Modest individually, but compounds across models and the
infrastructure enables further fusions.

The real payoff is **architectural**: new activations or normalizations
automatically fuse without new shader files.
