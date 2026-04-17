# Plan: Flash Attention — remaining work

## Done

**Forward kernel** (`generate_flash_attention_module` in codegen.rs):
- BQ = 256/head_dim query positions per workgroup (4 for hd=64)
- Shared-memory K staging, reused across BQ groups
- Per-position causal + sliding-window masking
- Automatic selection: compile.rs uses FA2 when q_seq >= BQ
- Falls back to BQ=1 kernel for large head_dim or short sequences

## Remaining: backward kernels

The backward shaders (`mha_grad_q.wgsl`, `mha_grad_kv.wgsl`) still dispatch
one workgroup per (position, head). Tiling the backward pass would:

1. Reduce dispatch count by BQ× (same as forward)
2. Enable score recomputation from LSE (no O(N^2) score buffer needed)
3. Share K/V loads across Q tiles in the dK/dV kernel

The dQ kernel would recompute S = Q @ K^T per tile, recover P from stored
LSE, compute dS = P * (dP - D), and accumulate dQ += dS @ K.

The dK/dV kernel iterates Q tiles per KV tile, accumulating dK and dV.

**Estimated effort:** ~1 day for both backward kernels + testing.
