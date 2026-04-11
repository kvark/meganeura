# Plan: Flash Attention

## Goal

Replace the current per-position attention kernel (1 workgroup per query
position, sequential KV loop, 7 barriers per KV via tree_reduce) with a
tiled Flash Attention kernel that processes blocks of KV positions in
shared memory.

## Current bottleneck

At chunk_size=50, attention is ~5% of forward time. At chunk_size=400, it
grows to ~28% because the kernel is O(q_seq * kv_seq * head_dim) with poor
parallelism:
- 64 threads per workgroup (= head_dim)
- Sequential loop over all KV positions
- 7 workgroup barriers per KV position (tree_reduce for dot product)
- Stores all q_seq * kv_seq scores to memory for backward (O(N^2) memory)

## Design

### Forward kernel: `flash_attention.wgsl`

**Workgroup**: 256 threads (4 warps of 64 on AMD, 8 warps of 32 on NVIDIA)
**Dispatch**: `[ceil(q_seq / BQ), num_heads, 1]`
**Tile sizes**: BQ=32 (query positions per WG), BKV=32 (KV positions per tile)

```
// Each workgroup processes BQ query positions
for each KV tile [t, t+BKV):
    // Load K tile [BKV, head_dim] into shared memory
    // Compute score block S[BQ, BKV] = Q_tile @ K_tile^T / sqrt(d)
    //   - tiled matmul in shared memory (BQ * BKV * head_dim FMAs)
    //   - each thread handles one (q, kv) pair or a subset
    // Online softmax: update running max, sum_exp, output accumulator
    // Load V tile [BKV, head_dim] into shared memory
    // Accumulate O += P @ V (softmax weights * values)
```

**Key differences from current kernel**:
- Processes BKV=32 KV positions per barrier instead of 1
- Score computation via tiled matmul in shared memory (no tree_reduce)
- O(BQ + BKV) shared memory per tile pair instead of O(1)
- No score storage — only LSE (max_score, log_sum_exp) per query position

**Shared memory layout** (~12KB per workgroup):
```
shared_k: array<f32, BKV * head_dim>     // 32 * 64 = 2048
shared_v: array<f32, BKV * head_dim>     // 32 * 64 = 2048
shared_s: array<f32, BQ * BKV>           // 32 * 32 = 1024
// Q can stay in registers (each thread holds one or more Q rows)
```

**Attention variants** (via template variables):
- Causal: skip KV tiles where all positions are masked
- Full (cross-attention): process all KV tiles
- Sliding window: restrict KV range per query position

### Backward kernel: `flash_attention_grad.wgsl`

The key insight from Flash Attention: **recompute scores** in the backward
pass instead of reading stored scores. This eliminates the O(N^2) score
buffer entirely.

**dQ kernel**: Dispatch `[ceil(q_seq / BQ), num_heads, 1]`
```
for each Q tile:
    load Q tile
    for each KV tile:
        recompute S = Q @ K^T / sqrt(d)
        recompute P from S using stored LSE (max, log_sum)
        dP = dO @ V^T         (need dO and V)
        dS = P * (dP - D)     (D = rowsum(dO * O), pre-computed)
        dQ += dS @ K * scale
```

**dK/dV kernel**: Dispatch `[ceil(kv_seq / BKV), num_kv_heads, 1]`
```
for each KV tile:
    load K, V tiles
    for each Q tile (that attends to this KV tile):
        load Q, dO tiles
        recompute S, P
        dK += (dS)^T @ Q * scale
        dV += P^T @ dO
```

**GQA handling**: When num_heads > num_kv_heads, the dK/dV kernel
accumulates gradients across the `heads_per_kv` query heads that share
each KV head.

### Memory savings

| | Current | Flash |
|---|---|---|
| Score buffer | q_seq * num_heads * kv_seq * 4B | 0 |
| LSE buffer | q_seq * num_heads * 2 * 4B | q_seq * num_heads * 2 * 4B |
| At N=400, 15 heads | 400 * 15 * 400 * 4 = 9.6MB | 0 |

### Implementation steps

1. Write `flash_attention.wgsl` forward kernel with BQ=32, BKV=32, WG=256
2. Add `ShaderGroup::FlashAttention` and `ShaderEntry::FlashAttention`
3. Update `compile.rs` to emit FlashAttention dispatch (no score buffer!)
4. Write `flash_attention_grad_q.wgsl` and `flash_attention_grad_kv.wgsl`
5. Update `autodiff.rs` to use the new gradient kernels
6. Remove score buffer allocation from the execution plan
7. Template variables for causal/full/sliding-window variants
8. Test: numerical gradient check against the existing attention backward

### Compatibility

The existing attention shaders remain for head_dim != 64 or other edge
cases. FlashAttention is selected when head_dim is a supported tile size
(64, 128) and sequences are long enough to benefit from tiling.

### Expected impact

- chunk_size=50: attention from ~0.44ms to ~0.15ms (3x faster, BKV tiles)
- chunk_size=400: attention from ~10ms to ~1.5ms (6-7x faster)
- Memory: eliminates O(N^2) score buffer (9.6MB at N=400)
- Barrier reduction: ~30 fewer barriers in backward (no per-position dispatch)
