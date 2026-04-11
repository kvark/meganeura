# Plan: KV-Cache for Autoregressive Inference

## Goal

Avoid recomputing all previous positions during autoregressive text
generation. Currently, generating token N recomputes the full [1..N]
forward pass. With KV-cache, the decode step processes only the new
token and reads cached K/V states from previous positions.

**Expected speedup**: ~10-20x for decode latency at typical sequence
lengths (32-512 tokens), since each step processes 1 position instead
of N.

## Current state

Infrastructure is **partially built** (see existing docs/kv-cache-plan.md):
- `Op::CacheWrite` and `Op::CachedAttention` are defined in graph.rs
- `ShaderGroup::CacheWrite`, `ShaderGroup::CachedAttention` exist in codegen.rs
- Compile dispatch stubs exist in compile.rs (lines 1450-1490)
- Shaders referenced but not yet written: `cache_write.wgsl`, `cached_attention.wgsl`

What's missing:
- The actual WGSL shaders
- Model-level `build_prefill_graph()` / `build_decode_graph()` methods
- Benchmark integration
- RoPE with dynamic position offset

## Design (two-phase inference)

### Phase 1: Prefill

Process the full prompt in one pass, outputting K/V caches per layer.

```
prefill_graph:
  token_ids: [prompt_len]
  → full transformer forward (existing build_graph)
  → outputs: logits[prompt_len, vocab], kv_cache[layers][prompt_len, kv_dim]
```

This reuses the existing forward graph but adds extra outputs for the K/V
projections at each layer. The K/V outputs are written to pre-allocated
cache buffers.

### Phase 2: Decode (per-token)

Process one token using cached K/V from all previous positions.

```
decode_graph:
  token_id: [1]
  kv_pos: [1] (u32, current position in cache)
  kv_caches: [layers][max_seq, kv_dim] (read/write)

  per layer:
    q = Linear(hidden, Wq)           // [1, q_dim]
    k_new = Linear(hidden, Wk)       // [1, kv_dim]
    v_new = Linear(hidden, Wv)       // [1, kv_dim]
    RoPE(q, k_new, pos=kv_pos)       // dynamic position
    CacheWrite(k_cache[layer], k_new, kv_pos)  // write at position
    CacheWrite(v_cache[layer], v_new, kv_pos)
    attn = CachedAttention(q, k_cache[layer], v_cache[layer], kv_pos)
    ...rest of layer (output proj, FFN)...
```

### New ops

**CacheWrite**: Write a [1, dim] vector to cache[pos, dim].
```wgsl
@compute @workgroup_size(256)
fn main(...) {
    let col = wgid.x * 256u + lid.x;
    if col < dim {
        cache[pos * dim + col] = new_kv[col];
    }
}
```
Dispatch: `[ceil(dim/256), 1, 1]`. Trivial kernel.

**CachedAttention**: Attend to all cached positions [0..pos+1].
```wgsl
@compute @workgroup_size(64)
fn main(...) {
    // Single query position, all cached KV positions
    let head = wgid.y;
    let tid = lid.x;  // head_dim element

    let q_val = q[head * head_dim + tid];
    var my_out = 0.0;
    var max_score = -1e30;
    var sum_exp = 0.0;

    for (var t = 0u; t <= pos; t++) {
        // Q·K dot product via tree reduction (same as current attention)
        // Online softmax accumulation
    }
    output[head * head_dim + tid] = my_out / sum_exp;
}
```
Dispatch: `[1, num_heads, 1]`. Similar to current attention but always
q_seq=1 and kv_seq=pos+1. Later, this becomes a Flash Attention variant
with tiled KV access for long sequences.

**RoPE with position offset**: Extend the existing `rope.wgsl` to accept
a `pos_offset` parameter. Currently, position = row index. With KV-cache,
the decode graph has row_index=0 but the actual position is `kv_pos`.

### Implementation steps

1. Write `cache_write.wgsl` — trivial row-write kernel
2. Write `cached_attention.wgsl` — single-query attention over cache
3. Extend `rope.wgsl` with `pos_offset` parameter for dynamic positioning
4. Add `build_prefill_graph()` to SmolLM2 model — same as `build_graph()`
   but outputs K/V per layer
5. Add `build_decode_graph()` — single-token graph with CacheWrite +
   CachedAttention
6. Create two `Session` instances in the benchmark: one for prefill, one
   for decode. Share the K/V cache buffers between them via `upload_param()`
7. Update `bench_meganeura.rs` to use prefill + decode loop
8. Measure: time-to-first-token (prefill) + per-token latency (decode)

### Session management

The key challenge: the decode session needs to READ buffers written by
the prefill session (the KV caches). Options:

a) **Single session**: Build one graph that handles both prefill and decode
   via conditional execution. Complex.

b) **Two sessions, shared buffers**: Prefill session writes KV caches.
   Copy cache data to decode session's buffers via `upload_param()`.
   Simple but requires a CPU roundtrip.

c) **Two sessions, GPU-shared memory**: Use Blade's buffer sharing or
   export/import to share GPU buffers between sessions. Most efficient
   but requires Blade API additions.

**Recommendation**: Start with (b) for correctness, then optimize to (c).
The prefill is a one-time cost; the per-token decode latency is what
matters.

### Expected impact

At 32-token generation (SmolLM2 benchmark):
- Current: ~32 * full_forward_latency ≈ 32 * 5ms = 160ms
- With KV-cache: 1 * prefill(5ms) + 31 * decode(~0.3ms) ≈ 14ms
- **~11x speedup** for the generation benchmark

The decode step processes [1, hidden] instead of [N, hidden], making all
matmuls tiny. At M=1, matmul is fully memory-bound — latency is dominated
by weight loading, not compute. This is where the per-barrier overhead
is relatively small (one token, same barrier count, but each dispatch is
fast because M=1).
