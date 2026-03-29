# KV Cache Implementation Plan

## Problem

During autoregressive generation, every decode step reruns the full forward
pass for all positions. PyTorch uses KV caching to process only the new token
after prefill. For SmolLM2-135M generating 32 tokens (seq_len=37), this means
Meganeura does ~37x more matmul work per decode step than necessary.

## Architecture

### Two-graph approach

Build separate **prefill** and **decode** graphs:

- **Prefill graph**: processes full prompt `[prompt_len, hidden]`, writes K/V to
  cache buffers, returns logits. This is essentially the current graph with K/V
  outputs added.
- **Decode graph**: processes 1 token `[1, hidden]`, reads/appends KV cache,
  returns logits for 1 position. Dispatch sizes drop dramatically (e.g. matmul
  workgroups go from `ceil(37/64)` to `ceil(1/64) = 1`).

### Pre-allocated KV cache

Cache buffers sized to `[max_seq_len, kv_dim]` per layer, allocated once at
session creation. A runtime `kv_pos` parameter tracks the write position.
For SmolLM2-135M (30 layers, kv_dim=192): 30 x 2 x max_seq x 192 x 4 bytes.

## New Graph Operations

### `CacheWrite`

Writes `[1, dim]` into row `kv_pos` of a `[max_seq, dim]` cache buffer.
Dispatch: `[ceil(dim/256), 1, 1]`.

```
inputs: [new_kv, cache_buf]
params: kv_pos (from runtime input buffer)
output: cache_buf (in-place write at row kv_pos)
```

### `CachedCausalAttention`

Attention with Q from current token and K/V from cache.

```
inputs: [q, k_cache, v_cache]    // q: [1, num_heads*head_dim]
params: kv_pos, num_heads, num_kv_heads, head_dim
dispatch: [1, num_heads, 1]
inner loop: for t in 0..kv_pos+1 (no causal mask needed, all past visible)
output: [1, num_heads*head_dim]
```

### `RoPE` with position offset

The decode graph processes `[1, dim]` tensors. RoPE currently derives position
from the row index (`pos = i / half_dim`), so for a single-row input `pos` is
always 0. Add a `pos_offset` uniform parameter to the RoPE shader:

```wgsl
let pos = i / half_dim + params.pos_offset;
```

For prefill, `pos_offset = 0` (backward compatible). For decode, `pos_offset = kv_pos`.

## New Shaders

### `cache_write.wgsl`

Simple copy kernel: writes `[1, dim]` into row `kv_pos` of `[max_seq, dim]`.

### `cached_attention.wgsl`

Based on `attention.wgsl` but:
- Q is `[1, num_heads*head_dim]` (single position)
- K/V are `[max_seq, kv_dim]` cache buffers
- Loop `for t in 0..kv_pos+1` with `kv_pos` read from a runtime input buffer
- Dispatch `[1, num_heads, 1]`

## Model Changes (`models/smollm2.rs`)

### `build_prefill_graph()`

Like current `build_graph()` but also outputs K/V tensors per layer as graph
outputs, so they can be read back and used to initialize the decode cache.

### `build_decode_graph()`

- Input: `token_ids: [1]`, `kv_pos: [1]` (u32)
- Per layer:
  1. Project Q/K/V from `[1, hidden]`
  2. Apply RoPE with offset = kv_pos
  3. `CacheWrite` K and V to cache at kv_pos
  4. `CachedCausalAttention` using cache
- Output: logits `[1, vocab]`
- Cache buffers as writable parameters (pre-allocated to max_seq_len)

## Runtime Changes

### Phase 1 (minimal, two separate Sessions)

Load weights into both prefill and decode sessions (duplicates parameter
memory but requires zero runtime changes):

```rust
// Prefill
prefill_session.set_input_u32("token_ids", &prompt_tokens);
prefill_session.step();
prefill_session.wait();
let logits = prefill_session.read_output(prompt_len * vocab);
let kv_caches = read_kv_outputs(&prefill_session); // new API

// Initialize decode session caches
decode_session.upload_kv_caches(&kv_caches);

// Decode loop
for step in 0..max_tokens {
    decode_session.set_input_u32("token_ids", &[next_token]);
    decode_session.set_input_u32("kv_pos", &[prompt_len + step]);
    decode_session.step();
    decode_session.wait();
    let logits = decode_session.read_output(vocab); // only 1 position
    next_token = argmax(&logits);
}
```

### Phase 2 (shared parameter buffers)

Add the ability for two Sessions to share GPU buffers for parameters,
eliminating the VRAM duplication.

### Phase 3 (single Session, two plans)

Integrate prefill and decode into a single Session with two execution plans
and shared buffers. The prefill plan writes directly into cache buffers used
by the decode plan.

## Compile Changes (`compile.rs`)

- New `ShaderEntry::CacheWrite`, `ShaderEntry::CachedCausalAttention`
- Compile `Op::CacheWrite`: dispatch `[ceil(dim/256), 1, 1]`, bind `kv_pos`
  input buffer
- Compile `Op::CachedCausalAttention`: dispatch `[1, num_heads, 1]`, bind
  `kv_pos` input buffer plus cache buffers
- Handle in-place buffer write for CacheWrite (output aliases cache input)

## Files to Modify

| File | Changes |
|------|---------|
| `src/graph.rs` | New `Op::CacheWrite`, `Op::CachedCausalAttention`, `Op::RoPE` with offset |
| `src/models/smollm2.rs` | `build_prefill_graph()`, `build_decode_graph()` |
| `src/shaders/cache_write.wgsl` | New shader |
| `src/shaders/cached_attention.wgsl` | New shader |
| `src/shaders/rope.wgsl` | Add `pos_offset` to params |
| `src/codegen.rs` | New `ShaderGroup` entries, generation functions |
| `src/compile.rs` | Compile cases for new ops, buffer aliasing |
| `src/runtime.rs` | New shader data structs, dispatch bindings |
| `bench/bench_meganeura.rs` | Prefill + decode loop |

## Expected Performance

For SmolLM2-135M generating 32 tokens: ~10-20x speedup on the decode phase.
Projection matmuls go from `[37, 576] x [576, 576]` to `[1, 576] x [576, 576]`.
Attention cost is linear in cached sequence length (scan over kv_pos entries).
