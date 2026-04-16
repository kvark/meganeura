# Rejected Optimizations

Approaches that were implemented, benchmarked on RTX 3050, and reverted
because they didn't improve performance or hurt correctness.

## Double-buffered coop matmul staging

**Idea:** Overlap next-tile memory loads with current-tile tensor core
compute by using two sets of shared memory (8 arrays instead of 4).
Reduces barriers from 2 to 1 per K-iteration.

**Result:** SmolVLA training went from 22ms to 28ms — *slower*.
With only 64 threads (2 warps) per workgroup, the GPU's warp scheduler
cannot overlap staging with compute. Both warps execute the same mixed
staging+compute instruction stream sequentially. The extra shared memory
(4KB → 8KB) and more complex control flow add overhead without benefit.

**When it would help:** GPUs with more SMs (RTX 5080+) where more
workgroups per SM provide the parallelism needed for overlap. Or with
larger workgroup sizes (256 threads = 8 warps).

## BKV=8 tiled attention backward (GradQ)

**Idea:** Process 8 KV positions per tree_reduce pass (like the forward
attention already does), reducing barriers by 8× in the inner loop.

**Result:** SmolVLA training unchanged or slightly worse (21.5ms vs 20.8ms).
For causal attention with seq=50, many positions have kv_len < 8, falling
through to the scalar tail path. The tiled reduction uses 512-element
shared memory per array (vs 64), and the extra inner loop overhead
outweighs the barrier savings for short sequences.

**When it would help:** Longer sequences (seq=512+) where most positions
have kv_len >> 8, amortizing the tiling overhead.

## Lower coop matmul threshold (< 20 workgroups)

**Idea:** Enable tensor cores for shapes like MatMul[32×192×576] that
have ceil(32/32)×ceil(192/32) = 18 coop workgroups, below the threshold.

**Result:** SmolLM2 training improved 44ms → 29ms (dramatic!), but the
end-to-end gradient finite-difference test failed. The coop matmul stages
f32 → f16 → tensor core → f32, and the f16 precision loss accumulates
across 30+ transformer layers. Whisper training (4 encoder layers) also
failed to decrease loss at threshold=20.

**Root cause:** VK_KHR_cooperative_matrix on NVIDIA Ampere only exposes
f16 tiles (f32_tile=0). There's no way to use tensor cores with f32
precision through Vulkan on this hardware.

**When it would help:** Hardware/drivers that expose f32 cooperative
matrix tiles, or a mixed-precision training strategy that's designed
to tolerate f16 rounding.

## Split-K for low-occupancy coop matmul

**Idea:** For shapes like MatMulBT[50×720×4096] with only 46 coop
workgroups (2.3 per SM), split the K dimension across wgid.z to create
more workgroups (e.g., 46×4=184). Each split writes a partial result,
then a SumRows reduction sums the partials.

**Result:** SmolVLA training went from 20.3ms to 22.0ms — *slower*.
The reduction dispatch overhead (SumRows kernel launch + barrier +
temp buffer allocation) exceeds the occupancy improvement on 20 SMs.

**When it would help:** Larger GPUs (80+ SMs) where occupancy matters
more and the fixed per-dispatch overhead is amortized over more compute.
Also beneficial with a persistent-kernel approach that avoids the
reduction dispatch entirely.
