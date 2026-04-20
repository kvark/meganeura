# Pipeline-stats branch retrospective

This document replaces the original "next big lever" plan with a
post-mortem of what shipped, what got reverted, and what to skip on
non-NVIDIA platforms.

## What shipped (kept)

| change                        | shader / file                            | win                                |
|-------------------------------|------------------------------------------|------------------------------------|
| LayerNorm workgroup reduction | `src/shaders/layer_norm.wgsl`            | Whisper inference -23%             |
| Flash attention fwd (coop)    | `codegen::generate_flash_attention_coop_module` | Whisper inference -53%      |
| Flash backward dQ (coop)      | `codegen::generate_flash_grad_q_coop_module`    | Whisper training -14%       |
| Flash backward dKV (coop)     | `codegen::generate_flash_grad_kv_coop_module`   | SmolLM2 train -29%, Whisper -21% |

All three coop kernels follow the same template — `coop_mat<f16>` for
the AB inputs, per-thread scalar registers for accumulators that span
the loop, `coop_mat<f32>` only for the per-tile MMA. They gate at
runtime on `gpu.capabilities().cooperative_matrix.f16_tile == 16`
(populated by `runtime::auto_tune` → `codegen::set_coop_caps`).

## What got tried and removed

* **`coop_mat<f32, C>` accumulator spanning the KV loop.** Naga emits
  it (no validation error), but on Blackwell it hits a race that
  produces exactly `2×` the scalar result at random elements. The
  fix is the per-thread register accumulator pattern used by all three
  shipped kernels. Don't try the cross-loop coop accumulator again —
  the bug is in the SPV backend, not in our shader.
* **`array<coop_mat<...>, N>`** is rejected by naga's SPV backend
  ("Expression [N] is not cached"). Unroll into N variables instead.
* **Per-flash-kernel EPT auto-tune** (`auto_tune_flash_ept`,
  `FlashEptConfig`, the `MEGANEURA_FLASH_*_EPT_CAP` env vars). The
  measurements showed register pressure didn't track wall-clock,
  and the combinatorial explosion (5 kernels × N caps × per-head_dim)
  made the tune pass take ~30s without measurable benefit. Replaced
  with a single `flash_ept_cap()` reader that defaults to 32 and
  takes `MEGANEURA_FLASH_EPT_CAP=N` for one-off experiments.
* **Register-aware fusion cost model** (`RegisterCostTable`,
  `set_fusion_register_costs`, `with_register_costs`,
  `measure_fused_op_register_costs`,
  `measure_pipeline_register_count`). Idea: penalize fusions whose
  driver-reported register count overshoots the occupancy threshold.
  In practice, `optimize` runs before any GPU exists in many test
  paths, the table was always empty, and on real workloads the
  unfused alternative was slower regardless. Plain `FusionCostModel`
  ships unchanged behavior.
* **Conv2dGradInputGemmCoopV2** (flash-coop-style implicit GEMM for
  conv backward). GPU profile showed -44% per dispatch; wall-clock
  showed +0.3 ms regression on ResNet — suspected CPU dispatch
  overhead from 14× more workgroups. The pre-existing legacy
  `Conv2dGradInputGemmCoop3x3` (opt-in via `MEGANEURA_CONV_COOP=1`)
  still wires through, in case the wall-clock story changes on a
  different driver.
* **Split `FlashGradK` / `FlashGradV` kernels.** Hypothesis: avoid
  the dKV kernel's register pressure by splitting work. Measured
  ~5 % regression because each kernel re-reads the same Q/dO/L tiles.
  Reverted; only `FlashGradKV` (the fused form) ships.

## Platform notes

* **NVIDIA Vulkan only.** The cooperative-matrix shaders use
  `requires fp16, language_extension; enable f16, subgroups`. The Apple
  Silicon path (`metal`) goes through `simdgroup_matrix` via blade's
  Metal backend, which is exercised by `matmul_coop_8x8_f32` already.
  AMD RDNA3 in theory has VK_KHR_cooperative_matrix but we haven't
  validated tile sizes 16x16x16; the gate (`f16_tile == 16`) will
  silently fall back to scalar if the driver reports a different tile.
* **`Conv2dGradInputGemmCoop3x3`** is hand-written WGSL with a 16x16
  tile baked in. It works on Blackwell and Ada; it has not been
  exercised on RDNA3 / Apple. Default-off (`MEGANEURA_CONV_COOP=1`).

## Non-trivial lessons that survived

1. **Per-thread register accumulators beat workgroup-cooperative
   accumulators when the accumulator state crosses a loop boundary.**
   The redundant per-row work (each row of a tile is recomputed by
   `BKV/8` threads) is cheaper than the shared-memory roundtrip and
   sidesteps the naga SPV bug.
2. **Pipeline-statistics register counts are a register-pressure
   signal, not an occupancy signal.** Two kernels at the same reg
   count can have wildly different wall-clock times depending on
   whether the workgroup count saturates the device. Use the stats
   for diagnosis, don't bake them into the optimizer.
3. **GPU-time wins do not always translate to wall-clock wins on
   small dispatches.** When per-dispatch GPU time drops below
   ~50 µs, CPU command-submission and synchronization dominate.
   Check `step()` wall-clock before declaring victory from the
   per-kernel profile alone.
