# Optimization Plan: Surpassing PyTorch ROCm

Baseline (AMD RDNA3, SmolVLA action expert, float32):
- meganeura: **24.5 ms/step** (40.9 steps/s)
- PyTorch ROCm: **24.1 ms/step** (41.6 steps/s)

GPU profiling (multi-pass mode, first 100 of 202 dispatches = ~8 transformer layers):
```
Silu:             8x   15.13ms  56.6%
MatMul:          48x    7.40ms  27.7%
FusedMatMulAdd:  13x    3.45ms  12.9%
CausalAttention:  4x    0.25ms   0.9%
RmsNorm:         14x    0.21ms   0.8%
CrossAttention:   3x    0.17ms   0.6%
Mul:              6x    0.10ms   0.4%
```

The Silu figure is inflated by multi-pass barrier overhead (each pass boundary = one ALL_COMMANDS
barrier + command-processor relaunch), but the dispatch count is real: 16 FFN layers each produce
separate Silu and Mul dispatches that could be one.

---

## [x] 1. SwiGLU fusion — `Mul(Silu(x), y)` → single kernel

Every FFN computes `gate_proj → Silu(gate) → Mul(silu, up_proj)`: two dispatches with an
intermediate roundtrip through 50×2048 global memory. A fused SwiGLU kernel reads gate and up
simultaneously, applies Silu in registers, writes one result. Removes 16 barrier groups per
forward pass, halves FFN intermediate memory traffic.

Implementation:
- Add `ShaderEntry::SwiGLU` and `ShaderGroup::SwiGLU`
- Generate a SwiGLU compute kernel in codegen.rs (two inputs, one output)
- Add e-graph rewrite rule: `Op::Mul` with one child being `Op::Silu` → `Op::SwiGLU`
- Emit dispatch in compile.rs

---

## [ ] 2. Fused MatMul output activation (`FusedMatMulSilu`, `FusedMatMulSwiGLU`)

The gate_proj MatMul always feeds Silu; the combined gate+up can feed SwiGLU directly. A
`FusedMatMulSwiGLU` kernel applies the activation while tiles are still in registers after
the cooperative MMA, before writing to global memory — eliminates the Silu/SwiGLU intermediate
buffer entirely. This is a structural advantage PyTorch's rocBLAS cannot match.

Implementation:
- Extend cooperative matmul codegen with an optional epilogue enum (None / Silu / SwiGLU)
- `FusedMatMulSwiGLU` takes A, B_gate, B_up (or two separate MatMuls with shared A) and fuses
  the SwiGLU into the output store loop
- Add e-graph rules to detect the pattern

---

## [ ] 3. Adaptive Layer Norm fusion (`AdaRmsNorm`)

Each transformer layer applies `RmsNorm(x)` then multiplies/adds timestep-conditioned scale and
shift before the Q/K/V projections. Currently 3+ separate barrier groups per layer. A fused
`AdaRmsNorm(x, weight, scale, shift)` kernel does all in one pass — specific to diffusion models,
not something PyTorch's generic kernels expose.

Implementation:
- Add `Op::AdaRmsNorm` to graph.rs and detect the pattern in the model builder
- Generate a fused kernel in codegen.rs
- Add e-graph rule or detect directly in smolvla.rs model graph

---

## [ ] 4. Double-buffered K-tile loop in cooperative matmul

The current K-tile loop: load A-tile and B-tile into shared memory → barrier → 4× CoopMMA →
barrier → next tile. With double buffering, tile N+1 is prefetched into a second shared memory
buffer while tile N is computing, hiding global memory latency. Can improve throughput 10–20%
for large K (e.g. K=2048 down-proj or K=720 attention projections).

Implementation:
- Allocate 2× shared memory (shared_a0/a1, shared_b0/b1) for ping-pong buffering
- Restructure gen_matmul_coop_inner to emit a prologue (load tile 0), main loop (compute tile i
  while loading tile i+1), and epilogue (compute last tile)

---

## [ ] 5. fp16 activations

RDNA3 has 2× fp16 throughput vs fp32. Using f16 for cooperative matmul inputs while accumulating
in f32 (standard WMMA pattern) would roughly halve matrix multiplication time. Requires:
- Converting input buffers to f16 before matmul (or storing activations as f16)
- Adapting cooperative matrix declarations in codegen (A/B in f16, C in f32)
- Careful handling of norms and attention scores which need f32 precision

---

## [ ] 6. GEMV kernel for M=1 shapes

The time MLP operates on [1, 720] and [1, 1440] — single-row matmuls. The 32×32 cooperative
tile dispatches only 1 workgroup in M, leaving 11 of 12 CUs idle. A dedicated GEMV kernel
parallelizes over the N dimension with each thread doing a dot-product reduction over K.

Implementation:
- Detect M==1 at compile time in compile.rs
- Generate a GEMV shader (1 thread per output element, reduction over K in shared memory)
- Fall back to standard matmul for M>1

---

## Notes

- Run `bash bench/compare.sh` after each step to verify correctness and record numbers
- Update README.md benchmark table and bench/results/ after each confirmed improvement
- Commit each optimization separately for clean history
