# Reduced-precision storage: status and plan (October 10, 2026)

Branch `experiment/f16-weight-storage-2026-10-10`, based on `3102683` (the
P3HPC October cohort revision).

## What exists

- `DType::F16` tensors, `Graph::parameter_f16`, and `Op::ToF16` (a cast whose
  backward is the identity, so an f32 master parameter receives the f32
  gradient).
- Matrix kernels read f16 **weights** (`WeightFormat::F16`) in the scalar,
  split-K, small-tile and matrix-vector paths, and `Embedding` reads an f16
  table. Uploads convert f32 data to f16 (`runtime/transfer.rs`).
- The accelerated contract feeds f16 *inputs* to cooperative matrix tiles,
  converting f32 operands inside the shader; storage stays f32.

## What does not

- **Activations are always f32.** Every intermediate buffer, attention,
  normalization, pointwise and convolution kernel reads and writes f32.
- **No bf16.** WGSL has no bf16 type; GGUF bf16 tensors are widened to f32.
- **f16 weights disabled cooperative tiles** at `3102683`: `Pipelines::key`
  checked reduced storage before cooperative kernels. Fixed in this branch.
- **No f16 training path.** Nothing keeps f32 master weights with f16 copies
  for forward and backward, and there is no loss scaling.

## In this branch

1. `Graph::store_weights_f16()` converts every f32 parameter whose only uses
   are the weight operand of `MatMul`, `MatMulBT`, `FusedMatMulAdd`,
   `FusedMatMulBTAdd` or an embedding table. Inferena's runner applies it to
   inference sessions under `INFERENA_F16_WEIGHTS=1` (accelerated contract
   only). Training is untouched.
2. f16-input cooperative tiles read f16 weights directly: `Variant::Coop`
   carries the weight format, the cooperative generators declare B as
   `array<vec4<f16>>`, and `kernels::matmul` admits f16 weights for plain
   (not compensated) f16 tiles. `tests/coop_f16_weights.rs` checks plain and
   transposed-B products against an f64 reference on the RTX 5070.

Measurements: `/x/Code/inferena-results/f16-weights-20261010`.

## Remaining stages

1. **Search over storage.** Let measured search choose f32 or f16 storage
   per weight instead of a session-wide switch.
2. **f16 activations at matrix boundaries.** A storage dtype per buffer in the
   memory plan; matrix kernels write f16 outputs with f32 accumulation;
   normalization, softmax and reductions read f16 and accumulate in f32.
   Pointwise kernels are generated, so f16 I/O is a codegen parameter there.
3. **Mixed-precision training.** f32 master parameters with f16 copies made
   once per step (`ToF16` already has the right backward), f16 activations
   saved for backward, f32 parameter gradients, and a static loss scale
   folded into the loss gradient. Derivative regions that need full precision
   keep their current marking.
4. **bf16.** Storage only: pack two bf16 values per `u32`; widening is
   `bitcast<f32>(bits << 16)`. bf16 cooperative tiles need
   `VK_KHR_shader_bfloat16`, which Naga does not expose.

Each stage needs its own numerical gate class: bf16 autocast in PyTorch
already misses the 1% sampled-output gate on SmolLM2 (1.02% on the RTX 5070).
