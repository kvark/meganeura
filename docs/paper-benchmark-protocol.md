# Meganeura paper benchmark protocol

Protocol name: `inferena-paper-v1`

This document defines the experiment before the final paper revision is
frozen. A result that violates the protocol may be useful for debugging but
must not appear as a valid paper cell.

## What “repairing the benchmark” means

Running every engine from one revision is necessary but not sufficient. The
repair covers:

- exact model architecture, tensor shapes, parameter names/layouts, frozen
  parameters, initialization, inputs, labels, and loss objective;
- equal inference and training scopes;
- explicit arithmetic permissions;
- matching compile/timing boundaries;
- warmups, repeated raw samples, and robust summary statistics;
- output samples distributed across the complete tensor rather than a
  convenient prefix;
- forward loss, total gradient norm, canonical trainable-parameter set, and
  per-parameter gradient-norm validation;
- independent validity for inference and training;
- exact code revisions, GPU selection, driver, and environment metadata.

The old harness could accept superficially close losses from different models
and sampled only the first flattened logits. `paper-v1` is designed to reject
that class of mistake.

## Audited engines

The paper comparison currently includes only:

- PyTorch;
- Meganeura.

Other Inferena runners remain available under `legacy`, but their model,
precision, compilation, and timing semantics have not been audited for this
paper. They must not be mixed into the main tables until audited to the same
standard.

## Workloads

| Workload | Audited graph | Full inference/training shape | Objective |
|---|---|---|---|
| SmolLM2-135M | 30-layer, d=576 decoder-only transformer | batch 1, sequence 128 | token cross entropy |
| SmolVLA | 16-layer, d=720 action expert, 99,848,592 parameters | batch 1, 50 action tokens, 16 VLM tokens | action-output MSE |
| Conditioned diffusion U-Net | three-level, base-64 latent U-Net with timestep injection and spatial self/cross-attention, 10,928,768 parameters | batch 1, 4×32×32 noisy latent, 64-D timestep embedding, 77×768 text context | noise-prediction MSE |
| ResNet-50 | matched ResNet-50 with inference-folded BN represented by identity scale and trainable channel bias, 25,530,472 parameters | batch 4, 3×224×224 | class cross entropy |
| Whisper-tiny encoder | four encoder layers, 8,208,384 total / 7,632,384 trainable parameters | batch 1, 80×3000 mel input, 1,500 encoded positions | mean square of encoder output |

The diffusion workload exercises the characteristic conditioning and
attention structure, but is scaled and is not checkpoint-compatible with
Stable Diffusion 1.5. Its batch size is currently one because Meganeura's
differentiable attention primitive represents one sequence per operation.
The Whisper workload has no decoder. The ResNet workload does not measure
training-mode BatchNorm statistics. The historical `stable-diffusion` and
`whisper` CLI names must be clarified in the paper.

## Precision contracts

| Property | `strict-f32` (primary) | `accelerated-f32` (secondary) |
|---|---|---|
| Persistent tensor storage | f32 | f32 |
| Output | f32 | f32 |
| PyTorch matmul/convolution | f32; TF32 disabled | `high`; TF32 permitted |
| Meganeura scalar paths | f32 | f32 |
| Meganeura cooperative inputs | f16 path disabled | f16 inputs permitted locally |
| Accumulation for accelerated matrix paths | f32 | f32 |
| Comparison meaning | same nominal arithmetic class | documented hardware-fast paths, not identical formats |

The secondary comparison is fair only if it is labeled accurately. PyTorch
TF32 and Meganeura f16 cooperative inputs are both normal fast paths, but they
are not the same number format. Neither result may be presented as strict f32.

An accelerated inference cell can pass while its training cell fails. A
historical Whisper run demonstrated this: local f16 conversion preserved the
forward output but corrupted derivative work. The current implementation
propagates a full-precision requirement through autodiff and keeps backward
matrices in f32. Every final cell must still pass the gate independently.

### Precision audit

The audited workload graphs contain no persistent f16 parameters,
activations, or explicit f16 embedding operations. Meganeura's accelerated
Vulkan path reads f32 buffers, rounds eligible matrix operands to f16 while
staging cooperative tiles, accumulates into an f32 cooperative matrix, and
stores f32 output. Strict mode removes the general f16 cooperative-matrix
opt-in and separately disables the cooperative f16 forward and backward
attention kernels. A device may still use an all-f32 cooperative matrix path
where the API exposes one.

PyTorch strict mode requests `highest`, explicitly disables both CUDA matmul
TF32 and cuDNN convolution TF32, and sets NVIDIA's process-level TF32 override
to zero before importing PyTorch. The result artifact records the effective
PyTorch flags, and the harness rejects a strict CUDA result unless every
control is false/disabled. Accelerated mode requests `high` and explicitly
permits TF32 for CUDA matmul and cuDNN.

TF32 and IEEE f16 have similar significand precision but different exponent
ranges: TF32 retains the f32 exponent range, while f16 can overflow or
underflow values that TF32 represents. Therefore accelerated mode is a
comparison of each engine's documented reduced-input/f32-accumulate fast path,
not an identical-arithmetic experiment. Numerical gates establish suitability
for each workload, not equivalence of the formats.

The paper should keep strict f32 as the primary fairness result. Accelerated
results answer a different, useful systems question—how the fastest
quality-validated hardware paths compare—and must remain a separate table.
A future matched-format experiment could add PyTorch autocast f16, but should
not replace the strict baseline and would need its own cast, accumulation, and
loss-scaling audit.

## Timing

Each series uses:

- five untimed warmup executions;
- at least 20 retained executions;
- median as the headline value;
- p25/p75 (IQR), minimum, maximum, and all raw samples in the artifact.

Scopes:

- **Compile:** graph construction, optimization, and GPU pipeline creation for
  inference, training, and latency sessions. Model download/loading and
  parameter upload are excluded or reported separately.
- **Inference:** complete no-gradient forward at the full workload shape.
- **Latency:** the agreed minimal shape (single token, action chunk, or
  batch-one equivalent), not an arbitrary engine-specific shortcut.
- **Training:** forward + scalar loss + backward; no optimizer update.

GPU work must be synchronized at measurement boundaries. Input generation,
model downloads, result readback for validation, and report serialization are
not timed.

## Correctness gates

PyTorch is the numerical reference. Both runners use deterministic inputs and
canonical name-seeded parameters, with explicit transposition where their
physical weight layouts differ.

Forward validity requires:

- the same output tensor shape;
- 256 deterministic samples distributed across the flattened output;
- relative output L2 error below 1%;
- symmetric relative scalar-loss error below 1%.

Training validity additionally requires:

- identical canonical trainable-parameter key sets;
- relative total-gradient-norm error below 5%;
- relative L2 error across the vector of canonical per-parameter gradient
  norms below 5%.

Exact full-output hashes are retained as a reproducibility fingerprint but
bitwise equality is not required across GPU APIs. A diagnostic `CLOSE` result
does not enter a publication table as valid.

## Revision and environment controls

Final artifacts must contain:

- Inferena commit;
- Meganeura commit;
- a clean/dirty flag;
- Rust, Python, PyTorch, CUDA/cuDNN, Vulkan/Metal, and driver versions;
- GPU marketing name, vendor/device identifiers, and memory;
- OS/kernel;
- precision and optimizer environment switches;
- workload and protocol metadata.

Naga IR validation and Vulkan-layer validation are separate checks. With the
currently pinned Naga revision, current Vulkan Validation Layers report
`VUID-StandaloneSpirv-None-10684` for explicit layout decorations in emitted
SPIR-V. This is the open upstream
[wgpu issue #7696](https://github.com/gfx-rs/wgpu/issues/7696); wgpu suppresses
that exact VUID while Blade exposes it. Execution and numerical validation
pass, but the frozen artifact must record the layer version and warning. It
may suppress only that exact known VUID, or consume a separately verified
upstream fix; other validation diagnostics remain failures.

During sibling-repository development, run:

```sh
cd ../inferena
VIRTUAL_ENV="$PWD/.venv" \
INFERENA_MEGANEURA_PATH=../meganeura \
./run.sh \
  --frameworks pytorch,meganeura \
  --protocol paper-v1 \
  --precision strict-f32 \
  --warmup-runs 5 \
  --measurement-runs 20 \
  --results-dir results/paper-v1-strict
```

Omitting `INFERENA_MEGANEURA_PATH` intentionally benchmarks Inferena's pinned
Meganeura dependency, not an uncommitted sibling working tree.

## Pre-freeze same-tree RTX 5080 iteration

This 2026-07-23 development sweep uses the current Meganeura and Inferena
working trees, five warmups, 20 retained samples, and the repaired workload
and precision protocol. Both repositories are dirty, so these are
same-revision engineering results rather than frozen publication cells. Raw
artifacts are in:

- `../inferena/results/pre-freeze-2026-07-23-strict/`;
- `../inferena/results/pre-freeze-2026-07-23-accelerated/`.

Every workload passes both the forward and training gates in both arithmetic
classes.

### Strict f32

| Workload | Engine | Compile (s) | Inference median (ms) | Latency (ms) | Training (ms) |
|---|---|---:|---:|---:|---:|
| SmolLM2-135M | PyTorch | 33.771 | 5.048 | 2.091 | 13.135 |
|  | Meganeura | 0.940 | 11.022 | 2.368 | 41.143 |
| SmolVLA | PyTorch | 16.266 | 1.790 | 0.966 | 4.425 |
|  | Meganeura | 0.780 | 3.560 | 1.516 | 10.920 |
| Conditioned diffusion U-Net | PyTorch | 20.489 | 1.941 | 1.947 | 4.493 |
|  | Meganeura | 0.490 | 2.152 | 2.190 | 8.569 |
| ResNet-50 | PyTorch | 9.231 | 4.951 | 3.392 | 11.109 |
|  | Meganeura | 0.671 | 5.271 | 3.657 | 31.935 |
| Whisper-tiny encoder | PyTorch | 5.416 | 2.571 | 2.587 | 8.615 |
|  | Meganeura | 0.555 | 7.058 | 7.043 | 23.200 |

### Accelerated f32

| Workload | Engine | Compile (s) | Inference median (ms) | Latency (ms) | Training (ms) |
|---|---|---:|---:|---:|---:|
| SmolLM2-135M | PyTorch | 33.813 | 2.789 | 2.095 | 8.832 |
|  | Meganeura | 1.765 | 8.686 | 2.325 | 38.491 |
| SmolVLA | PyTorch | 15.522 | 1.259 | 0.967 | 4.391 |
|  | Meganeura | 1.615 | 3.617 | 1.476 | 10.991 |
| Conditioned diffusion U-Net | PyTorch | 21.219 | 1.166 | 1.187 | 4.471 |
|  | Meganeura | 1.319 | 1.859 | 1.781 | 8.104 |
| ResNet-50 | PyTorch | 9.247 | 1.787 | 1.180 | 5.289 |
|  | Meganeura | 1.529 | 5.312 | 3.535 | 32.011 |
| Whisper-tiny encoder | PyTorch | 5.407 | 1.847 | 1.845 | 6.918 |
|  | Meganeura | 1.129 | 3.207 | 3.200 | 20.292 |

The accelerated Whisper result is the direct confirmation of the precision
repair: forward relative L2 error is 0.0175%, total-gradient error is 0.0228%,
and the per-parameter gradient-norm-vector error is 0.0364%. The conditioned
diffusion result is also fully valid; its corresponding errors are 0.471%,
1.30%, and 1.73%.

## Archived RTX 5080 strict-f32 iteration

This iteration predates the conditioned diffusion workload, the
full-precision autodiff policy, and the final same-revision rerun. It is
retained only to document development history and must not be copied into a
publication table.

Environment: NVIDIA GeForce RTX 5080 (16,303 MiB), driver 595.71.05, Vulkan
1.4.329 device API, PyTorch 2.13.0+cu130, CUDA 13.0, cuDNN 9.20.0 (reported
version 92000), Python 3.14.4, Rust 1.97.1. Meganeura revision is
`a7ced10-dirty`.

All five workloads pass the repaired forward and training gates.

| Workload | Engine | Compile (s) | Inference median (ms) | Inference IQR (ms) | Latency (ms) | Training (ms) |
|---|---|---:|---:|---:|---:|---:|
| SmolLM2-135M | PyTorch | 34.323 | 5.045 | 5.042–5.047 | 2.130 | 13.174 |
|  | Meganeura | 2.045 | 11.082 | 10.999–11.110 | 2.341 | 41.135 |
| SmolVLA | PyTorch | 16.201 | 1.793 | 1.792–1.797 | 0.961 | 4.503 |
|  | Meganeura | 1.394 | 3.390 | 3.291–3.614 | 1.513 | 10.739 |
| Former convolution-only U-Net | PyTorch | 10.753 | 1.349 | 1.348–1.350 | 1.276 | 1.962 |
|  | Meganeura | 0.516 | 0.847 | 0.843–0.848 | 0.742 | 4.523 |
| ResNet-50 | PyTorch | 9.227 | 4.954 | 4.950–4.957 | 3.398 | 11.116 |
|  | Meganeura | 1.114 | 5.437 | 5.200–5.558 | 3.671 | 31.978 |
| Whisper-tiny encoder | PyTorch | 5.432 | 2.575 | 2.570–2.580 | 2.583 | 8.614 |
|  | Meganeura | 0.736 | 6.730 | 6.681–6.758 | 6.744 | 23.311 |

These are iteration measurements. Documentation and harness changes occurred
afterward, so the final paper must rerun from clean pinned commits.

## Archived RTX 5080 accelerated-f32 iteration

The same working tree was run with PyTorch's documented TF32-capable
`high` policy and Meganeura's f16-input/f32-accumulate cooperative paths.

| Workload | Engine | Compile (s) | Inference median (ms) | Inference IQR (ms) | Latency (ms) | Training (ms) | Validation |
|---|---|---:|---:|---:|---:|---:|---|
| SmolLM2-135M | PyTorch | 33.990 | 2.809 | 2.807–2.810 | 2.116 | 8.885 | reference |
|  | Meganeura | 2.860 | 8.692 | 8.528–8.707 | 2.363 | 31.113 | forward + training pass |
| SmolVLA | PyTorch | 15.546 | 1.264 | 1.262–1.266 | 0.969 | 4.409 | reference |
|  | Meganeura | 2.212 | 3.615 | 3.468–3.931 | 1.493 | 10.187 | forward + training pass |
| Former convolution-only U-Net | PyTorch | 10.767 | 0.667 | 0.666–0.670 | 0.657 | 1.297 | reference |
|  | Meganeura | 1.359 | 0.838 | 0.836–0.839 | 0.732 | 4.650 | forward + training pass |
| ResNet-50 | PyTorch | 9.236 | 1.786 | 1.783–1.787 | 1.179 | 5.267 | reference |
|  | Meganeura | 1.966 | 5.324 | 5.248–5.527 | 3.691 | 31.757 | forward + training pass |
| Whisper-tiny encoder | PyTorch | 5.403 | 1.850 | 1.846–1.854 | 1.857 | 6.909 | reference |
|  | Meganeura | 1.311 | 3.143 | 3.094–3.228 | 3.179 | **invalid: 15.029** | forward pass; training fails |

In this archived run, Whisper's accelerated forward relative error was about
0.017%, but its total gradient norm differed by 22.2% and its per-parameter
gradient-norm vector by 42.5%. That result led to the full-precision autodiff
policy. A later 2-warmup/5-sample development check reduced those errors to
0.023% and 0.030%, respectively; the final 5-warmup/20-sample sweep still has
to establish the publication result.

## Failure policy

- A failed runner remains in raw artifacts with its reason.
- A forward failure invalidates compile/inference/latency cells for the
  comparison.
- A backward-only failure invalidates only training.
- Unsupported is distinct from incorrect and from process failure.
- No value is silently replaced by a historical value.
- Outliers remain in raw samples; any exclusion requires a recorded,
  predeclared machine-state rule.

## Final artifact layout

Suggested immutable structure:

```text
artifact/
  manifest.json
  environment/
  strict-f32/<device>/<workload>/<engine>.json
  accelerated-f32/<device>/<workload>/<engine>.json
  ablations/<device>/...
  profiles/<device>/...
  plots/
  analysis/
```

The analysis script should regenerate every paper table and plot from the raw
JSON without hand-entered performance values.
