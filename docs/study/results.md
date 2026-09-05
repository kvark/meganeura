# Results: the evidence and its limits

This chapter is a reading guide to the frozen experiment, not a new benchmark.
The September audit replays files already in [paper/results](../../paper/results/).
All ratios below are Meganeura time / PyTorch time; smaller is better.

## Exactly what was frozen

| Item | Identity or scope |
|---|---|
| Meganeura matrix revision | `7561a64ec5a7e4bcdcd2c719aaaffe5912ed5e85` |
| Inferena harness revision | `7ca9c5c7b2cd614343a3de3dcc86999ced66e8c0` |
| Public snapshot tag | `paper-arxiv-1` in both repositories |
| Profile revision | Meganeura `b1405a3a52fabf9858aca5cbd80e246811cb6a58`; sidecars identify their own configuration |
| Matrix | 5 machines × 5 workloads × 2 arithmetic modes = 50 paired cells |
| Files | 100 individual engine JSONs, 50 joined JSONs, 10 SVGs, 5 profile JSONs = 165 |
| Sampling | One process-level run per cell, 20 synchronized timing samples per series; 5 warmups, except Intel's 4 |
| Timed series | Full forward; minimal-shape forward; forward + loss + backward without optimizer |
| Compilation | Construction/rewrites/autodiff/pipelines for all three sessions, not Rust compilation or a single shader |

Five workloads are SmolLM2-135M, the SmolVLA action expert, a scaled
text/timestep-conditioned diffusion U-Net, ResNet-50, and the Whisper-tiny
encoder. This is not the full SmolVLA perception system, a checkpoint-
compatible SD 1.5, or full Whisper speech recognition. ResNet normalization
is folded, so this is not standard batch-norm training. The LLM minimal shape
is one stateless token, without KV cache; do not call its latency a serving
or sustained decoding result.

| Machine | Meganeura path | Frozen PyTorch reference |
|---|---|---|
| RTX 5070 | Vulkan | CUDA, torch 2.13.0+cu130, compiled |
| RX 7900 XT | Vulkan/RADV | ROCm 7.1, torch 2.10.0, compiled |
| Radeon 780M | Vulkan/RADV | ROCm 7.14, torch 2.12.0, compiled |
| Intel RPL-U iGPU | Vulkan/ANV | CPU, torch 2.11.0+xpu with no usable XPU |
| Apple M3 | Metal | MPS, torch 2.11.0, eager |

The frozen Linux runner calls `torch.compile(model)` in its default mode.
An older/manual CUDA-graph branch elsewhere in Inferena is not the paper
protocol. No manual CUDA capture was added. Default compiled execution is a
useful baseline, but not every automatic PyTorch optimization configuration.
Future comparison should include documented `reduce-overhead` and
`max-autotune` modes where applicable, with compile time and memory charged
separately. These are ordinary compiler options, not hand-specialized models.
[PyTorch compile documentation](https://docs.pytorch.org/docs/2.14/generated/torch.compile.html)

## What the validation actually checks

Forward validity requires matching full output shape, relative L2 error below
1% over retained output samples, and symmetric relative scalar-loss error
below 1%. Each output record retains 256 evenly spaced flattened values
(or all values if the tensor is smaller), not the whole output tensor. The
PyTorch sample serialization rounds to six decimal places. A full-output hash
is a reproducibility diagnostic, not a tolerant numerical comparison.

Backward validity additionally requires matching canonical parameter names,
total gradient norm within 5%, and relative L2 error below 5% over the vector
of per-parameter gradient norms. This vector has one scalar per parameter
tensor; it is not a vector of all gradient elements. Gradients `g` and `-g`
have identical norms. A sparse output error can also miss all 256 sampled
positions. Existing numerical gates are useful screens, not a convergence
proof or complete elementwise validation.

Relative vector error is `||candidate - reference||₂ / ||reference||₂`, with
the harness's zero-denominator protection. Scalar loss uses a symmetric
normalization. Replaying the stored comparisons verifies the recorded
diagnostics, not unretained tensor elements or the original GPU execution.
The next protocol should retain seeded gradient projections/samples, use
full outputs where affordable, and add independent finite differences on
small adversarial graphs plus short optimizer-backed trajectories.

| Comparable-cell worst error | Strict | Accelerated | Gate |
|---|---:|---:|---:|
| Sampled output L2 | 0.00432% | 0.588% | 1% |
| Total gradient norm | 0.0811% | 0.177% | 5% |
| L2 of parameter-norm vector | 0.639% | 0.639% | 5% |

All 50 cells pass the forward gate. Forty-eight pass the local backward gate.
The two remaining cells are the two arithmetic modes of one
780M–Whisper backward disagreement, handled below. Do not say that only 48
forward results are valid.

## Why the oracle-disputed pair is excluded

The local 780M pair disagrees by about 8.9% in total norm and 17.8% in the
strict per-parameter-norm vector. A cross-backend audit finds the other four
PyTorch backends tightly clustered, and all five Meganeura backends close to
that cluster. The 780M PyTorch norm vector is 16.3% from the other PyTorch
backends; Meganeura is about 0.035% from the good strict reference cluster.

That supports treating the designated local reference as disputed. It does
not identify a root cause inside PyTorch, ROCm or the driver, nor prove the
remaining implementations' elementwise gradients correct. The analysis keeps
both raw records and excludes the pair from training ratios and aggregation
for both engines. Valid forward measurements remain. This post-hoc rule is
explicitly disclosed; it was not preregistered.

Sensitivity matters: the former one-sided assignment would reduce
Meganeura's mean strict training portability score from 0.62 to 0.52 while
leaving PyTorch's rounded 0.93 unchanged. Symmetric exclusion is defensible,
but should not be concealed as an inconsequential analysis choice. An
independent small reference and full gradient checks are the route to resolving
the dispute, not a stronger assertion about which project is at fault.

## Read the ratios using the correct population

The GPU-reference population excludes Intel, where the reference is CPU.

| Mode / workload phase | Valid GPU pairs | Meganeura wins | Median ratio | Worst ratio |
|---|---:|---:|---:|---:|
| Strict full inference | 20 | 8 | 1.21 | 2.65 |
| Strict minimal latency | 20 | 12 | 0.93 | 2.64 |
| Strict F+L+B | 19 | 5 | 1.78 | 2.87 |
| Accelerated full inference | 20 | 8 | 1.40 | 2.87 |
| Accelerated minimal latency | 20 | 11 | 0.89 | 2.91 |
| Accelerated F+L+B | 19 | 5 | 1.69 | 4.63 |

Including the complete five-machine set, strict inference has 13/25 wins
and median 0.98; minimal latency has 16/25 wins and median 0.89; training has
8/24 wins and median 1.50. Therefore "median training 1.8×" and "mean training
portability 0.62" use different populations and aggregation procedures. State
their labels rather than inviting the reader to infer a shared denominator.

The strongest device-level story is discrete AMD: four of five strict full
inference ratios are below 1.11 and three training workloads win. The
SmolLM2 ratio rounds to 1.10 but slightly exceeds an exact 1.10 cutoff; use
"about 10%" or a 1.11 bound rather than asserting ≤1.10. ResNet remains a
large loss. On NVIDIA, strict inference wins two of five and minimal latency
three of five; training remains slower. On Apple, every training workload
loses, at 1.20–2.84× even against eager MPS. The earlier roadmap's universal
Apple/AMD/Intel lead does not survive this matched protocol.

The accelerated contract does not always help: AMD SmolVLA inference becomes
slower after allowing reduced-input cooperative work. Capability is not
profitability, and occupancy heuristics can still select a poor candidate.
Conversely, larger PyTorch TF32 gains can worsen a ratio even when Meganeura
itself improves. Always inspect the two absolute timings before explaining
a changed ratio.

## The performance-portability score

For a workload and fixed platform set `H`, let `t(a,p)` be implementation
`a`'s valid time and `b(p)` the best valid time of the two implementations.
Application efficiency is `e(a,p) = b(p) / t(a,p)`. The workload score is:

`P(a,H) = |H| / Σ[p in H] (1 / e(a,p))`

if the implementation is valid and supported on every member; otherwise it
is zero. This harmonic construction penalizes low efficiency. For example,
efficiencies `[1, 1, 0.5]` produce `3/(1+1+2) = 0.75`, not the arithmetic
mean 0.833. This follows the platform-set discipline of
[Pennycook, Sewall and Lee](https://arxiv.org/abs/1611.07409).

Here a platform means the complete machine and installed stack, so the
labeled Intel CPU path remains supported. Whisper training uses four machines
because the local oracle pair is disputed. The final reported mean is an
arithmetic mean of five workload-level harmonic scores, not one harmonic
mean over all cells and not a median speed ratio.

| Mean workload score | Meganeura strict | PyTorch strict | Meganeura accelerated | PyTorch accelerated |
|---|---:|---:|---:|---:|
| Inference | 0.75 | 0.81 | 0.71 | 0.81 |
| Minimal latency | 0.83 | 0.73 | 0.82 | 0.74 |
| F+L+B | 0.62 | 0.93 | 0.58 | 0.94 |

These are efficiencies relative to the better of two observations, not
fractions of peak FLOPS, a roofline limit, or the world's best engine. A new
stronger implementation changes the denominator. An invalid forward result
must zero the corresponding inference score; the audit corrected the table
generator to enforce that policy even though every frozen forward cell passes.

## Profiles: useful localization, not a promised speedup

The NVIDIA accelerated ResNet training profile's normal control is 36.81 ms,
close to the frozen 36.77 ms. Backward spatial convolution takes 73% of
timestamped GPU time and forward convolution another 16.3%. Three derivative
shaders contribute 28.5 ms; the small weight-gradient variant contributes
12.6 ms over 12 dispatches. This prioritizes convolution derivatives and
their layout/tiling before unrelated compiler complexity.

The Apple Whisper profile is explicitly pre-optimization: roughly 330 ms,
58.3% in attention backward, with fused dK/dV at 141.9 ms. The frozen cell is
about 274 ms after the profile-guided pass, a 1.20× improvement. Do not present
that older breakdown as a profile of the final 274 ms execution.

Profiles serialize/instrument dispatches differently from normal grouped
execution, and timestamp attribution can include synchronization. They locate
work but are not additive predictions of step wall time. Hypothetically, if
73% of actual step time were accelerated by 2×, Amdahl's law gives
`1 / (0.27 + 0.73/2) = 1.57×` total; infinitely fast treatment would cap the
gain at 3.70×. Those are conditional calculations, not measured headroom.

## Compilation, footprint and deployment

Frozen Meganeura compilation ranges from 0.08–1.52 s in strict mode and up to
2.36 s accelerated. Compiled PyTorch runs span roughly 6.5–96 s. Eager Apple
and Intel do not have a comparable compiler cost. No claim of universally
subsecond Meganeura compilation is supported.

The frozen stripped runner is 12.8 MiB versus 1.75 GiB for PyTorch+Triton
packages and 4.66 GiB for the CUDA runtime dependency closure. Weights, Python
itself, OS and driver are excluded from the compared footprint bases as stated
in the paper. This is a deployment-closure comparison, not equal feature
coverage or a developer-productivity ratio. The frozen 34.5 K Rust and 6.1 K
WGSL line counts also do not describe today's refactored source.

DinoVision provides a separate train-to-deploy application: a host-trained
2.01 M-parameter decoder, a Quest 3S inference graph shared with a renderer,
and separately versioned evidence. It is outside the 50-cell matrix, has no
matched PyTorch Android result and does not demonstrate Quest training.
The [frozen section](../../paper/dinovision-section.tex) is imported from that
project and must be updated at its source, not rewritten in this repository.

## Replay and claim ledger

From the repository root, these commands require no GPU:

```sh
python3 paper/p3hpc/artifact/verify.py --repository --show-facts
python3 -m unittest discover -s paper/p3hpc/artifact -p 'test_*.py'
```

The verifier checks inventory, identity/revisions, protocols, arithmetic
settings, retained samples, medians, raw-versus-summary equality, stored
sampled-output/gradient-norm diagnostics, validity decisions, and byte-identical
regeneration of the six table fragments. Packaged mode additionally checks
the manifest and expected fact block. It does not execute GPUs or prove that
every sentence in a TeX file agrees with the facts.

| Load-bearing claim | Evidence to open | Review caveat |
|---|---|---|
| 12/20 minimal wins; 1.78× training | Verifier's `gpuref strict` fact rows | Different valid pair counts: 20 and 19. |
| 0.62 versus 0.93 training score | `PP means` facts and `mktables.py` | Five workload scores, Whisper over four machines. |
| 48/50 backward-valid; 50/50 forward-valid | Raw validation records and replayed gates | Sampled outputs, norm-only gradients. |
| 780M oracle dispute | Cross-backend facts and oracle table | Post-hoc, symmetric, root cause unresolved. |
| Conv/attention explain priorities | `results/*/profiles/` | Sidecar revisions and instrumentation basis. |
| 12.8 MiB deployment closure | Paper footprint methodology/table | Frozen build, excluded dependencies, unequal breadth. |
| Native training-to-renderer path | DinoVision source/evidence cited in paper | Separate revision and experiment, not matrix speedup. |

Before final submission, manually walk this ledger through abstract,
contributions, tables, captions, discussion and conclusion. The matrix contains
one run per cell; within-run samples do not establish population-level
confidence, thermal robustness or driver-wide generality.
