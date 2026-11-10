# GTC 2027 poster submission

Deadline: Tuesday, November 10, 2026, 5 p.m. PT. Upload `poster.pdf`
(44 × 35 in landscape, built by `build.sh`); NVIDIA adds the 2 in header band.

## Title (15 words maximum)

Training and Inference on NVIDIA GPUs Through Vulkan: Where the Gap to CUDA Lies

## Topics (up to two, from the portal's list)

Pick the closest entries to these:

1. Performance optimization / GPU programming
2. AI frameworks, training and inference

## Poster description (100 words maximum)

Can machine learning on NVIDIA GPUs run without CUDA? Meganeura, a 19 MiB Rust
engine, compiles training and inference graphs to Vulkan compute shaders. It
passes every numerical check on an RTX 3050 under Windows and on an RTX 5070
and an H100 under Linux. Against compiled PyTorch 2.13 with CUDA graphs, it matches
strict-f32 inference on the RTX 5070 (1.02×) and trails training by 1.5–1.8×,
or 2.8–5.2× with BF16. Single-operation probes locate the losses in f32 GEMM
kernels and host submission.

## Extended abstract and results

**Problem.** On NVIDIA GPUs, machine learning means CUDA. The same GPUs also
run Vulkan, which exposes compute shaders, subgroup operations and tensor cores
through `VK_KHR_cooperative_matrix`, and which ships with the driver on nearly
every desktop and mobile GPU. We ask how close a compact engine gets to CUDA
PyTorch through that path, and where the remaining time goes.

**Approach.** Meganeura is a compiler and runtime in about 54k lines of Rust
(19.4 MiB executable; the PyTorch CUDA dependency closure is 4.42 GiB). It
differentiates tensor graphs, keeps graph rewrites and matrix schedules in an
e-graph, and lowers the selected programs to 113 WGSL compute shaders, which
naga translates to SPIR-V for Vulkan. Programs are chosen by timing candidates
on the device, and a candidate counts only if it passes full-output and
full-gradient checks. The same Linux binary runs on AMD and Intel GPUs through
Vulkan, and a macOS build of the same source targets Apple GPUs through Metal;
the poster covers NVIDIA.

**Three NVIDIA GPUs.** The core suite is five workloads under two precisions
and three phases, 30 cells per GPU. Every native cell passes the numerical
gates: 30 on the RTX 3050 (Ampere, Windows 11, driver 610.88), 36 on the
RTX 5070 (Blackwell, Linux, driver 595.91.07, adding SmolLM2-360M), and 42 on
the H100 80GB HBM3 (Hopper, Linux, 595.91.07, adding SmolLM2-360M and 1.7B;
1.7B does not fit the RTX 5070's 12 GB). The RTX 3050 cells are checked
against CPU PyTorch f32 and have no GPU timing comparison.

**Benchmarking methodology.** Timing uses the RTX 5070 and H100, with driver
595.91.07 serving both Vulkan and CUDA. The reference is PyTorch 2.13 with
`torch.compile` and CUDA graph replay. The five workloads are SmolLM2-135M
(public checkpoint), the SmolVLA action expert, a scaled 10.9M-parameter
Stable Diffusion 1.x U-Net, ResNet-50 with folded normalization, and the
Whisper-tiny encoder; all but SmolLM2 use synthetic weights. Each runs a batch
or 128-token prefill forward, a minimal-shape forward (for SmolLM2, one token
without a KV cache, which is not decode latency), and forward, loss and
backward without an optimizer update. Under strict f32, TF32 is
off. Under reduced precision, PyTorch runs autocast BF16 and Meganeura may
select F16/BF16 cooperative-matrix products, keeping parameters, loss and
accumulation in f32. Each engine passes its own gates before timing.
Meganeura checks full outputs and full single-step gradients against an f32
reference build, using token KL for the language model at reduced precision.
PyTorch checks its forward against eager f32 and its training against a
matched SGD trajectory. Across engines, sampled outputs, loss and gradient
norms must agree. Timing uses a 10 s warmup, 20 samples and three processes. We report Meganeura time
divided by PyTorch time, so values below 1 favor Meganeura. Seventy-two NVIDIA
ratios qualified: 57 in the core suite and 15 for the larger SmolLM2 models.

**Model results.** Strict-f32 geometric means are 1.02 (inference), 0.96
(minimal shape) and 1.49 (training) on RTX 5070, and 1.30, 1.06 and 1.81 on
H100. Meganeura is faster on RTX 5070 for ResNet-50 (0.74) and U-Net (0.80)
inference and for the one-token SmolLM2 forward (0.84). Under reduced precision, the
training ratios become 2.81 and 5.24. Over the same qualified models, BF16
speeds PyTorch up 2.1–3.0× and Meganeura's searched precision policy 1.1–1.4×.
These are whole-model times that include unconverted operations, partial
backward coverage and conversions, so they do not measure tensor-core kernels.
The
ratios compose exactly: RTX 5070 training goes from 1.49 to 1.49 × 2.27 / 1.20
= 2.81. Whisper is the outlier, at 3.2× on RTX 5070 and 7.4× on H100 under
reduced precision; the search has not found a competitive long-sequence
attention. Larger models do not close the gap on H100: strict
training goes from 1.86 at SmolLM2-135M to 1.96 at 360M and 3.77 at 1.7B
(135.0 against 35.8 ms). On the RTX 5070, 360M strict ratios are 1.11
(prefill), 0.93 (one token) and 1.78 (training).

**Gap analysis.** A single-operation study times strict-f32 GEMMs,
convolutions and dispatches on RTX 5070, with RX 7900 XT and Arc B570 for
context; both engines run 30 serial copies. On the RTX 5070:

- Dense f32 GEMM is 1.3–1.7× cuBLAS. With TF32 off, cuBLAS also runs CUTLASS
  SIMT kernels on CUDA cores, so the arithmetic matches: 11.8 against 19.9
  TFLOP/s at 4096³. A 128×64 tile with an 8×4 register block cut this from 2.06×.
- On small long-K GEMMs, cuBLAS splits K. Adding split-K took 64×1024→256 from
  4.40× to 1.88×, adding one reduction dispatch per product.
- Winograd 3×3 convolutions take 0.34–0.64× of cuDNN's batch-1 time.
- A dependent dispatch with a barrier costs 0.97 µs, against 0.85 µs per op
  inside a CUDA graph. One host submit-and-wait takes 19.4 µs against 6.9 µs,
  so submitting work costs more than running it.
- These probes run with cooperative matrices off, so tensor-core kernel
  efficiency is not yet isolated. The search already times 16×16 F16/BF16
  subgroup tiles and larger workgroup tiles from `VK_NV_cooperative_matrix2`,
  but the path has no TF32 inputs and no control of Hopper warpgroup MMA, TMA
  or clusters. How much of the H100 gap these explain needs kernel-level profiles.

These ratios locate implementation losses. They do not partition model time.

**Driver finding.** On driver 595, a dynamically indexed nested function-local
array (`array<array<f32,4>,8>`) reads zero for outer indices at or above the
inner length. Our 8×4 GEMM tile first looked fast because it dropped half its
rows. A flat accumulator avoids it. A raw-Vulkan reproducer is reported.

**Significance.** Through Vulkan, an NVIDIA GPU already runs these
strict-f32 inference workloads at CUDA PyTorch's speed, without vendor
libraries, in a 19 MiB Linux binary that also drives other vendors' GPUs. The
measured gaps point to the next steps. Persistent megakernels, now in
development, run a region of the graph in one launch and schedule dependent
tiles on the GPU through device-side queues, to cut submissions and barriers
per step. Kernel discovery would let the search compose new kernels from tile,
staging and fusion choices, beyond the hand-written variants it times today.
Fused attention targets the long-sequence Whisper gap.

**Availability.** Source: https://github.com/kvark/meganeura. Full protocol,
archives and offline replay are in the companion P3HPC workshop paper at SC26,
"Meganeura: Vulkan and Metal as a Performance-Portability Layer for GPU Training
and Inference". Model results come from the October 31 – November 1, 2026
cohort (Meganeura `567028b8`). Single operations were measured on November 8,
2026 (Meganeura `d32a495`).
