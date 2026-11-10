# GTC 2027 poster submission

Deadline: Tuesday, November 10, 2026, 5 p.m. PT. Upload `poster.pdf`, built by
`build.sh`.

## Poster file guidelines

From "Poster Submission Guidelines for NVIDIA GTC 2027"
(https://register.nvidia.com/flow/nvidia/gtc27/cfsresources/page/posters):

- The document is 43 in wide × 24 in high with a 1 in white margin, so the
  design area is 41 × 22 in.
- No NVIDIA branding or logos.
- High-resolution (300 dpi), print-ready PDF at full size.
- Structure: problem statement or goal, application of NVIDIA technology,
  benchmarking methods, results and outcomes. Cite relevant work.
- No product promotion, marketing language or confidential material.

`poster.pdf` is one 43 × 24 in page with vector figures and embedded fonts, and
all of its ink stays inside the 41 × 22 in design area. Sections 1–6 follow
that structure and a reference list cites prior work. Accents are teal; the
poster carries no NVIDIA logo or brand color.

Review criteria on the call page: proven technical outcomes with practical
impact; significant technical innovation or improvement enabled by accelerated
computing; and the research framework above.

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
and an H100 under Linux. Against compiled PyTorch 2.13 with CUDA graphs on all
three, strict-f32 inference takes 1.02–1.30× as long and training 1.5–1.8×, or
2.4–5.2× with BF16. Single-operation probes locate the losses in f32 GEMM
kernels and host submission.

## Extended abstract and results

**Problem and goal.** On NVIDIA GPUs, machine learning means CUDA. The same
GPUs also run Vulkan, which exposes compute shaders, subgroup operations and
tensor cores through `VK_KHR_cooperative_matrix`, and which ships with the
driver on Windows and Linux. The goal is to measure how close a compact engine
gets to CUDA PyTorch for training and inference through that path alone, and
to locate where the remaining time goes.

**Approach.** Meganeura is a compiler and runtime in about 54k lines of Rust
(19.4 MiB executable; the PyTorch CUDA dependency closure is 4.42 GiB). It
differentiates tensor graphs, keeps graph rewrites and matrix schedules in an
e-graph, and lowers the selected programs to 113 WGSL compute shaders, which
naga translates to SPIR-V for Vulkan. Programs are chosen by timing candidates
on the device, and a candidate counts only if it passes full-output and
full-gradient checks. The same Linux binary runs on AMD and Intel GPUs through
Vulkan, and a macOS build of the same source targets Apple GPUs through Metal;
the poster covers NVIDIA.

**Application of NVIDIA technology.** The study runs on three NVIDIA GPU
generations: an RTX 3050 8 GB (Ampere, Windows 11, driver 610.88), an RTX 5070
12 GB (Blackwell, Linux, driver 595.91.07) and an H100 80GB HBM3 (Hopper,
Linux, 595.91.07). All work reaches the GPU through NVIDIA's Vulkan driver.
Tensor cores are used through `VK_KHR_cooperative_matrix`, as 16×16×16 F16 and
BF16 tiles with f32 accumulation, and through the larger workgroup-scope tiles
of `VK_NV_cooperative_matrix2`. The compiler times these kernels against its
other candidates on the device, and matrix kernels end up in 36–39 of each
GPU's 42 reduced-precision sessions. The reference is NVIDIA's CUDA stack:
PyTorch 2.13 on CUDA 13 with `torch.compile` and CUDA graph replay, and cuBLAS
and cuDNN in the single-operation probes.

**Benchmarking methodology.** On each GPU one driver serves both Vulkan and
CUDA: 610.88 under Windows and 595.91.07 under Linux. The reference is
PyTorch 2.13.0+cu130 with `torch.compile` and CUDA graph replay, and one
open-source harness runs both engines in fresh, alternating processes. The
five workloads are SmolLM2-135M (public checkpoint), the SmolVLA action
expert, a scaled 10.9M-parameter Stable Diffusion 1.x U-Net, ResNet-50 with
folded normalization, and the Whisper-tiny encoder; all but SmolLM2 use
synthetic weights, and inputs are deterministic. Each runs a batch or
128-token prefill forward, a minimal-shape forward (for SmolLM2, one token
without a KV cache, which is not decode latency), and forward, loss and
backward without an optimizer update. Under strict f32, TF32 is off. Under
reduced precision, PyTorch runs autocast BF16 and Meganeura may select
F16/BF16 cooperative-matrix products, keeping parameters, loss and
accumulation in f32. Each engine passes its own gates before timing.
Meganeura checks full outputs and full single-step gradients against an f32
reference build, using token KL for the language model at reduced precision.
PyTorch checks its forward against eager f32 and its training against a
matched SGD trajectory. Across engines, sampled outputs, loss and gradient
norms must agree. Timing uses a 10 s warmup, 20 samples and three processes.
We report Meganeura time divided by PyTorch time, so values below 1 favor
Meganeura.

**Results and outcomes: models.** The core suite is five workloads under two
precisions and three phases, 30 cells per GPU. Every native cell passes the
numerical gates: 30 on the RTX 3050, 36 on the RTX 5070 (adding SmolLM2-360M)
and 42 on the H100 (adding SmolLM2-360M and 1.7B; 1.7B does not fit the
RTX 5070's 12 GB). All three GPUs are timed against CUDA PyTorch, and 102
ratios qualified: 87 in the core suite and 15 for the larger SmolLM2 models.
Strict-f32 geometric means are 1.16 (inference), 0.97 (minimal shape) and 1.53
(training) on the RTX 3050, 1.02, 0.96 and 1.49 on the RTX 5070, and 1.30,
1.06 and 1.81 on the H100. The Ampere card under Windows lands close to the
Blackwell card under Linux. Meganeura is faster on RTX 5070 for ResNet-50
(0.74) and U-Net (0.80) inference and for the one-token SmolLM2 forward
(0.84), and on RTX 3050 for U-Net (0.95) and ResNet-50 (0.98) inference. Under
reduced precision, the training ratios become 2.42, 2.81 and 5.24. Over the
same qualified models, BF16 speeds PyTorch's inference and training up
1.9–3.0× and Meganeura's searched precision policy 1.1–1.4×. These are
whole-model times that include unconverted operations, partial backward
coverage and conversions, so they do not measure tensor-core kernels. The
ratios compose exactly: RTX 5070 training goes from 1.49 to 1.49 × 2.27 / 1.20
= 2.81. Whisper is the outlier, at 3.1× on RTX 3050, 3.2× on RTX 5070 and 7.4×
on H100 under reduced precision; the search has not found a competitive
long-sequence attention. Larger models do not close the gap on H100: strict
training goes from 1.86 at SmolLM2-135M to 1.96 at 360M and 3.77 at 1.7B
(135.0 against 35.8 ms). On the RTX 5070, 360M strict ratios are 1.11
(prefill), 0.93 (one token) and 1.78 (training).

**Results and outcomes: gap analysis.** A single-operation study times
strict-f32 GEMMs, convolutions and dispatches on RTX 5070, with RX 7900 XT and
Arc B570 for context; both engines run 30 serial copies. On the RTX 5070:

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
rows. A flat accumulator avoids it, and a raw-Vulkan reproducer isolates it.

**Practical impact.** Through Vulkan, NVIDIA GPUs from three generations run
these strict-f32 inference workloads at 1.02–1.30× CUDA PyTorch's time, under
Windows and Linux, with the GPU driver as the only dependency. The Linux
binary is 19 MiB and also drives other vendors' GPUs. The probes have already
turned into fixes: split-K and the larger GEMM tile above.

**Innovation.** One compiler keeps graph rewrites and matrix schedules in an
e-graph, admits a candidate only after full-output and full-gradient checks,
and picks programs by timing them on the device, so tensor-core kernels run
where they win.

**Next steps.** Persistent megakernels, now in development, run a region of
the graph in one launch and schedule dependent tiles on the GPU through
device-side queues, to cut submissions and barriers per step. Kernel discovery
would let the search compose new kernels from tile, staging and fusion
choices, beyond the hand-written variants it times today. Fused attention
targets the long-sequence Whisper gap.

**Availability.** Source: https://github.com/kvark/meganeura. Full protocol,
archives and offline replay are in the companion P3HPC workshop paper at SC26,
"Meganeura: Vulkan and Metal as a Performance-Portability Layer for GPU Training
and Inference". Model results come from the October 31 – November 1, 2026
cohort (Meganeura `567028b8`); the RTX 3050 was collected on November 10 at the
same revisions. Single operations were measured on November 8,
2026 (Meganeura `d32a495`).

**References** (as on the poster).

1. A. Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep
   Learning Library," NeurIPS 2019.
2. Khronos Group, Vulkan 1.4 Specification, with `VK_KHR_cooperative_matrix`
   and `VK_NV_cooperative_matrix2`.
3. Y. Zhang et al., "Better Together: Unifying Datalog and Equality
   Saturation," PLDI 2023.
4. SmolLM2, arXiv:2502.02737; SmolVLA, arXiv:2506.01844; Latent Diffusion,
   CVPR 2022; ResNet, CVPR 2016; Whisper, arXiv:2212.04356.
5. X. Cheng et al., "MPK: A Compiler and Runtime for Mega-Kernelizing Tensor
   Programs," arXiv:2512.22219.
6. D. Malyshau, "Meganeura: Vulkan and Metal as a Performance-Portability
   Layer for GPU Training and Inference," P3HPC at SC26.
