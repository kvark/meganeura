# meganeura

[![CI](https://github.com/kvark/meganeura/actions/workflows/ci.yml/badge.svg)](https://github.com/kvark/meganeura/actions/workflows/ci.yml)
[![Docs](https://docs.rs/meganeura/badge.svg)](https://docs.rs/meganeura)
[![Crates.io](https://img.shields.io/crates/v/meganeura.svg?label=meganeura)](https://crates.io/crates/meganeura)

**Neural network training and inference in Rust, on any GPU.** Vulkan, Metal, DX12 — Linux, Windows, macOS, iOS, Android. No CUDA, no Python in the loop, routinely faster than PyTorch on AMD and Apple GPUs.

[![logo](https://github.com/kvark/meganeura/raw/main/etc/logo.png)](/kvark/meganeura/blob/main/etc/logo.png)

> **Status:** actively developed, APIs still in motion, but already running real workloads (SmolLM2, SmolVLA, ResNet-50, Whisper-tiny, Stable Diffusion U-Net). Issues and pull requests welcome.

Define a graph, call `build_session`, train. Meganeura handles autodiff, kernel fusion via [e-graph](https://egraphs-good.github.io/) equality saturation, [Naga IR](https://docs.rs/naga/latest/naga/) shader codegen, and GPU dispatch automatically.

```rust
use meganeura::{Graph, Trainer, TrainConfig, build_session};

let mut g = Graph::new();
let x = g.input("x", &[32, 784]);
let labels = g.input("labels", &[32, 10]);

let w1 = g.parameter("w1", &[784, 128]);
let h = g.relu(g.matmul(x, w1));
let w2 = g.parameter("w2", &[128, 10]);
let logits = g.matmul(h, w2);

let loss = g.cross_entropy_loss(logits, labels);
g.set_outputs(vec![loss]);

// autodiff + e-graph optimize + compile + GPU init
let session = build_session(&g);
let mut trainer = Trainer::new(session, TrainConfig::default());
trainer.train(&mut data, /* epochs = */ 10); // data loader: see examples/mnist.rs
```

A two-layer MLP, trained end to end on the GPU, in one screen.

## Why Meganeura

**Fast.** Inference against PyTorch on matching model configs, from the cross-framework benchmark at [Inferena](https://inferena.tech):

|Platform   |Workload              |Meganeura|PyTorch         |              |
|-----------|----------------------|---------|----------------|--------------|
|Apple M3   |Stable Diffusion U-Net|**10 ms**|534 ms (MPS)    |**53× faster**|
|Apple M3   |SmolVLA               |37 ms    |172 ms (MPS)    |4.6× faster   |
|Apple M3   |SmolLM2-135M          |62 ms    |247 ms (MPS)    |4× faster     |
|Radeon 890M|SmolLM2-135M          |34 ms    |67 ms (ROCm)    |2× faster     |
|Radeon 890M|SmolVLA               |15 ms    |25 ms (ROCm)    |1.7× faster   |
|RTX 5080   |SmolVLA               |3 ms     |3 ms (CUDA 13.0)|parity        |
|RTX 5080   |SmolLM2-135M          |7 ms     |4 ms (CUDA 13.0)|within 1.75×  |

Training shows the same shape on non-NVIDIA: on Radeon 890M, Meganeura trains SmolLM2-135M in 87 ms/step vs PyTorch ROCm’s 123 ms, and SmolVLA in 35 ms vs 41 ms. On NVIDIA, PyTorch CUDA still leads on training workloads.

Meganeura also runs on GPUs PyTorch doesn’t target, including Radeon 780M and Intel integrated graphics (RPL-U).

The wedge isn’t every workload: on ResNet-50 and Whisper-tiny, Meganeura currently trails. Full cross-framework tables — including the losses — at [inferena.tech](https://inferena.tech). Reproduce locally with `./run.sh -m <Model>`.

**Portable.** GPU access via [blade-graphics](https://github.com/kvark/blade/tree/main/blade-graphics) — one backend, five platforms. Works on anything with Vulkan (including Mesa’s [Lavapipe](https://www.phoronix.com/news/Lavapipe-CPU-Vulkan-Windows) for headless CI), Metal on Apple silicon, and DX12 on Windows. No CUDA, no ROCm, no vendor lock-in at any layer of the stack.

**Lean.** A handful of [kernel archetypes](https://github.com/kvark/meganeura/blob/main/docs/kernel-archetypes.md) — pointwise, reduction, matmul, attention — compose into specialized GPU shaders at compile time. An e-graph equality-saturation pass discovers fusions (e.g. `x * sigmoid(x)` → Silu, `Silu(gate) * up` → SwiGLU) with a cost-model-driven extractor, rather than relying on hand-written fused kernels for every pattern. The codebase is small enough to read end to end.

## How it compares

|                                                 |GPU backends                        |Training      |Approach                                |
|-------------------------------------------------|------------------------------------|--------------|----------------------------------------|
|**Meganeura**                                    |blade-graphics (Vulkan, Metal, DX12)|yes           |graph IR + e-graph fusion + Naga codegen|
|[Candle](https://github.com/huggingface/candle)  |CUDA, Metal, CPU                    |limited       |eager tensors, hand-written kernels     |
|[Burn](https://github.com/tracel-ai/burn)        |CUDA, wgpu, NDArray, LibTorch       |yes           |modular multi-backend                   |
|[tch-rs](https://github.com/LaurentMazare/tch-rs)|CUDA, CPU (via libtorch)            |yes           |PyTorch FFI bindings                    |
|[Luminal](https://github.com/luminal-ai/luminal) |CUDA, Metal                         |inference-only|e-graph IR                              |

Meganeura’s wedge: **training, on non-NVIDIA hardware, without writing the kernels by hand.**

## Install

```
cargo add meganeura
```

Worked examples live in [`examples/`](https://github.com/kvark/meganeura/tree/main/examples):

- [`mnist.rs`](https://github.com/kvark/meganeura/blob/main/examples/mnist.rs) — MNIST training end to end.
- [`smollm2.rs`](https://github.com/kvark/meganeura/blob/main/examples/smollm2.rs) — LLM inference with HuggingFace weights.

Pretrained models can be loaded from ONNX or NNEF via `meganeura::load_onnx(...)` / `meganeura::load_nnef(...)`. Both lower through Meganeura’s IR, so e-graph fusions apply to imported graphs as well as hand-built ones.

## System requirements

Meganeura runs best where cooperative matrix operations are hardware-accelerated:

- **Vulkan** — [`VK_KHR_cooperative_matrix`](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_KHR_cooperative_matrix.html) (NVIDIA Volta+, AMD RDNA3+, Intel Arc).
- **Metal** — simdgroup matrix (Apple M1+).

Falls back to scalar matmul on older hardware. Headless Lavapipe works for CI.

## Profiling

```
MEGANEURA_TRACE=trace.pftrace cargo run --example mnist
```

Open the trace in [Perfetto](https://ui.perfetto.dev):

[![perfetto trace](https://github.com/kvark/meganeura/raw/main/etc/example-trace.png)](/kvark/meganeura/blob/main/etc/example-trace.png)

## Contributing

Early project, small API surface, small community — a good time to show up. Open an issue before starting anything sizeable so we can align on direction.