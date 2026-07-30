# meganeura

[![CI](https://github.com/kvark/meganeura/actions/workflows/ci.yml/badge.svg)](https://github.com/kvark/meganeura/actions/workflows/ci.yml)
[![Docs](https://docs.rs/meganeura/badge.svg)](https://docs.rs/meganeura)
[![Crates.io](https://img.shields.io/crates/v/meganeura.svg?label=meganeura)](https://crates.io/crates/meganeura)

**Portable neural-network training and inference in Rust.** Meganeura uses
Vulkan on Linux, Windows, and Android and Metal on Apple platforms. It does
not require CUDA, ROCm, or Python at runtime.

[![logo](https://github.com/kvark/meganeura/raw/main/etc/logo.png)](/kvark/meganeura/blob/main/etc/logo.png)

> **Status:** actively developed; APIs and benchmark methodology are still in
> motion. Current workloads include SmolLM2, a SmolVLA action expert,
> ResNet-50, the Whisper-tiny encoder, and a scaled, timestep- and
> text-conditioned latent-diffusion U-Net. Issues and pull requests are
> welcome.

Define a graph, call `build_session`, train. Meganeura handles autodiff,
graph rewrites, WGSL specialization, Naga parsing and validation, and GPU
dispatch automatically. The rewrite engine supports a fast deterministic
greedy mode and experimental equality-saturation modes.

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

// autodiff + graph rewrite + compile + GPU init
let session = build_session(&g);
let mut trainer = Trainer::new(session, TrainConfig::default());
trainer.train(&mut data, /* epochs = */ 10); // data loader: see examples/mnist.rs
```

A two-layer MLP, trained end to end on the GPU, in one screen.

## Why Meganeura

**Fast.** Meganeura is competitive with vendor-native ML stacks on selected
workloads while retaining one Vulkan/Metal implementation. Results vary
substantially by model, device, precision policy, and driver. The historical
[Inferena](https://inferena.tech) tables are useful exploratory data, but they
predate the audited paper protocol and should not be treated as
publication-grade comparisons. The new protocol reports raw samples, uses
matched workloads and full forward-plus-backward timing, and separates strict
f32 from reduced-input accelerated modes.

PyTorch CUDA currently leads several RTX 5080 workloads, especially training.
Meganeura's strongest result is therefore not universal speed superiority; it
is how much of that performance can be reached through a portable execution
stack that also runs on AMD, Intel, and Apple GPUs.

**Portable.** GPU access is provided by
[blade-graphics](https://github.com/kvark/blade/tree/main/blade-graphics):
Vulkan on Linux, Windows, and Android, and Metal on Apple platforms. Mesa's
Lavapipe provides a software Vulkan target for headless CI. The compute stack
does not require CUDA or ROCm, although performance and feature availability
still depend on each vendor's driver.

**Composable.** A small set of
[kernel archetypes](https://github.com/kvark/meganeura/blob/main/docs/kernel-archetypes.md)
— pointwise, reduction, matmul, convolution, and attention — compose into
specialized GPU shaders at compile time. The rewrite set recognizes
equivalent fused forms (for example, `x * sigmoid(x)` → SiLU and
`SiLU(gate) * up` → SwiGLU).
It can run either as a deterministic greedy pass or through equality
saturation with a traffic-aware extraction cost. Current ablations find the
same selected graph for the benchmark rewrite set, so equality saturation is
research infrastructure rather than a claimed source of runtime speedup.
Consolidating the remaining hand-written WGSL variants into parameterized
generators is active work.

## How it compares

|                                                 |GPU backends                        |Training      |Approach                                |
|-------------------------------------------------|------------------------------------|--------------|----------------------------------------|
|**Meganeura**                                    |blade-graphics (Vulkan, Metal)      |yes           |graph IR + rewrites + specialized WGSL         |
|[Candle](https://github.com/huggingface/candle)  |CUDA, Metal, CPU                    |limited       |eager tensors, hand-written kernels     |
|[Burn](https://github.com/tracel-ai/burn)        |CUDA, wgpu, NDArray, LibTorch       |yes           |modular multi-backend                   |
|[tch-rs](https://github.com/LaurentMazare/tch-rs)|CUDA, CPU (via libtorch)            |yes           |PyTorch FFI bindings                    |

Meganeura's wedge is a uniform graph, autodiff, compiler, and runtime stack for
both training and inference across desktop and edge-class Vulkan/Metal
devices.

## Install

```
cargo add meganeura
```

Worked examples live in [`examples/`](https://github.com/kvark/meganeura/tree/main/examples):

- [`mnist.rs`](https://github.com/kvark/meganeura/blob/main/examples/mnist.rs) — MNIST training end to end.
- [`train_deploy.rs`](https://github.com/kvark/meganeura/blob/main/examples/train_deploy.rs) — optimizer-backed training, checkpoint save, and reload into a fresh inference session.
- [`smollm2.rs`](https://github.com/kvark/meganeura/blob/main/examples/smollm2.rs) — LLM inference with HuggingFace weights.

Pretrained models can be loaded from ONNX or NNEF via `meganeura::load_onnx(...)` / `meganeura::load_nnef(...)`. Both lower through Meganeura’s IR, so the same graph rewrites apply to imported graphs and hand-built ones.

## System requirements

Meganeura runs best when the selected driver exposes hardware-accelerated
cooperative matrix operations:

- **Vulkan** —
  [`VK_KHR_cooperative_matrix`](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_KHR_cooperative_matrix.html).
- **Metal** — simdgroup matrix support.

Falls back to scalar matmul on older hardware. Headless Lavapipe works for CI.

When several adapters are present, select one with its backend-reported numeric
device ID (on Vulkan this is normally the PCI device ID, not an adapter ordinal):

```sh
MEGANEURA_DEVICE_ID=0x744c cargo run --release --example mnist
```

Decimal IDs are accepted too. `MEGANEURA_DISABLE_COOP=1` forces the portable
scalar path for regression diagnosis; `MEGANEURA_FLASH_FWD_COOP=0` and
`MEGANEURA_FLASH_BWD_COOP=0` disable only cooperative flash attention.
Device-local intermediate storage and lifetime aliasing are default-on, with
`MEGANEURA_NO_DEVICE_LOCAL=1` and `MEGANEURA_NO_ALIAS=1` as diagnostic escape
hatches.

Hosted CI executes Linux Vulkan (Lavapipe) and macOS Metal tests and compile-checks
Windows. Generated Vulkan shaders, including the f16 cooperative-matrix path used
on supported NVIDIA and AMD adapters, are also validated and translated to SPIR-V
offline. Real AMD and NVIDIA hardware testing remains part of release
qualification because software drivers cannot reproduce vendor-driver behavior.

## Profiling

For a repeatable per-dispatch JSON profile:

```sh
MEGANEURA_GPU_TIMING=1 \
  cargo run --release --example profile_session -- gap-profile.json
```

The report retains raw hardware-timestamp samples, selected pipeline variants,
workgroup geometry, forward/backward and kernel-family aggregates, device and
memory metadata, and the instrumentation overhead relative to normal grouped
execution. See [structured performance profiling](docs/performance-profiling.md)
for the Inferena harness and interpretation rules.

For a CPU/GPU timeline:

```
MEGANEURA_TRACE=trace.pftrace cargo run --example mnist
```

Open the trace in [Perfetto](https://ui.perfetto.dev):

[![perfetto trace](https://github.com/kvark/meganeura/raw/main/etc/example-trace.png)](/kvark/meganeura/blob/main/etc/example-trace.png)

## Citation

Machine-readable author and project metadata is available in
[`CITATION.cff`](https://github.com/kvark/meganeura/blob/main/CITATION.cff).
The paper citation and archival identifier will be added after publication.

## Contributing

Early project, small API surface, small community — a good time to show up. Open an issue before starting anything sizeable so we can align on direction.
