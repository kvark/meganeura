# P3HPC presentation

A 20-minute talk: 21 main slides, a backup index and 10 backups. It introduces
Vulkan, Metal and WGSL, then follows the workshop's three parts:
productivity (API, correctness, graph alternatives, program search),
portability (hardware access, limits, DinoVision, the MI300X driver) and
performance (cohort ratios, single-op kernel gaps, larger models, progress
since the paper). The "From PyTorch" backup describes a planned exporter that
is not implemented. Each slide's notes carry a short script and likely
questions. Earlier `presentation-full`, `presentation-alt`, `presentation-api`
and `presentation-story` drafts stay outside the repository.

From the repository root:

```sh
bash paper/p3hpc/presentation/build.sh
```

This needs `uv`, Node/npm and a Chromium-compatible browser; no GPU or raw
archive. The build pins Matplotlib 3.11.2 and Marp CLI 4.5.1 and regenerates
every figure. SVGs, HTML, PDF and notes are ignored build products. The HTML
needs the adjacent `figures/`, `assets/` and
`../../figures/dinovision-quest3s.jpg`; the PDF embeds them.

## Evidence

The paper and talk share the October 31–November 1, 2026 cohort: Inferena
`a0799e42`, Meganeura `567028b8`, Blade `eae0c6f7`, Naga `a489554c`, PyTorch
2.13.0 (`cf30153c`). Charts read the paper analyzer's audited outputs in
`../tables/`; the [result ledger](../RESULTS.md) documents the audit and
failures, and the [artifact README](../artifact/README.md) gives the offline
replay. The ratio is **Meganeura time / PyTorch time**; below one is faster.

Talk-only evidence in `data/`:

- **Single operations:** `op-bench.tsv`, from `examples/op_bench.rs` and
  `op-bench/torch_ops.py` on RTX 5070, Arc B570, and RX 7900 XT (November 8).
  Strict f32, tuner on, cooperative matrices off; each native
  row is the median of three processes. Both engines time 30 serial ops and
  divide by 30. `op-bench/run_all.sh` reruns every shape.
- **Shared memory:** `b570-unet-shared-memory.txt` records 16.4 KiB for the
  largest generated U-Net kernel at `99e3de2`, below the 48 KiB limit.
- **Hardware:** `hardware.json` holds device queries and the native runner's
  identity; `limits.json` separates queried from documented limits.
- **Since the paper:** `since_paper.py` compares the tagged September 25
  supplement (`paper-p3hpc-work`) with current results on the five shared
  timed GPUs, restricted to the same qualified conditions.
- **DinoVision:** the separately pinned
  [study](https://github.com/kvark/dinovision/blob/dc35cdf1c7c910cdd93c5b5362846842ae469a21/REPORT.md)
  trains with Adam on desktop Vulkan and deploys beside headset rendering.

The Unreal Tournament screenshot is attributed in
[assets/README.md](assets/README.md).
