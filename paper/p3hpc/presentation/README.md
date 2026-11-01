# P3HPC presentation

The main talk has 19 slides with 19:05 of suggested speaking time, a backup
index and 24 backups. It combines the Vulkan/Metal deployment argument, the
compiler's architecture and search, hardware access and limits, and measured
performance. The Rust API example and server scaling are in the main talk; the compiler
map and importer example are in backup. The alternatives diagram shows the actual matmul/add fusion
rewrite, with operator e-nodes pointing to shared operand e-classes. The paper
and talk use the same October 31–November 1, 2026 cohort. Earlier `presentation-full`, `presentation-alt`, `presentation-api`
and `presentation-story` drafts remain local references with older evidence;
they are not regenerated with this deck.

From the repository root:

```sh
bash paper/p3hpc/presentation/build.sh
```

This needs `uv`, Node/npm and a Chromium-compatible browser. The build pins
Matplotlib 3.11.2 and Marp CLI 4.5.1 and regenerates every figure. SVGs, HTML,
PDF and notes are ignored build products. The HTML needs the adjacent
`figures/`, `assets/` and `../../figures/dinovision-quest3s.jpg`; the PDF embeds
them. No GPU or raw archive is needed to build. Each slide's notes carry a
short script, qualifications and sources. The backup index links to numbered
slides in HTML.

## Framework cohort

- Inferena `a0799e4210eafa25161ec50d475b186c4b19d3da`.
- Meganeura `567028b805425375148e88a7e1d464fafdd99777`.
- Blade `eae0c6f7aa3167c79d1135050cf8e17c17fabd0a`.
- Naga `a489554c96f87cbc5252ee00350730f6e93869f9`.
- PyTorch 2.13.0, source `cf30153c4c131c8164ee7798e5022d810682e2cb`.

`data/results.csv`, `preparation.csv` and `summary.json` copy the paper
analyzer's audited outputs. `cohort.sha256` identifies all sixteen raw
archives. `preparation-cache.csv` transcribes its six-row preparation table.
These compact chart inputs do not replace full qualification receipts.
The paper's [result ledger](../RESULTS.md) documents the audit and failures;
[artifact README](../artifact/README.md) gives the offline replay command.

All 330 core native cells qualify across eleven GPUs. Six devices provide
replicated performance results; RTX 3050 joins the three integrated GPUs
and Quest as correctness-only. Core plus larger models provide 222 native
timing cells and 201 ratios. B570 360M completes all native cells and all
accelerated reference phases; one strict reference process fails gradient
repeatability. No old result fills a missing new cell.

The ratio is **Meganeura time / PyTorch time**; below one is faster. Teal
marks native wins and brown losses. The three ratio-table slides place their
legends in the top-right title band, outside the SVGs. Repeated figure footer
notes are omitted; the comparison slide states the M3 timing caveat once.
Missing ratios mean a failed reference
or incomplete qualifying repeats, never zero or infinite speedups. A sample
is one host-wall model call ending after GPU completion. Each measured
process takes twenty samples after ten seconds' warmup. The engine time is
the median of three process medians from sequential pairs with alternating
engine order. M3 process spread remains high (median 15.7%, 43/60 groups over
10%); small differences are uncertain.

Main medians are strict 1.46 inference / 1.59 training, accelerated 1.67 /
1.92. The final slide's 1.60 median covers 112 core inference/training
comparisons, both profiles (58 strict, 54 accelerated), excluding duplicated
minimal shapes. Its geometric mean is 1.66; 71/112 ratios are below two.
The aggregate is not an upper bound.

Native strict/accelerated speedup medians are 1.19 inference / 1.05 training,
PyTorch 1.46 / 1.64, using each engine's own eligible pairs. Precision-gain
bars instead hold both engines and both profiles to the same model set per
GPU: 27 inference and 25 training conditions globally. Their native medians
are 1.19 / 1.04. `gap_analysis.py` uses geometric means on those identical
sets: ratios 1.24 → 1.66 and 1.54 → 2.21, multipliers 1.34 / 1.44. This
accounting is not a causal CPU/GPU/API/driver partition.

Matrix kernels are selected on every timed GPU, including 36/42 accelerated
B570 sessions. Backward coverage is partial: selected AT/BT products may use
F16/BF16; NN derivatives remain protected. Private matrix storage is searched
as well. The archive's generic dtype summary has older backward-f32 wording;
selected descriptions and the pinned compiler establish the actual paths.
Public tensors, masters, loss and accumulation remain f32. These facts
replace the old deck's “Intel selects no matrices” and “derivatives stay f32.”

## Separate evidence

- **Single operations:** `data/op-bench.tsv`, three native processes and one reference
  process per shape on RX 7900
  XT, RTX 5070 and B570, October 31. Native `5b232bd` plus the tuner scratch
  fix present in cohort revision `567028b8`; strict f32, tuner on, cooperative
  matrices off. PyTorch 2.13, TF32 off, `cudnn.benchmark` on. Native GPU time
  sums pass timestamps; reference device events time a graph of 30 serial
  ops divided by 30, including dispatch gaps. This favors native most on
  small ops and cannot partition cohort model time. Shapes, reference runner
  and driver are in `data/op-bench/`; native runner is `examples/op_bench.rs`.
  The table's small-GEMM losses support testing split-K and smaller tiles,
  not a claim that those proposed changes already recover the gap.
- **Shared memory:** `data/b570-unet-shared-memory.txt` at `99e3de2` records
  16.4 KiB for the largest generated U-Net kernel, below the 48 KiB ceiling.
- **Hardware:** `hardware.json` keeps the October 15 query and October 19
  CUDA/Level Zero limits, plus capability fields and native runner identity
  from the new cohort. The runner is 19.4 MiB, 14.9 stripped; source is
  54,271 Rust and 7,025 WGSL lines. `limits.json` distinguishes queried and
  documented values. RTX 5070/B570 f16 tile shapes were queried October 30;
  H100/3050 shapes assume the corresponding NVIDIA driver's offerings.
- **Since the paper:** `since_paper.py` compares the tagged September 25
  supplement with current results on the five shared timed GPUs. Both
  populations are restricted to the same qualified conditions. The tag
  `paper-p3hpc-work` (`c62a8ec`) and its supplement remain unchanged; the
  protocol changes prevent attributing every improvement to the compiler.
- **Warmup:** `warmup.csv` / `warmup-protocol.json` remain the separate
  October 19 diagnostic at Meganeura `c19cb1c`, not the new cohort revision.
  It keeps a program fixed over cumulative 5/15/30/60-second checkpoints,
  two processes per GPU, after search; no universal warmup claim follows.
- **Compile stages:** the September 10 diagnostic in `docs/experiments.md`
  remains historical, separately labeled in the backup.
- **DinoVision:** the separately pinned
  [study](https://github.com/kvark/dinovision/blob/dc35cdf1c7c910cdd93c5b5362846842ae469a21/REPORT.md)
  trains with Adam on desktop Vulkan and deploys beside headset rendering.
  Cohort training times exclude optimizer updates.

The opening's real Unreal Tournament screenshot is attributed in
[assets/README.md](assets/README.md). The model/API snippets follow the
public API, with `init` and `mnist` standing for application data.
