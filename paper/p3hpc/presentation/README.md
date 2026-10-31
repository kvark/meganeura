# P3HPC presentation

This is the main 20-minute talk: 17 main slides with about 17:50 of suggested
speaking time, followed by a backup index and 26 backups. It asks whether
Vulkan and Metal can carry ML training and inference, and answers in the
workshop's three parts. The header on every main slide names the parts and
highlights the current one.

- **Productivity:** one small stack; writing and training a model in Rust,
  with the ONNX, NNEF and GGUF importers; how we know it is right; the search
  that adapts the same code to each GPU.
- **Portability:** which hardware features we reach; same GPU, different
  limits; the headset application; the MI300X driver.
- **Performance:** the five models and eleven GPUs, and the measurement
  protocol; strict f32 and reduced-precision results for inference and
  training; single GEMMs and convolutions against the vendor libraries;
  changes since the paper.
- **Conclusion:** two slides. What Meganeura shows: one compact binary
  trains and runs on every GPU we tried, with no runtime dependencies, within
  2× of PyTorch. Whether Vulkan and Metal reach the vendor stacks: some GPUs
  already match, and the rest splits into our kernels and what the drivers
  expose.

The code on "How do we write a model?" and on the "Importing a model" backup
type-checks against the crate's public API (`Graph`, `nn::Linear`,
`build_session`, `Trainer`; `load_onnx`, `build_inference_session` and the
`Session` methods), as in `examples/mnist.rs` and `examples/onnx_compare.rs`.
`init` and `mnist` stand for the application's initial weights and dataset.

The conclusion's 1.70× is the median `ratio` in `data/results.csv` over the
five core models' inference and training rows, both precisions: 132
comparisons (68 strict, 64 reduced precision) on the seven timed GPUs. Their
geometric mean is 1.81×; reduced precision alone is 2.06× and 2.21×.

`data/whisper-profile.txt` is a local per-pass profile of Whisper-tiny
inference on the RTX 5070 (October 30, Meganeura `99e3de2`, not the cohort
revision). Attention over 1,500 positions takes 2.98 of 5.77 ms with matrix
units off, and the matrix-unit attention kernel takes 3.30 ms. It explains
the Whisper notes on the two results slides; it is not part of the cohort.
Without Whisper, the strict medians are 1.24 and 1.55 and the reduced ones
1.97 and 2.09.

The f16 tile shapes on "Which hardware features do we reach?" were queried
through `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR` on the RTX 5070
and B570 on October 30; the H100 and RTX 3050 rows assume the same NVIDIA
driver shapes.

`presentation-full` is the earlier 24-slide version of this talk, with the
same data. The `presentation-alt`, `presentation-api` and `presentation-story`
siblings remain local reference drafts.

From the repository root:

```sh
bash paper/p3hpc/presentation/build.sh
```

The build needs `uv`, Node/npm and a Chromium-compatible browser. It pins
Matplotlib 3.11.2 and Marp CLI 4.5.1, and passes `--no-stdin` so Marp does not
wait on an open pipe. `--figures` remains accepted for existing commands;
every build regenerates the figures. Outputs are ignored by Git:
`figures/*.svg`, `slides.html`, `slides.pdf` and `slides-notes.txt`. The HTML
needs its adjacent `figures/`, `assets/`, and
`../../figures/dinovision-quest3s.jpg`; the PDF embeds them. In the HTML, the
backup index links to each backup. No GPU or full benchmark archive is needed
to build. `figures.py` imports `../artifact/gap_analysis.py` and reads
`../artifact/attention-followup.json` for two backups.

Each slide's notes hold a short script, an "If asked" list and the sources.
The results tables show inference and training; the minimal shape is a backup.

The framework comparison tables use the final October 19, 2026 cohort:

- Inferena `add19182e5479e3f7549894c524ac6691d3dc363`.
- Meganeura `c19cb1caedd90be7abc61de61f6197018711398e`.
- Blade `4845d6529c38b07718a624bf3c4f2ab4317c9bf4`.
- Naga `13f582ddd2792befb46952498010248bd65818e6`.
- PyTorch 2.13.0 at source `cf30153c`.

The camera-ready paper is tag `paper-p3hpc-work` (`c62a8ec`) and reports the
September 25 runs. The chart inputs in `data/` are frozen separately; do not
regenerate them from an older paper checkout.

`data/results.csv`, `preparation.csv` and `summary.json` preserve the audited
final-cohort analyzer outputs; CSV line endings are normalized to LF, with
every field unchanged. `cohort.sha256` identifies all sixteen raw archives.
`preparation-cache.csv` transcribes the analyzer's seven-row preparation table
without additional rounding. These are presentation inputs, not substitutes
for the full qualification receipts in the archives.

`warmup.csv` and `warmup-protocol.json` are the separately versioned SmolLM2
diagnostic, copied from the paper's warmup artifact. `hardware.json` records
the October 15 Vulkan feature query, selected cohort capability fields,
October 19 CUDA/Level Zero limit checks and the measured native runner's
size and hash.

`limits.json` feeds "Same GPU, different limits" and gives each value's
source. The RTX 5070 and B570 values were queried. The RTX 3050, H100, AMD and
Apple values come from vendor documentation and a public RADV report, not from
our hosts; `vulkaninfo` and a HIP device query on the RX 7900 XT and MI300X
hosts would confirm the AMD row. The "Only through the vendor API" column
lists matrix inputs and Hopper features that Vulkan does not expose; for the
AMD GPUs it is BF16 matrices, which the RADV builds in these runs do not
report (`cooperative_bf16_f32_shapes` is empty in their run records).

The feature tiers on "Which hardware features do we reach?" come from Blade's
`Capabilities` at the pinned revision and their uses in Meganeura `c19cb1c`;
the exposed-but-unused features are the RTX 5070's October 15 query.

The ratio is **Meganeura time ÷ PyTorch time**; below 1, Meganeura is faster.
Teal marks Meganeura faster and brown slower, in the charts and in the text.
Every missing ratio is a PyTorch result that failed its checks, or a later
phase not attempted after such a failure; all native results pass.

A synchronized sample means one host-wall timing of a model call, ending after
GPU completion. A process pair means fresh Meganeura and PyTorch processes for
the same condition, run sequentially. Each process takes 20 samples after
warmup; the reported engine time is the median of three process medians.
Engine order alternates between pairs.

The search line's 306 of 588 sessions (52%) choosing a program other than
ordinary extraction comes from the search table in `../main.tex`. The
matrix-dispatch counts in the backups come from the `cooperative_dispatches`
field of each selected session in the raw archives. The compile-stage backup
is the separate September 10 study in `docs/experiments.md`.

`since_paper.py` reproduces the "Since the paper" numbers. It compares
`data/results.csv` with the camera-ready supplement, replayed by the tagged
analyzer; its docstring lists the steps. The protocol changed between the two
runs, so the comparison does not isolate a compiler speedup.

"Where is the gap? One op at a time" comes from `data/op-bench.tsv`, an
October 30 run at 99e3de2 on the RTX 5070 and B570: one strict-f32 GEMM or
batch-1 convolution per process, Meganeura with tile search on and
cooperative matrices off against PyTorch 2.13.0 with TF32 off and
`cudnn.benchmark` on. Times are medians of summed GPU kernel time over 30
operations. The shape list, the PyTorch runner and the driver script are in
`data/op-bench/`, and the Meganeura runner is `examples/op_bench.rs` at the
repository root. The slide's ranges group the rows of the TSV. The 5.45× in the takeaway is the cohort's
B570 U-Net strict inference ratio. The notes' 16 KiB comes from
`data/b570-unet-shared-memory.txt`, and the earlier U-Net diagnostics (286
against 276 dispatches, the F16 Winograd ablation) from `data/intel-gap.json`.

The B570's zero selected matrix sessions belongs to the October 19 runs. The
October 21 local follow-up at PR #274 revision
`a0cd01b806363bd30ea81ef75510882883fc3394` selects F16 Winograd kernels for
U-Net inference and training; the tables do not include it. Local receipt:
`/x/Code/inferena-results/pr273-274-review-20261021/review-summary.json`.

The gap-accounting backup holds the qualified model set fixed across both
engines and both arithmetic profiles. `../artifact/gap_analysis.py` computes
an exact accounting identity and geometric means on 32 inference / 30
training conditions. These are not the main table's medians and do not
allocate time to CPU, GPU, API or driver causes.

The "Change one path" backup uses the separate October 24 data in
`../artifact/attention-followup.json`, with all eight model/device
comparisons. It is one process per condition and a within-native shader
ablation; it updates none of the cohort ratios or coverage.

The opening uses a real Unreal Tournament screenshot. Its source and
attribution are in [assets/README.md](assets/README.md).

The headset and full training-cycle evidence comes from the separately pinned
[DinoVision study](https://github.com/kvark/dinovision/blob/dc35cdf1c7c910cdd93c5b5362846842ae469a21/REPORT.md).
It trains with Adam on a desktop Vulkan GPU and deploys on Quest; the
framework training timings exclude optimizer updates.
