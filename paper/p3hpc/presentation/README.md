# P3HPC presentation

This is the main 20-minute talk: 22 main slides with 18:00 of suggested
speaking time, followed by 15 backups. It asks whether Vulkan and Metal can
carry ML training and inference, shows how Meganeura is built, then measures it
against PyTorch and ends on the headset. The `presentation-alt`,
`presentation-api` and `presentation-story` siblings remain local reference
drafts.

From the repository root:

```sh
bash paper/p3hpc/presentation/build.sh
```

The build needs `uv`, Node/npm and a Chromium-compatible browser. It pins
Matplotlib 3.11.2 and Marp CLI 4.5.1. `--figures` remains accepted for existing
commands; every build regenerates the figures. Outputs are ignored by Git:
`figures/*.svg`, `slides.html`, `slides.pdf` and `slides-notes.txt`.
The HTML needs its adjacent `figures/`, `assets/`, and `../../figures/dinovision-quest3s.jpg`;
the PDF embeds them. Markdown, CSS, Python, the build script and the compact
data inputs are the sources. No GPU or full benchmark archive is needed to build.

The main sequence: one GPU, two stacks → how we reach every GPU → the compiler
map → models → equivalent forms → kernels and build time → testing → matrix
units → API limits → search → measurement protocol → the MI300X driver →
strict f32 results → changes since the paper → reduced-precision results →
who gains from reduced precision → larger models → why the gap → setup cost →
the headset → what works. Strict f32 comes first because both engines then do
the same arithmetic. Each results table has one panel per phase: inference,
minimal shape and training.

Each slide's notes hold a short script, an "If asked" list and the sources.
Backups cover matrix use by GPU, the reduced-precision rules, numerical checks,
coverage, timing variability, warmup, compile stages, saved kernel decisions,
portability scores, TF32, SPIR-V precision, the Rust API, gradients, headset
queue measurements and the motivation.

The framework results use the final October 19, 2026 cohort:

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
every field unchanged. `cohort.sha256` identifies all sixteen
raw archives. `preparation-cache.csv` transcribes the analyzer's seven-row
preparation table without additional rounding. These are presentation inputs,
not substitutes for the full qualification receipts in the archives.

`warmup.csv` and `warmup-protocol.json` are the separately versioned SmolLM2
diagnostic, copied from the paper's warmup artifact. `hardware.json` records
the October 15 Vulkan feature query, selected cohort capability fields,
October 19 CUDA/XPU limit checks and the measured native runner's size/hash.
API documentation and pinned source links are in the relevant slide notes.

The ratio is **Meganeura time ÷ PyTorch time**; below 1, Meganeura is faster.
Teal marks Meganeura faster and brown slower, in the charts and in the text.
Every missing ratio is a PyTorch result that failed its checks, or a later
phase not attempted after such a failure; all native results pass.

A synchronized sample means one host-wall timing of a model call, ending after
GPU completion. A process pair means fresh Meganeura and PyTorch processes for
the same condition, run sequentially. Each process takes 20 samples after warmup;
the reported engine time is the median of three process medians. Engine order
alternates between pairs.

The matrix-dispatch counts on the "Who gains from reduced precision?" slide and
its backup come from the `cooperative_dispatches` field of each selected session
in the raw archives. The compile-stage backup is the separate September 10 study
in `docs/experiments.md`.

`since_paper.py` reproduces the "Since the paper" numbers. It compares
`data/results.csv` with the camera-ready supplement, replayed by the tagged
analyzer; its docstring lists the steps. The protocol changed between the two
runs, so the comparison does not isolate a compiler speedup.

The B570's zero selected matrix sessions belongs to the October 19 runs. The
October 21 local follow-up at PR #274 revision
`a0cd01b806363bd30ea81ef75510882883fc3394` selects F16 Winograd kernels for
U-Net inference and training; the tables do not include it. Local receipt:
`/x/Code/inferena-results/pr273-274-review-20261021/review-summary.json`.

The opening uses a real Unreal Tournament screenshot. Its source and attribution
are in [assets/README.md](assets/README.md).

The headset and full training-cycle evidence comes from the separately pinned
[DinoVision study](https://github.com/kvark/dinovision/blob/dc35cdf1c7c910cdd93c5b5362846842ae469a21/REPORT.md).
It trains with Adam on a desktop Vulkan GPU and deploys on Quest; the framework
training timings exclude optimizer updates.
