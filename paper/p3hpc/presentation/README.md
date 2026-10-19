# P3HPC presentation

This is the main 20-minute talk: 19 main slides with 19:15 of suggested
speaking time, followed by 11 backups. It combines the graphics-API argument,
the compiler's design, hardware access and limits, measured results, and
train-to-deploy conclusions. The `presentation-alt`, `presentation-api` and
`presentation-story` siblings remain local reference drafts, outside this commit.

From the repository root:

```sh
bash paper/p3hpc/presentation/build.sh
```

The build needs `uv`, Node/npm and a Chromium-compatible browser. It pins
Matplotlib 3.11.2 and Marp CLI 4.5.1. `--figures` remains accepted for existing
commands; every build regenerates the figures. Outputs are ignored by Git:
`figures/*.svg`, `slides.html`, `slides.pdf` and `slides-notes.txt`.
The HTML needs its adjacent figures and `../../figures/dinovision-quest3s.jpg`;
the PDF embeds them. Markdown, CSS, Python, the build script and the compact
data inputs are the sources. No GPU or full benchmark archive is needed to build.

The main sequence is: shared graphics stack → Blade → compiler map and graph
choices → hardware access and API limits → measured search → comparison and
qualification → inference/training/accelerated results → selected paths →
large models → preparation → headset integration → remaining gaps → what works.
Detailed coverage, variability, numerical controls and TF32 belong to backup.
Notes provide sources and qualifications beyond the short text on each slide.

The framework results use the final October 19, 2026 cohort:

- Inferena `add19182e5479e3f7549894c524ac6691d3dc363`.
- Meganeura `c19cb1caedd90be7abc61de61f6197018711398e`.
- Blade `4845d6529c38b07718a624bf3c4f2ab4317c9bf4`.
- Naga `13f582ddd2792befb46952498010248bd65818e6`.
- PyTorch 2.13.0 at source `cf30153c`.

The submitted-paper tags identify an earlier collection. The chart inputs in
`data/` are frozen separately so this talk can be committed and built before
the paper revision. Do not regenerate them from an older paper checkout.

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

The ratio is **Meganeura / PyTorch time**, lower is better. Missing qualified
references stay empty. The deck names the four correctness-only devices,
describes the unequal numerical gates and setup scopes, and retains M3's
timing uncertainty. Matrix-capability tables distinguish driver exposure from
compiler support and selected programs. Development effort is author context,
not a measured allocation of the performance gap.

The headset and full training-cycle evidence comes from the separately pinned
[DinoVision study](https://github.com/kvark/dinovision/blob/dc35cdf1c7c910cdd93c5b5362846842ae469a21/REPORT.md).
It trains with Adam on a desktop Vulkan GPU and deploys on Quest; the framework
training timings exclude optimizer updates. The headset scheduling result is
an observation: source inspection permits queue interleaving, but the archive
has no trace establishing the cause of the lower worker latency.
