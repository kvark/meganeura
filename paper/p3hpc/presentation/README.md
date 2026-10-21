# P3HPC presentation

This is the main 20-minute talk: 22 main slides with 18:50 of suggested
speaking time, followed by 14 backups. It combines the graphics-API argument,
the compiler's design, hardware access and limits, measured results, and
train-to-deploy conclusions. The `presentation-alt`, `presentation-api` and
`presentation-story` siblings remain intact as local reference drafts.

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

The main sequence is: game graphics → reaching the GPU → compiler map → models,
kernels and graph choices → compiler correctness → hardware access and API
limits → measured search → comparison and qualification → accelerated arithmetic
and results → selected paths → large models → preparation → headset integration
→ changes since submission → remaining gaps → what works.
Accelerated inference and training share a table: each cell shows the inference
time ratio, with training in parentheses. Strict results, the Rust API example,
gradients, detailed coverage, variability, numerical controls, TF32 and headset
queue measurements belong to backup. The CUDA/Vulkan limits stay in the main talk.
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

A synchronized sample means one host-wall timing of a model call, ending after
GPU completion. A process pair means fresh Meganeura and PyTorch processes for
the same condition, run sequentially. Each process takes 20 samples after warmup;
the reported engine time is the median of three process medians. Engine order
alternates between pairs. These definitions are also in the timing slide notes.

The B570's zero selected matrix sessions belongs to the frozen cohort, not a
claim that its matrix extension lacks hardware support. The October 21 local
follow-up at PR #274 revision `a0cd01b806363bd30ea81ef75510882883fc3394` selects
F16 Winograd kernels for U-Net inference and training. This is identified as
separate work; the table is not patched with those timings. Local receipt:
`/x/Code/inferena-results/pr273-274-review-20261021/review-summary.json`.
The “Since the paper” slide compares the submitted manuscript at
`paper-p3hpc-work` (`c62a8ec`) with the October cohort, and labels later kernel work
separately. Changed numerical policies prevent treating aggregate changes as a
controlled compiler speedup.

The opening uses a real Unreal Tournament screenshot. Its source and attribution
are in [assets/README.md](assets/README.md).

The headset and full training-cycle evidence comes from the separately pinned
[DinoVision study](https://github.com/kvark/dinovision/blob/dc35cdf1c7c910cdd93c5b5362846842ae469a21/REPORT.md).
It trains with Adam on a desktop Vulkan GPU and deploys on Quest; the framework
training timings exclude optimizer updates. The main headset slide demonstrates
mobile inference sharing a graphics context and queue with OpenXR rendering.
Scheduling measurements and their interpretation remain in backup notes.
