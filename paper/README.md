# Meganeura papers

## P3HPC (SC26 workshop)

The current paper is [p3hpc/main.tex](p3hpc/main.tex). It and the
[presentation](p3hpc/presentation/slides.md) use the final October 2026 cohort:
Inferena `add19182e5479e3f7549894c524ac6691d3dc363`, Meganeura
`c19cb1caedd90be7abc61de61f6197018711398e`, and PyTorch 2.13.0 at `cf30153c`.
The older `paper-p3hpc-2026-final` tags do **not** identify this collection.
No earlier result fills a missing cell.

[RESULTS.md](p3hpc/RESULTS.md) records the audit, failures, numerical-policy
limits and timing uncertainty. [QUALIFICATION.md](p3hpc/QUALIFICATION.md)
separates the four correctness-only platforms from the seven timed GPUs.
All 330 core native cells pass; core plus completed large-model runs supply
246 native performance cells and 226 qualified paired comparisons.
Raw archives, PDFs and ZIPs stay outside Git.

### Presentation

The main [talk](p3hpc/presentation/slides.md) combines the graphics-API argument,
the compiler architecture, hardware access and limits, and measured results.
It has 22 main slides (18:00 of suggested speaking time) and 15 backups.
The alternative decks remain local references and are not part of this commit.

The talk uses the later October 19 cohort: Inferena `add19182`, Meganeura
`c19cb1c`, and PyTorch 2.13.0 at `cf30153c`. Its versioned chart inputs are
independent of the submitted-paper tags and of the ongoing paper revision.
See its [README](p3hpc/presentation/README.md) for exact pins and evidence scope.

```sh
bash paper/p3hpc/presentation/build.sh
```

This regenerates the figures with Matplotlib 3.11.2 via `uv` and the deck with
Marp CLI 4.5.1 via `npx` (Node and a Chromium-compatible browser required).
Generated SVGs, HTML, PDF and notes are ignored; only source and chart inputs
are committed. No GPU or raw benchmark archive is needed for this build.

### Replaying the final evidence

Use standard-library Python 3.12+ and a clean Inferena checkout at the exact
revision above. No GPU, PyTorch installation, model downloads or new measurements
are needed for replay:

```sh
python3 paper/p3hpc/artifact/final_cohort.py "$HOME/Downloads/p3hpc" \
  --inferena /path/to/frozen/inferena \
  --check paper/p3hpc/tables --output target/p3hpc-final
```

The analyzer verifies all sixteen archive hashes, imports the frozen
collector's qualification routines, and regenerates eight TeX fragments,
`summary.json`, `results.csv` and `preparation.csv`. The result CSV includes all
retained process medians and paired ratios; the preparation CSV retains native
setup, search and reference compilation with their recorded cache states.
Additional audit JSON is written only to the output directory.
The ratio convention is **Meganeura / PyTorch time**, below one favors Meganeura.
See the [artifact README](p3hpc/artifact/README.md) for qualification and
packaging limits. The old `cohort.py`, `search_ablation.py` and `ablation.tex`
belong to historical studies and are not inputs to the final results.

### Template and build

The [workshop instructions](https://p3hpc.org/workshop/2026/submissions/)
require IEEE proceedings format, at most 12 content pages and 16 pages
including references and appendices. The paper uses the vendored IEEEtran
conference class, version 1.8b, and does not change its margins or font sizes.

The artifact description follows the
[SC26 author template at b5195e6](https://github.com/jennfshr/sc26-repro/tree/b5195e67d9ad0b5d07e8b6840558c7251c73b3c0/for-paper-authors).
[p3hpc/sc26repro.sty](p3hpc/sc26repro.sty) is copied from that revision;
its IEEEtran class matches ours apart from whitespace.
[p3hpc/artifact-description.tex](p3hpc/artifact-description.tex) uses the
template's contribution/artifact map and six subsections per artifact.
It is appended after the bibliography, with no authors added to its heading,
no example/explanation text, and no optional AE or badge claim.

Build from `paper/p3hpc`:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

IEEEtran needs the PostScript base fonts (Times, Courier and Helvetica).
For the small TeX Live container, install `collection-fontsrecommended`.
Do not substitute `lmodern`, which changes the IEEE font.

The bibliography combines shared `paper/references.bib` and workshop additions
in `paper/p3hpc/references.bib`; their citation keys are distinct.
A source ZIP must retain both files and their relative paths, as well as
`artifact-description.tex`, `sc26repro.sty`, the class/style and figures/tables.

After building the PDF, package the source and supplementary ZIPs together:

```sh
python3 paper/p3hpc/artifact/submission.py "$HOME/Downloads/p3hpc" \
  --inferena /path/to/frozen/inferena \
  --pdf paper/p3hpc/main.pdf --output paper/p3hpc/submission \
  --submission-id <SC submission number>
```

The source ZIP carries both bibliography files, `main.bbl`, the class/style,
SC's `fancyhdr.sty`, the AD, figures and tables with their relative paths.
SC26 wants auxiliary materials as one `<SC submission number>aux.zip` with a
short `readme.txt`; ours is [artifact/SUPPLEMENT.md](p3hpc/artifact/SUPPLEMENT.md)
and sits at the archive root with the sixteen original campaign archives,
the final analyzer and generated inputs, and a `MANIFEST.sha256`. The compact
warmup summary and its distinct protocol are included separately.

The first-page copyright block follows the
[SC26 Instructions for Workshop Authors](https://submissions.supercomputing.org/static_resources/SC26_Instructions_for_Workshop_Authors.pdf)
(Step 3a) with SC's `fancyhdr.sty`. The camera-ready PDF must also pass IEEE
PDF eXpress (conference ID 72768X) and stay under 4 MB. Do not use the legacy
packager below for the current cohort.

## Original arXiv report

`main.tex`, `tables/` and `results/` at this directory level belong to
[arXiv:2608.01563](https://arxiv.org/abs/2608.01563).
Its frozen data use Meganeura `7561a64` and Inferena `7ca9c5c7`, both tagged
`paper-arxiv-1`. They are separate from the current P3HPC evidence.

From `paper/`, `python3 mktables.py` regenerates those historical tables.
From the repository root, the legacy checker remains available:

```sh
python3 paper/p3hpc/artifact/verify.py --repository --show-facts
```

The original report's 780M Whisper backward pair is excluded from training
ratios by its oracle-consistency audit. That finding does not describe the
new P3HPC cohort. `artifact/package.py` packages only this original evidence.
The separately versioned `dinovision-section.tex` comes from
`kvark/dinovision/experiments`; update it from there, not in place.
