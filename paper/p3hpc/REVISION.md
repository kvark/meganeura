# P3HPC camera-ready working plan

Prepared September 5, 2026, following acceptance reported by the author.
This is a revised working draft and preparation kit, not a declaration that
the camera-ready has been approved or submitted. Reviewer text and any
acceptance-specific conditions were not supplied during the audit.

## Dates and constraints

The official page lists September 25 for camera-ready submission and
November 15–20 for the SC26 workshop week. It specifies IEEE proceedings
format, at most 12 main-text pages excluding references/appendices, and
5–16 pages overall. Exact talk day/time/length and any author-specific upload
instructions still need confirmation. The page encourages artifact
description/evaluation appendices and explaining cross-architecture
configuration differences.
[Official P3HPC submission instructions](https://p3hpc.org/workshop/2026/submissions/)

Target a reviewed content freeze by September 19, leaving six days before the
official deadline. Keep the current frozen measurements unless a clearly
separated new experiment can be completed, validated and reviewed in time.

## Changes already made

| Issue | Revision | Evidence |
|---|---|---|
| Ambiguous validation population in abstract | State 50/50 forward-valid and 48/50 backward-valid; exclude only disputed backward pairs | Replayed raw diagnostics. |
| Forward L2 read as full-tensor validation | Explicitly say 256 evenly spaced flattened output samples | Frozen harness sampling and retained records. |
| Gradient vectors read as elementwise gradients | Name parameter-gradient-norm vectors and limitations, including sign errors | Norm-based comparison implementation. |
| 1.8× median lacked reference-set label | Explicit GPU-reference population | 19 valid GPU-referenced strict training pairs, median 1.78. |
| Rounded 1.10 asserted as exact threshold | Use 1.11 bound for four discrete-AMD inference workloads | SmolLM2 ratio slightly exceeds 1.10 before rounding. |
| Runtime described as having no dynamic allocation | Narrow to no per-step tensor allocation | Runtime source still contains host-side allocations. |
| Stronger PyTorch baselines framed as manual specialization | Name untested automatic compiler modes; identify frozen default mode | Frozen runner and current official API documentation. |
| Profiles treated as proof of no API ceiling | Present kernel/policy targets without excluding API/driver limits | Profiles localize work, not theoretical ceilings. |
| Cooperative regression explained as having no profitability heuristics | Describe the unprofitable heuristic choice and propose measured selection | Frozen selection code and accelerated regressions. |
| Verifier purported to assert every prose number | Describe record/gate/median/table/fact replay and manual prose audit | Expanded verifier and explicit scope. |
| Incorrect RAJA author list; stale SYCL revision label | Correct RAJA authors/order and name current specification revision 12 | Publisher-deposited Crossref metadata, ECP publication record and Khronos specification. |
| Stale project documentation | Update roadmap/default precision/alternatives; add study kit | Source history and primary project documentation. |

Corresponding sampled-validation, ratio-scope and threshold corrections also
appear in the companion `paper/main.tex`. No benchmark JSON, generated numeric
table, frozen source tag or DinoVision imported fragment was modified.

## One-to-two-week schedule

| Window | Work | Done when |
|---|---|---|
| Sep 5–7 | Read reviewer feedback; map each request to evidence and a proposed change; study architecture/results | A response matrix exists with no unexplained reviewer item. |
| Sep 7–10 | Tighten narrative and related work; verify bibliography metadata and every load-bearing number | Abstract, contributions, captions and conclusion use the same populations and qualifications. |
| Sep 10–13 | Author reads the full technical argument; rehearse the short talk and challenge questions | Can explain precision rollback, norm gate, oracle exclusion and portability equation without notes. |
| Sep 13–16 | Optional narrowly scoped additional evidence only if hardware is free and protocol stable; otherwise keep frozen results | New results are separately versioned or explicitly deferred; no mixed-revision speed claims. |
| Sep 16–19 | Clean PDF build, visual review, artifact replay, final language pass and reviewer-response closure | Author-approved content freeze; clean reviewed commit ready for packaging. |
| Sep 19–25 | Build deterministic archive, verify extracted bundle, complete venue forms/format checks and submit | Author verifies final upload and all venue requirements; only the author merges/publishes. |

The critical path is claim precision and author understanding, not a new
performance breakthrough. If a reviewer asks for a materially larger scope,
follow their/organizer guidance rather than treating this schedule as
permission to rewrite the study or silently add incomparable measurements.

## Reviewer-response template

When feedback arrives, add one row per concern, preserving the reviewer's
meaning and distinguishing required changes from suggestions:

| Review / concern | Response and evidence | Manuscript location | Status |
|---|---|---|---|
| Awaiting supplied reviewer text | Do not invent reviewer requests | — | Pending author input |

Common questions and prepared responses are in
[the rehearsal guide](../../docs/study/p3hpc-questions.md); they are anticipated
discussion topics, not a substitute for actual review comments.

## Evidence and build checks

From the repository root:

```sh
python3 paper/p3hpc/artifact/verify.py --repository --show-facts
python3 -m unittest discover -s paper/p3hpc/artifact -p 'test_*.py'
```

Both pass after the audit: 50 paired cells, 5 profiles, 165 files, six table
fragments unchanged, six verifier tests. See the
[claim ledger](../../docs/study/results.md) for the manual prose check.

The local draft PDF is generated at
`target/p3hpc-audit/main.pdf` (from the repository root), with auxiliary files
beside it. It is a working artifact, not a committed publication file. The
normal build remains `latexmk -pdf main.tex` inside `paper/p3hpc`, or
`pdflatex`, `bibtex`, then two more `pdflatex` passes.

Audit rendering check: 10 US-letter pages, with the main argument ending on
page 8 and acknowledgments/artifact description/references on pages 9–10.
All 20 fonts are embedded Type 1. The final log has no undefined references,
citation warnings, font substitutions or overfull boxes; underfull layout
notices remain. All ten pages were visually inspected. The final reference
columns use `\IEEEtriggeratref{24}`; revisit that split if the bibliography
changes. The companion report also builds at `target/paper-audit/main.pdf`.

Current draft SHA-256:
`4ad1d060c224a1a193718f3143adcb521e8d8e1a716d675e2d88122895c03419`.
Rebuilding changes PDF metadata and may change this hash; record a fresh hash
for the final submission artifact.

This machine lacked the recommended fonts, TikZ and several LaTeX packages.
They were downloaded/extracted under `target/tex-packages/`, not installed
system-wide. To reproduce this local out-of-tree setup, run from `paper/p3hpc`:

```sh
export TEXMFHOME=/x/Code/meganeura/target/tex-packages/extracted/usr/share/texlive/texmf-dist
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../../target/p3hpc-audit main.tex
openout_any=a bibtex ../../target/p3hpc-audit/main
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../../target/p3hpc-audit main.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../../target/p3hpc-audit main.tex
```

The process-local BibTeX `openout_any` override allows writing the explicitly
chosen auxiliary directory outside its current directory. It is unnecessary
for a normal in-directory build. Use a complete TeX installation or the
documented container for the final archive build; do not substitute fonts to
hide a missing dependency.

## Final author checklist

- [ ] Incorporate actual reviews and acceptance-specific instructions.
- [ ] Read the complete draft, especially numerical contracts and exclusions.
- [ ] Check bibliography author order, publication metadata and URLs against
  primary sources; a successful BibTeX build checks syntax, not truth.
- [ ] Check every numeric prose claim against the fact ledger and inspect all
  captions, including Intel CPU labels and pre-optimization profile revision.
- [ ] Inspect the final PDF page by page: embedded fonts, diagrams, tables,
  cross-references, last-column balance and the venue's PDF requirements.
- [ ] Confirm disclosure text accurately describes the final workflow and
  the author's actual review/verification; retain responsibility statements
  only after performing that review.
- [ ] Review/commit changes; package with the existing clean-tree guard intact.
- [ ] Extract the archive elsewhere and run `verify.sh`; record hashes and
  provenance. Do not claim independent evaluation unless it actually occurred.
- [ ] Resolve permanent artifact hosting/DOI, rights and upload requirements
  through the venue's instructions. Nothing was uploaded in this audit.
- [ ] Rehearse with [48 questions and exercises](../../docs/study/p3hpc-questions.md),
  then adapt slides to the confirmed talk length.

The bounded tuning-foundation engineering phase is closed with split-K promotion
deferred after numerical rejection. No further development benchmark is required
for this checklist. Separately retained development results must not be described
as improvements to the frozen paper measurements.
