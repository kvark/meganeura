# P3HPC camera-ready working plan

Prepared September 5, 2026; updated September 8 after reading all three reviews.
This is a revised working draft and preparation kit, not a declaration that
the camera-ready has been approved or submitted. Acceptance-specific upload
conditions and talk instructions still need confirmation.

September 10: the deployment discussion now connects short specialization to
bounded on-device search, with a [study explanation](../../docs/study/performance-plan.md#cheap-compilation-as-a-search-budget).
It distinguishes today's opt-in, post-rewrite kernel selection from joint
graph-and-kernel search and keeps it outside the frozen matrix. Neither a
100 µs end-to-end compile nor a 10,000× Triton advantage is claimed. The
[stage experiment](../../docs/experiments.md#compiler-stages--september-10)
now measures 141–172 µs median WGSL parsing on three actual workloads, separately
charging native driver compilation and reporting cold/reused-cache controls.
It is a serial compilation diagnostic, not a replacement for the frozen tables
or the new collection cohort. A matched-domain GEMM follow-up measures about
31 ms native versus 100–167 ms Triton for a cold candidate after compiler warmup.
Separately, six-pair AB/BA confirmation finds 1.092× real 135M prefill gain,
amortized after about 80 prefills, while 360M finds no gain. These development
results inform the argument without silently changing the paper's frozen data.

The follow-up states the positive architectural argument explicitly: empirical
tuning runs automatically inside session construction when enabled, without a
separate offline workflow. The study guide now limits the negative transfer
evidence to the one-device, narrow tile-search experiment; it is not an
all-model/all-platform result, nor a comparison against greedy graph rewriting.

September 11: the controlled RTX 5070 campaign completed all 120 paired entries
(30 qualification and 90 replicated measurements) at Inferena `6f5f94cf`. On
both AMD systems, PyTorch max-autotune instead fails while compiling the
diffusion training graph:
generated convolution candidates exhaust local memory or time out, and the
remaining ATen fallback is malformed. The paper now treats automatic compiler
search as part of the portability surface. AMD default-mode collection remains
valid as an explicitly labeled availability subset; no eager timing replaces
the failed condition.

The Arc B570 follow-up at Inferena `f4255c4b` completes all 10 default/no-graph
qualification pairs and 30 replicated measurements. Its pinned XPU build has a
repeatable native dense-embedding backward defect; a shape-derived probe selects
and records the qualified dense `index_add` autograd equivalent for SmolLM2.
Max-autotune did not finish its first ResNet qualification within one hour, so
the B570 data is also an availability subset. This is useful portability
evidence, but the workaround and omitted condition must remain visible in any
new table.

The response matrix below paraphrases `review1.txt`, `review2.txt` and
`review3.txt` supplied at `/home/kvark/Documents/P3HPC`; private review text
is not copied into Git. The CUDA Graph concern is confirmed in the frozen
runner; see [the diagnosis and collection plan](CUDA-GRAPHS.md).

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
official deadline. A controlled, CUDA-Graph-qualified comparison is now the
first evidence priority. Keep it separate from the frozen measurements; if
it cannot be completed and reviewed in time, narrow the baseline claims
explicitly rather than treating the review concern as resolved.

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
| Sep 8 | Read all three reviews; map requests to evidence and changes | Response matrix below populated; responses are not yet closed. |
| Sep 8–10 | Tighten narrative and related work; verify bibliography metadata and every load-bearing number | Abstract, contributions, captions and conclusion use the same populations and qualifications. |
| Sep 8–13 | Qualify the clean Inferena baseline, then collect a separately versioned, replicated comparison and representative overhead profiles | Graph replay and all validity gates pass; same-campaign engine results, preparation costs and memory are disclosed. |
| Sep 10–13 | Author reads the full technical argument; rehearse the short talk and challenge questions | Can explain precision rollback, norm gate, oracle exclusion and portability equation without notes. |
| Sep 13–16 | Analyze the new cohort and address review items; explicitly defer unavailable platforms or scale extensions | No mixed-revision speed claims, unsupported causal explanations or implied datacenter validation. |
| Sep 16–19 | Clean PDF build, visual review, artifact replay, final language pass and reviewer-response closure | Author-approved content freeze; clean reviewed commit ready for packaging. |
| Sep 19–25 | Build deterministic archive, verify extracted bundle, complete venue forms/format checks and submit | Author verifies final upload and all venue requirements; only the author merges/publishes. |

The critical path is baseline validity, claim precision and author understanding,
not a new performance breakthrough. Do not expand this into a datacenter or
distributed-training campaign without an explicit scope decision.

## Reviewer-response matrix

Status distinguishes a source diagnosis or pilot from completed manuscript
changes and new evidence. A qualification pilot does not close a performance
comparison request.

| Review / concern | Response and evidence | Manuscript location | Status |
|---|---|---|---|
| R1, R2: mixed reference versions, execution modes and CPU fallback | Distinguish installed-stack availability from a controlled engine comparison. Pin a common PyTorch release where supported and disclose exceptions separately. | Evaluation setup; result tables; portability metric | RTX controlled cohort complete at Inferena `6f5f94cf`; Arc B570 default-mode cohort complete at `f4255c4b` with its probe-selected PyTorch workaround labelled; AMD default-mode collection pending. AMD and XPU max-autotune limitations are retained as availability results. |
| R1: missing CUDA Graph baseline for low latency | Paper-v1 bypassed explicit capture. Inferena now qualifies whole-phase forward, minimal forward and forward/loss/backward replay before a paired campaign. | Abstract; methodology; minimal-latency results; limitations; `CUDA-GRAPHS.md` | RTX 5070 campaign complete: 30 qualification and 90 measurement pairs across default/no-graph, default/graph and max-autotune/graph. No new timings have yet replaced the frozen tables. |
| R1, R2: small workloads and consumer hardware do not establish datacenter scaling | State the consumer/edge question, reduced shapes and absent optimizer/distributed work. | Abstract; introduction; workloads; limitations; conclusion | Consumer/edge framing and datacenter/distributed-work exclusion revised. No scale extension claimed. |
| R1, R2, R3: deployment efficiency is not programmer productivity | Separate compile/footprint metrics from qualitative application-author, backend-maintainer and debugging costs. | Deployment/productivity section; introduction; conclusion | Section retitled and rewritten, including the eager-PyTorch comparison and static-debugging costs. [Study guide](../../docs/study/observability.md) supplies detail. No productivity score claimed. |
| R2: separate host/runtime overhead from GPU kernel costs | Pair synchronized wall time with host/GPU timelines and kernel-family profiles in separate diagnostic runs. Overlap prevents treating wall time minus summed dispatch medians as exact CPU time. | Methodology; minimal-latency results; gap analysis | Qualified Systems/Graphics diagnostics now separate host encoding, GPU execution and parameter placement on RTX 5070. Resident 135M/1.7B pairs retain full CUDA Graph gates. No removable-barrier percentage established; [development analysis](../../docs/experiments.md) remains separate from the frozen matrix. |
| R2: explain what can be done about the performance gaps | Propose general candidate selection, convolution decompositions and safe synchronization changes; preserve validated fallbacks. | Locating the Gaps; conclusion | Remedies added; demonstrated Metal change distinguished from proposals. No new frozen-matrix speedup claimed. [Development conclusions](../../docs/experiments.md) remain separate. |
| R2, R3: explain the edge experiment and reference Figure 2 | Explain the reconstruction, host/device agreement and renderer interference before implementation detail. | Edge Deployment Beyond the Matrix | Rewritten around those two experimental questions, with an explicit casting-figure reference. |
| R3: abstract, terminology, abbreviations/cells, vague contribution, crowded captions | Remove opening KLOC clutter; use a consistent compiler/runtime description; define comparisons and core abbreviations; link arithmetic contracts to Section III.C. State what profiling explains. | Abstract; introduction; contributions; captions | First prose pass implemented. Short captions now precede tables in IEEE style; corrected tables and opening visually checked. Author readability pass remains. |
| R3: abstract compile-time range differs from the results table | Frozen records give 0.083–1.518 s strict and 0.083–2.358 s accelerated; the strict-results table contains only the first population. | Abstract; results discussion; deployment section | All three prose locations now state 0.08–1.52 s strict and 0.08–2.36 s accelerated. Records and generated tables unchanged. |
| Prevent two review misreadings | Blade is not wgpu; shared Naga does not imply a shared runtime. Cross-engine gradient norms are not elementwise gradient comparisons. | Architecture; numerical contracts | Blade/Naga/WGSL distinction now explicit; norm-versus-elementwise caveats retained. |

Common questions and prepared responses are in
[the rehearsal guide](../../docs/study/p3hpc-questions.md); they are anticipated
discussion topics, not a substitute for actual review comments.

Next evidence gate: complete the default-mode AMD availability cohorts and
qualify Windows/RTX 3050 and macOS locally. Keep graph/uncaptured controls on
the same device, rotate order across fresh processes, and recollect both engines in
the same campaign. The larger-model placement experiments are not a silent
change to that collection tag. Do not interpret 20 samples from one process as 20
independent experimental replicates. Preserve source refs and concise results,
with generated records and binaries outside main.

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
notices remain. All ten pages were visually inspected in that audit. The
companion report also builds at `target/paper-audit/main.pdf`.

The September 8 reviewer-correction pass also builds to 10 pages without
undefined references, citation warnings or overfull boxes. The obsolete forced
bibliography break was removed; the opening, revised table placement and final
reference columns were visually checked. A final whole-document author review
is still required. No working PDF or
its changing metadata hash belongs in Git; the final submission package should
record its own checksum separately from the reviewed source revision.

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

- [x] Read all three reviews and map concerns to evidence and proposed changes.
- [ ] Complete the response matrix and confirm acceptance-specific instructions.
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
