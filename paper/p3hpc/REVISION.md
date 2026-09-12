# P3HPC camera-ready revision status

Updated September 12, 2026 after auditing the final supplied cohort.
The manuscript now uses the common-revision campaign rather than the
original five-machine tables. See [RESULTS.md](RESULTS.md) for the evidence
ledger, crash analysis, scaling limits, and Intel-server recommendation.

All nine archives use Inferena `17d13a3`, Meganeura `fcdd76d1`,
Python 3.13.13, and PyTorch 2.13.0 at source `cf30153`.
Seven campaigns complete their selected conditions (330 paired processes);
Windows and the H100 extension retain 36 additional valid pairs and two
failures. The data and protocol were not changed during analysis.
Archives, logs, and generated binaries remain outside Git.

## Reviewer-response matrix

The three private reviews are paraphrased, not copied into the repository.
“Addressed” identifies a manuscript/evidence change, not reviewer approval.

| Concern | Camera-ready response | Location / remaining limit |
|---|---|---|
| R1/R2: four PyTorch versions confound the comparison | All nine manifests and every successful PyTorch record share the same release/source revision and Python version. Backend wheels and drivers remain explicitly different. | Methodology; environment table. MPS eager, ROCm/XPU default-only, and CPU execution remain declared conditions. |
| R1: missing CUDA Graphs makes low-latency comparison unfair | Both complete NVIDIA campaigns include default/no-graph, default/replay, and max-autotune/replay. Whole-phase forward, minimal forward, and F+L+B replay is qualified against uncaptured PyTorch, including full gradients. Replay improves H100 135M token latency 3.32× and reverses that comparison. | Methodology; replay ablation; searched table. The old minimal-shape headline is removed. ROCm replay remains unqualified in this protocol and is disclosed. |
| R1/R2: consumer-only hardware and small models | Add H100 80GB, native Windows 3050 evidence, and the completed strict 360M/1.7B pairs. Show actual replicate counts and the failed condition. | Scaling table; scope and threats. Larger models have one successful process per condition, no accelerated extension, no optimizer/distributed work. Partially addresses scale; does not claim production training. |
| R2: Intel CPU fallback obscures GPU comparisons | RPL-U occupies a clearly labeled CPU support block and is excluded from GPU scores. Arc B570 supplies a separate XPU GPU comparison with a recorded qualified embedding workaround. | Device table; light table; availability discussion. No GPU-versus-GPU claim for RPL-U. |
| R1/R2: practical availability matters | Include requested ROCm/XPU search failures, 780M overrides, intermittent CUDA capture failures, and the adverse MI300X Vulkan result. Unreached conditions are unmeasured. | Availability and scaling sections. Final manifests prove omissions; earlier bring-up failures and MI300X have separately identified evidence. |
| R1/R2/R3: deployment efficiency is not programmer productivity | Keep preparation and historical closure measurements distinct from qualitative author/maintainer effort. Explain eager PyTorch inspection versus static-graph preservation, provenance, first-nonfinite attribution, and same-kernel debugging. | Deployment/productivity section; study guide. No developer-hour, user-study, or equivalent-port productivity score claimed. |
| R2: separate CPU/runtime from GPU kernel performance | Add qualified RTX 5070 resident-parameter Nsight timeline table: host wall, CPU recording, GPU span, and CUDA kernel intervals. Describe directly measured CPU downclock sensitivity and kernel-family diagnostics. | Gap analysis. This is representative development evidence with explicit revisions; no systematic all-platform breakdown or barrier percentage is claimed. |
| R2: propose or demonstrate solutions for performance gaps | Final cohort shows 1.38× H100 and 1.09× 5070 native 135M prefill gains from compile-time search. Separate numerically qualified convolution constant-specialization experiments demonstrate ~1.32× F+L+B improvement with preparation charged. | Search/preparation and gap sections. Prototype gains are not inserted into final-cohort timings. GEMV, representation, and convolution choices remain general search opportunities. |
| R2: explain Figure 2 and edge experiment | Keep the explicit figure reference and explain host/device agreement and renderer co-tenancy before implementation detail. Label the case as separately versioned deployment evidence. | Edge deployment section. No matched Android performance or headset-training claim. |
| R3: abrupt abstract, inconsistent terminology, undefined abbreviations, dense writing | Rewrite abstract/contributions around one “compact native compiler and runtime”; explain comparison populations, arithmetic/preparation axes, and limits. Expand key abbreviations and use short table captions with definitions in text. | Abstract, introduction, architecture, methodology, captions. Author readability review remains. |
| R3: inconsistent compilation range | State light-policy native medians as 0.09–3.52 s strict and 0.12–8.31 s accelerated; abstract gives their combined 0.09–8.31 s range. Search costs have their own table. | Abstract; preparation; generated tables. These are three-phase setup fields, not full application startup. |
| Avoid inherited evidence mistakes | Remove the old 780M oracle exclusion and old ratio/portability claims. Keep historical footprint, Quest evidence, and profiles labeled by provenance. Blade is not wgpu; cross-engine gradient norms are not elementwise validation. | Throughout; legacy artifact instructions clearly distinguished. |
| Environment provenance | Disclose H100's recorded 570 graphics driver alongside CUDA 13 wheels and the missing loaded compatibility-library identity. | Threats; results guide. NVIDIA documents a forward-compatibility path; the version combination alone does not diagnose the crash. |

## What to say about the results

The corrected baseline is stronger and the performance claims are narrower.
Across the six complete GPU-reference configurations under light preparation,
strict median inference/minimal/training ratios are 1.66/1.26/2.41.
Searched PyTorch wins every measured CUDA phase comparison, while native
search demonstrably helps and costs much less preparation on these workloads.
The case is compact deployment and usable graphics paths with explicit
performance limits, not general superiority to vendor libraries.

H100 scaling partly supports amortization: training ratios narrow
4.95 → 4.73 → 3.73, but prefill stays around 3.5–3.9 and token ratios widen
at 1.7B. The partial extension is worth reporting without fitting a scaling law.
Another Intel server campaign is not a prerequisite for this revision.

## Evidence and build

~~~sh
python3 paper/p3hpc/artifact/cohort.py "$HOME/Downloads/p3hpc" \
  --check paper/p3hpc/tables --output target/p3hpc-final-data
~~~

This audits the external archives and regenerates the six camera-ready
fragments without a GPU. The original `artifact/verify.py --repository`
and its tests still check the companion report's historical dataset.
Do not use the legacy packager as the final-cohort artifact.

The working PDF is `target/p3hpc-final/main.pdf`. Build from `paper/p3hpc`
with a normal IEEE-compatible TeX installation, or use this machine's
existing local package tree:

~~~sh
export TEXMFHOME=/x/Code/meganeura/target/tex-packages/extracted/usr/share/texlive/texmf-dist
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../../target/p3hpc-final main.tex
openout_any=a bibtex ../../target/p3hpc-final/main
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../../target/p3hpc-final main.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../../target/p3hpc-final main.tex
~~~

## Remaining author decisions

Local checks passed: all 366 pairs also pass the frozen Inferena checker;
all seven complete campaigns reproduce its retained gradient-replication
reports exactly. The offline analyzer rejects altered timing, gradient, and
capture evidence. Generated tables, legacy artifact verification and its
six tests, and the 11-page PDF build pass. The PDF was visually checked;
fonts are embedded, with no undefined references or overflowing boxes.

The recorded camera-ready deadline is September 25; check the
[venue instructions](https://p3hpc.org/workshop/2026/submissions/) and any
acceptance-specific requirements before upload.

- Read the revised argument, tables, crash interpretation, and AI disclosure.
- Review the final PDF and confirm talk date/length, rights, and upload rules.
- Choose permanent hosting for the small external result archives and logs;
  archive source refs with the final submission, then tag the settled revision.
- Approve and merge the source changes. Only the author merges or submits.

The paper does not need another development benchmark or a speculative Intel
server rental to complete these decisions.
