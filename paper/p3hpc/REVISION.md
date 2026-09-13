# P3HPC camera-ready review

Updated September 13, 2026 from the final cohort. This is preparation for
author approval, not a claim of reviewer approval or an uploaded submission.
[RESULTS.md](RESULTS.md) is the numerical/provenance ledger;
[SUBMISSION.md](SUBMISSION.md) is the final handoff checklist.

**Status correction:** the following matrix describes the superseded v7 draft,
not acceptance of the author's requested protocol. Native tuning, strict
native-f32 tiles, ROCm/XPU replay, MPS compilation and bounded preparation are
being implemented and qualified in Inferena; see [NEXT-COHORT.md](NEXT-COHORT.md).
New data are required before this matrix or the manuscript can claim those
conditions were measured. A separate Future Work section now discusses neural
accelerator backends and persistent megakernels, not these required fixes.

The paper now uses Inferena efb1e520 / Meganeura 75dfe901: eight complete
device campaigns (390 paired processes), plus five valid H100 extension
pairs and one failed searched 1.7B attempt. All share one PyTorch source.
Raw archives, traces, binaries, and model weights stay outside Git.

## Reviewer and author response matrix

Private reviews are paraphrased here rather than copied into source history.

| Concern | Final manuscript response | Remaining boundary |
|---|---|---|
| R1/R2: inconsistent PyTorch versions | Same 2.13.0 release/source and Python 3.13.13 in all manifests and successful records; wheels/drivers disclosed. | Common source does not erase backend/library differences. |
| R1: missing CUDA Graphs | Qualified full-phase replay on all three NVIDIA configurations, with uncaptured ablations. H100 token replay gains 2.74x. | ROCm/XPU replay and MPS compilation remain untested protocol alternatives, explicitly disclosed. |
| R1/R2: HPC hardware and larger models | H100 completes the main matrix; strict 360M and 1.7B have valid forward/backward pairs. | Larger points have one process; searched 1.7B capture fails; no optimizer, long-context or distributed scaling claim. |
| R1/R2: support failures are meaningful results | RPL-U runs native GPU workloads but PyTorch only on CPU; AMD/XPU searched bring-up failures and the current H100 capture failure are operational findings. | Omitted and unreached conditions are not newly observed failures. Final Windows light/replay is complete and successful. |
| R2: CPU reference contaminates GPU comparisons | RPL-U gets a separate GPU-versus-CPU table block; seven shared GPU-reference configurations enter all primary scores. | MI300X and partial large-model results stay outside that score. |
| R2: separate host/runtime and GPU causes | Same-pin RTX ResNet table reports host recording/wait, grouped Vulkan span, and CUDA kernel intervals; calibrated dispatch profiles and native counters identify derivative hotspots. | Single-process diagnostic, 16% per-dispatch instrumentation overhead; intervals overlap. No removable-barrier fraction is claimed. |
| R2: propose or demonstrate remedies | Production shape-specialized convolution candidates are in the measured pin; H100/5070 ResNet inference search gains 1.20x/1.07x. Separate six-pair NVIDIA/Intel qualification is labeled. | Tuner visits 5/71 classes in the diagnostic and misses expensive small-output gradients; prioritization and parallel reductions remain future work. |
| R1/R2/R3: productivity vs footprint | Preparation and separately versioned deployment closure are operational metrics; author/maintainer/debugging costs are qualitative. | No developer-hours, learnability or equivalent-port productivity score. |
| R2: explain edge experiment / figure reference | Explicitly explain host/device reconstruction agreement and renderer co-tenancy; shared context helps hybrid graphics/AI applications. | Independently versioned Quest case, not matched Android speed or on-headset training. |
| R3: abstract, terms, abbreviations, tables | Consistent compiler/runtime description; SGD, arithmetic/preparation axes and validation explained; short captions and aligned grouped headings; SmolLM2 absolute-time figure. | Author's final readability review remains. |
| R3: contradictory compilation ranges | Light native medians 0.09–3.65 s strict, 0.12–3.63 s accelerated; abstract gives combined 0.09–3.65 s. | Compiler/first-execution fields exclude some startup; not time-to-first-answer. |
| Author: hours of PyTorch search hidden | Cumulative preparation table: H100 searched compiler time 170.3 min versus 114.7 s native; capture and research qualification separate. | Unequal searches and boundaries; no equal-deadline claim. |
| Author: always tune native / strict native-f32 tiles | State actual light/search policy and conservative strict exclusion without suggesting they are inherent numerical requirements. | These proposed protocol changes were not implemented; see NEXT-COHORT.md. No further cohort started. |
| Author: MI300X framing | Treat compute-focused CDNA as a missing working Vulkan driver path, not an intrinsic impossibility of tensor compute through Vulkan. | A separately supplied report, not a final-revision timing; no claim a RADV fix is trivial. |
| Author: memory and CPU downclocking | Label plan sizes vs allocator peaks; keep measured host clock sensitivity separate from CPU energy policy. | No comparable process VRAM peak or claim normal downclocking is a malfunction. |

The final paper has no inherited Windows crash count, old H100 cuBLAS
attribution, obsolete H100 570-driver issue, or superseded H100 transient.
It does not mix preprint table values with the final cohort. Windows is
now a full replicated result. The latest capture log identifies a Triton
launch after an unspecified earlier capture error, not the initiating defect.

## Verification

From the repository root:

~~~sh
python3 paper/p3hpc/artifact/cohort.py "$HOME/Downloads/p3hpc" \
  --check paper/p3hpc/tables --output target/p3hpc-final-data
python3 -m unittest discover -s paper/p3hpc/artifact -p 'test_*.py'
python3 paper/p3hpc/artifact/verify.py --repository
~~~

All 395 successful pairs pass the frozen Inferena check_pair function as
well as the evidence replay; all eight complete gradient-replication reports
reproduce exactly. Offline checks reject modified timing, source, policy,
gradient, and replay evidence. The legacy tests concern the companion
report, not a second GPU measurement or a claim of external artifact evaluation.

Build and final package checks are recorded in SUBMISSION.md. Sources are
reviewable on the paper branch; publication files remain outside Git.
The author alone approves the text, chooses artifact hosting, merges, and
submits. The corrected cohort must not repeat unbounded max-autotune runs.
