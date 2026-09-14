# P3HPC camera-ready review

Updated September 14, 2026 after auditing the completed v9 cohort.
This is preparation for author approval, not reviewer approval or an uploaded
submission. [RESULTS.md](RESULTS.md) is the numerical/provenance ledger;
[SUBMISSION.md](SUBMISSION.md) describes the three publication files.

The paper uses Inferena fa5a04e1 / Meganeura 428fc2d2: 240 main paired
processes and 12 fully replicated H100 extension pairs. All pass.
The former protocol omissions are now checked against executed evidence,
not inferred from flags. Raw records and publication binaries stay outside Git.

## Reviewer and author response matrix

Private reviews are paraphrased, not copied into source history.

| Concern | Manuscript response | Remaining boundary |
|---|---|---|
| R1/R2: different PyTorch versions | One 2.13.0 source revision and Python 3.13.13 across every record; backend/driver differences disclosed. | Common source does not equalize vendor libraries. |
| R1/author: missing command-graph replay | Full-phase qualified CUDA, HIP and XPU replay; MPS and CPU compile all requested phases. | MPS has no equivalent public whole-phase replay API. |
| Author: always tune native and permit strict f32 tiles | Every one of 708 sessions reports search; M3 demonstrates native-f32 cooperative dispatches under strict arithmetic. | No final-revision search-off/cooperative-off ablation. |
| Author: broader search before collection | No class-count cap, 60 s soft/session, 1 GiB scratch plus device guard; table reports 13,322/14,085 class instances and all rejections. | 24 ResNet sessions reach the deadline; not all kernel/graph families are searched. |
| Author: stop multi-hour reference search | All platforms use default compilation with a 120 s watchdog; cumulative preparation and qualification reported separately. | Native tuning can cost more than default PyTorch; limits have different boundaries. |
| Author: explain the omission of max-autotune | Methodology now gives the H100 cumulative cost and Radeon/Arc bring-up findings at the policy choice, with references to the detailed evidence. | Default compilation is a deliberate common deployment policy, not an estimate of the best result after unlimited search. |
| Author: show tuning benefit and multi-hour cost | Separate three-process H100 pilot demonstrates a 1.36x native prefill improvement and 170.3 min cumulative searched PyTorch compilation. | Pilot uses a separately identified revision/two-second native policy; not a main-cohort ablation. |
| R1/R2: HPC hardware and larger models | H100 main matrix plus 360M and 1.7B, both arithmetic contracts, three processes each; all requested runs complete. | Small batch/context, stateless token, no optimizer state or distributed training. |
| R1/R2: support failures are results | Native RPL-U GPU support versus PyTorch CPU; XPU compatibility workarounds and ROCm consumer overrides; separate MI300X driver boundary. | Do not claim final Windows/H100 crashes or infer failures from omitted arms. |
| R2: CPU reference contaminates GPU aggregate | Separate GPU-versus-CPU block; seven shared GPU-reference systems in every primary score. | Extra H100 sizes and MI300X stay outside that aggregate. |
| R2: distinguish host and GPU costs | Pinned RTX timeline/table, calibrated per-dispatch profiles and native counters localize convolution derivatives. | Diagnostic predates expanded search; instrumentation adds 16%; no removable-barrier fraction. |
| R2/author: remedies rather than unexplained gaps | Production convolution candidates and expanded tuning are included; final NVIDIA coverage is complete within the implemented domain. | GEMV, reduced-input paths and parallel reductions remain optimization targets. |
| Author: wider matrix instructions such as WGMMA | Cite Hopper's asynchronous warpgroup API, the measured Blade subgroup-only tile filter, and Vulkan's unused vendor workgroup extension. | Possible contributor to accelerated H100 results; no matched ISA attribution or fundamental Vulkan ceiling is claimed. |
| R1/R2/R3: productivity vs footprint | Deployment costs and programmer/maintainer/debugging tradeoffs are separate; Inferena's Rust-framework comparison origin explained. | No developer-hours, learnability or equivalent-port productivity score. |
| R2/author: edge/hybrid application story | Explain host/device reconstruction gate, figure and shared rendering/ML context. | Independently versioned Quest case, not Android comparative speed or on-device training. |
| R3: terminology, abstract, tables | Expand SGD and other acronyms; aligned arithmetic headings and absolute SmolLM2 figure; current preparation ranges. | Author's final prose review remains. |
| Author: CPU downclock and VRAM | Disclose clock sensitivity without calling power saving a malfunction; distinguish planner allocation, allocator peaks and missing NVML metrics. | No matched continuous peak-VRAM measurement. |
| Author: future work | Neural accelerators and persistent megakernels have a separate section. | Required protocol repairs are completed, not deferred to future work. |

The paper does not claim that native tuning always fits inside PyTorch's
compilation time. M3's noisy/mixed changes and accelerated H100 1.7B regression
are reported, not hidden by aggregate wins. The overall strict training
ratio is 2.47, while native ResNet improves substantially across cohorts.

## Verification and handoff

The frozen Inferena checker accepts all 252 pairs. Independent replay
recomputes retained medians, cross-engine errors, per-tensor/whole-gradient
replay bounds, search receipts and all nine replication reports.
The compact evidence preserves every JSON value, including raw/joined
records and full replay statistics. Mutated evidence is checked separately
without adding a large fixture/test suite to engine builds.

The original-submission verifier remains explicitly legacy. It must not be
mistaken for verification of this cohort. Final build, font, page-limit,
source-ZIP rebuild and supplementary replay checks are in SUBMISSION.md.

The author alone approves the manuscript, merges and submits. At the author's
request, the annotated paper-p3hpc-2026 tag now identifies the measured engine.
Paper code/report links use full revisions; prose was reviewed for repetitive
framing and authorial em-dashes while preserving the AI disclosure.
No GPU run, system change or portal upload is part of this update.
