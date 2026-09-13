# Corrected-protocol acceptance checklist

The author's recovered feedback (`~/Downloads/p3hpc-v2/feedback.txt`) is the
requirements source, together with the original review in the conversation.
The former decision to defer these protocol changes was incorrect. The v7
data (Inferena `efb1e520`, Meganeura `75dfe901`) remain valid observations of
their declared conditions, but do not implement the requested comparison.
The current paper tables and camera-ready packages still describe v7 and
must not be uploaded as the corrected study.

Collection instructions live in Inferena's `EXPERIMENT.md` on
`experiment/p3hpc-cuda-graphs`. This checklist tracks implementation and
evidence separately; a mode flag or successful generic test is not proof
that a full workload used the requested path.

| Requirement | Implementation / acceptance evidence |
|---|---|
| Always tune Meganeura, independently of PyTorch | Collector enables it for every pair; each actual session reports policy, search coverage, decisions and cost. Missing/disabled receipts fail. |
| Strict permits native-f32 cooperative tiles | Meganeura merged `428fc2d` adds `NativeF32`, filtering both planning and runtime capabilities. Scalar f32 remains the baseline; f16-input kernels stay forbidden. Local GPUs have no native-f32 tiles, so positive hardware use requires the Mac receipt. |
| Explore the legal search space | Remove eight-class cutoff; include current dense/convolution domains; raise scratch ceiling to 1 GiB while retaining the device-memory guard. CUDA ResNet covers all classes under the 60-second/session ceiling; B570 can exhaust it. Report coverage, not exhaustive search. |
| Stop multi-hour reference preparation | Default compiled reference without max-autotune; 120-second first-specialization watchdog kills compiler descendants and retains failure evidence. Optional max-autotune uses the same limit. No automatic eager fallback. |
| Replay on every applicable backend | CUDA/HIP `CUDAGraph` and XPU `XPUGraph` use one preparation/run stream and unchanged full-tensor qualification. CUDA and XPU pass all five models in both arithmetic classes and small live-input/weight replay checks. ROCm and Windows hardware qualification is still required. |
| MPS compilation and timing | Remove eager bypass; compile requested forward/backward/minimal phases and include synchronized first specializations in `compile_s`. Routing/failure checks pass locally, but a Mac must qualify actual execution. MPS has no equivalent public whole-phase replay API. |
| Reduce collection time | Primary cohort is 30 paired processes per device; graph ablation and max-autotune are opt-in. No duplicate qualification campaign is required before every measurement. |
| No weakened validation | Retain v7 cross-engine/replicated-gradient and fixed full-gradient replay gates; malformed execution evidence is rejected. |
| Production convolution choices | Already merged and included in the pin. Search legal shape/staging alternatives; report invalid candidates and uncovered classes rather than treating heuristics as measurements. |
| Honest memory and timing interpretation | Existing distinct native allocation/Torch allocator/MPS endpoint measurements retained; no fictitious common peak-VRAM or subtraction-based barrier metric. |

## Paper acceptance after collection

- Recompute tables, preparation costs and the SmolLM2 figure from one v8 source;
  retain incomplete conditions as operational findings, not favorable retries.
- State actual platform failures and successful native execution directly;
  distinguish driver support, compile failure, budget exhaustion, and omitted
  experiments. Do not reintroduce superseded Windows or ROCm failure anecdotes.
- Keep H100 as an unoptimized transfer/scaling test, not a tuned target. Collect
  360M/1.7B only on the cloud GPU, and disclose replication and residency limits.
- Keep the shared graphics-context use case, expanded abbreviations, readable
  grouped table headings, observability and native-profile attribution.
- Add neural accelerators and persistent megakernels as genuine future work;
  they are now in the draft. Required benchmark corrections are not future work.
- Replace the v7 limitations and artifacts only when new executed evidence
  supports the replacement. Retain the cache-history threat to validity.
- Rebuild and inspect the IEEE-format PDF and independent source ZIP. Prefer
  <2 MB and stay below the portal's 4 MB rejection limit. Supplementary material
  needs ZIP, not the existing 23 MB evidence tarball; raw evidence needs hosting.

No new collection tag or cloud run has been created during protocol repair.

## Local qualification findings (September 13)

CUDA completes all five models in both arithmetic classes with native tuning
independent of default-compiled graph replay. The initial 10-second native
ceiling covered all non-ResNet classes locally, but only 22/71 strict and
25/59 accelerated ResNet training classes. A 60-second ceiling reached every
ResNet class in 29/24 seconds and passed the cross-engine gates. This is
qualification, not replicated speedup evidence; actual native preparation
can exceed default PyTorch compilation on ResNet.

XPU ResNet completes compiled whole-phase replay. Native strict search visits
51/71 training classes before its 60-second deadline on the B570's present
host/PCIe configuration; full native preparation is 89 seconds versus 39
seconds for PyTorch compilation. Other native sessions finish early. Do not
claim every search is exhaustive or faster than default reference compilation.

XPU SmolLM2's default fused attention fails inside capture with an event-wait
error after successful compilation and ordinary execution. An isolated
grouped-query attention reproducer fails likewise; the public PyTorch math
SDPA setting passes complete forward/backward replay validation. Inferena
records that setting explicitly for XPU, including its uncaptured control,
and all ten model/arithmetic pairs pass full qualification. This is a reference-stack
graph-compatibility workaround, not disabling replay or relaxing a gate.

Final-default CUDA ResNet qualification passes at Inferena `1430d0d`; the full
XPU qualification at `d8335a8` also passes that revision's receipt checker.
Six broad execution/contract tests pass with each local vendor wheel, together
with nine Rust harness tests. No qualification output is publication data.

The Mac must still demonstrate real MPS compilation timings and native-f32
cooperative use, ROCm must qualify the new replay path, and Windows must check
the new compilation/watchdog workflow before an expensive common-source
collection is released. The local routing/receipt tests cannot substitute for
those hardware checks.
