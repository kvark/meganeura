# Corrected-protocol acceptance: complete

Closed September 14, 2026 by the final v9 cohort at Inferena fa5a04e1 /
Meganeura 428fc2d2. The recovered author feedback and all three reviewer
notes have been checked against executed records. The v7 paper packages
remain superseded; the current manuscript and handoff now use v9.

| Requirement | Executed evidence |
|---|---|
| Always tune Meganeura | All 252 pairs, 708 sessions report measured search independently of reference mode. |
| Strict native-f32 cooperative tiles | Every strict receipt permits NativeF32 and forbids f16 inputs; M3 actually uses cooperative dispatches in all five workloads. |
| Broaden legal search | All scope, no class cap, 1 GiB scratch with device guard, 60 s soft deadline. 684 sessions fully cover their eligible classes; 24 ResNet sessions reach the deadline. |
| Bound PyTorch preparation | Default compilation, no max-autotune, enforced 120 s watchdog in every record. All finish under the limit. |
| Replay on applicable backends | CUDA/HIP/XPU capture and full-element replay qualification pass all requested workloads and both contracts, including Windows and large H100 models. |
| MPS compilation and timing | All 30 M3 pairs report compilation, first specializations and nonzero compile time. No public equivalent whole-phase replay API is assumed. |
| Reduce cohort size | 30 paired processes per main device, plus 12 large-model H100 pairs; no duplicate qualification cohort or searched arm. |
| Explicit numerical acceptance | V9 fixed per-output maximum/RMS bounds are recorded and replayed. Cross-engine and replicated-gradient gates pass. |
| Production convolution choices | Included in the measured Meganeura pin and searched on NVIDIA, AMD, Intel and Apple. |
| Honest memory and attribution | Plan sizes and allocator peaks stay distinct; missing H100 NVML telemetry is disclosed; no invented barrier percentage. |

The v9 output gate is an intentional change from v8's near-zero-sensitive
pointwise rule: maximum and RMS errors must each satisfy fixed 1e-4 / 1e-6
bounds. No tolerance is fitted to the run. Pointwise mismatch counts remain
diagnostics. The failed v8 AMD Whisper run was recollected under v9, not
relabelled or retrospectively accepted.

XPU uses qualified dense-index-add embedding backward and the public math
SDPA setting for capture compatibility. Those accommodations remain in the
paper and metadata. They are not silent eager fallback or disabled replay.

[RESULTS.md](RESULTS.md) records numerical outcomes and timing variability.
[REVISION.md](REVISION.md) maps reviewer/author concerns to the manuscript.
[SUBMISSION.md](SUBMISSION.md) describes final files and author upload checks.
No further cohort is needed to support the paper's stated scope; broader
search domains, neural accelerators and megakernels are genuinely future work.
