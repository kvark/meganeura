# Replay methodology and final outcomes

The frozen final source is Inferena fa5a04e1; [RESULTS.md](RESULTS.md)
identifies the observations. Compiler graph capture and command-graph replay
are different; the latter is explicitly qualified here.

| Backend | Primary reference execution |
|---|---|
| CUDA, including Windows | Default compilation, whole-phase torch.cuda.CUDAGraph replay |
| ROCm | Default compilation, HIP graph replay through torch.cuda.CUDAGraph |
| XPU | Default compilation, math SDPA, whole-phase torch.xpu.XPUGraph replay |
| MPS | Default compilation of all requested phases; no equivalent public whole-phase replay API |
| CPU support control | Explicit default-compiled CPU execution |

Meganeura always performs bounded measured kernel selection during session
construction. Inductor's internal triton.cudagraphs is disabled, leaving one
explicit replay owner. Preparation and execution share one dedicated stream
on CUDA/HIP/XPU. XPU's embedding-backward and attention accommodations are
recorded; they are not an unmodified native XPU path.

Full inference, minimal forward and forward/loss/backward are captured
separately. Qualification compares every participating PyTorch output and
gradient element with the first uncaptured result: two ordinary repeats
(eight for accelerated training) and two replays. V9 uses fixed per-output
maximum/RMS bounds, strict per-parameter and whole-gradient bounds, and
accelerated whole-gradient bounds. These are not fitted to repeat noise
and are distinct from cross-engine sampled-output / gradient-norm checks.

Five warmups precede twenty synchronized samples. Loading, readback,
compilation, capture and validation are outside steady-state samples.
Default compilation has an enforced 120-second first-specialization deadline;
graph preparation and research qualification are timed separately.
No failure substitutes eager or CPU timings.

All 240 main pairs and 12 H100 extension pairs pass. Both larger models
have three processes under both arithmetic contracts. No final Windows or
H100 crash is carried over from preceding collections.

Each process has an empty private Inductor/Triton cache. Persistent driver
and vendor-library state remains as found: disabling max-autotune does not
guarantee no reuse of lower-level choices. This remains a threat to validity,
not a reason to change or repeat the completed protocol.

The main paper separately labels an H100 search pilot and RTX profiles at
older revisions/policies. Their observations are not current replay ablations
or replacements for the final timing matrix.
