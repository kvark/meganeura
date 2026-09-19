# Cached-attention sequence tuning

Implementation: `861d87b9a85f666a5a131f82d07c0b66d9131e16`, based on
`52027a8fc9e243a50039967b4d0dee0c51a05897`. Ordinary command recording throughout.
Model, llama.cpp revision, hardware and sampling follow [the matched GGUF
diagnostic](gguf_latency.md). This does not change the frozen Inferena cohort.

The existing tuner now compares complete attention implementations: one dispatch,
or split/combine at 2/4/8/16 splits. The original count is retained as a control,
including non-power-of-two counts. It uses the same deadline, scratch cap,
alternating trials, qualification and noise/improvement guards as matrix search.
The candidate set contains no device names or model-specific thresholds.

Private scratch matches binding placement. Qualification checks two nonzero
input patterns, including tiny f32 values, against sampled f64 attention
references and full-output cross-variant/finite checks. Long-to-short reuse
also checks empty partials. No live KV state is read or modified. The allocation
planner keeps external bindings live across the entire operation so either
implementation is legal. Partial growth preserves old allocation bytes; repeated
retuning retains buffer identities. Original barrier boundaries remain intact.

## Three-process comparison

Milliseconds; median of three fresh-process medians, with rotated engine order.
Both Meganeura sessions allow 30 seconds, 64 classes and 256 MiB of scratch.
`dense` selects only the existing matrix search, isolating the lifetime/plumbing
change from the new attention choices. All saved logits are checked below.

| GPU | Phase | Before | New lifetime plan, dense search | Full search | llama.cpp |
| --- | --- | ---: | ---: | ---: | ---: |
| RTX 5070 | Prefill | 8.794 | 8.816 | 8.758 | 7.087 |
| RTX 5070 | Decode/token | 3.131 | 3.130 | 2.973 | 1.376 |
| Arc B570 | Prefill | 22.825 | 22.752 | 21.287 | 15.649 |
| Arc B570 | Decode/token | 5.595 | 5.612 | 5.506 | 3.438 |

Every run selected 16 decode splits on NVIDIA and unsplit prefill on Intel.
NVIDIA prefill and Intel decode retained eight splits. Intel prefill removes
30 combine dispatches (454 to 424); decode remains 394 dispatches on both GPUs.

NVIDIA decode improves about 5% by the median, but one process regressed:
full-search medians span 2.970--3.239 ms, versus 3.127--3.137 ms before.
CPU recording, waiting and readback stages vary too; clocks were not fixed.
Do not present this as a uniformly repeatable whole-call win or subtract these
overlapping stages to infer GPU barrier cost. Intel prefill improves in all
three trials: 21.257--21.347 ms versus 22.805--22.839 ms, about 7% by the median.
Intel decode's small difference is not an attention-selection gain.

The attention comparisons themselves cost about 0.46 seconds across the two
NVIDIA sessions and 1.40 seconds on Intel. Median total preparation, including
loading, session setup and all tuning, is 3.93/11.21 seconds versus 3.46/9.83
before. No session exhausted its tuning budget. These are process starts with
existing driver caches, not cache-cold compilation measurements.

All 33 saved next-token choices match the independent CPU f32 reference in all
24 process runs. Maximum per-row relative L2 error for full search is 0.00000994
on NVIDIA and 0.000177 on Intel. Matching these tokens is a numerical sanity
check, not a language-quality evaluation or proof of identical intermediate
precision across engines.

The gap is not closed: whole-call decode remains 2.16x/1.60x llama.cpp and
Intel prefill 1.36x. Further work should address matrix/GEMV layout, fusions and
cooperative-matrix coverage, not reintroduce command replay.

## Scope and verification

The search chooses one geometry per shape/placement class, using the mean cost
at short, middle and full cache positions. It does not adapt on each token or
know an application's position distribution. Matrix classes are visited first;
both families share the same class/time budget. Reports expose visited/excluded
classes and incomplete searches. Use `TuneScope::Attention` to isolate this axis,
or the diagnostic's optional `all|dense|attention` final argument.

453 CPU tests pass. The existing cached-attention regression covers live-output
preservation and report round-tripping on both GPUs. An opt-in test forces
repeated 1/2/4/16 transitions, partial growth, profiling-window remapping and
stable allocation footprint on both GPUs. No new test executable or tolerance
relaxation. All six CI jobs passed for the measured revision in
[run 737](https://github.com/kvark/meganeura/actions/runs/35467384352).
The known Naga Workgroup ArrayStride validation warning remains unchanged.
Raw timings, logits and binaries stay outside Git.
