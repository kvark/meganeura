# Matched Vulkan GGUF diagnostic

`gguf_latency.rs` and `gguf_latency.cpp` compare the same GGUF tensors and
token IDs through Meganeura and llama.cpp. This is a kernel/runtime diagnostic,
not a replacement for the frozen Inferena paper cohort or a language-quality
evaluation.

Both use a 128-token prefill, 32 subsequent cached decode steps, context capacity
256, f32 K/V caches, three warmups and seven measured sequences. Prefill produces
only the last row of logits. Every measured call includes submission, waiting,
and copying the full vocabulary's logits to CPU memory. The first measured
sequence saves 33 logit vectors for comparison. llama.cpp uses Vulkan with all
layers offloaded and flash attention enabled; it refuses a missing Vulkan device.
The helpers intentionally do not tokenize text.

Build against a recorded llama.cpp checkout:

```sh
cargo build --release --features gguf --example gguf_latency
cmake -S ../llama.cpp -B ../llama.cpp/build -DGGML_VULKAN=ON
cmake --build ../llama.cpp/build -j2
c++ -O2 -std=c++17 bench/gguf_latency.cpp \
  -I ../llama.cpp/include -I ../llama.cpp/ggml/include \
  -L ../llama.cpp/build/bin -Wl,-rpath,"$PWD/../llama.cpp/build/bin" \
  -lllama -lggml -lggml-base -o /tmp/llama-latency

# Select exactly one GPU's ICD. For Intel use intel_icd.json instead.
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export GGML_VK_VISIBLE_DEVICES=0
MEGANEURA_COOP_F16=1 target/release/examples/gguf_latency model.gguf /tmp/meg 30
/tmp/llama-latency model.gguf /tmp/llama
```

The optional final argument bounds tuning to 30 seconds **per session**;
the diagnostic allows 256 MiB of scratch so the vocabulary projection is not
silently excluded by the default 64 MiB limit. Reports record candidates,
qualification, selections, preparation time and CPU record/wait/readback stages.
No precision tolerances are changed. Run engines and GPUs sequentially, without
profilers or builds competing with the timed runs. Repeat in fresh processes
and reverse their order. First-read mapped/staged qualification is absorbed
by warmup, not included in steady-state latency.

For separate attribution captures, add `MEGANEURA_GPU_TIMING=1` to the Meganeura
command or `GGML_VK_PERF_LOGGER=1` to llama.cpp. Profiled numbers are not substitute
latencies: instrumentation changes execution, and Meganeura's pass intervals
include the transition to the next pass. In particular, wall time minus their
sum is **not** a measurement of barrier cost.

## September 19, 2026 checkpoint

Source: Meganeura `4195be1b869cbb8b9eaf4b0403c3f48875d1e675`, based on
`da0e28424395904ae9a8cb53e7da32e15012ec1a`. llama.cpp:
`05f2dcfdba3879c55f735efa0f124b1a56f7ed11`. The original unfinished experiments
are preserved separately at `experiment/llama-prototypes-2026-09-17`
(`78e2ff448030fdbd7932cd06c76ae282bcef6542`), not in this branch's ancestry.

Model: `HuggingFaceTB/SmolLM2-135M` at
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, converted with the recorded
llama.cpp revision's official converter:

```sh
hf download HuggingFaceTB/SmolLM2-135M \
  --revision 93efa2f097d58c2a74874c7e644dbc9b0cee75a2 --local-dir hf-smollm2
# Install llama.cpp's converter requirements in a separate venv first.
python ../llama.cpp/convert_hf_to_gguf.py hf-smollm2 \
  --outtype f16 --outfile smollm2-official-f16.gguf
```

GGUF SHA-256:
`56df427e4aa9a57d67b207d45e1a2b51e80b956b11cd9fbf0c162429f95168c7`.
All 272 tensors were checked against the HF source after the converter's Q/K
permutation and storage conversion. An inherited local GGUF lacked that
permutation; its preliminary results were discarded, not mixed into this table.

The split-attention combine on the upstream base multiplied a partial-row
stride twice. Correcting that and clearing empty partials restored all 33
next-token predictions against llama.cpp on both GPUs. The regression exercises
multiple heads, query rows, cache reuse and a split context; the submitted paper's
older Inferena path is not this newly added GGUF path.

An independent CPU f32 eager run of the pinned HF model produced the same 33
next-token choices. Maximum per-row relative L2 logit errors against it were
0.0000093 (Meganeura) and 0.01121 (llama.cpp) on the 5070, and 0.000175 and
0.01314 on the B570. Matching weight storage and f32 caches does not guarantee
matching intermediate arithmetic. Top-1 agreement on these fixed tokens is a
sanity check, not a language-quality evaluation.

### Unprofiled latency

Milliseconds; median of three fresh-process medians, each with seven prefill
and 224 decode observations. Process order was rotated. The control is
`d0bb9669bab48942a3ab873e38911090e27e9002`, including the necessary attention
fix; timing incorrect upstream outputs would not be a useful baseline.

| GPU | Phase | Corrected control | This branch | llama.cpp |
| --- | --- | ---: | ---: | ---: |
| RTX 5070 | Prefill, 128 tokens | 13.47 | 11.57 | 7.12 |
| RTX 5070 | Decode, per token | 5.83 | 4.35 | 1.38 |
| Arc B570 | Prefill, 128 tokens | 86.39 | 27.62 | 15.65 |
| Arc B570 | Decode, per token | 22.81 | 6.18 | 3.45 |

This is not parity: the remaining decode ratios are 3.16x and 1.79x.
The B570 control was variable (61.26--86.64 ms prefill, 15.17--22.85 ms decode);
the new branch was steadier (27.62--27.64 and 6.16--6.18 ms). CPU/GPU clocks were
not fixed. Both engines used the same six physical i5-12400F cores, pinned with
`taskset -c 0,2,4,6,8,10`; the CPU retained its powersave governor. NVIDIA driver:
595.91.07. The secondary B570's reported link was PCIe 2.5 GT/s x1, so its
host-readback numbers must not be generalized to a full-bandwidth installation.

The branch's median preparation times, including model loading, both sessions
and tuning, were 3.45 s (5070) and 9.79 s (B570). llama.cpp took 0.26 and 0.50 s.
Neither Meganeura session exhausted its 30 s tuning budget. The control used
the previous 64 MiB tuning scratch cap, which omitted the vocabulary GEMV;
the branch allowed 256 MiB. These are process starts with existing driver caches,
not cache-cold compilation measurements.

### What changed, and what remains

Mapped VRAM was the main avoidable host-read cost. The runtime now measures direct
and staged reads, checks bit-exact agreement, caches the choice per allocation
and byte count, and reuses at most 16 MiB of download staging. Dense F16 and packed
weights can also enter the existing scalar tile search with unchanged decoding
and numerical qualification.

The 5070's median CPU-side readback stage fell from 3.20 to 0.37 ms. The B570's
fell from 13.78 to 0.51 ms. The whole-call improvements are smaller than these
differences suggest: CPU record/submit rose from 0.37 to 1.71 ms on NVIDIA and
0.62 to 1.36 ms on Intel, while Intel's wait stage also changed. These overlapping
CPU/GPU stages under variable clocks are not an additive causal gap budget.

Separate instrumented Meganeura captures locate the remaining GPU work:

| GPU / phase | Matrix pass intervals | Cached-attention intervals | Other intervals |
| --- | ---: | ---: | ---: |
| 5070 / decode | 51% | 30% | 19% |
| 5070 / prefill | 57% | 35% | 8% |
| B570 / decode | 39% | 31% | 30% |

Decode still issues 394 dispatches in 274 dependency groups. Cached attention
uses 30 split/combine pairs. llama.cpp's own Vulkan timestamp logger reports
about 0.18 ms for its 30 decode attention operations on NVIDIA, versus about
0.98 ms in Meganeura's attention pass intervals. The instrumentation differs,
so those numbers identify a kernel target, not a directly subtractable barrier
cost. An Nsight Systems 2025.5.2 Vulkan/OS-runtime capture of Meganeura completed.
The llama.cpp injection exited abnormally; that capture was excluded from claims.
Uninstrumented runs and llama.cpp's own timestamp logger completed on both GPUs.

Cooperative matrices remain a concrete implementation gap. This B570 advertises
8x16x16 f16-input/f32-accumulator tiles; Blade's capability filter and pinned
Naga WGSL frontend accept only square 8x8 or 16x16 tiles. llama.cpp reports
`KHR_coopmat` there, and `NV_coopmat2` on the 5070. Meganeura also excludes
reduced-storage weights from its cooperative matmul path. Scalar tile tuning
does not resolve either limitation. Cooperative coverage is a prefill target;
decode additionally needs better cached attention, GEMV and host submission.
Sharing Vulkan alone does not make these kernel implementations equivalent.

A tilewise softmax-rescaling trial is retained at
`experiment/cached-attention-tile-softmax-2026-09-19`
(`bd25688a585ee6e093ea0beb8ae98edfe56fca88`), not included in the production
branch. It passed the numerical checks but gave only a small NVIDIA prefill
gain (11.64 to 11.17 ms) and no Intel gain in three reversed-order pairs.

### Fixed-head attention follow-up

`dd9f56f9f9d334f791ce64ae7b0163d6ed631058` compiles cached attention at the graph's
known head dimension, using the existing per-head pipeline mechanism. It removes
unused per-thread values and dynamic head-width branches without changing the
algorithm or precision.
Three fresh-process pairs, reversing order, compared it with `5aeb856`:

| GPU | Phase | Before | Fixed head width |
| --- | --- | ---: | ---: |
| RTX 5070 | Prefill | 11.58 | 8.82 |
| RTX 5070 | Decode | 4.29 | 3.20 |
| Arc B570 | Prefill | 27.63 | 22.79 |
| Arc B570 | Decode | 6.09 | 5.66 |

Units and sampling are unchanged. All 33 token predictions match the independent
CPU reference on both devices; maximum per-row relative L2 error remains below
0.000010 on NVIDIA and 0.000178 on Intel. The existing cache regression now also
covers head widths 64, 80 and 512, including partially occupied thread lanes.
NVIDIA attention pass intervals fell from 0.98 to 0.49 ms for decode and from
3.80 to 1.66 ms for prefill in separate captures. This is still not parity with
llama.cpp; matrix kernels and host submission remain substantial costs.

### Serialized command replay

The follow-up pins Blade `4befad4f5fbd427c1aca4b1b001fb8b7acd8e109` and reuses
an unchanged Vulkan inference recording after waiting for its previous execution.
Input contents remain dynamic. Rebinding, tuning, profiling, changing submission
chunks, and other uses of the encoder invalidate the recording. Training,
timestamped sessions, Metal and GLES keep ordinary recording. The normal
`Session::step` path selects this automatically; no benchmark-specific toggle
or numerical change is involved.

Three fresh-process trials per engine, with rotated order and the same diagnostic:

| GPU | Phase | Fixed head, recording | Fixed head, replay | llama.cpp |
| --- | --- | ---: | ---: | ---: |
| RTX 5070 | Prefill | 8.65 | 7.23 | 7.11 |
| RTX 5070 | Decode | 3.13 | 1.82 | 1.38 |
| Arc B570 | Prefill | 22.89 | 21.31 | 15.61 |
| Arc B570 | Decode | 5.64 | 4.48 | 3.42 |

The NVIDIA recording control varied from 2.28 to 3.15 ms per decode; replay
ranged from 1.81 to 1.85 ms. Intel replay ranged from 4.45 to 4.54 ms. CPU clocks
remain uncontrolled. The CPU record/submit stage fell from 1.26 to 0.040 ms on
NVIDIA and from 1.42 to 0.044 ms on Intel. Replay does not remove GPU barriers
or change kernel arithmetic. All 33 next-token choices still match the CPU
reference; maximum relative L2 errors are 0.0000102 and 0.000177 respectively.
Median preparation remains 3.46 s and 9.79 s, including tuning.

NVIDIA prefill is now within 2% of llama.cpp on this case. Decode remains about
32% slower on NVIDIA and 31% on Intel; Intel prefill remains 37% slower.
This is not general parity and does not establish an improvement in the paper's
training workloads. Keep the frozen paper cohort unchanged until the remaining
kernel work and measurements on the paper's own graphs justify recollection.

The timing-only precursor is preserved on `experiment/llama-replay-2026-09-19`
in Meganeura (`5af8494`) and Blade (`1b4157a`). CPU-stage instrumentation is on
Meganeura `experiment/llama-cpu-record-2026-09-19` (`29e3cd4`). These source-only
experiments are not part of the production branch's ancestry.

### Verification

Formatting and all-target/all-feature Clippy passed. CPU unit tests: 449 passed,
three ignored. Readback bit preservation, cached-attention reference checks and
the opt-in reduced-storage tile-tuning regression passed on both GPUs. The
existing cache regression now includes nonuniform scores, sliding windows and
long-to-short cache reuse rather than adding another test binary.

NVIDIA's GGUF-feature regression/smoke run passed 183/81 tests, with 13 ignored.
On this driver, repeated context destruction/recreation eventually fails with
`ERROR_INCOMPATIBLE_DRIVER`, including on unchanged main. Keeping the ICD loaded
with a process-local `LD_PRELOAD=libGLX_nvidia.so.0` made the suite pass. This is
a test workaround, not a production global-context cache or a profiling setting.
Intel's broader model-feature run passed 184 regression and 94 smoke tests;
one pairwise-distance gradient comparison failed by 2.1e-7 and reproduced on
unchanged main with the same dependency lock. Its tolerance was not weakened.

Raw timings, logits, profiles and binaries stay outside Git. Do not update the
paper or launch another full cohort on the strength of this one model and shape.
In particular, the frozen Inferena cohort times GPU-resident work; these
host-readback savings do not automatically improve that timed window.
