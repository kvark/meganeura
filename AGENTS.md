# Kindle current-upstream qualification

Base this candidate on upstream 986f49a, including its timing integration,
windowed profiling and quantized-kernel fixes. Use Blade 92553493 with matched
allocation observability (2accfeee); no local timing-enabled accessor is needed.
Retain the alias-order/immediate-zero initialization hypothesis and flushed
observations from 070f4b51. It is not a proven historical fault fix. Check all
GPU waits: disabled timing is valid, but an incomplete fence or device error
must stop execution. Do not change learning math, precision or cooperative policy.

The September 16 user resumes bounded GPU work with NVML temporarily disabled.
Use fresh declarations, the direct-child host-only guard and actual-device
checks; review each GPU result before follow-up. No NVML, blind retry or host
recovery. Preserve completed sources, writers, raw evidence and the Pong hold.
CPU builds use fresh private targets, one CPU, 2 GiB and no swap. The full
library suite contains an unignored GPU test; use reviewed CPU module filters.
Main remains ce80. Full correctness, pixel/state/memory and same-backend block
throughput gates still precede adoption and held Pong work.
