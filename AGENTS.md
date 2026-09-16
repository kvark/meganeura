# Kindle timing candidate

Base this candidate on current upstream 5a570099, retaining packed-weight and
all other runtime fixes. Restore alias-order allocation with immediate Shared
zeroing after each creation, preserving the explicit parameter-zero opt-out.
This is distinct from quarantined 0a98775, which restores allocation order but
defers all host zeroing. The hypothesis is not a proven historical fault cause.

Use the same flushed initialization/upload/wait observations as control
9b9e7ee7, with the three fail-fast waits. Pin shared Blade current upstream plus
matched allocation observability. Do not change math, default precision, memory
placement, cooperative policy or the production graph. Main remains ce80.

The September 16 user resumes bounded GPU work with NVML disabled. Adapt the
runtime to current Blade last_timing, retain owned pass durations and record
borrowed timestamps immediately. Never call timing APIs on a disabled encoder.
Fail fast on unsuccessful waits; do not change learning arithmetic or dispatch
grouping. This candidate is separate from completed driver-580 comparisons.

Meganeura's full library test suite
contains an unignored GPU test; invoke only reviewed CPU module filters.
A separately declared timing-on/off native regression precedes the remaining
production hardware/state/pixel/throughput gates. Use fresh host/upstream checks,
actual-device assertions and the direct-child host guard. No NVML or blind retry.
Review each result before follow-up. No reset/reload/reboot or other recovery without
user approval. Preserve old writers, raw control/failure evidence and Pong hold.
