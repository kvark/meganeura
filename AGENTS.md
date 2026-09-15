# Combined-initialization hypothesis only

Base this candidate on current upstream 5a570099, retaining packed-weight and
all other runtime fixes. Restore alias-order allocation with immediate Shared
zeroing after each creation, preserving the explicit parameter-zero opt-out.
This is distinct from quarantined 0a98775, which restores allocation order but
defers all host zeroing. The hypothesis is not a proven historical fault cause.

Use the same flushed initialization/upload/wait observations as control
9b9e7ee7, with the three fail-fast waits. Pin shared Blade current upstream plus
matched allocation observability. Do not change math, default precision, memory
placement, cooperative policy or the production graph. Main remains ce80.

Prepare CPU source/fixture checks only. Meganeura's full library test suite
contains an unignored GPU test; invoke only reviewed CPU module filters.
A separately declared combined LeVJEPA/first-world initialization test is the
next possible GPU boundary, never a training or full pixel retry. Use fresh
host/upstream checks, actual-device/memory gates and direct-child guard, with
no automatic successor. No reset/reload/reboot or other host recovery without
user approval. Preserve old writers, raw control/failure evidence and Pong hold.
