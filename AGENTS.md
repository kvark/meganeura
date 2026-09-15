# Combined initialization control, CPU preparation only

This is the ce80 control with flushed initialization/constant-upload traces and
three checked waits. Preserve its allocation order, immediate Shared zeroing,
math, policies and settings. The parent failed bundle remains quarantined.

Prepare only CPU code and tests here. No GPU execution is authorized by this
branch or preparation. A direct initialization-only fixture needs a new pinned
declaration, source-matched Blade control, current upstream and host checks,
actual-device assertions, guard and at least 2 GiB directly free.
Never retry old writers or queues, or perform host recovery without approval.
