# Deferred methodology work

The September 13 cohort is final for this camera-ready preparation:
Inferena `efb1e520`, Meganeura `75dfe901`.
This file records follow-up work, **not an instruction to recollect**.
The [results guide](RESULTS.md) and manuscript describe the policies actually
measured. No existing observation has been relabeled as a proposed policy.

The convolution production gate is complete: immutable-parameter specialization
and alternative staging are ordinary bounded tuning candidates, qualified on
RTX 5070 and B570 and included in the final Meganeura pin.

The remaining author-review proposals were **not implemented in this protocol**:

- Permit qualified native-f32 cooperative tiles under strict arithmetic.
- Qualify whole-phase replay on ROCm and XPU, and compiled MPS. API availability
  alone is not model qualification; reuse the full-output/full-gradient gates.
- Compare bounded native search with default PyTorch compilation as a practical
  startup policy, or supervise a genuinely common preparation deadline.
  Present mode flags do not implement an equal-time-budget comparison.
- Broaden bounded search coverage and prioritize measured expensive classes.
  The same-pin NVIDIA diagnostic shows the current arithmetic-volume order
  misses small-output, long-reduction convolution hotspots.
- Measure comparable process/device peak memory, rather than comparing native
  plan sizes directly with PyTorch allocator high-water marks.

These opportunities motivate future work and explicit limitations in the paper.
They do not justify silently changing the final cohort, claiming unsupported
backends, or spending another multi-hour H100 search during submission work.
