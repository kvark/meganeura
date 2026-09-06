# Corrected convolution tile crossover

## Predeclared correction, before measurement

This is a new cohort, not a replacement for the [original six attempts](../conv-tiles-2026-09-06/README.md).
Archive review found that the new small-case builder violated `Graph::conv2d`'s
documented flat NCHW operand contract: it supplied 4-D inputs and weights.
Forward compilation inferred batch from `shape[0]`, yielding zero workgroups
for the first convolution. Both sessions therefore had zero loss, gradients
and moments; advancing Adam's counter did not establish actual training.
Their original `complete` status reflects insufficient runner checks. Those
cases are disqualified as training/performance evidence. The ResNet case used
the proper flat layout and nonzero gradients; its six inconclusive records
remain separately interpretable. No raw files or evidence tags are rewritten.

Correct only the small-case operand shapes to flat arrays of identical declared
element counts. The public convolution helper now rejects nonflat/mismatched
operands early instead of silently accepting them; it does not add a new layout.
The full f64 oracle now checks every forward output as well as both derivatives.
The distinct-state convolution swap test uses proper flat operands and requires
a real parameter update. No tuning eligibility, shader arithmetic, candidates,
numerical tolerances, priorities, budgets or performance gates change.

Strengthen the runner: reject any zero-workgroup convolution cohort plan before
the prefix; require nonzero loss and gradient norms after the matched prefix;
require actual parameter changes in both optimizer-backed sessions by the end.
Keep full tensor/moment/counter comparisons, exact search/swap isolation and
all timed loss checks. Full source and this correction are committed before
measurement, with a new immutable evidence tag and executable identity.

Retain six fresh serial processes, seeds 1..6, running the same ordered three
cases (ResNet F+L+B, two-layer convolution Adam, same graph SGD), initializers,
strict-f32 policies, eight-class/64 MiB/ten-second search and role-reversed
whole-step protocol documented in the original predeclared protocol. The
same 5% plus twice-paired-MAD guard applies. No further reruns, case changes,
budget expansion or threshold fitting after these results. If a new validity
issue arises, retain/disqualify it rather than silently repairing its records.

Use the same [Blade 0.9 lock](../readback-2026-09-06/Cargo.lock), SHA-256
`72cecddd0f090e82d5db46bc6f7f80429a058b2afa0eaec84a9b27e7340a4a80`.
Build before measurements; no other build, qualification or heavy host analysis
runs during the cohort. Telemetry, create-new incremental records, full raw
dispatch contracts and all evidence limitations remain unchanged.

```sh
cargo build --release --locked --example tune_crossover
for seed in 1 2 3 4 5 6; do
    target/release/examples/tune_crossover "new-corrected-conv-${seed}.json" "$seed" --conv-derivatives
done
```
