#!/usr/bin/env python3
"""Compare CI latency benchmark results against a baseline.

Reads bench/ci_latency_baseline.json (checked in) and compares against
a fresh run. Exits non-zero if any metric regresses beyond the threshold.

Usage:
  python3 bench/check_latency.py bench/results/ci_latency.json
"""

import json
import os
import sys

BASELINE_PATH = os.path.join(os.path.dirname(__file__), "ci_latency_baseline.json")
# Allow 20% regression before failing (lavapipe is noisy)
THRESHOLD = 0.20

METRICS = [
    ("forward_median_ms", "Forward median"),
    ("train_step_median_ms", "Train step median"),
]


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <results.json>")
        sys.exit(1)

    results_path = sys.argv[1]

    if not os.path.exists(BASELINE_PATH):
        print(f"No baseline found at {BASELINE_PATH} — saving current results as baseline.")
        with open(results_path) as f:
            data = json.load(f)
        with open(BASELINE_PATH, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        print("Baseline created. Re-run to compare.")
        sys.exit(0)

    with open(BASELINE_PATH) as f:
        baseline = json.load(f)
    with open(results_path) as f:
        current = json.load(f)

    print(f"{'Metric':<24} {'Baseline':>12} {'Current':>12} {'Change':>10} {'Status':>8}")
    print("-" * 70)

    regressions = []
    for key, label in METRICS:
        base_val = baseline.get(key)
        curr_val = current.get(key)
        if base_val is None or curr_val is None:
            print(f"{label:<24} {'N/A':>12} {'N/A':>12} {'':>10} {'SKIP':>8}")
            continue

        if base_val > 0:
            change = (curr_val - base_val) / base_val
        else:
            change = 0.0  # no baseline yet, skip regression check

        status = "OK"
        if base_val == 0:
            status = "NEW"
        elif change > THRESHOLD:
            status = "REGRESS"
            regressions.append((label, base_val, curr_val, change))
        elif change < -THRESHOLD:
            status = "FASTER"

        print(
            f"{label:<24} {base_val:>10.2f}ms {curr_val:>10.2f}ms "
            f"{change:>+9.1%} {status:>8}"
        )

    print()
    if regressions:
        print(f"FAIL: {len(regressions)} metric(s) regressed beyond {THRESHOLD:.0%} threshold:")
        for label, base, curr, change in regressions:
            print(f"  {label}: {base:.2f}ms -> {curr:.2f}ms ({change:+.1%})")
        sys.exit(1)
    else:
        print("PASS: no latency regressions detected.")


if __name__ == "__main__":
    main()
