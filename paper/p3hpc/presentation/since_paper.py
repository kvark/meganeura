#!/usr/bin/env python3
"""Numbers for the "Since the paper" slide: the camera-ready runs against this talk's.

The camera-ready paper is tag paper-p3hpc-work. Replay its September 25 runs from
the supplement with the tagged analyzer, each in a new, empty directory:

    unzip ../submission/meganeura-p3hpc-aux.zip -d AUX
    git show paper-p3hpc-work:paper/p3hpc/artifact/cohort.py > TOOL/cohort.py
    git show paper-p3hpc-work:paper/p3hpc/artifact/cohort.sha256 > TOOL/cohort.sha256
    python3 -I TOOL/cohort.py AUX/records.jsonl.xz --output PAPER --check AUX/tables
    python3 -I since_paper.py PAPER/conditions.csv

Standard library only. The talk's side is data/results.csv.
"""
import csv
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHARED = ["nvidia-5070", "nvidia-h100", "nvidia-3050", "amd-7900xt", "intel-b570", "apple-m3"]
STABLE = SHARED[:-1]  # Apple M3 times vary by tens of percent between processes.
MODELS = ["SmolLM2-135M", "SmolVLA", "StableDiffusion", "ResNet-50", "Whisper-tiny"]
PHASES = ["inference", "latency", "training"]


def number(text):
    return float(text) if text not in (None, "") else None


def main(paper_csv):
    paper = {(r["device"], r["precision"], r["model"]): r for r in csv.DictReader(open(paper_csv))}
    talk = {(r["archive"], r["model"], r["precision"], r["phase"]): r
            for r in csv.DictReader(open(HERE / "data" / "results.csv"))}

    def ratios(devices, precision, phase):
        old = [number(paper[d, precision, m][f"ratio_{phase}"]) for d in devices for m in MODELS]
        new = [number(talk[d, m, precision, phase]["ratio"]) for d in devices for m in MODELS]
        return [x for x in old if x], [x for x in new if x]

    for label, devices in (("six GPUs in both", SHARED), ("without the M3", STABLE)):
        for precision in ("strict", "accelerated"):
            cells = []
            for phase in PHASES:
                old, new = ratios(devices, precision, phase)
                cells.append(f"{phase} {statistics.median(old):.2f} -> {statistics.median(new):.2f}")
            print(f"{label:17s} {precision:11s} " + " | ".join(cells))

    print("\nStrict time change, October / September (median over workloads):")
    for engine, paper_key, talk_key in (("Meganeura", "meganeura", "native_ms"), ("PyTorch", "pytorch", "reference_ms")):
        for phase in PHASES:
            per_gpu, cells = [], []
            for d in SHARED:
                changes = [number(talk[d, m, "strict", phase][talk_key]) / number(paper[d, "strict", m][f"{paper_key}_{phase}"])
                           for m in MODELS
                           if number(talk[d, m, "strict", phase][talk_key]) and number(paper[d, "strict", m][f"{paper_key}_{phase}"])]
                per_gpu.append(f"{d} {statistics.median(changes):.3f}")
                if d in STABLE:
                    cells += changes
            print(f"  {engine:9s} {phase:9s} without M3 {statistics.median(cells):.3f}; " + ", ".join(per_gpu))

    print("\nMeganeura strict cells more than 5% slower than in September:")
    for d in SHARED:
        for m in MODELS:
            for phase in PHASES:
                old = number(paper[d, "strict", m][f"meganeura_{phase}"])
                new = number(talk[d, m, "strict", phase]["native_ms"])
                if old and new and new / old > 1.05:
                    print(f"  {d} {m} {phase}: {old:.3f} -> {new:.3f} ms")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
