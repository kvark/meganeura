#!/usr/bin/env python3
"""Account for the strict/accelerated gap on identical qualified core cells.

Standard-library only. Geometric means preserve the per-cell identity
R_acc = R_strict * (PyTorch speedup / Meganeura speedup). These are an
accounting of recorded policies, not a causal CPU/GPU or API decomposition.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
DEVICES = {"nvidia-5070": "RTX 5070", "nvidia-h100": "H100", "nvidia-3050": "RTX 3050",
           "amd-7900xt": "RX 7900 XT", "amd-mi300": "MI300X", "intel-b570": "Arc B570", "apple-m3": "Apple M3"}
MODELS = ("SmolLM2-135M", "SmolVLA", "StableDiffusion", "ResNet-50", "Whisper-tiny")


def geometric(values):
    return math.exp(statistics.mean(math.log(v) for v in values))


def analyze(rows):
    lookup = {(r["archive"], r["model"], r["precision"], r["phase"]): r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError("duplicate cohort condition")
    cells, groups = [], []
    keys = ("strict_ratio", "accelerated_ratio", "torch_gain", "native_gain", "gap_multiplier")
    for phase in ("inference", "training"):
        phase_cells = []
        for device in DEVICES:
            selected = []
            for model in MODELS:
                strict, accelerated = (lookup[device, model, p, phase] for p in ("strict", "accelerated"))
                # A retained time alone does not establish an eligible comparison.
                if not strict["ratio"] or not accelerated["ratio"]:
                    continue
                ns, ps = float(strict["native_ms"]), float(strict["reference_ms"])
                na, pa = float(accelerated["native_ms"]), float(accelerated["reference_ms"])
                cell = dict(device=device, model=model, phase=phase, strict_ratio=ns / ps,
                            accelerated_ratio=na / pa, torch_gain=ps / pa, native_gain=ns / na)
                cell["gap_multiplier"] = cell["torch_gain"] / cell["native_gain"]
                if not math.isclose(cell["accelerated_ratio"], cell["strict_ratio"] * cell["gap_multiplier"], rel_tol=1e-12):
                    raise ValueError("gap identity does not close")
                selected.append(cell)
            if selected:
                groups.append(dict(device=device, phase=phase, n=len(selected),
                                   **{k: geometric([c[k] for c in selected]) for k in keys}))
            phase_cells.extend(selected)
        groups.append(dict(device="all", phase=phase, n=len(phase_cells),
                           **{k: geometric([c[k] for c in phase_cells]) for k in keys}))
        cells.extend(phase_cells)
    return dict(statistic="geometric means over identical qualified core conditions; not the main-table medians",
                interpretation="accounting identity, not causal attribution to hardware, driver, CPU or GPU",
                groups=groups, conditions=cells)


def followups(path):
    data = json.loads(path.read_text())
    for row in [*data["conditions"], *data.get("auxiliary_conditions", [])]:
        for key in ("baseline", "candidate"):
            samples = row[key + "_samples_ms"]
            if len(samples) != 100 or any(not math.isfinite(x) or x <= 0 for x in samples):
                raise ValueError("invalid follow-up sample inventory")
            # The diagnostic runner uses the upper median for an even count.
            if sorted(samples)[len(samples) // 2] != row[key + "_ms"]:
                raise ValueError("follow-up median differs from retained samples")
        if row["qualification_before"] != {"Ok": None} or row["qualification_after"] != {"Ok": None}:
            raise ValueError("follow-up failed qualification")
        row["change_percent"] = 100 * (row["candidate_ms"] / row["baseline_ms"] - 1)
    return data


def render(rows, local):
    summary = analyze(rows)
    summary["input_sha256"] = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    lookup = {(r["device"], r["phase"]): r for r in summary["groups"]}
    table = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
             r"Conditions & $n$ & $R_s$ & $S_P$ & $S_M$ & $S_P/S_M$ & $R_a$ \\", r"\midrule"]
    groups = [("All inference", "all", "inference"), ("All training", "all", "training")]
    groups.extend((name + " train", device, "training") for device, name in DEVICES.items())
    for i, (name, device, phase) in enumerate(groups):
        if i == 2:
            table.append(r"\midrule")
        g = lookup[device, phase]
        table.append(f"{name} & {g['n']} & " + " & ".join(f"{g[k]:.2f}" for k in
                     ("strict_ratio", "torch_gain", "native_gain", "gap_multiplier", "accelerated_ratio")) + r" \\")
    table.extend([r"\bottomrule", r"\end{tabular}"])
    local_lookup = {(r["gpu"], r["model"]): r for r in local["conditions"]}
    second = [r"\begin{tabular}{@{}lrr@{}}", r"\toprule", r"Model & RTX 5070 & Arc B570 \\", r"\midrule"]
    for model, label in [("SmolLM2-135M", "SmolLM2-135M"), ("Whisper-tiny", "Whisper encoder"),
                         ("StableDiffusion", "U-Net-small"), ("SmolVLA", "SmolVLA expert")]:
        values = [local_lookup[g, model]["change_percent"] for g in ("rtx5070", "b570")]
        second.append(label + " & " + " & ".join(f"${v:+.1f}\\%$" for v in values) + r" \\")
    second.extend([r"\bottomrule", r"\end{tabular}"])
    return {"gap.json": json.dumps(summary, indent=2) + "\n", "gap.tex": "\n".join(table) + "\n",
            "attention-followup.tex": "\n".join(second) + "\n"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=HERE.parent / "tables/results.csv")
    p.add_argument("--followups", type=Path, default=HERE / "attention-followup.json")
    p.add_argument("--output", type=Path)
    p.add_argument("--check", type=Path)
    args = p.parse_args()
    outputs = render(list(csv.DictReader(args.input.open())), followups(args.followups))
    if not args.output and not args.check:
        p.error("specify --output and/or --check")
    for name, content in outputs.items():
        if args.check and (args.check / name).read_text() != content:
            raise SystemExit(f"stale gap analysis: {args.check / name}")
        if args.output:
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / name).write_text(content)
    print("Gap identity and follow-up sample medians verified.")


if __name__ == "__main__":
    main()
