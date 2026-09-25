#!/usr/bin/env python3
"""Summarize the same-revision native search ablation (no GPU).

The ablation reran Meganeura alone on the RTX 5070 / Arc B570 host with the
cohort's revisions and drivers, in two faster preparation policies: no search,
and the library's default two-second tuner. The 60-second measured search is
not rerun; its timings come from the cohort itself. Every ablation output is
checked against the cohort's PyTorch outputs for the same GPU, workload and
replicate, using the cross-engine gates of the paper.
"""

import argparse
import csv
import hashlib
import json
import lzma
import math
from pathlib import Path, PurePosixPath
import statistics
import tarfile

import cohort

DEVICES = {"nvidia-5070": ("RTX 5070", 12036), "intel-b570": ("Arc B570", 57868)}
ARMS = {"off": "No search", "tune2s": "2\\,s tuner"}
CONDITION = ("strict", "default-graph1")


def geomean(values):
    return math.exp(statistics.mean(math.log(value) for value in values))


def load_cohort(source, wanted):
    """Audit the wanted cohort campaigns from a directory or a records bundle."""
    digests = {}
    for line in (cohort.HERE / "cohort.sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        digests[name] = digest
    names = {name for name in digests if Path(name).stem in wanted}
    cohort.require(len(names) == len(wanted), "missing cohort campaign")
    result = {}
    if source.is_dir():
        for name in sorted(names):
            path = source / name
            with path.open("rb") as file:
                cohort.require(hashlib.file_digest(file, "sha256").hexdigest() == digests[name],
                               "input digest differs: " + name)
            records = cohort.read_records(cohort.archive_records(path))
            result[path.stem] = cohort.load_campaign(path, records)
        return result
    with lzma.open(source, "rt") as bundle:
        cohort.require(json.loads(next(bundle)) == {"format": "p3hpc-files-v3", "archives": digests},
                       "bundle provenance differs")
        for name in digests:
            cohort.require(json.loads(next(bundle)) == name, "bundle archive order differs")
            if name not in names:
                for line in bundle:
                    if line.strip() == "null":
                        break
                continue
            records = cohort.read_records(iter(lambda: json.loads(next(bundle)), None))
            result[Path(name).stem] = cohort.load_campaign(Path(name), records)
    return result


def ablation_records(path):
    with tarfile.open(path, mode="r:*") as archive:
        for member in archive:
            if member.isfile() and member.name.endswith(".json"):
                yield member.name, json.loads(archive.extractfile(member).read().decode("utf-8"))


def validate(record, reference):
    a, b = record["outputs"], reference["outputs"]
    cohort.require(a["output_shape"] == b["output_shape"], "output shape differs")
    names = sorted(b["gradient_norms"])
    cohort.require(sorted(a["gradient_norms"]) == names, "gradient names differ")
    errors = {
        "output_relative_l2_error": cohort.relative_l2(a["logits_sample"], b["logits_sample"]),
        "loss_relative_error": abs(a["loss"] - b["loss"]) / max(abs(a["loss"]), abs(b["loss"]), 1e-12),
        "total_gradient_relative_error": abs(a["grad_norm"] - b["grad_norm"]) / max(abs(b["grad_norm"]), 1e-12),
        "parameter_gradient_relative_l2_error": cohort.relative_l2(
            [a["gradient_norms"][name] for name in names], [b["gradient_norms"][name] for name in names]),
    }
    cohort.require(errors["output_relative_l2_error"] < 0.01 and errors["loss_relative_error"] < 0.01,
                   "forward gate failed")
    cohort.require(errors["total_gradient_relative_error"] < 0.05
                   and errors["parameter_gradient_relative_l2_error"] < 0.05, "gradient gate failed")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", type=Path, help="campaign .tgz directory or records.jsonl.xz bundle")
    parser.add_argument("ablation", type=Path, help="search-ablation.tgz")
    parser.add_argument("--output", type=Path, help="write ablation.tex and ablation.csv here")
    parser.add_argument("--check", type=Path, help="compare ablation.tex against this directory")
    args = parser.parse_args()
    expected = (cohort.HERE / "search-ablation.sha256").read_text().split()[0]
    with args.ablation.open("rb") as file:
        cohort.require(hashlib.file_digest(file, "sha256").hexdigest() == expected, "ablation digest differs")
    campaigns = load_cohort(args.cohort, DEVICES)
    records = dict(ablation_records(args.ablation))
    manifest = records.pop(next(name for name in records if name.endswith("results/manifest.json")))
    cohort.require(manifest["meganeura"] == cohort.MEGANEURA and manifest["inferena"].startswith(cohort.INFERENA)
                   and manifest["precision"] == "strict" and manifest["samples"] == 20
                   and set(manifest["arms"]) == set(ARMS), "ablation manifest differs")
    cohort.require(len(manifest["runs"]) == len(DEVICES) * len(cohort.MODELS) * len(ARMS) * 3
                   and all(run["status"] == "valid" and run["returncode"] == 0 for run in manifest["runs"]),
                   "incomplete or failed ablation run")
    measured = {}
    worst = {}
    for name, record in records.items():
        if not name.endswith("_meganeura.json"):
            continue
        device, replicate, model, arm = PurePosixPath(name).parts[-5:-1]
        label, device_id = DEVICES[device]
        cohort.require(record["status"] == "ok" and cohort.MEGANEURA.startswith(record["framework_rev"])
                       and record["environment"]["gpu_device_id"] == device_id
                       and not record["environment"]["gpu_software_emulated"], "ablation identity differs")
        cohort.require(record["precision"]["comparison_class"] == "strict-f32"
                       and record["protocol"]["measurement_runs"] == 20 and record["protocol"]["warmup_runs"] == 5
                       and record["optimizer"]["mode"] == "egglog-outlined"
                       and record["optimizer"]["measured_construction"] is False, "ablation protocol differs")
        cohort.require(all(session["ablation"] == {"ordinary_tune": arm == "tune2s"} and session["search"] is None
                           for session in record["optimizer"]["sessions"]), "ablation arm differs")
        campaign, groups, _ = campaigns[device]
        runs = {run["replicate"]: run for run in groups[CONDITION[0], model, CONDITION[1]]}
        errors = validate(record, runs[replicate]["pair"]["pytorch"])
        for key, value in errors.items():
            worst[key] = max(worst.get(key, 0.0), value)
        for phase in cohort.PHASES:
            samples = record["timing_samples_ms"][phase]
            cohort.require(len(samples) == 20 and all(math.isfinite(x) and x > 0 for x in samples),
                           "invalid ablation samples")
            measured.setdefault((device, model, arm, phase), []).append(statistics.median(samples))
        measured.setdefault((device, model, arm, "compile"), []).append(record["timings"]["compile_s"])
    rows = []
    for device in DEVICES:
        campaign, groups, _ = campaigns[device]
        cohort_rows = cohort.aggregate(groups)
        for model in cohort.MODELS:
            row = cohort_rows[CONDITION[0], model, CONDITION[1]]
            cohort.require(row["meganeura_training_replicates"] == 3, "cohort search arm incomplete")
            entry = {"device": device, "model": model, "search_compile_s": row["meganeura_compile"]}
            for phase in cohort.PHASES:
                entry["search_" + phase] = row["meganeura_" + phase]
            for arm in ARMS:
                values = measured[device, model, arm, "compile"]
                cohort.require(len(values) == 3, "missing ablation replicate")
                entry[arm + "_compile_s"] = statistics.median(values)
                for phase in cohort.PHASES:
                    entry[arm + "_" + phase] = statistics.median(measured[device, model, arm, phase])
            rows.append(entry)
    lines = []
    for device, (label, _) in DEVICES.items():
        data = [row for row in rows if row["device"] == device]
        for index, (arm, name) in enumerate((*ARMS.items(), ("search", "60\\,s search"))):
            cells = [label if index == 0 else "", name,
                     f"{statistics.median(row[arm + '_compile_s'] for row in data):.1f}"]
            for phase in cohort.PHASES:
                speedups = [row["off_" + phase] / row[arm + "_" + phase] for row in data]
                cells.append(f"{geomean(speedups):.2f}")
                if arm != "off":
                    print(label, name, phase, "speedup range", f"{min(speedups):.2f}-{max(speedups):.2f}",
                          {row["model"]: round(value, 2) for row, value in zip(data, speedups)})
            lines.append(cells)
    table = cohort.tex_table("llrrrr", r"GPU & Policy & Prep.\,s & Inf. & Min. & F+L+B", lines)
    print("maximum ablation errors against the cohort PyTorch outputs:", worst)
    for row in rows:
        print(row["device"], row["model"], *(f"{arm}: " + " ".join(
            f"{row[arm + '_' + phase]:.3f}" for phase in cohort.PHASES) + f" ({row[arm + '_compile_s']:.1f}s)"
            for arm in ("off", "tune2s", "search")))
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "ablation.tex").write_text(table)
        with (args.output / "ablation.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    if args.check:
        cohort.require((args.check / "ablation.tex").read_text() == table, "generated table differs: ablation.tex")


if __name__ == "__main__":
    main()
