#!/usr/bin/env python3
"""Audit the final campaign archives and regenerate camera-ready tables (no GPU)."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import tarfile


HERE = Path(__file__).resolve().parent
INFERENA = "17d13a3a94cc5bdfaa63f0eb2cb7e8ee7593c053"
MEGANEURA = "fcdd76d1a4cd0e3d10507e56ea3f2412378a2ba7"
TORCH = "cf30153c4c131c8164ee7798e5022d810682e2cb"
MODELS = ("SmolLM2-135M", "SmolVLA", "StableDiffusion", "ResNet-50", "Whisper-tiny")
PHASES = ("inference", "latency", "training")
DEVICES = {
    "nvidia-5070": "RTX 5070",
    "nvidia-h100": "H100",
    "amd-dgpu": "RX 7900 XT",
    "amd-igpu": "Radeon 780M",
    "intel-dgpu": "Arc B570",
    "apple-m3": "Apple M3",
    "intel-igpu": "Intel RPL-U (CPU ref.)",
    "nvidia-3050": "RTX 3050 (Windows)",
    "nvidia-h100-large": "H100 extension",
}
ENGINES = ("meganeura", "pytorch")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def relative_l2(actual, reference):
    require(len(actual) == len(reference) > 0, "vector length mismatch")
    require(all(math.isfinite(x) for x in (*actual, *reference)), "nonfinite vector")
    delta = sum((x - y) ** 2 for x, y in zip(actual, reference))
    norm = sum(x * x for x in reference)
    return math.sqrt(delta / norm) if norm else (math.inf if delta else 0.0)


def audit_replay(validation, training, accelerated):
    require(validation["rtol"] == 1e-4 and validation["atol"] == 1e-6
            and validation["accelerated_gradient_rtol"] == 0.01, "replay bounds changed")
    for stage, count in (("uncaptured", validation["uncaptured_repeats"]), ("replays", 2)):
        tensors = validation[stage]
        require(len(tensors) == count, "missing replay evidence")
        for sample in tensors:
            require(sum(name.startswith("gradient ") for name in sample) == validation["gradient_tensors"],
                    "gradient inventory differs in replay")
            for name, row in sample.items():
                require(row["elements"] > 0 and all(math.isfinite(x) and x >= 0 for x in row.values()),
                        "invalid replay metric")
                if name.startswith("gradient ") and not accelerated:
                    require(row["rms_error"] <= row["rms_bound"] and row["max_abs_error"] <= row["max_abs_bound"],
                            "strict gradient replay failed")
        totals = validation["full_gradient"][stage]
        require(len(totals) == (count if training else 0), "missing whole-gradient evidence")
        for sample, total in zip(tensors, totals):
            gradients = [row for name, row in sample.items() if name.startswith("gradient ")]
            elements = sum(row["elements"] for row in gradients)
            require(elements == total["elements"] > 0, "whole-gradient element count differs")
            rtol = 0.01 if accelerated else 1e-4
            for metric in ("rms", "max_abs"):
                for part in ("error", "reference"):
                    key = metric + "_" + part
                    value = (math.sqrt(sum(row[key] ** 2 * row["elements"] for row in gradients) / elements)
                             if metric == "rms" else max(row[key] for row in gradients))
                    require(math.isclose(value, total[key], rel_tol=1e-12, abs_tol=1e-12), "whole-gradient summary differs")
                bound = 1e-6 + rtol * total[metric + "_reference"]
                require(total[metric + "_error"] <= bound and math.isclose(bound, total[metric + "_bound"]),
                        "whole-gradient replay failed")


def audit_pair(pair, campaign, run):
    require(set(pair) == set(ENGINES), "missing engine")
    mg, pt = (pair[engine] for engine in ENGINES)
    for engine, record in pair.items():
        require(record["status"] == "ok", "failed engine in a valid pair")
        require(INFERENA.startswith(record["benchmark_rev"]), "wrong harness revision")
        require(record["protocol"]["warmup_runs"] == 5, "warmup count changed")
        require(record["protocol"]["measurement_runs"] == 20, "sample count changed")
        require(record["protocol"]["training_requested"] is True, "training scope changed")
        require(record["protocol"]["diagnostic"] is False, "profiled timing in campaign")
        for phase in PHASES:
            samples = record["timing_samples_ms"][phase]
            require(len(samples) == 20 and all(math.isfinite(x) and x > 0 for x in samples),
                    "invalid timing samples")
            median = statistics.median(samples)
            require(math.isclose(median, record["timing_summary_ms"][phase]["median"],
                                 rel_tol=0, abs_tol=1e-6), "stored median differs")
            require(abs(median - record["timings"][phase + "_ms"]) <= 0.000501,
                    "rounded timing differs")
    require(MEGANEURA.startswith(mg["framework_rev"]), "wrong Meganeura revision")
    require(pt["environment"]["torch_git_version"] == TORCH, "wrong PyTorch revision")
    require(pt["torch_version"] == campaign["torch"]["version"], "wheel changed")
    require(pt["backend"].split()[0].lower() == campaign["args"]["backend"], "reference backend changed")
    require(mg["precision"]["comparison_class"] == pt["precision"]["comparison_class"], "arithmetic contracts differ")
    require(pt["environment"]["python_version"] == "3.13.13", "Python changed")
    require(mg["environment"]["gpu_device_id"] == campaign["native_device"]["device_id"],
            "native GPU changed")
    require(mg["optimizer"]["measured_kernel_search"] == (run["mode"] == "max-autotune"),
            "native search policy differs")
    execution = pt["execution"]
    require(execution["requested_mode"] == run["mode"], "reference mode differs")
    require(execution["compiled"] == (run["mode"] != "eager"), "compiler fallback")
    require(execution["cuda_graphs"]["requested"] == run["graphs"], "graph request differs")
    for phase in PHASES:
        graph = execution["cuda_graphs"]["phases"][phase]
        require(graph["status"] == ("captured-and-validated" if run["graphs"] else "not-requested"),
                "unqualified replay")
        if run["graphs"]:
            v = graph["validation"]
            repeats = 8 if phase == "training" and pt["precision"]["reduced_precision_allowed"] else 2
            require(v["policy"] == "fixed-full-gradient-v3" and v["uncaptured_repeats"] == repeats
                    and v["uncaptured_calls"] == repeats + 1 and v["consecutive_replays"] == 2,
                    "replay validation policy changed")
            audit_replay(v, phase == "training", phase == "training" and pt["precision"]["reduced_precision_allowed"])
    a, b = mg["outputs"], pt["outputs"]
    require(a["output_shape"] == b["output_shape"], "output shape mismatch")
    require(len(a["logits_sample"]) == len(b["logits_sample"]) == 256, "sample inventory differs")
    require(a["gradient_norms"].keys() == b["gradient_norms"].keys(), "gradient names differ")
    names = sorted(b["gradient_norms"])
    errors = {
        "output_relative_l2_error": relative_l2(a["logits_sample"], b["logits_sample"]),
        "loss_relative_error": abs(a["loss"] - b["loss"]) / max(abs(a["loss"]), abs(b["loss"]), 1e-12),
        "total_gradient_relative_error": abs(a["grad_norm"] - b["grad_norm"]) / max(abs(b["grad_norm"]), 1e-12),
        "parameter_gradient_relative_l2_error": relative_l2(
            [a["gradient_norms"][name] for name in names], [b["gradient_norms"][name] for name in names]),
    }
    for key, value in errors.items():
        require(math.isfinite(value) and math.isclose(value, mg["validation"][key], rel_tol=1e-9, abs_tol=1e-12),
                "retained evidence disagrees with " + key)
    require(errors["output_relative_l2_error"] < 0.01 and errors["loss_relative_error"] < 0.01,
            "forward gate failed")
    limit = 0.10 if mg["precision"]["reduced_precision_allowed"] else 0.05
    require(all(errors[key] < limit for key in errors if "gradient" in key), "gradient sample gate failed")
    require(mg["validation"]["forward_valid"] is True, "stored forward gate differs")
    require(mg["validation"]["training_valid"] == all(errors[key] < 0.05 for key in errors if "gradient" in key),
            "stored gradient gate differs")
    return errors


def load_campaign(path):
    with tarfile.open(path) as archive:
        records = {member.name: json.load(archive.extractfile(member)) for member in archive
                   if member.isfile() and member.name.endswith(".json")}
    manifests = [name for name in records if name.endswith("/campaign.json")]
    require(len(manifests) == 1, "expected one campaign in " + path.name)
    prefix = str(PurePosixPath(manifests[0]).parent) + "/"
    campaign = records[manifests[0]]
    require(campaign["source"] == INFERENA and campaign["meganeura"]["rev"] == MEGANEURA,
            "campaign revision mismatch")
    require(campaign["torch"]["git_version"] == TORCH, "campaign PyTorch mismatch")
    require(campaign["protocol"] == "p3hpc-paired-campaign-v7", "campaign protocol mismatch")
    require(campaign["args"]["replicates"] == 3, "replicate policy changed")
    groups = defaultdict(list)
    failed = []
    seen = set()
    for run in campaign["runs"]:
        folder = run["path"].replace("\\", "/")
        require(folder not in seen, "duplicate run")
        seen.add(folder)
        stage, replicate, precision, model, condition = folder.split("/")
        require(stage == "measurement", "qualification included as measurement")
        require(condition == run["mode"] + "-graph" + str(int(run["graphs"])), "condition path differs")
        require(replicate in ("r1", "r2", "r3"), "unexpected replicate")
        summary = records[prefix + folder + "/" + model + "_summary.json"]
        pair = {record["framework"]: record for record in summary}
        for engine, record in pair.items():
            require(record == records[prefix + folder + "/" + model + "_" + engine + ".json"],
                    "joined record differs from raw record")
        if run["status"] != "valid":
            failed.append({"path": folder, "engines": {engine: record["status"] for engine, record in pair.items()}})
            continue
        errors = audit_pair(pair, campaign, run)
        require(pair["meganeura"]["precision"]["reduced_precision_allowed"] == (precision == "accelerated"),
                "precision label differs")
        groups[precision, model, condition].append({"replicate": replicate, "pair": pair, "errors": errors})
    if campaign["status"] == "complete":
        require(not failed, "complete campaign contains failure")
        expected = len(campaign["args"]["models"]) * len(campaign["args"]["precisions"]) * len(campaign["reference_conditions"]["selected"])
        require(len(groups) == expected, "missing selected condition")
        require(all(len(runs) == 3 for runs in groups.values()), "incomplete replication")
        require(campaign["replicated_gradient_validation"]["status"] == "pass", "replicated gate failed")
    for runs in groups.values():
        for key in ("parameter_gradient_relative_l2_error", "total_gradient_relative_error"):
            if len(runs) == 3:
                require(statistics.median(run["errors"][key] for run in runs) < 0.05, "replicated gradient gate failed")
    return campaign, groups, failed


def aggregate(groups):
    rows = {}
    for key, runs in groups.items():
        row = {"replicates": len(runs)}
        for engine in ENGINES:
            for phase in PHASES:
                values = [statistics.median(run["pair"][engine]["timing_samples_ms"][phase]) for run in runs]
                row[engine + "_" + phase] = statistics.median(values)
                row[engine + "_" + phase + "_min"] = min(values)
                row[engine + "_" + phase + "_max"] = max(values)
                row[engine + "_" + phase + "_spread"] = (max(values) - min(values)) / statistics.median(values)
            row[engine + "_compile"] = statistics.median(run["pair"][engine]["timings"]["compile_s"] for run in runs)
        graphs = [run["pair"]["pytorch"]["execution"]["cuda_graphs"]["phases"] for run in runs]
        for part in ("capture", "validation"):
            row["pytorch_" + part] = statistics.median(sum(phase.get(part + "_s", 0) for phase in graph.values()) for graph in graphs)
        for phase in PHASES:
            row["ratio_" + phase] = row["meganeura_" + phase] / row["pytorch_" + phase]
        rows[key] = row
    return rows


def primary_condition(campaign):
    if campaign["args"]["backend"] == "cuda":
        return "default-graph1"
    return "eager-graph0" if campaign["args"]["backend"] in ("mps", "cpu") else "default-graph0"


def tex_table(columns, heading, lines):
    return "\n".join([r"\begin{tabular}{" + columns + "}", r"\toprule", heading + r" \\",
                      r"\midrule", *[" & ".join(row) + r" \\" for row in lines], r"\bottomrule", r"\end{tabular}", ""])


def ratio(value):
    text = f"{value:.2f}"
    return r"\textbf{" + text + "}" if value < 1 else text


def tables(campaigns, rows):
    output = {}
    lines = []
    for device, label in DEVICES.items():
        if device == "nvidia-h100-large":
            continue
        c = campaigns[device]
        backend = c["args"]["backend"]
        driver = c["native_device"]["driver_info"].split("-")[0] or "Metal"
        valid = sum(run["status"] == "valid" for run in c["runs"])
        expected = 3 * 10 * len(c["reference_conditions"]["selected"])
        mode = "eager" if backend in ("mps", "cpu") else "compiled"
        coverage = "partial" if c["status"] != "complete" else c["reference_conditions"]["coverage"]
        coverage = "light only" if coverage == "availability-subset" else coverage
        lines.append([label, backend.upper() + "/" + mode, driver, f"{valid}/{expected}", coverage])
    output["devices.tex"] = tex_table("lllll", "Device & PyTorch path & Graphics driver & Valid/selected pairs & Coverage", lines)
    lines = []
    for device, label in DEVICES.items():
        c = campaigns[device]
        if c["status"] != "complete":
            continue
        condition = primary_condition(c)
        if device == "intel-igpu":
            lines.append([r"\multicolumn{8}{l}{\emph{GPU-versus-CPU support comparison; excluded from GPU aggregates}}"])
        for model in MODELS:
            values = [ratio(rows[device][precision, model, condition]["ratio_" + phase])
                      for precision in ("strict", "accelerated") for phase in PHASES]
            lines.append([label if model == MODELS[0] else "", model, *values])
    output["ratios.tex"] = tex_table("llrrrrrr", r"Device & Workload & \multicolumn{3}{c}{Strict: inf. / min. / F+L+B} & \multicolumn{3}{c}{Accelerated: inf. / min. / F+L+B}", lines)
    lines = []
    for device in ("nvidia-5070", "nvidia-h100"):
        for model in MODELS:
            values = [ratio(rows[device][precision, model, "max-autotune-graph1"]["ratio_" + phase])
                      for precision in ("strict", "accelerated") for phase in PHASES]
            lines.append([DEVICES[device] if model == MODELS[0] else "", model, *values])
    output["searched-ratios.tex"] = tex_table("llrrrrrr", r"Device & Workload & \multicolumn{3}{c}{Strict: inf. / min. / F+L+B} & \multicolumn{3}{c}{Accelerated: inf. / min. / F+L+B}", lines)
    lines = []
    for device in ("nvidia-5070", "nvidia-h100"):
        for model in MODELS:
            base = rows[device]["strict", model, "default-graph0"]
            light = rows[device]["strict", model, "default-graph1"]
            searched = rows[device]["strict", model, "max-autotune-graph1"]
            lines.append([DEVICES[device] if model == MODELS[0] else "", model,
                          *[f"{base['pytorch_' + phase] / light['pytorch_' + phase]:.2f}" for phase in PHASES],
                          *[f"{light[engine + '_inference'] / searched[engine + '_inference']:.2f}" for engine in ENGINES],
                          *[f"{light[engine + '_compile']:.2f}/{searched[engine + '_compile']:.2f}" for engine in ENGINES]])
    output["search.tex"] = tex_table("llrrrrrrr", r"Device & Workload & \multicolumn{3}{c}{PyTorch replay gain} & \multicolumn{2}{c}{Search inf. gain} & \multicolumn{2}{c}{Compile seconds: light/searched} \\ & & Inf. & Min. & F+L+B & M & P & M & P", lines)
    lines = []
    for model, device in (("SmolLM2-135M", "nvidia-h100"), ("SmolLM2-360M", "nvidia-h100-large"), ("SmolLM2-1.7B", "nvidia-h100-large")):
        for condition in ("default-graph1", "max-autotune-graph1"):
            key = ("strict", model, condition)
            if key not in rows[device]:
                lines.append([model, "searched", "0", *["--"] * 8])
                continue
            row = rows[device][key]
            lines.append([model, "light" if condition.startswith("default") else "searched", str(row["replicates"]),
                          *[f"{row[engine + '_' + phase]:.2f}" for phase in PHASES for engine in ENGINES],
                          *[f"{row[engine + '_compile']:.2f}" for engine in ENGINES]])
    output["scaling.tex"] = tex_table("llrrrrrrrrr", r"Model & Policy & $n$ & \multicolumn{2}{c}{Prefill ms} & \multicolumn{2}{c}{One token ms} & \multicolumn{2}{c}{F+L+B ms} & \multicolumn{2}{c}{Compile s} \\ & & & M & P & M & P & M & P & M & P", lines)
    lines, scores = [], []
    for model in MODELS:
        values = []
        for phase in PHASES:
            ratios = [rows[device]["strict", model, primary_condition(c)]["ratio_" + phase]
                      for device, c in campaigns.items() if c["status"] == "complete" and c["args"]["backend"] != "cpu"]
            values.extend((len(ratios) / sum(max(x, 1) for x in ratios),
                           len(ratios) / sum(max(1 / x, 1) for x in ratios)))
        scores.append(values)
        lines.append([model, *[f"{x:.2f}" for x in values]])
    lines.append(["Workload mean", *[f"{statistics.mean(col):.2f}" for col in zip(*scores)]])
    output["portability.tex"] = tex_table("lrrrrrr", r"Workload & \multicolumn{2}{c}{Inference} & \multicolumn{2}{c}{Minimal} & \multicolumn{2}{c}{F+L+B} \\ & M & P & M & P & M & P", lines)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", type=Path, help="directory containing the supplied .tgz campaigns")
    parser.add_argument("--output", type=Path, help="write generated tables and per-condition CSV here")
    parser.add_argument("--check", type=Path, help="compare generated tables against this directory")
    args = parser.parse_args()
    campaigns, rows, all_groups = {}, {}, {}
    for line in (HERE / "cohort.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        require(Path(name).name == name, "manifest path is not a filename")
        with (args.archives / name).open("rb") as file:
            require(hashlib.file_digest(file, "sha256").hexdigest() == expected, "input digest differs: " + name)
    print("Archive and supplied-log SHA-256 digests match")
    for device in DEVICES:
        path = args.archives / (device + ".tgz")
        c, groups, failed = load_campaign(path)
        campaigns[device], rows[device], all_groups[device] = c, aggregate(groups), groups
        print(device, c["status"], dict(Counter(run["status"] for run in c["runs"])), "failed:", failed)
    generated = tables(campaigns, rows)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        for name, content in generated.items():
            (args.output / name).write_text(content)
        flat = [{"device": device, "precision": key[0], "model": key[1], "condition": key[2], **row}
                for device, data in rows.items() for key, row in sorted(data.items())]
        with (args.output / "conditions.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, flat[0].keys())
            writer.writeheader()
            writer.writerows(flat)
    if args.check:
        for name, content in generated.items():
            require((args.check / name).read_text() == content, "generated table differs: " + name)
    for precision in ("strict", "accelerated"):
        print("\n", precision, "complete GPU/light cohort")
        for phase in PHASES:
            values = [rows[device][precision, model, primary_condition(c)]["ratio_" + phase]
                      for device, c in campaigns.items() if c["status"] == "complete" and c["args"]["backend"] != "cpu"
                      for model in MODELS]
            print(phase, "wins", sum(x < 1 for x in values), "/", len(values), "median ratio", statistics.median(values))
    errors = [(run["errors"], device, key) for device, groups in all_groups.items() for key, runs in groups.items() for run in runs]
    for key in errors[0][0]:
        print("maximum", key, max((e[key], device, group) for e, device, group in errors))


if __name__ == "__main__":
    main()
