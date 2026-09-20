#!/usr/bin/env python3
"""Audit the final campaign archives and regenerate camera-ready tables (no GPU)."""

import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
import csv
import hashlib
import json
import lzma
import math
from pathlib import Path, PurePosixPath
import statistics
import tarfile


HERE = Path(__file__).resolve().parent
INFERENA = "fa5a04e1c1b38405cfa371a27c5dcef1319835d5"
MEGANEURA = "428fc2d2322229e5338f5d80a10d700340d593cd"
TORCH = "cf30153c4c131c8164ee7798e5022d810682e2cb"
MODELS = ("SmolLM2-135M", "SmolVLA", "StableDiffusion", "ResNet-50", "Whisper-tiny")
PHASES = ("inference", "latency", "training")
DEVICES = {
    "nvidia-5070": "RTX 5070",
    "nvidia-h100": "H100",
    "nvidia-3050": "RTX 3050 (Windows)",
    "amd-dgpu": "RX 7900 XT",
    "amd-igpu": "Radeon 780M",
    "intel-b570": "Arc B570",
    "apple-m3": "Apple M3",
    "intel-igpu": "Intel RPL-U",
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
    anchor = validation["uncaptured"][0]
    for stage, count in (("uncaptured", validation["uncaptured_repeats"]), ("replays", 2)):
        tensors = validation[stage]
        require(len(tensors) == count, "missing replay evidence")
        for sample in tensors:
            require(sample.keys() == anchor.keys(), "replay tensor inventory changed")
            require({name for name in sample if not name.startswith("gradient ")}
                    == ({"output 0", "output 1"} if training else {"output 0"}), "missing output evidence")
            require(sum(name.startswith("gradient ") for name in sample) == validation["gradient_tensors"],
                    "gradient inventory differs in replay")
            for name, row in sample.items():
                require(all(row[key] == anchor[name][key] for key in ("elements", "max_abs_reference", "rms_reference")),
                        "replay reference changed between calls")
                require(row["elements"] > 0 and all(math.isfinite(x) and x >= 0 for x in row.values()),
                        "invalid replay metric")
                for metric in ("max_abs", "rms"):
                    require(math.isclose(row[metric + "_bound"], 1e-6 + 1e-4 * row[metric + "_reference"],
                                         rel_tol=1e-12), "per-tensor bound changed")
                if not (name.startswith("gradient ") and accelerated):
                    require(row["rms_error"] <= row["rms_bound"] and row["max_abs_error"] <= row["max_abs_bound"],
                            "tensor replay failed")
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
    require(not mg["environment"]["gpu_software_emulated"], "software GPU substituted")
    backend = campaign["args"]["backend"]
    require(mg["gpu_name"] == campaign["native_device"]["name"], "native device name changed")
    if backend in ("cuda", "rocm", "xpu"):
        require(pt["environment"]["triton_backend"] == {"cuda": "nvidia", "rocm": "amd", "xpu": "intel"}[backend],
                "wrong reference compiler backend")
        require(pt["device"] == pt["gpu_name"] and pt["device"].lower() != "cpu", "CPU fallback")
    require(mg["optimizer"]["measured_kernel_search"] is True, "native search disabled")
    strict = not mg["precision"]["reduced_precision_allowed"]
    require(mg["precision"]["cooperative_matrix_policy"] == ("NativeF32" if strict else "Auto: protect full-precision derivative regions"),
            "cooperative matrix policy changed")
    require((not strict or mg["precision"]["native_f32_cooperative_matrix_permitted"] is True)
            and mg["precision"]["f16_cooperative_matrix_permitted"] == (not strict), "cooperative permissions differ")
    sessions = mg["optimizer"]["sessions"]
    require(Counter(s["mode"] for s in sessions) == {"Inference": 1 if mg["model"] == "Whisper-tiny" else 2, "Training": 1},
            "missing native sessions")
    for session in sessions:
        require(session["cooperative_matrix_policy"] == ("NativeF32" if strict else "Auto"), "session arithmetic changed")
        search = session["search"]
        require(search["scope"] == "All" and search["class_limit"] is None and not search["class_limit_reached"]
                and search["max_seconds"] == 60.0 and search["max_scratch_bytes"] == 1024**3, "search policy changed")
        require(0 <= search["visited_classes"] <= search["eligible_classes"]
                and (search["visited_classes"] == search["eligible_classes"] or search["time_budget_exhausted"]),
                "unexplained search truncation")
        require(math.isfinite(search["elapsed_seconds"]) and search["elapsed_seconds"] >= 0
                and sum(search["decisions"].values()) == search["comparisons"], "invalid search receipt")
    execution = pt["execution"]
    require(execution["requested_mode"] == run["mode"] == "default" and execution["compiled"], "compiler fallback")
    require(execution["compile_budget_seconds"] == 120.0 and execution["compile_budget_enforced"], "watchdog missing")
    require(not any(execution["compiler_options"].values()), "unexpected compiler search or internal replay")
    require(execution["sdpa_policy"] == ("math" if backend == "xpu" else "auto"), "attention policy changed")
    allowed = {"MATH"} if backend == "xpu" else {"MATH", "CUDNN_ATTENTION", "FLASH_ATTENTION", "EFFICIENT_ATTENTION", "OVERRIDEABLE"}
    require(set(execution["sdpa_enabled_backends"]) == allowed, "attention backends differ")
    require(execution["graph_replay"]["requested"] == run["graphs"] == (backend in ("cuda", "rocm", "xpu")),
            "graph request differs")
    require(pt["protocol"]["name"] == "inferena-graph-replay-v4", "reference protocol changed")
    for phase in PHASES:
        graph = execution["graph_replay"]["phases"][phase]
        require(graph["status"] == ("captured-and-validated" if run["graphs"] else "not-requested"),
                "unqualified replay")
        if run["graphs"]:
            require(graph["api"] == ("torch.xpu.XPUGraph" if backend == "xpu" else "torch.cuda.CUDAGraph")
                    and execution["stream_policy"] == "single dedicated preparation/run stream", "wrong replay API/stream")
            v = graph["validation"]
            repeats = 8 if phase == "training" and pt["precision"]["reduced_precision_allowed"] else 2
            require(v["policy"] == "fixed-full-tensor-v4" and v["uncaptured_repeats"] == repeats
                    and v["output_metric"] == "per-tensor RMS and maximum absolute error"
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


def read_records(path):
    with tarfile.open(path) as archive:
        return {member.name: json.load(archive.extractfile(member)) for member in archive
                if member.isfile() and member.name.endswith(".json")}


def load_campaign(path, records=None):
    records = read_records(path) if records is None else records
    manifests = [name for name in records if name.endswith("/campaign.json")]
    require(len(manifests) == 1, "expected one campaign in " + path.name)
    prefix = str(PurePosixPath(manifests[0]).parent) + "/"
    campaign = records[manifests[0]]
    require(campaign["source"] == INFERENA and campaign["meganeura"]["rev"] == MEGANEURA,
            "campaign revision mismatch")
    require(campaign["torch"]["git_version"] == TORCH, "campaign PyTorch mismatch")
    require(campaign["protocol"] == "p3hpc-paired-campaign-v9", "campaign protocol mismatch")
    require(campaign["args"]["replicates"] == 3, "replicate policy changed")
    expected_models = ["SmolLM2-360M", "SmolLM2-1.7B"] if path.stem == "nvidia-h100-large" else list(MODELS)
    require(campaign["args"]["models"] == expected_models
            and campaign["args"]["precisions"] == ["strict", "accelerated"], "campaign workload coverage changed")
    require(campaign["native_policy"] == {"measured_kernel_search": True, "scope": "All", "class_limit": None,
            "max_scratch_bytes": 1024**3, "search_seconds_per_session": 60.0, "strict_coop": "NativeF32"},
            "manifest native policy differs")
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
        expected = {(precision, model, condition["mode"] + "-graph" + str(int(condition["graph_replay"])))
                    for precision in campaign["args"]["precisions"] for model in campaign["args"]["models"]
                    for condition in campaign["reference_conditions"]["selected"]}
        require(set(groups) == expected, "missing selected condition")
        require(all({run["replicate"] for run in runs} == {"r1", "r2", "r3"} and len(runs) == 3
                    for runs in groups.values()), "incomplete replication")
        require(campaign["replicated_gradient_validation"]["status"] == "pass", "replicated gate failed")
    for runs in groups.values():
        for key in ("parameter_gradient_relative_l2_error", "total_gradient_relative_error"):
            if len(runs) == 3:
                require(statistics.median(run["errors"][key] for run in runs) < 0.05, "replicated gradient gate failed")
    report = campaign["replicated_gradient_validation"]
    require(report["policy"] == "replicated-gradient-median-v1" and len(report["groups"]) == len(groups),
            "replicated report policy/inventory differs")
    checked = set()
    for group in report["groups"]:
        key = (group["precision"], group["model"], group["mode"] + "-graph" + str(int(group["graph_replay"])))
        require(key not in checked and key in groups, "duplicate or unknown replicated group")
        checked.add(key)
        require(group["status"] == "pass" and group["median_limit"] == 0.05
                and group["sample_limit"] == (0.1 if key[0] == "accelerated" else 0.05), "replication bounds changed")
        require(set(group["metrics"]) == {"parameter_gradient_relative_l2", "total_gradient_relative"},
                "missing replicated metric")
        for metric, stored in group["metrics"].items():
            values = [run["errors"][metric + "_error"] for run in groups[key]]
            require(len(stored["samples"]) == 3
                    and all(math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12) for a, b in zip(values, stored["samples"]))
                    and math.isclose(statistics.median(values), stored["median"], rel_tol=1e-9, abs_tol=1e-12),
                    "replicated report differs from retained errors")
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
            for field, name in (("allocated_bytes", engine + "_memory"), ("peak_reserved_bytes", engine + "_reserved")):
                values = [max(phase[field] for phase in memory["phases"].values())
                          for run in runs if (memory := run["pair"][engine].get("memory"))
                          and all(field in phase for phase in memory["phases"].values())]
                row[name] = statistics.median(values) if len(values) == len(runs) else None
        graphs = [run["pair"]["pytorch"]["execution"]["graph_replay"]["phases"] for run in runs]
        for part in ("capture", "validation"):
            row["pytorch_" + part] = statistics.median(sum(phase.get(part + "_s", 0) for phase in graph.values()) for graph in graphs)
        for phase in PHASES:
            row["ratio_" + phase] = row["meganeura_" + phase] / row["pytorch_" + phase]
        rows[key] = row
    return rows


def primary_condition(campaign):
    return "default-graph" + str(int(campaign["args"]["backend"] in ("cuda", "rocm", "xpu")))


def tex_table(columns, heading, lines):
    return "\n".join([r"\begin{tabular}{" + columns + "}", r"\toprule", heading + r" \\",
                      r"\midrule", *[" & ".join(row) + r" \\" for row in lines], r"\bottomrule", r"\end{tabular}", ""])


def ratio(value):
    text = f"{value:.2f}"
    return r"\textbf{" + text + "}" if value < 1 else text


def tables(campaigns, rows, groups):
    output = {}
    paired_devices = {device: label for device, label in DEVICES.items()
                      if campaigns[device]["args"]["backend"] != "cpu"}
    lines = []
    for device, label in paired_devices.items():
        if device == "nvidia-h100-large":
            continue
        c = campaigns[device]
        backend = c["args"]["backend"]
        driver = c["native_device"]["driver_info"].split("-")[0] or "Metal"
        valid = sum(run["status"] == "valid" for run in c["runs"])
        expected = 3 * 10 * len(c["reference_conditions"]["selected"])
        replay = {"cuda": "CUDA Graph", "rocm": "HIP graph", "xpu": "XPU graph"}.get(backend, "no public replay" if backend == "mps" else "none")
        lines.append([label, backend.upper() + "/compiled", driver, replay, f"{valid}/{expected}"])
    output["devices.tex"] = tex_table("llllr", "Device & PyTorch path & Graphics driver & Explicit replay & Valid/selected pairs", lines)
    lines = []
    for device, label in DEVICES.items():
        if device in paired_devices:
            continue
        conditions = len(groups[device])
        lines.append([label, "Vulkan", f"{conditions}/{conditions}", "Unavailable"])
    output["qualification.tex"] = tex_table("llrl",
        "Device & Meganeura path & Qualified conditions & PyTorch GPU", lines)
    lines = []
    for device, label in paired_devices.items():
        c = campaigns[device]
        if device == "nvidia-h100-large":
            continue
        condition = primary_condition(c)
        for model in MODELS:
            values = [ratio(rows[device][precision, model, condition]["ratio_" + phase])
                      for precision in ("strict", "accelerated") for phase in PHASES]
            lines.append([label if model == MODELS[0] else "", model, *values])
    ratio_heading = (r"Device & Workload & \multicolumn{3}{c}{Strict} & \multicolumn{3}{c}{Accelerated} \\"
                     r" \cmidrule(lr){3-5}\cmidrule(lr){6-8}"
                     r" & & Inf. & Min. & F+L+B & Inf. & Min. & F+L+B")
    output["ratios.tex"] = tex_table("llrrrrrr", ratio_heading, lines)
    lines = []
    for device, label in paired_devices.items():
        runs = [run for batch in groups[device].values() for run in batch]
        compile_s = [sum(run["pair"][engine]["timings"]["compile_s"] for run in runs) for engine in ENGINES]
        graph_s = [sum(phase.get(part + "_s", 0) for run in runs
                       for phase in run["pair"]["pytorch"]["execution"]["graph_replay"]["phases"].values())
                   for part in ("capture", "validation")]
        lines.append([label, str(len(runs)), *[f"{value / 60:.2f}" for value in (*compile_s, *graph_s)]])
    output["preparation.tex"] = tex_table("lrrrrr",
        r"Device & Pairs & M compile+tune & P compile & P graph prep. & P qualification", lines)
    lines = []
    for device, label in paired_devices.items():
        searches = [session["search"] for batch in groups[device].values() for run in batch
                    for session in run["pair"]["meganeura"]["optimizer"]["sessions"]]
        decisions = sum((Counter(s["decisions"]) for s in searches), Counter())
        lines.append([label, str(len(searches)),
                      f"{sum(s['visited_classes'] for s in searches)}/{sum(s['eligible_classes'] for s in searches)}",
                      str(sum(s["time_budget_exhausted"] for s in searches)),
                      str(decisions["FasterCandidate"]), str(decisions["InvalidOutput"])])
    output["search.tex"] = tex_table("lrrrrr", r"Device & Sessions & Classes visited/eligible & Deadlines & Faster & Rejected", lines)
    lines = []
    for model, device in (("SmolLM2-135M", "nvidia-h100"), ("SmolLM2-360M", "nvidia-h100-large"), ("SmolLM2-1.7B", "nvidia-h100-large")):
        for precision in ("strict", "accelerated"):
            key = (precision, model, "default-graph1")
            row = rows[device][key]
            lines.append([model, precision, str(row["replicates"]),
                          *[f"{row[engine + '_' + phase]:.2f}" for phase in PHASES for engine in ENGINES],
                          *[f"{row[engine + '_compile']:.2f}" for engine in ENGINES]])
    output["scaling.tex"] = tex_table("llrrrrrrrrr", r"Model & Arithmetic & $n$ & \multicolumn{2}{c}{Prefill ms} & \multicolumn{2}{c}{One token ms} & \multicolumn{2}{c}{F+L+B ms} & \multicolumn{2}{c}{Compile s} \\ & & & M & P & M & P & M & P & M & P", lines)
    lines = []
    for model, device in (("SmolLM2-135M", "nvidia-h100"), ("SmolLM2-360M", "nvidia-h100-large"), ("SmolLM2-1.7B", "nvidia-h100-large")):
        row = rows[device]["strict", model, "default-graph1"]
        lines.append([model, str(row["replicates"]), *[f"{row[key] / 2**30:.2f}"
                      for key in ("meganeura_memory", "pytorch_memory", "pytorch_reserved")]])
    output["memory.tex"] = tex_table("lrrrr",
        r"Model & $n$ & M plan & P allocated & P reserved", lines)
    lines, scores = [], []
    for model in MODELS:
        values = []
        for phase in PHASES:
            ratios = [rows[device]["strict", model, primary_condition(c)]["ratio_" + phase]
                      for device, c in campaigns.items() if device != "nvidia-h100-large" and c["args"]["backend"] != "cpu"]
            values.extend((len(ratios) / sum(max(x, 1) for x in ratios),
                           len(ratios) / sum(max(1 / x, 1) for x in ratios)))
        scores.append(values)
        lines.append([model, *[f"{x:.2f}" for x in values]])
    lines.append(["Workload mean", *[f"{statistics.mean(col):.2f}" for col in zip(*scores)]])
    output["portability.tex"] = tex_table("lrrrrrr", r"Workload & \multicolumn{2}{c}{Inference} & \multicolumn{2}{c}{Minimal} & \multicolumn{2}{c}{F+L+B} \\ & M & P & M & P & M & P", lines)
    # Paired bars, not a stack: forward and F+L+B are separate measurements.
    chart = [r"\begin{tikzpicture}[x=1mm,y=1mm,font=\scriptsize]"]
    devices = [device for device, c in campaigns.items() if device != "nvidia-h100-large" and c["args"]["backend"] != "cpu"]
    bottom = 1 - len(devices) * 10
    for panel, (phase, title) in enumerate((("inference", "Prefill (ms)"), ("training", "F+L+B (ms)"))):
        maximum = max(rows[device]["strict", "SmolLM2-135M", primary_condition(campaigns[device])][engine + "_" + phase + "_max"]
                      for device in devices for engine in ENGINES)
        limit = math.ceil(maximum / 30) * 30
        origin = 32 + panel * 94
        chart.append(rf"\node at ({origin + 29},7) {{{title}}};")
        for tick in range(4):
            x = origin + tick * 18
            chart.extend([rf"\draw[black!15] ({x},1) -- ({x},{bottom});",
                          rf"\node[below] at ({x},{bottom}) {{{limit * tick // 3}}};"])
        for i, device in enumerate(devices):
            y = -i * 10
            row = rows[device]["strict", "SmolLM2-135M", primary_condition(campaigns[device])]
            chart.append(rf"\node[anchor=east] at ({origin - 1},{y - 3}) {{{DEVICES[device]}}};")
            for j, (engine, color) in enumerate(zip(ENGINES, ("blue!65!black", "orange!80!black"))):
                value = row[engine + "_" + phase]
                end = origin + value / limit * 54
                top = y - j * 3.5
                low = origin + row[engine + "_" + phase + "_min"] / limit * 54
                high = origin + row[engine + "_" + phase + "_max"] / limit * 54
                chart.extend([rf"\fill[{color}] ({origin},{top}) rectangle ({end:.4f},{top - 2.8});",
                              rf"\node[anchor=west,inner sep=1pt] at ({high:.4f},{top - 1.4}) {{{value:.1f}}};"])
                chart.append(rf"\draw[black,|-|] ({low:.4f},{top - 1.4}) -- ({high:.4f},{top - 1.4});")
    chart.extend([rf"\fill[blue!65!black] (49,{bottom - 11}) rectangle (53,{bottom - 8});",
                  rf"\node[anchor=west] at (54,{bottom - 9.5}) {{Meganeura}};",
                  rf"\fill[orange!80!black] (85,{bottom - 11}) rectangle (89,{bottom - 8});",
                  rf"\node[anchor=west] at (90,{bottom - 9.5}) {{PyTorch}};",
                  r"\end{tikzpicture}", ""])
    output["smollm2.tex"] = "\n".join(chart)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", type=Path, help="campaign .tgz directory or lossless records.jsonl.xz bundle")
    parser.add_argument("--output", type=Path, help="write generated tables and per-condition CSV here")
    parser.add_argument("--check", type=Path, help="compare generated tables against this directory")
    parser.add_argument("--bundle", type=Path, help="export all original JSON content, losslessly compressed (no logs)")
    args = parser.parse_args()
    require(args.bundle is None or args.bundle.resolve() != args.archives.resolve(), "cannot overwrite the input bundle")
    campaigns, rows, all_groups = {}, {}, {}
    digests = {}
    for line in (HERE / "cohort.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        require(Path(name).name == name, "manifest path is not a filename")
        digests[name] = expected
        if args.archives.is_dir():
            with (args.archives / name).open("rb") as file:
                require(hashlib.file_digest(file, "sha256").hexdigest() == expected, "input digest differs: " + name)
    print("Campaign archive SHA-256 digests match" if args.archives.is_dir() else "Auditing lossless JSON-content bundle")
    input_hashes = {}
    with (nullcontext(None) if args.archives.is_dir() else lzma.open(args.archives, "rt")) as source, \
            (lzma.open(args.bundle, "wt") if args.bundle else nullcontext(None)) as bundle:
        if source:
            require(json.loads(next(source)) == digests, "bundle provenance differs")
        if bundle:
            bundle.write(json.dumps(digests) + "\n")
        for device in DEVICES:
            path = args.archives / (device + ".tgz")
            records = json.loads(next(source)) if source else read_records(path)
            c, groups, failed = load_campaign(path, records)
            require(c["status"] == "complete" and not failed, "incomplete final campaign")
            for name, digest in c["sha256"].items():
                name = name.replace("\\", "/")
                require(input_hashes.setdefault(name, digest) == digest, "input identity differs: " + name)
            campaigns[device], all_groups[device] = c, groups
            rows[device] = aggregate(groups) if c["args"]["backend"] != "cpu" else {}
            if bundle:
                bundle.write(json.dumps(records, separators=(",", ":")) + "\n")
            print(device, c["status"], dict(Counter(run["status"] for run in c["runs"])), "failed:", failed)
        if source:
            require(not source.read().strip(), "unexpected trailing campaign")
    generated = tables(campaigns, rows, all_groups)
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
        print("\n", precision, "complete GPU-reference cohort")
        for phase in PHASES:
            values = [rows[device][precision, model, primary_condition(c)]["ratio_" + phase]
                      for device, c in campaigns.items() if device != "nvidia-h100-large" and c["args"]["backend"] != "cpu"
                      for model in MODELS]
            print(phase, "wins", sum(x < 1 for x in values), "/", len(values), "median ratio", statistics.median(values))
    errors = [(run["errors"], device, key) for device, groups in all_groups.items() for key, runs in groups.items() for run in runs]
    for key in errors[0][0]:
        print("maximum", key, max((e[key], device, group) for e, device, group in errors))
    searches = [session["search"] for device, groups in all_groups.items()
                if campaigns[device]["args"]["backend"] != "cpu"
                for runs in groups.values() for run in runs
                for session in run["pair"]["meganeura"]["optimizer"]["sessions"]]
    print("GPU-paired native search:", len(searches), "sessions;",
          sum(s["visited_classes"] == s["eligible_classes"] for s in searches), "complete;",
          sum(s["visited_classes"] for s in searches), "/",
          sum(s["eligible_classes"] for s in searches), "classes; longest",
          max(s["elapsed_seconds"] for s in searches), "seconds")
    print("GPU-paired decisions:", sum((Counter(s["decisions"]) for s in searches), Counter()))


if __name__ == "__main__":
    main()
