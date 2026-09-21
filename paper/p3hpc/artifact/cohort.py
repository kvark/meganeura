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
INFERENA = "7b8fcb72e55410d8e33bbe948f89be50c691fcf7"
MEGANEURA = "0dbfcc0029bf98b33a03fc792e1f4e90ead17f23"
TORCH = "cf30153c4c131c8164ee7798e5022d810682e2cb"
MODELS = ("SmolLM2-135M", "SmolVLA", "StableDiffusion", "ResNet-50", "Whisper-tiny")
PHASES = ("inference", "latency", "training")
DEVICES = {
    "nvidia-5070": "RTX 5070",
    "nvidia-h100": "H100",
    "nvidia-3050": "RTX 3050",
    "amd-7900xt": "RX 7900 XT",
    "intel-b570": "Arc B570",
    "apple-m3": "Apple M3",
}
OPERATING_SYSTEMS = {"nvidia-3050": "Windows", "apple-m3": "macOS"}
MESA_DRIVERS = {"amd": "RADV", "intel": "ANV"}
BACKENDS = {"cuda": "CUDA", "rocm": "ROCm", "xpu": "XPU", "mps": "MPS"}
MODEL_LABELS = {"StableDiffusion": "SD U-Net"}
QUALIFICATIONS = {"amd-780m-qualify": "Radeon 780M",
                  "amd-9600x-qualify": "Ryzen 9600X iGPU",
                  "intel-rplu-qualify": "Intel RPL-U"}
ENGINES = ("meganeura", "pytorch")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def seconds(duration):
    return duration["secs"] + duration["nanos"] / 1e9


def roundtrip_equal(a, b):
    # The diagnostic reserializes JSON through Rust; decimal parsing can move
    # a float by one ULP. This is a file-identity check, not a numerical gate.
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(roundtrip_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(roundtrip_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= 2 * max(math.ulp(a), math.ulp(b))
    return a == b


def audit_search(session):
    search = session["search"]
    options, trials = search["options"], search["trials"]
    tuning, memory = options["tuning"], session["memory_budget"]
    require(seconds(options["max_time"]) == 60 and options["max_graphs"] == 16
            and options["max_programs"] == 64 and options["warmup_runs"] == 2
            and seconds(options["warmup_time"]) == 0.25
            and memory["plan_fraction_of_available"] == 0.75
            and options["max_plan_bytes"] == (memory["device_budget_bytes"] - memory["device_usage_bytes"]) // 4 * 3
            and options["max_plan_bytes"] > 0, "construction limits changed")
    require(tuning["scope"] == "All" and tuning["max_classes"] == 2**64 - 1
            and seconds(tuning["max_time"]) == 60 and tuning["max_scratch_bytes"] == 1024**3,
            "kernel search limits changed")
    require(trials and 0 <= search["selected"] < len(trials), "no selected program")
    selected = trials[search["selected"]]
    require(selected["outcome"]["qualified"] is True and selected["kernel_tuning"] is not None,
            "selected program was not tuned and qualified")
    qualification = session["qualification"]
    require(qualification["policy"] == "fixed-full-tensor-v4"
            and qualification["reference"] == "ordinary untuned construction"
            and qualification["rtol"] == 1e-4 and qualification["atol"] == 1e-6
            and qualification["accelerated_gradient_rtol"] == 0.01
            and qualification["qualified_calls"] >= 2 and qualification["output_elements"] > 0
            and (session["mode"] != "Training" or qualification["gradient_elements"] > 0),
            "missing native full-output/gradient qualification")
    decisions, phases = Counter(), Counter()
    visited = eligible = rejections = 0
    for trial in trials:
        outcome, kernel = trial["outcome"], trial["kernel_tuning"]
        require(outcome["decision"] != "FasterCandidate" or outcome["qualified"] is True,
                "unqualified program accepted")
        rejections += outcome["decision"] == "InvalidOutput"
        for field in ("construction_time", "initialization_time", "lowering_time", "qualification_time", "state_copy_time"):
            phases[field] += seconds(trial[field])
        phases["program_elapsed"] += seconds(outcome["elapsed"])
        if kernel is None:
            continue
        k = kernel["options"]
        require(k["scope"] == "All" and k["max_classes"] == tuning["max_classes"]
                and k["max_scratch_bytes"] == 1024**3 and 0 <= seconds(k["max_time"]) <= 60
                and not kernel["class_limit_reached"], "program kernel limits changed")
        require(0 <= kernel["visited_classes"] <= kernel["eligible_classes"]
                and (kernel["visited_classes"] == kernel["eligible_classes"] or kernel["time_budget_exhausted"]),
                "unexplained kernel search truncation")
        visited += kernel["visited_classes"]
        eligible += kernel["eligible_classes"]
        phases["kernel_search"] += seconds(kernel["elapsed"])
        for outcome in kernel["outcomes"]:
            require(outcome["decision"] != "FasterCandidate" or outcome["qualified"] is True,
                    "unqualified kernel accepted")
            decisions[outcome["decision"]] += 1
    require(math.isfinite(seconds(search["elapsed"])) and seconds(search["elapsed"]) >= 0,
            "invalid construction duration")
    return {"elapsed_seconds": seconds(search["elapsed"]), "graphs": len(search["graphs"]),
            "programs": len(trials), "truncated": search["truncated"],
            "extraction_truncated": search["extraction_truncated"], "skipped_regions": search["skipped_regions"],
            "visited_classes": visited, "eligible_classes": eligible, "decisions": dict(decisions),
            "program_rejections": rejections, "selected_graph": int(selected["description"].split(",")[0].split("=")[1]),
            "selected_description": selected["description"], "phase_seconds": dict(phases)}


def relative_l2(actual, reference):
    require(len(actual) == len(reference) > 0, "vector length mismatch")
    require(all(math.isfinite(x) for x in (*actual, *reference)), "nonfinite vector")
    delta = sum((x - y) ** 2 for x, y in zip(actual, reference))
    norm = sum(x * x for x in reference)
    return math.sqrt(delta / norm) if norm else (math.inf if delta else 0.0)


def audit_replay(validation, training, accelerated, capture):
    require(validation["rtol"] == 1e-4 and validation["atol"] == 1e-6
            and validation["accelerated_gradient_rtol"] == 0.01, "replay bounds changed")
    anchor = validation["uncaptured"][0]
    for stage, count in (("uncaptured", validation["uncaptured_repeats"]), ("replays", 2 if capture else 0)):
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


def audit_pair(pair, campaign, run, phases=PHASES, oracle=False):
    require(set(pair) == set(ENGINES), "missing engine")
    mg, pt = (pair[engine] for engine in ENGINES)
    count = 20 if campaign["args"]["collect"] else 1
    for engine, record in pair.items():
        require(record["status"] in ("ok", "partial"), "failed engine in a valid phase")
        require(record["model"] == run["path"].replace("\\", "/").split("/")[3], "run model differs")
        require(campaign["source"].startswith(record["benchmark_rev"]), "wrong harness revision")
        require(record["protocol"]["warmup_runs"] == 5, "warmup count changed")
        samples_per_phase = 1 if oracle and engine == "pytorch" else count
        require(record["protocol"]["measurement_runs"] == samples_per_phase, "sample count changed")
        require(record["protocol"]["training_requested"] is True, "training scope changed")
        require(record["protocol"]["diagnostic"] == (oracle and engine == "pytorch"), "diagnostic timing mixed with campaign")
        require(record["protocol"]["synthetic_parameter_init"] == "name-index-uniform-v1"
                and record["protocol"]["warmup_seconds"] == 2, "initialization or warmup policy changed")
        for phase in phases:
            warmup = record["protocol"]["warmup"][phase]
            require(warmup["runs"] >= 5 and math.isfinite(warmup["seconds"]) and warmup["seconds"] >= 2,
                    "insufficient workload warmup")
            samples = record["timing_samples_ms"][phase]
            require(len(samples) == samples_per_phase and all(math.isfinite(x) and x > 0 for x in samples),
                    "invalid timing samples")
            median = statistics.median(samples)
            require(math.isclose(median, record["timing_summary_ms"][phase]["median"],
                                 rel_tol=0, abs_tol=1e-6), "stored median differs")
            require(abs(median - record["timings"][phase + "_ms"]) <= 0.000501,
                    "rounded timing differs")
    require(campaign["meganeura"]["rev"].startswith(mg["framework_rev"]), "wrong Meganeura revision")
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
    require(mg["optimizer"].get("measured_construction") is True,
            "native search disabled")
    require(mg["optimizer"]["mode"] == "egglog-outlined", "wrong graph optimizer")
    require(mg["protocol"]["name"] == "inferena-paper-v3", "native protocol changed")
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
    execution = pt["execution"]
    mode = "default" if campaign["args"]["collect"] and not oracle else "eager"
    require(execution["requested_mode"] == mode and (oracle or run["mode"] == mode)
            and execution["compiled"] == (mode == "default"), "compiler fallback")
    require(execution["compile_budget_seconds"] == 120.0 and execution["compile_budget_enforced"], "watchdog missing")
    if mode == "default":
        require(not any(execution["compiler_options"].values()), "unexpected compiler search or internal replay")
    require(execution["sdpa_policy"] == ("math" if backend == "xpu" or oracle else "auto"), "attention policy changed")
    require(execution["sdpa_compile"] == ("compiled" if mode == "default" else "eager"), "SDPA compilation changed")
    allowed = {"MATH"} if backend == "xpu" or oracle else {"MATH", "CUDNN_ATTENTION", "FLASH_ATTENTION", "EFFICIENT_ATTENTION", "OVERRIDEABLE"}
    require(set(execution["sdpa_enabled_backends"]) == allowed, "attention backends differ")
    capture = backend in ("cuda", "rocm", "xpu") and not oracle
    require(execution["graph_replay"]["requested"] == capture and (oracle or run["graphs"] == capture),
            "graph request differs")
    require(pt["protocol"]["name"] == "inferena-graph-replay-v7", "reference protocol changed")
    for phase in phases:
        graph = execution["graph_replay"]["phases"][phase]
        require(graph["status"] == ("captured-and-validated" if capture else "validated-uncaptured"),
                "unqualified replay")
        if capture:
            require(graph["api"] == ("torch.xpu.XPUGraph" if backend == "xpu" else "torch.cuda.CUDAGraph")
                    and execution["stream_policy"] == "single dedicated preparation/run stream", "wrong replay API/stream")
        v = graph["validation"]
        repeats = 8 if phase == "training" and pt["precision"]["reduced_precision_allowed"] else 2
        require(v["policy"] == "fixed-full-tensor-v4" and v["uncaptured_repeats"] == repeats
                and v["output_metric"] == "per-tensor RMS and maximum absolute error"
                and v["uncaptured_calls"] == repeats + 1 and v["consecutive_replays"] == (2 if capture else 0),
                "replay validation policy changed")
        audit_replay(v, phase == "training", phase == "training" and pt["precision"]["reduced_precision_allowed"], capture)
    a, b = mg["outputs"], pt["outputs"]
    require(a["output_shape"] == b["output_shape"], "output shape mismatch")
    require(len(a["logits_sample"]) == len(b["logits_sample"]) == 256, "sample inventory differs")
    names = sorted(b["gradient_norms"])
    errors = {
        "output_relative_l2_error": relative_l2(a["logits_sample"], b["logits_sample"]),
        "loss_relative_error": abs(a["loss"] - b["loss"]) / max(abs(a["loss"]), abs(b["loss"]), 1e-12),
    }
    if "training" in phases:
        require(a["gradient_norms"].keys() == b["gradient_norms"].keys(), "gradient names differ")
        errors.update(total_gradient_relative_error=abs(a["grad_norm"] - b["grad_norm"]) / max(abs(b["grad_norm"]), 1e-12),
                      parameter_gradient_relative_l2_error=relative_l2(
                          [a["gradient_norms"][name] for name in names], [b["gradient_norms"][name] for name in names]))
    for key, value in errors.items():
        require(math.isfinite(value) and math.isclose(value, mg["validation"][key], rel_tol=1e-9, abs_tol=1e-12),
                "retained evidence disagrees with " + key)
    require(errors["output_relative_l2_error"] < 0.01 and errors["loss_relative_error"] < 0.01,
            "forward gate failed")
    limit = 0.10 if mg["precision"]["reduced_precision_allowed"] and campaign["args"]["collect"] and not oracle else 0.05
    require(all(errors[key] < limit for key in errors if "gradient" in key), "gradient sample gate failed")
    require(mg["validation"]["forward_valid"] is True, "stored forward gate differs")
    if "training" in phases and not oracle:
        require(mg["validation"]["training_valid"] == all(errors[key] < 0.05 for key in errors if "gradient" in key),
                "stored gradient gate differs")
    return errors


def archive_records(path):
    if path.suffix == ".txt":
        yield path.name, path.read_text()
        return
    with tarfile.open(path, mode="r|gz") as archive:
        for member in archive:
            if member.isfile() and member.name.endswith((".json", ".log", ".txt")):
                data = archive.extractfile(member).read().decode("utf-8")
                yield member.name, json.loads(data) if member.name.endswith(".json") else data


def read_records(entries, bundle=None):
    # Full optimizer receipts total several GiB. Check one JSON file at a time;
    # keep only audited summaries in memory, while the bundle retains every value.
    records = {}
    for name, value in entries:
        require(name not in records, "duplicate archive member: " + name)
        if bundle:
            bundle.write(json.dumps([name, value], separators=(",", ":")) + "\n")
        for record in value if isinstance(value, list) else [value]:
            if not isinstance(record, dict) or "framework" not in record:
                continue
            record["source_digest"] = hashlib.sha256(json.dumps(record, sort_keys=True,
                separators=(",", ":")).encode()).hexdigest()
            if record["framework"] == "meganeura" and record["status"] == "ok":
                for session in record["optimizer"]["sessions"]:
                    session["search"] = audit_search(session)
        records[name] = value
    return records


def load_campaign(path, records):
    manifests = [name for name in records if name.endswith("/campaign.json")]
    require(len(manifests) == 1, "expected one campaign in " + path.name)
    prefix = str(PurePosixPath(manifests[0]).parent) + "/"
    campaign = records[manifests[0]]
    require(campaign["source"] == INFERENA and campaign["meganeura"]["rev"] == MEGANEURA,
            "campaign revision mismatch")
    require(campaign["torch"]["git_version"] == TORCH, "campaign PyTorch mismatch")
    require(campaign["protocol"] == "p3hpc-paired-campaign-v14"
            and campaign["synthetic_parameter_init"] == "name-index-uniform-v1",
            "campaign protocol mismatch")
    require(campaign["args"]["replicates"] == 3, "replicate policy changed")
    require(campaign["warmup"] == {"minimum_runs": 5, "minimum_seconds": 2.0}
            and campaign["reference_compile_seconds"] == 120.0
            and campaign["failure_policy"] == {"recoverable": ["numerical", "capture", "cross-engine-mismatch"],
                "eager_diagnostic_attempts": 1, "eager_sdpa": "math", "timing_substitution": False,
                "retry_failed_primary": False}, "preparation or failure policy changed")
    expected_models = (["SmolLM2-360M", "SmolLM2-1.7B"] if path.stem == "nvidia-h100-large"
                       else ["SmolLM2-360M"] if "large" in path.stem else list(MODELS))
    require(campaign["args"]["models"] == expected_models
            and campaign["args"]["precisions"] == ["strict", "accelerated"], "campaign workload coverage changed")
    policy = {"scope": "All", "class_limit": None, "max_scratch_bytes": 1024**3,
              "search_seconds_per_session": 60.0, "strict_coop": "NativeF32"}
    policy.update({
        "measured_construction": True, "optimizer": "egglog-outlined", "kernel_seconds_per_program": 60.0,
        "max_graphs": 16, "max_programs": 64, "plan_fraction_of_available": 0.75,
        "warmup_pairs": 2, "warmup_seconds": 0.25,
        "qualification": "fixed-full-tensor-v4"})
    require(campaign["native_policy"] == policy,
            "manifest native policy differs")
    groups = defaultdict(list)
    failed = []
    seen = set()
    for run in campaign["runs"]:
        folder = run["path"].replace("\\", "/")
        require(folder not in seen, "duplicate run")
        seen.add(folder)
        stage, replicate, precision, model, condition = folder.split("/")
        require(model in expected_models and precision in campaign["args"]["precisions"], "unexpected workload or precision")
        require(stage == ("measurement" if campaign["args"]["collect"] else "qualification"), "wrong run stage")
        require(condition == run["mode"] + "-graph" + str(int(run["graphs"])), "condition path differs")
        require(replicate in ("r1", "r2", "r3"), "unexpected replicate")
        summary = records.get(prefix + folder + "/" + model + "_summary.json", [])
        pair = {record["framework"]: record for record in summary}
        for engine, record in pair.items():
            require(record == records[prefix + folder + "/" + model + "_" + engine + ".json"],
                    "joined record differs from raw record")
        require(set(pair) == set(ENGINES) and pair["meganeura"]["status"] == "ok", "missing native result")
        if run["status"] != "failed" and campaign["args"]["collect"]:
            preparation = records[prefix + folder + "/torch-preparation.json"]
            require(preparation["status"] == "complete" and preparation["budget_seconds"] == 120
                    and 0 <= preparation["elapsed_seconds"] <= 120
                    and 0 <= pair["pytorch"]["timings"]["compile_s"] <= 120,
                    "reference compilation receipt differs")
        native = pair["meganeura"]
        require(MEGANEURA.startswith(native["framework_rev"])
                and native["environment"]["gpu_device_id"] == campaign["native_device"]["device_id"]
                and not native["environment"]["gpu_software_emulated"], "native identity differs")
        if run["status"] == "failed":
            require(pair["pytorch"]["status"] == "error" and campaign["status"] == "incomplete",
                    "unrecognized hard failure")
            failed.append({"path": folder, "engines": {engine: record["status"] for engine, record in pair.items()},
                           "errors": {engine: record.get("error") for engine, record in pair.items()
                                      if record["status"] != "ok"},
                           "preparation": records.get(prefix + folder + "/torch-preparation.json")})
            continue
        phases = tuple(phase for phase in PHASES if run["phases"][phase]["status"] == "valid")
        require(run["status"] == ("valid" if len(phases) == 3 else "partial"), "phase status differs")
        errors = audit_pair(pair, campaign, run, phases)
        diagnostic_errors = None
        if len(phases) != 3:
            pt = pair["pytorch"]
            failure = pt["execution"]["failure"]
            require(pt["status"] == "partial" and failure["kind"] == "numerical", "unclassified partial result")
            metrics = failure["metrics"]
            require(metrics["max_abs_error"] > metrics["max_abs_bound"] or metrics["rms_error"] > metrics["rms_bound"],
                    "numerical failure does not exceed its bounds")
            for phase in set(PHASES) - set(phases):
                require(pt["timings"][phase + "_ms"] is None and pt["timing_samples_ms"][phase] is None,
                        "failed or unreached phase has timings")
                require((run["phases"][phase]["status"] == "failed") == (phase == failure["phase"]),
                        "failure attributed to the wrong phase")
                require(run["phases"][phase]["status"] == pt["execution"]["graph_replay"]["phases"][phase]["status"],
                        "manifest phase status differs")
            diagnostic = run["eager_diagnostic"]
            require(diagnostic["status"] == "valid" and diagnostic["timing_substituted"] is False
                    and diagnostic["mode"] == "eager" and not diagnostic["graph_replay"], "diagnostic substituted")
            diag_prefix = prefix + folder + "/diagnostic-eager/"
            comparison = {r["framework"]: r for r in records[diag_prefix + "comparison.json"]}
            for engine, original in (("meganeura", native), ("pytorch", records[diag_prefix + model + "_pytorch.json"])):
                require(roundtrip_equal(
                        {k: v for k, v in comparison[engine].items() if k not in ("validation", "source_digest")},
                        {k: v for k, v in original.items() if k not in ("validation", "source_digest")}),
                        "diagnostic changed original data")
            diagnostic_errors = audit_pair(comparison, campaign, run, oracle=True)
            failed.append({"path": folder, "failure": failure, "eager_diagnostic_errors": diagnostic_errors})
        require(pair["meganeura"]["precision"]["reduced_precision_allowed"] == (precision == "accelerated"),
                "precision label differs")
        groups[precision, model, condition].append({"replicate": replicate, "pair": pair, "errors": errors,
                                                    "phases": phases, "diagnostic_errors": diagnostic_errors})
    if campaign["status"] in ("complete", "complete-with-failures"):
        require((not failed) == (campaign["status"] == "complete"), "campaign failure status differs")
        expected = {(precision, model, condition["mode"] + "-graph" + str(int(condition["graph_replay"])))
                    for precision in campaign["args"]["precisions"] for model in campaign["args"]["models"]
                    for condition in campaign["reference_conditions"]["selected"]}
        require(set(groups) == expected, "missing selected condition")
        replicates = {"r1", "r2", "r3"} if campaign["args"]["collect"] else {"r1"}
        require(all({run["replicate"] for run in runs} == replicates and len(runs) == len(replicates)
                    for runs in groups.values()), "incomplete replication")
        for phase in PHASES:
            require(campaign["phase_coverage"][phase] == {"planned": len(campaign["runs"]),
                    "valid": sum(phase in run["phases"] for runs in groups.values() for run in runs)},
                    "phase coverage differs")
    for runs in groups.values():
        runs = [run for run in runs if "training" in run["phases"]]
        for key in ("parameter_gradient_relative_l2_error", "total_gradient_relative_error"):
            if len(runs) == 3:
                require(statistics.median(run["errors"][key] for run in runs) < 0.05, "replicated gradient gate failed")
    report = campaign.get("replicated_gradient_validation")
    if report is None:
        require(not campaign["args"]["collect"] or campaign["status"] != "complete", "missing replication report")
        return campaign, groups, failed
    require(report["policy"] == "replicated-gradient-median-v1" and len(report["groups"]) == len(groups)
            and report["status"] == ("pass" if all(g["status"] == "pass" for g in report["groups"]) else "fail"),
            "replicated report policy/inventory differs")
    checked = set()
    for group in report["groups"]:
        key = (group["precision"], group["model"], group["mode"] + "-graph" + str(int(group["graph_replay"])))
        require(key not in checked and key in groups, "duplicate or unknown replicated group")
        checked.add(key)
        runs = [run for run in groups[key] if "training" in run["phases"]]
        if len(runs) != 3:
            require(group["status"] == "incomplete" and group["valid_replicates"] == len(runs)
                    and group["required_replicates"] == 3, "invalid incomplete-replication report")
            continue
        require(group["status"] == "pass" and group["median_limit"] == 0.05
                and group["sample_limit"] == (0.1 if key[0] == "accelerated" else 0.05), "replication bounds changed")
        require(set(group["metrics"]) == {"parameter_gradient_relative_l2", "total_gradient_relative"},
                "missing replicated metric")
        for metric, stored in group["metrics"].items():
            values = [run["errors"][metric + "_error"] for run in runs]
            require(len(stored["samples"]) == 3
                    and all(math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12) for a, b in zip(values, stored["samples"]))
                    and math.isclose(statistics.median(values), stored["median"], rel_tol=1e-9, abs_tol=1e-12),
                    "replicated report differs from retained errors")
    return campaign, groups, failed


def aggregate(groups):
    rows = {}
    for key, runs in groups.items():
        row = {"replicates": len(runs)}
        for phase in PHASES:
            row["replicates_" + phase] = sum(phase in run["phases"] for run in runs)
        for engine in ENGINES:
            for phase in PHASES:
                values = [statistics.median(run["pair"][engine]["timing_samples_ms"][phase])
                          for run in runs if phase in run["phases"]
                          or (engine == "meganeura" and run["diagnostic_errors"] is not None)]
                row[engine + "_" + phase + "_replicates"] = len(values)
                if not values:
                    for suffix in ("", "_min", "_max", "_spread"):
                        row[engine + "_" + phase + suffix] = None
                    continue
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
            row["ratio_" + phase] = (row["meganeura_" + phase] / row["pytorch_" + phase]
                                    if row["replicates_" + phase] else None)
        rows[key] = row
    return rows


def primary_condition(campaign):
    return ("eager" if campaign["args"]["eager"] else "default") + "-graph" + str(int(campaign["args"]["backend"] in ("cuda", "rocm", "xpu")))


def spike_sensitivity(groups):
    # Post-hoc sensitivity only; no sample or primary aggregate is changed.
    samples = dropped = changed = 0
    largest_change = 0.0
    for runs in groups.values():
        for engine in ENGINES:
            for phase in PHASES:
                original, trimmed = [], []
                for run in runs:
                    values = run["pair"][engine]["timing_samples_ms"][phase]
                    median = statistics.median(values)
                    kept = [value for value in values if value <= 3 * median]
                    samples += len(values)
                    dropped += len(values) - len(kept)
                    original.append(median)
                    trimmed.append(statistics.median(kept))
                change = abs(statistics.median(trimmed) / statistics.median(original) - 1)
                changed += change != 0
                largest_change = max(largest_change, change)
    return {"samples": samples, "dropped": dropped, "changed_aggregates": changed,
            "largest_relative_change": largest_change}


def complete_devices(campaigns, rows, precision):
    return [device for device, c in campaigns.items() if c["args"]["backend"] != "cpu"
            and all(rows[device].get((precision, model, primary_condition(c)), {}).get("replicates_" + phase) == 3
                   for model in MODELS for phase in PHASES)]


def tex_table(columns, heading, lines):
    return "\n".join([r"\begin{tabular}{" + columns + "}", r"\toprule", heading + r" \\",
                      r"\midrule", *[" & ".join(row) + r" \\" for row in lines], r"\bottomrule", r"\end{tabular}", ""])


def ratio(value):
    if value is None:
        return "--"
    text = f"{value:.2f}"
    return r"\textbf{" + text + "}" if value < 1 else text


def portability_score(rows, engine, phase):
    field = engine + "_" + phase
    if any(row[field + "_replicates"] != 3 for row in rows):
        return 0.0
    costs = []
    for row in rows:
        best = min(row[e + "_" + phase] for e in ENGINES
                   if row[e + "_" + phase + "_replicates"] == 3)
        costs.append(row[field] / best)
    return len(costs) / sum(costs)


def label(model):
    return MODEL_LABELS.get(model, model)


def driver_label(device, campaign):
    info = campaign["native_device"]["driver_info"].split("-")[0]
    if not info:
        return "built-in"
    return f"{info} ({MESA_DRIVERS[device.split('-')[0]]})" if info.startswith("Mesa") else info


def ratio_plot(campaigns, rows):
    # Strict Meganeura/PyTorch ratios on a log axis, one row per GPU and one
    # marker per workload. A phase without a qualified reference is omitted.
    low, high, width, gap, left, pitch = 0.15, 6.0, 50, 7, 21, 6
    marks = {"SmolLM2-135M": ("circle", "0072B2"), "SmolVLA": ("square", "E69F00"),
             "StableDiffusion": ("triangle", "009E73"), "ResNet-50": ("diamond", "D55E00"),
             "Whisper-tiny": ("cross", "CC79A7")}

    def x_of(origin, value):
        return origin + math.log(value / low) / math.log(high / low) * width

    def marker(shape, color, x, y):
        c = "p" + color
        if shape == "circle":
            return rf"\fill[{c}] ({x:.3f},{y:.3f}) circle (0.8);"
        if shape == "square":
            return rf"\fill[{c}] ({x - 0.72:.3f},{y - 0.72:.3f}) rectangle ({x + 0.72:.3f},{y + 0.72:.3f});"
        if shape == "triangle":
            return rf"\fill[{c}] ({x:.3f},{y + 0.95:.3f}) -- ({x - 0.9:.3f},{y - 0.7:.3f}) -- ({x + 0.9:.3f},{y - 0.7:.3f}) -- cycle;"
        if shape == "diamond":
            return (rf"\fill[{c}] ({x:.3f},{y + 1:.3f}) -- ({x + 0.85:.3f},{y:.3f}) -- "
                    rf"({x:.3f},{y - 1:.3f}) -- ({x - 0.85:.3f},{y:.3f}) -- cycle;")
        return (rf"\draw[{c},line width=0.45mm] ({x - 0.75:.3f},{y - 0.75:.3f}) -- ({x + 0.75:.3f},{y + 0.75:.3f}) "
                rf"({x - 0.75:.3f},{y + 0.75:.3f}) -- ({x + 0.75:.3f},{y - 0.75:.3f});")

    devices = list(DEVICES)
    top, bottom = pitch / 2, pitch / 2 - len(devices) * pitch
    chart = [r"\begin{tikzpicture}[x=1mm,y=1mm,font=\scriptsize]",
             *[rf"\definecolor{{p{color}}}{{HTML}}{{{color}}}" for _, color in marks.values()]]
    for i in range(1, len(devices), 2):
        chart.append(rf"\fill[black!5] ({left - 20},{top - i * pitch}) rectangle "
                     rf"({left + 3 * width + 2 * gap},{top - (i + 1) * pitch});")
    for i, device in enumerate(devices):
        chart.append(rf"\node[anchor=east] at ({left - 1},{-i * pitch}) {{{DEVICES[device]}}};")
    for panel, (phase, title) in enumerate((("inference", "Inference"), ("latency", "Minimal shape"),
                                            ("training", "F+L+B"))):
        origin = left + panel * (width + gap)
        chart.append(rf"\node at ({origin + width / 2},{top + 3}) {{{title}}};")
        for tick in (0.25, 0.5, 1, 2, 4):
            x = x_of(origin, tick)
            style = "black!55,semithick" if tick == 1 else "black!15"
            chart.extend([rf"\draw[{style}] ({x:.3f},{top}) -- ({x:.3f},{bottom});",
                          rf"\node[below] at ({x:.3f},{bottom}) {{{tick:g}}};"])
        chart.append(rf"\draw[black!40] ({origin},{bottom}) rectangle ({origin + width},{top});")
        for i, device in enumerate(devices):
            row_y = -i * pitch
            for j, model in enumerate(MODELS):
                value = rows[device]["strict", model, primary_condition(campaigns[device])]["ratio_" + phase]
                if value is None:
                    continue
                require(low < value < high, "ratio outside the plotted range")
                shape, color = marks[model]
                chart.append(marker(shape, color, x_of(origin, value), row_y + (2 - j) * 0.95))
    legend_y = bottom - 9
    x = left + 12
    for model in MODELS:
        shape, color = marks[model]
        chart.extend([marker(shape, color, x, legend_y),
                      rf"\node[anchor=west] at ({x + 1.3},{legend_y}) {{{label(model)}}};"])
        x += 31
    chart.extend([r"\end{tikzpicture}", ""])
    return "\n".join(chart)


def tables(campaigns, rows, groups):
    output = {}
    paired_devices = DEVICES
    lines = []
    for device, name in paired_devices.items():
        c = campaigns[device]
        backend = c["args"]["backend"]
        valid = [sum(phase in run["phases"] for batch in groups[device].values() for run in batch) for phase in PHASES]
        replay = {"cuda": "CUDA Graph", "rocm": "HIP graph", "xpu": "XPU graph"}.get(backend, "none")
        lines.append([name, OPERATING_SYSTEMS.get(device, "Linux"), "Metal" if backend == "mps" else "Vulkan",
                      driver_label(device, c), BACKENDS[backend], replay, "/".join(map(str, valid))])
    output["devices.tex"] = tex_table("lllllll", "GPU & OS & API & Driver & PyTorch & Replay & Pairs (inf./min./F+L+B)", lines)
    lines = []
    for device, name in QUALIFICATIONS.items():
        conditions = len(groups[device])
        reference = {"intel-rplu-qualify": r"no XPU device$^\dagger$",
                     "amd-9600x-qualify": r"ROCm crash, wrong output$^\dagger$",
                     "amd-780m-qualify": "HIP launch failure"}[device]
        lines.append([name, f"Vulkan {conditions}/10", reference])
    lines.append([r"MI300X$^\dagger$", "no Vulkan driver", "strict 135M ran"])
    output["qualification.tex"] = tex_table("lll", "GPU & Meganeura & PyTorch GPU path", lines)
    lines = []
    for device, name in paired_devices.items():
        c = campaigns[device]
        condition = primary_condition(c)
        for model in MODELS:
            values = []
            for precision in ("strict", "accelerated"):
                row = rows[device][precision, model, condition]
                require(row["replicates"] == 3, "paired group lacks three processes")
                for phase in PHASES:
                    value = ratio(row["ratio_" + phase])
                    if row["ratio_" + phase] is None and row["meganeura_" + phase + "_replicates"] == 3:
                        value = rf"{row['meganeura_' + phase]:.2f}\,ms$^\dagger$"
                    values.append(value)
            lines.append([name if model == MODELS[0] else "", label(model), *values])
    ratio_heading = (r"GPU & Workload & \multicolumn{3}{c}{Strict} & \multicolumn{3}{c}{Accelerated} \\"
                     r" \cmidrule(lr){3-5}\cmidrule(lr){6-8}"
                     r" & & Inf. & Min. & F+L+B & Inf. & Min. & F+L+B")
    output["ratios.tex"] = tex_table("llrrrrrr", ratio_heading, lines)
    lines = []
    for device, name in paired_devices.items():
        runs = [run for batch in groups[device].values() for run in batch]
        compile_s = [sum(run["pair"][engine]["timings"]["compile_s"] for run in runs) for engine in ENGINES]
        graph_s = [sum(phase.get(part + "_s", 0) for run in runs
                       for phase in run["pair"]["pytorch"]["execution"]["graph_replay"]["phases"].values())
                   for part in ("capture", "validation")]
        lines.append([name, str(len(runs)), *[f"{value / 60:.2f}" for value in (*compile_s, *graph_s)]])
    output["preparation.tex"] = tex_table("lrrrrr",
        r"GPU & Processes & M prepare & P compile & P capture & P check", lines)
    lines = []
    for device, name in paired_devices.items():
        searches = [session["search"] for batch in groups[device].values() for run in batch
                    for session in run["pair"]["meganeura"]["optimizer"]["sessions"]]
        lines.append([name, str(len(searches)),
                      str(sum(s["programs"] for s in searches)), str(sum(s["truncated"] for s in searches)),
                      str(sum(s["selected_graph"] != 0 for s in searches)),
                      str(sum(s["program_rejections"] for s in searches))])
    output["search.tex"] = tex_table("lrrrrr",
        r"GPU & Sessions & Programs & Out of time & Changed graph & Rejected", lines)
    lines = []
    for device in ("nvidia-5070", "amd-7900xt", "nvidia-h100"):
        first = True
        for model in ("SmolLM2-135M", "SmolLM2-360M", "SmolLM2-1.7B"):
            if model.endswith("1.7B") and device != "nvidia-h100":
                continue
            for precision in ("strict", "accelerated"):
                source = device if model.endswith("135M") else device + "-large"
                row = rows[source][precision, model, "default-graph1"]
                require(row["replicates"] == 3, "size-study group lacks three processes")
                lines.append([DEVICES[device] if first else "",
                              model.removeprefix("SmolLM2-") if precision == "strict" else "",
                              "S" if precision == "strict" else "A",
                              *[f"{row['meganeura_' + phase]:.2f} ({ratio(row['ratio_' + phase])})" for phase in PHASES]])
                first = False
    output["scaling.tex"] = tex_table("lllrrr", r"GPU & Size & & Prefill & One token & F+L+B", lines)
    lines, scores = [], []
    for model in MODELS:
        values = []
        for phase in PHASES:
            data = [rows[device]["strict", model, primary_condition(campaigns[device])]
                    for device in DEVICES]
            values.extend(portability_score(data, engine, phase) for engine in ENGINES)
        scores.append(values)
        lines.append([label(model), *[f"{x:.2f}" for x in values]])
    lines.append(["Workload mean", *[f"{statistics.mean(col):.2f}" for col in zip(*scores)]])
    output["portability.tex"] = tex_table("lrrrrrr", r"Workload & \multicolumn{2}{c}{Inference} & \multicolumn{2}{c}{Minimal} & \multicolumn{2}{c}{F+L+B} \\ & M & P & M & P & M & P", lines)
    output["ratio-plot.tex"] = ratio_plot(campaigns, rows)
    return output

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", type=Path, help="campaign .tgz directory or lossless records.jsonl.xz bundle")
    parser.add_argument("--output", type=Path, help="write generated tables and per-condition CSV here")
    parser.add_argument("--check", type=Path, help="compare generated tables against this directory")
    parser.add_argument("--bundle", type=Path, help="export original JSON values and text logs, losslessly compressed")
    args = parser.parse_args()
    require(args.bundle is None or args.bundle.resolve() != args.archives.resolve(), "cannot overwrite the input bundle")
    campaigns, rows, all_groups, failures = {}, {}, {}, {}
    digests = {}
    for line in (HERE / "cohort.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        require(len(PurePosixPath(name).parts) == 1 and name.endswith((".tgz", ".txt")),
                "unexpected manifest path")
        digests[name] = expected
        if args.archives.is_dir():
            with (args.archives / name).open("rb") as file:
                require(hashlib.file_digest(file, "sha256").hexdigest() == expected, "input digest differs: " + name)
    print("Campaign archive SHA-256 digests match" if args.archives.is_dir() else "Auditing lossless JSON-content bundle")
    input_hashes = {}
    log_count = 0
    with (nullcontext(None) if args.archives.is_dir() else lzma.open(args.archives, "rt")) as source, \
            (lzma.open(args.bundle, "wt") if args.bundle else nullcontext(None)) as bundle:
        if source:
            require(json.loads(next(source)) == {"format": "p3hpc-files-v3", "archives": digests}, "bundle provenance differs")
        if bundle:
            bundle.write(json.dumps({"format": "p3hpc-files-v3", "archives": digests}) + "\n")
        for name in digests:
            path = args.archives / name
            if source:
                require(json.loads(next(source)) == name, "bundle archive order differs")
                entries = iter(lambda: json.loads(next(source)), None)
            else:
                entries = archive_records(path)
            if bundle:
                bundle.write(json.dumps(name) + "\n")
            records = read_records(entries, bundle)
            log_count += sum(isinstance(value, str) for value in records.values())
            if bundle:
                bundle.write("null\n")
            if name.endswith(".txt"):
                continue
            c, groups, failed = load_campaign(path, records)
            for name, digest in c["sha256"].items():
                name = name.replace("\\", "/")
                require(input_hashes.setdefault(name, digest) == digest, "input identity differs: " + name)
            device = path.stem
            campaigns[device], all_groups[device], failures[device] = c, groups, failed
            rows[device] = aggregate(groups) if c["args"]["backend"] != "cpu" else {}
            print(device, c["status"],
                  dict(Counter(run["status"] for run in c["runs"])), flush=True)
            del records
        if source:
            require(not source.read().strip(), "unexpected trailing campaign")
    print(log_count, "text logs retained; source, lockfile and checkpoint identities agree")
    generated = tables(campaigns, rows, all_groups)
    print("M3 spike-removal sensitivity (not applied):", spike_sensitivity(all_groups["apple-m3"]))
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        for name, content in generated.items():
            (args.output / name).write_text(content)
        flat = [{"device": device, "precision": key[0], "model": key[1], "condition": key[2],
                 "fully_replicated": all(row["replicates_" + phase] == 3 for phase in PHASES), **row}
                for device, data in rows.items() for key, row in sorted(data.items())]
        with (args.output / "conditions.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, flat[0].keys())
            writer.writeheader()
            writer.writerows(flat)
        (args.output / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    if args.check:
        for name, content in generated.items():
            require((args.check / name).read_text() == content, "generated table differs: " + name)
    for precision in ("strict", "accelerated"):
        devices = complete_devices(campaigns, rows, precision)
        print("\n", precision, "fully replicated configurations:", devices)
        for phase in PHASES:
            values = [rows[device][precision, model, primary_condition(c)]["ratio_" + phase]
                      for device in devices for c in [campaigns[device]]
                      for model in MODELS]
            print(phase, "wins", sum(x < 1 for x in values), "/", len(values), "median ratio", statistics.median(values))
            available = [row["ratio_" + phase] for device in DEVICES
                         for (p, model, condition), row in rows[device].items()
                         if p == precision and row["replicates_" + phase] == 3]
            print("  all qualified phase groups:", sum(x < 1 for x in available), "/", len(available),
                  "nominal wins; median ratio", statistics.median(available))
    errors = [(run["errors"], device, key) for device, groups in all_groups.items() for key, runs in groups.items() for run in runs]
    for key in set().union(*(e.keys() for e, _, _ in errors)):
        print("maximum", key, max((e[key], device, group) for e, device, group in errors if key in e))
    searches = [session["search"] for device, groups in all_groups.items()
                if device in DEVICES
                for runs in groups.values() for run in runs
                for session in run["pair"]["meganeura"]["optimizer"]["sessions"]]
    print("GPU-paired native search:", len(searches), "sessions;",
          sum(s["truncated"] for s in searches), "truncated;",
          sum(s["visited_classes"] for s in searches), "/",
          sum(s["eligible_classes"] for s in searches), "classes; longest",
          max(s["elapsed_seconds"] for s in searches), "seconds")
    print("GPU-paired decisions:", sum((Counter(s["decisions"]) for s in searches), Counter()))
    if args.output:
        (args.output / "search-summary.json").write_text(json.dumps({device: [
            {"precision": key[0], "model": key[1], "replicate": run["replicate"], **session}
            for key, runs in groups.items() for run in runs
            for session in run["pair"]["meganeura"]["optimizer"]["sessions"]]
            for device, groups in all_groups.items()}, indent=2) + "\n")


if __name__ == "__main__":
    main()
