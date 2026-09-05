#!/usr/bin/env python3
"""Generate the LaTeX result tables in paper/tables/ from paper/results/.

Every numeric table in main.tex is emitted by this script; rerun it after
refreshing paper/results/ and the tables update in place. It also prints a
FACTS block with the aggregate numbers cited in prose, so a text sweep can
be checked against the frozen artifacts.
"""

import json
import math
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
TABLES = os.path.join(HERE, "tables")

# (dir, display name, meganeura backend note, pytorch note)
PLATFORMS = ["nvidia", "amd-d", "amd-i", "intel", "mac"]
GPU_REF = ["nvidia", "amd-d", "amd-i", "mac"]  # PyTorch ran on the GPU here
MODES = ["strict", "accelerated"]
MODELS = [
    ("SmolLM2-135M", "SmolLM2-135M"),
    ("SmolVLA", "SmolVLA"),
    ("StableDiffusion", "Diffusion U-Net"),
    ("ResNet-50", "ResNet-50"),
    ("Whisper-tiny", "Whisper-tiny"),
]
PHASES = ["inference_ms", "latency_ms", "training_ms"]

# The Radeon 780M PyTorch/ROCm Whisper backward result is inconsistent with
# PyTorch on CUDA, discrete ROCm, CPU, and MPS, while Meganeura reproduces the
# cross-backend consensus on all five devices.  The raw times and diagnostics
# remain in the artifact, but this device/workload/phase is not a valid paired
# comparison and is excluded symmetrically for both engines.
ORACLE_DISPUTES = {("amd-i", "Whisper-tiny", "training_ms")}


def oracle_disputed(platform, model, phase):
    return (platform, model, phase) in ORACLE_DISPUTES


def load(platform, mode, model):
    path = os.path.join(RESULTS, platform, f"paper-v1-{mode}", f"{model}_summary.json")
    with open(path) as f:
        return {e["framework"]: e for e in json.load(f)}


def t(entry, phase):
    return entry["timings"][phase]


def fnum(x):
    if x >= 100:
        return f"{x:.0f}"
    return f"{x:.2f}"


def ratio_cell(r):
    s = f"{r:.2f}"
    return rf"\textbf{{{s}}}" if float(s) < 1.00 else s


def device_header(platform):
    s = load(platform, "strict", "SmolLM2-135M")
    env = s["meganeura"]["environment"]
    pt = s["pytorch"]
    name = env["gpu_device_name"]
    return name, env, pt


def check_validation():
    total = passed = disputed = 0
    worst = {"strict": {}, "accelerated": {}}
    dispute_metrics = {"strict": {}, "accelerated": {}}
    for platform in PLATFORMS:
        for mode in MODES:
            for model, _ in MODELS:
                v = load(platform, mode, model)["meganeura"]["validation"]
                total += 1
                if v["forward_valid"] and v["training_valid"]:
                    passed += 1
                elif oracle_disputed(platform, model, "training_ms"):
                    disputed += 1
                for key in ("output_relative_l2_error",
                            "total_gradient_relative_error",
                            "parameter_gradient_relative_l2_error"):
                    if oracle_disputed(platform, model, "training_ms"):
                        dispute_metrics[mode][key] = v[key]
                    else:
                        cur = worst[mode].get(key, 0.0)
                        worst[mode][key] = max(cur, v[key])
    return total, passed, disputed, worst, dispute_metrics


def gradient_relative_l2(vector, reference):
    """Relative L2 over named per-parameter gradient norms."""
    if vector.keys() != reference.keys():
        raise ValueError("gradient parameter sets differ in oracle audit")
    numerator = math.sqrt(sum(
        (vector[key] - reference[key]) ** 2 for key in vector
    ))
    denominator = math.sqrt(sum(value ** 2 for value in reference.values()))
    return numerator / denominator


def oracle_audit(mode):
    """Cross-backend evidence for the one declared oracle dispute."""
    records = {
        platform: load(platform, mode, "Whisper-tiny")
        for platform in PLATFORMS
    }
    reference_platforms = [p for p in PLATFORMS if p != "amd-i"]
    references = [
        records[p]["pytorch"]["outputs"]["gradient_norms"]
        for p in reference_platforms
    ]
    candidates = [
        records[p]["meganeura"]["outputs"]["gradient_norms"]
        for p in PLATFORMS
    ]
    disputed = records["amd-i"]["pytorch"]["outputs"]

    def max_internal(vectors):
        return max(
            gradient_relative_l2(a, b)
            for a in vectors for b in vectors if a is not b
        )

    reference_norm = statistics.mean(
        records[p]["pytorch"]["outputs"]["grad_norm"]
        for p in reference_platforms
    )
    return {
        "reference_internal": max_internal(references),
        "candidate_internal": max_internal(candidates),
        "candidate_to_reference": max(
            gradient_relative_l2(candidate, reference)
            for candidate in candidates for reference in references
        ),
        "disputed_to_reference": max(
            gradient_relative_l2(disputed["gradient_norms"], reference)
            for reference in references
        ),
        "disputed_total_norm": (
            abs(disputed["grad_norm"] - reference_norm) / reference_norm
        ),
    }


def emit(fname, colspec, header, body_lines):
    """Write a complete tabular environment; \\input-ing a bare table body
    inside an alignment breaks TeX's row scanner, so the file owns it all."""
    lines = [rf"\begin{{tabular}}{{{colspec}}}", r"\toprule"]
    lines += header
    lines += body_lines
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(TABLES, fname), "w") as f:
        f.write("\n".join(lines) + "\n")


def shorten_name(name):
    return (name.replace(" (RADV NAVI31)", " (RADV)")
                .replace(" Graphics (RADV PHOENIX)", " (RADV)")
                .replace("(R)", ""))


def write_devices():
    rows = []
    meta = {
        "nvidia": ("discrete, 12\\,GB", "compiled"),
        "amd-d": ("discrete, 20\\,GB", "compiled"),
        "amd-i": ("integrated", "compiled"),
        "intel": ("integrated", "eager"),
        "mac": ("unified SoC", "eager"),
    }
    for platform in PLATFORMS:
        name, env, pt = device_header(platform)
        cls, ptmode = meta[platform]
        if platform == "mac":
            osname = "macOS " + pt["environment"]["platform"].split("-")[1]
            backend = "Metal"
        else:
            osname = "Linux"
            drv = env["gpu_driver_info"].replace("-1ubuntu1", "")
            drv = drv.replace("-0ubuntu0.24.04.1", "")
            backend = f"Vulkan, {drv}"
        backend_short = (pt["backend"]
                         .replace("ROCm 7.1.25424", "ROCm 7.1")
                         .replace("ROCm 7.14.60850", "ROCm 7.14"))
        ptdesc = f"{pt['framework_rev']} ({backend_short}, {ptmode})"
        rows.append(f"{shorten_name(name)} & {cls} & {osname} & {backend} & {ptdesc} \\\\")
    header = [r"Device & Class & OS & \system{} backend & "
              r"PyTorch build (backend, mode) \\", r"\midrule"]
    emit("devices.tex", "l l l l l", header, rows)


def write_results_table(mode, fname, with_compile):
    lines = []
    ncols = 12 if with_compile else 10
    for platform in PLATFORMS:
        name, env, pt = device_header(platform)
        name = shorten_name(name).replace(" (RADV)", "")
        api = "Metal" if platform == "mac" else "Vulkan"
        ref = {"nvidia": "CUDA, compiled", "amd-d": "ROCm, compiled",
               "amd-i": "ROCm, compiled",
               "intel": "CPU fallback, eager", "mac": "MPS, eager"}[platform]
        lines.append(r"\midrule")
        lines.append(
            rf"\multicolumn{{{ncols}}}{{l}}{{\textbf{{{name}}}"
            rf" --- \system{{}} {api} vs.\ PyTorch {ref}}} \\"
        )
        for model, disp in MODELS:
            s = load(platform, mode, model)
            mg, pt_e = s["meganeura"], s["pytorch"]
            cells = [disp]
            if with_compile:
                cells += [f"{t(mg, 'compile_s'):.2f}", f"{t(pt_e, 'compile_s'):.1f}"]
            for phase in PHASES:
                a, b = t(mg, phase), t(pt_e, phase)
                cell = ratio_cell(a / b)
                if oracle_disputed(platform, model, phase):
                    cell = r"--$^\ddagger$"
                elif phase == "training_ms" and not mg["validation"]["training_valid"]:
                    cell = f"{a / b:.2f}$^\\dagger$"
                cells += [fnum(a), fnum(b), cell]
            lines.append(" & ".join(cells) + r" \\")
    groups = (r" & \multicolumn{2}{c}{Compile (s)}" if with_compile else "") + \
        r" & \multicolumn{3}{c}{Full / prefill (ms)}" \
        r" & \multicolumn{3}{c}{Minimal / one-token (ms)}" \
        r" & \multicolumn{3}{c}{F+L+B (ms)} \\"
    if with_compile:
        cmid = r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}\cmidrule(lr){7-9}\cmidrule(lr){10-12}"
        head = r"Workload & Ours & PT & Ours & PT & $\times$ & Ours & PT & $\times$ & Ours & PT & $\times$ \\"
        colspec = "l rr rrr rrr rrr"
    else:
        cmid = r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}"
        head = r"Workload & Ours & PT & $\times$ & Ours & PT & $\times$ & Ours & PT & $\times$ \\"
        colspec = "l rrr rrr rrr"
    emit(fname, colspec, [groups, cmid, head], lines)


def write_memory():
    lines = []
    for model, disp in MODELS:
        ph = load("nvidia", "strict", model)["meganeura"]["memory"]["phases"]
        cells = [disp]
        for phase in ("training", "inference"):
            log = ph[phase]["plan_logical_bytes"] / 2**20
            phys = ph[phase]["allocated_bytes"] / 2**20
            cells += [f"{log:.0f}", f"{phys:.0f}", f"{100 * (1 - phys / log):.0f}\\%"]
        lines.append(" & ".join(cells) + r" \\")
    header = [
        r" & \multicolumn{3}{c}{Training (MiB)} & \multicolumn{3}{c}{Inference (MiB)} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"Workload & Logical & Phys. & Saved & Logical & Phys. & Saved \\",
        r"\midrule",
    ]
    emit("memory.tex", "l rrr rrr", header, lines)


def efficiencies(mode, phase, model):
    eff = {"meganeura": [], "pytorch": []}
    for platform in PLATFORMS:
        if oracle_disputed(platform, model, phase):
            continue
        s = load(platform, mode, model)
        tmg, tpt = t(s["meganeura"], phase), t(s["pytorch"], phase)
        # An invalid implementation result scores zero and does not set the
        # best-observed bar. Oracle-disputed pairs are removed above for both
        # engines rather than being assigned to either one.
        gate = "training_valid" if phase == "training_ms" else "forward_valid"
        valid = s["meganeura"]["validation"][gate]
        best = min(tmg, tpt) if valid else tpt
        eff["meganeura"].append(best / tmg if valid else 0.0)
        eff["pytorch"].append(best / tpt)
    return eff


def pennycook(mode, phase, model):
    eff = efficiencies(mode, phase, model)
    return {k: (0.0 if 0.0 in v else len(v) / sum(1 / x for x in v))
            for k, v in eff.items()}


def write_portability():
    lines = []
    sums = {(p, e): 0.0 for p in PHASES for e in ("meganeura", "pytorch")}
    for model, disp in MODELS:
        label = disp + (r"$^\ddagger$" if model == "Whisper-tiny" else "")
        cells = [label]
        for phase in PHASES:
            pp = pennycook("strict", phase, model)
            for engine in ("meganeura", "pytorch"):
                sums[(phase, engine)] += pp[engine]
                val = f"{pp[engine]:.2f}"
                if pp[engine] == 0.0:
                    val = r"0$^\dagger$"
                cells.append(val)
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    cells = [r"\textit{mean}"]
    for phase in PHASES:
        for engine in ("meganeura", "pytorch"):
            cells.append(f"{sums[(phase, engine)] / len(MODELS):.2f}")
    lines.append(" & ".join(cells) + r" \\")
    header = [
        r" & \multicolumn{2}{c}{Inference} & \multicolumn{2}{c}{Minimal} & "
        r"\multicolumn{2}{c}{F+L+B} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"Workload & Ours & PT & Ours & PT & Ours & PT \\",
        r"\midrule",
    ]
    emit("portability.tex", "l rr rr rr", header, lines)


def facts():
    print("== FACTS ==")
    total, passed, disputed, worst, dispute_metrics = check_validation()
    print(f"validation: {passed}/{total} paired comparisons pass; "
          f"{disputed} excluded because the PyTorch/ROCm 780M Whisper "
          "backward record is oracle-disputed")
    for mode in MODES:
        w = worst[mode]
        d = dispute_metrics[mode]
        print(f"  worst comparable {mode}: "
              f"out_l2={100*w['output_relative_l2_error']:.3g}% "
              f"grad_tot={100*w['total_gradient_relative_error']:.3g}% "
              f"grad_vec={100*w['parameter_gradient_relative_l2_error']:.3g}%")
        print(f"  oracle dispute {mode}: "
              f"out_l2={100*d['output_relative_l2_error']:.3g}% "
              f"grad_tot={100*d['total_gradient_relative_error']:.3g}% "
              f"grad_vec={100*d['parameter_gradient_relative_l2_error']:.3g}%")
        audit = oracle_audit(mode)
        print(f"  cross-backend audit {mode}: "
              f"PT-good={100*audit['reference_internal']:.3g}% "
              f"Meganeura={100*audit['candidate_internal']:.3g}% "
              f"Meganeura-to-PT-good={100*audit['candidate_to_reference']:.3g}% "
              f"PT-780M-to-PT-good={100*audit['disputed_to_reference']:.3g}% "
              f"PT-780M-total-norm={100*audit['disputed_total_norm']:.3g}%")

    for scope, plats in (("all5", PLATFORMS), ("gpuref", GPU_REF)):
        for mode in MODES:
            for phase in PHASES:
                rs = []
                for p in plats:
                    for model, _ in MODELS:
                        s = load(p, mode, model)
                        if oracle_disputed(p, model, phase):
                            continue
                        if phase == "training_ms" and \
                                not s["meganeura"]["validation"]["training_valid"]:
                            continue  # invalid implementation cells never enter a count
                        rs.append(t(s["meganeura"], phase) / t(s["pytorch"], phase))
                w2 = sum(1 for r in rs if r <= 2.0)
                fast = sum(1 for r in rs if r < 1.0)
                print(f"{scope:>6} {mode:>11} {phase:<13} n={len(rs)} within2x={w2} "
                      f"faster={fast} median={statistics.median(rs):.2f} max={max(rs):.2f}")

    print("compile ranges (s):")
    for platform in PLATFORMS:
        for mode in MODES:
            mg = [t(load(platform, mode, m)["meganeura"], "compile_s") for m, _ in MODELS]
            pt = [t(load(platform, mode, m)["pytorch"], "compile_s") for m, _ in MODELS]
            print(f"  {platform:<7} {mode:<11} mg {min(mg):.2f}-{max(mg):.2f}  pt {min(pt):.1f}-{max(pt):.1f}")

    print("strict->accelerated inference change (mg, pt) per device:")
    for platform in PLATFORMS:
        for model, _ in MODELS:
            a = load(platform, "strict", model)
            b = load(platform, "accelerated", model)
            dmg = t(a["meganeura"], "inference_ms") / t(b["meganeura"], "inference_ms")
            dpt = t(a["pytorch"], "inference_ms") / t(b["pytorch"], "inference_ms")
            print(f"  {platform:<7} {model:<16} mg x{dmg:.2f}  pt x{dpt:.2f}")

    print("PP means (strict, accelerated):")
    for mode in MODES:
        for phase in PHASES:
            ms = [pennycook(mode, phase, m)["meganeura"] for m, _ in MODELS]
            ps = [pennycook(mode, phase, m)["pytorch"] for m, _ in MODELS]
            print(f"  {mode:>11} {phase:<13} mg={sum(ms)/5:.2f} pt={sum(ps)/5:.2f}")


FIG_DEVICES = [
    ("nvidia", "RTX 5070 (CUDA)"),
    ("amd-d", "RX 7900 XT (ROCm)"),
    ("amd-i", "Radeon 780M (ROCm)"),
    ("intel", "Intel RPL-U (CPU ref.)"),
    ("mac", "Apple M3 (MPS)"),
]


def write_figure():
    """Strict-mode ratio chart: two aligned panels (inference, F+L+B),
    horizontal bars on a log2 axis anchored at parity."""
    import math
    xmin, xmax = 0.4, 3.4
    panel_w = 6.9          # cm of bar area per panel
    panel_gap = 1.0
    row_h = 0.34           # cm per workload row
    group_gap = 0.22
    label_w = 2.55         # cm reserved left of panel A for workload labels

    def x(r, off):
        return off + panel_w * (math.log2(r / xmin) / math.log2(xmax / xmin))

    lines = [r"\begin{tikzpicture}[baseline]"]
    offs = {0: label_w, 1: label_w + panel_w + panel_gap}
    titles = {0: "Full / prefill inference", 1: "Forward--loss--backward"}
    phases = {0: "inference_ms", 1: "training_ms"}

    n_rows = len(FIG_DEVICES) * len(MODELS)
    height = n_rows * row_h + len(FIG_DEVICES) * (group_gap + 0.30)
    for p in (0, 1):
        for guide, style in ((0.5, "densely dotted"), (1.0, "semithick"),
                             (2.0, "densely dotted")):
            gx = x(guide, offs[p])
            lines.append(
                rf"\draw[{style}, black!60] ({gx:.2f},0.35) -- ({gx:.2f},{-height:.2f});")
            gl = f"{guide:g}$\\times$"
            lines.append(
                rf"\node[font=\scriptsize, black!70] at ({gx:.2f},{-height - 0.25:.2f}) {{{gl}}};")
        lines.append(
            rf"\node[font=\small\bfseries] at ({offs[p] + panel_w / 2:.2f},0.62) {{{titles[p]}}};")

    y = 0.0
    for plat, dispname in FIG_DEVICES:
        y -= 0.30
        lines.append(
            rf"\node[anchor=west, font=\scriptsize\bfseries] at (0,{y:.2f}) {{{dispname}}};")
        y -= group_gap
        for model, disp in MODELS:
            s = load(plat, "strict", model)
            yc = y - row_h / 2
            lines.append(
                rf"\node[anchor=west, font=\tiny] at (0.15,{yc:.2f}) {{{disp}}};")
            for p in (0, 1):
                r = t(s["meganeura"], phases[p]) / t(s["pytorch"], phases[p])
                disputed = oracle_disputed(plat, model, phases[p])
                if disputed:
                    x0 = x(1.0, offs[p])
                    lines.append(
                        rf"\node[font=\tiny, violet!80!black] at ({x0:.2f},{yc:.2f}) "
                        rf"{{$\ddagger$}};")
                    continue
                invalid = (p == 1 and not s["meganeura"]["validation"]["training_valid"])
                x0 = x(1.0, offs[p])
                x1 = x(max(min(r, xmax), xmin), offs[p])
                color = "teal!70!black" if r < 1 else "orange!85!black"
                lo, hi = min(x0, x1), max(x0, x1)
                if invalid:
                    lines.append(
                        rf"\draw[{color}] ({lo:.2f},{yc - 0.10:.2f}) "
                        rf"rectangle ({hi:.2f},{yc + 0.10:.2f});")
                else:
                    lines.append(
                        rf"\fill[{color}] ({lo:.2f},{yc - 0.10:.2f}) "
                        rf"rectangle ({hi:.2f},{yc + 0.10:.2f});")
                anchor = "east" if r < 1 else "west"
                tx = x1 + (0.06 if r >= 1 else -0.06)
                mark = f"{r:.2f}" + ("$^\\dagger$" if invalid else "")
                lines.append(
                    rf"\node[anchor={anchor}, font=\tiny, {color}] at ({tx:.2f},{yc:.2f}) {{{mark}}};")
            y -= row_h
    lines.append(r"\end{tikzpicture}")
    with open(os.path.join(TABLES, "figratios.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    os.makedirs(TABLES, exist_ok=True)
    write_devices()
    write_results_table("strict", "strict.tex", with_compile=True)
    write_results_table("accelerated", "accel.tex", with_compile=False)
    write_memory()
    write_portability()
    write_figure()
    facts()
    print("tables written to", TABLES)


if __name__ == "__main__":
    main()
