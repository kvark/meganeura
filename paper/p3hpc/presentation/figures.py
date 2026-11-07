# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib==3.11.2"]
# ///
"""Slide figures from the final cohort CSV."""
import ast
import csv
import math
from pathlib import Path
import statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
INPUT = HERE / "data"
TABLES = HERE.parent / "tables"
OUTPUT = HERE / "figures"
INK, TEAL, ORANGE, CANVAS, MUTED = "#23343b", "#006b70", "#a44616", "#faf9f6", "#637679"
DEVICES = {"nvidia-5070": "RTX 5070", "nvidia-h100": "H100",
           "amd-7900xt": "RX 7900 XT", "amd-mi300": "MI300X", "intel-b570": "Arc B570", "apple-m3": "Apple M3"}
MODELS = ["SmolLM2-135M", "SmolVLA", "StableDiffusion", "ResNet-50", "Whisper-tiny"]
SHORT = ["SmolLM2", "SmolVLA", "U-Net", "ResNet", "Whisper"]
# Ratio-panel columns: what each workload is, then which model.
KINDS = ["LLM", "Robot", "Diffusion", "Vision", "Speech"]
PHASES = ["inference", "latency", "training"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "figure.facecolor": CANVAS, "axes.facecolor": CANVAS,
    "svg.fonttype": "none", "svg.hashsalt": "meganeura-p3hpc-final"})
ROWS = list(csv.DictReader((TABLES / "results.csv").open()))
DATA = {(r["archive"], r["model"], r["precision"], r["phase"]): r for r in ROWS}
assert len(DATA) == 372
assert sum(bool(r["ratio"]) for r in ROWS) == 201


def save(fig, name):
    OUTPUT.mkdir(exist_ok=True)
    fig.savefig(OUTPUT / name, metadata={"Date": None})
    plt.close(fig)


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#d1dad8", linewidth=.7)
    ax.set_axisbelow(True)


def ratio(device, model, profile, phase):
    raw = DATA[device, model, profile, phase]["ratio"]
    return float(raw) if raw else None


def shade(value):
    """Teal below parity, brown above, deeper with distance from 1 (capped at 4×)."""
    if value is None:
        return to_rgb("#e9ebe8"), MUTED
    amount = .12 + .6 * min(abs(math.log2(value)) / 2, 1)
    base = to_rgb(TEAL if value < 1 else ORANGE)
    fill = tuple(1 - amount + amount * c for c in base)
    return fill, "#ffffff" if amount > .5 else INK


def ratio_panels(panels, name):
    """Core comparisons, one panel per (profile, phase), one cell per GPU and model."""
    fig = plt.figure(figsize=(11.56, 4.38))
    left, gap = .15, .025
    width = (.985 - left - gap * (len(panels) - 1)) / len(panels)
    for panel, (profile, phase, title) in enumerate(panels):
        ax = fig.add_axes((left + panel * (width + gap), .03, width, .76))
        for row, device in enumerate(DEVICES):
            for col, model in enumerate(MODELS):
                value = ratio(device, model, profile, phase)
                fill, ink = shade(value)
                ax.add_patch(Rectangle((col + .04, row + .06), .92, .88, facecolor=fill, linewidth=0))
                ax.text(col + .5, row + .5, "—" if value is None else f"{value:.2f}",
                        ha="center", va="center", fontsize=15, color=ink)
        ax.set(xlim=(0, 5), ylim=(len(DEVICES), 0))
        ax.set_xticks([c + .5 for c in range(5)], [f"{kind}\n{short}" for kind, short in zip(KINDS, SHORT)], fontsize=12.5)
        ax.xaxis.tick_top()
        ax.set_yticks([r + .5 for r in range(len(DEVICES))], list(DEVICES.values()) if panel == 0 else [""] * len(DEVICES), fontsize=13.5)
        ax.tick_params(length=0, pad=5)
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.text(left + panel * (width + gap), .94, title, fontsize=16, weight="bold", va="bottom")
    save(fig, name)


def search_loop():
    """The search: the four steps for each candidate program, and where candidates leave."""
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(11.56, 4.38))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set(xlim=(0, 1156), ylim=(438, 0))
    ax.axis("off")
    top, height, width, gap, start = 150, 112, 200, 30, 236
    middle = top + height / 2
    ax.text(196, middle - 14, "candidate\nprograms", ha="right", va="center", fontsize=17, weight="bold",
            color=INK, linespacing=1.15)
    ax.text(196, middle + 30, "from the e-graph", ha="right", va="center", fontsize=12.5, color=MUTED)
    ax.annotate("", xy=(start - 4, middle), xytext=(204, middle),
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.8, shrinkA=0, shrinkB=0))
    steps = [("Build", "one candidate", "program"), ("Validate", "outputs + gradients", "against the f32 build"),
             ("Tune", "time kernels,", "remember winners"), ("Measure", "time the program,", "keep it if it wins")]
    for i, (title, *lines) in enumerate(steps):
        x = start + i * (width + gap)
        ax.add_patch(FancyBboxPatch((x, top), width, height, boxstyle="round,pad=0,rounding_size=9",
                                   facecolor="#e7eeeb", edgecolor="#b6cdca", linewidth=1))
        ax.text(x + 14, top + 32, f"{i + 1}", fontsize=15, weight="bold", color=ORANGE, va="center")
        ax.text(x + 36, top + 32, title, fontsize=19, weight="bold", color=TEAL, va="center")
        for j, line in enumerate(lines):
            ax.text(x + 14, top + 66 + 24 * j, line, fontsize=12.5, va="center")
        if i < 3:
            ax.annotate("", xy=(x + width + gap - 3, middle), xytext=(x + width + 3, middle),
                        arrowprops=dict(arrowstyle="->", color=TEAL, lw=2.2))
    first, last = start + width / 2, start + 3 * (width + gap) + width / 2
    ax.plot([last, last, first], [top - 3, 112, 112], color=TEAL, lw=2.2, solid_capstyle="butt")
    ax.annotate("", xy=(first, top - 3), xytext=(first, 111),
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=2.2, shrinkA=0, shrinkB=0))
    ax.text((first + last) / 2, 96, "next candidate, until the 30s budget runs out", ha="center", va="center",
            fontsize=14, color=TEAL, backgroundcolor=CANVAS)
    check = start + width + gap + width / 2
    ax.annotate("", xy=(check, 338), xytext=(check, top + height + 3),
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.8))
    ax.text(check, 360, "drop invalid", ha="center", va="center", fontsize=14, color=ORANGE)
    ax.annotate("", xy=(last, 338), xytext=(last, top + height + 3),
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=2.2))
    ax.text(last, 372, "fastest checked\nprogram", ha="center", va="center", fontsize=15, weight="bold",
            color=TEAL, linespacing=1.25)
    save(fig, "search.svg")


def egraph():
    """Matmul/add fusion: operator e-nodes point to shared operand e-classes."""
    from matplotlib.patches import FancyBboxPatch, PathPatch
    from matplotlib.path import Path as PlotPath
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set(xlim=(0, 720), ylim=(460, 0))
    ax.axis("off")

    def eclass(x, y, width, height, label):
        ax.add_patch(FancyBboxPatch((x, y), width, height,
                     boxstyle="round,pad=0,rounding_size=12", facecolor=CANVAS,
                     edgecolor=MUTED, linewidth=1.5, linestyle=(0, (4, 3)), zorder=1))
        ax.text(x + 12, y - 9, label, fontsize=13, weight="bold", color=MUTED, va="bottom")

    def enode(x, y, width, label, added=False):
        ax.add_patch(FancyBboxPatch((x - width / 2, y - 19), width, 38,
                     boxstyle="round,pad=0,rounding_size=6",
                     facecolor="#dcebe8" if added else "#eceeec",
                     edgecolor=TEAL if added else MUTED, linewidth=1.6, zorder=3))
        ax.text(x, y, label, ha="center", va="center", fontsize=15,
                weight="bold" if added else "normal", color=INK, zorder=4)

    def edge(points, added=False):
        color = TEAL if added else MUTED
        ax.add_patch(PathPatch(PlotPath(points), facecolor="none", edgecolor=color,
                               linewidth=1.8, joinstyle="round", zorder=2))
        ax.annotate("", xy=points[-1], xytext=points[-2], zorder=2,
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.8,
                                    shrinkA=0, shrinkB=1))

    # Dashed boxes are equivalence classes; only operator nodes have outgoing edges.
    eclass(30, 38, 660, 100, "Y: one output e-class, two alternatives")
    eclass(40, 200, 210, 80, "P")
    eclass(550, 200, 140, 80, "bias")
    eclass(40, 348, 170, 64, "input")
    eclass(340, 348, 170, 64, "weights")
    enode(145, 88, 104, "Add")
    enode(480, 88, 244, "FusedMatMulAdd", added=True)
    enode(145, 240, 146, "MatMul")
    enode(620, 240, 72, "b")
    enode(125, 380, 72, "X")
    enode(425, 380, 72, "W")

    edge([(130, 107), (130, 200)])
    edge([(175, 107), (205, 166), (590, 166), (590, 200)])
    edge([(126, 259), (125, 348)])
    edge([(179, 259), (390, 348)])
    # The fused alternative shares X, W and b; it has no dependency on P.
    edge([(410, 107), (300, 180), (300, 310), (170, 348)], added=True)
    edge([(480, 107), (455, 348)], added=True)
    edge([(550, 107), (650, 200)], added=True)

    ax.text(30, 448, "Dashed box: e-class  ·  Solid box: e-node  ·  Teal: added by the rewrite",
            fontsize=12.5, va="center", color=MUTED)
    save(fig, "egraph.svg")






def path_admission():
    """Admission checkpoints, with separate rejection branches at each stop."""
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(11.56, 4.38))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set(xlim=(0, 1156), ylim=(438, 0))
    ax.axis("off")
    stages = [
        ("Expose", "Driver + compiler", ["Instruction unavailable", "Operand type missing"]),
        ("Lower", "Shape + resources", ["No matching tile / layout", "Too much shared memory"]),
        ("Qualify", "Numerical checks", ["Forward error too large", "Gradient error too large"]),
        ("Measure", "Whole-program timing", ["Budget runs out", "Fallback is faster"]),
    ]
    for i, (title, detail, rejected) in enumerate(stages):
        x, width = 12 + i * 239, 207
        ax.add_patch(FancyBboxPatch((x, 34), width, 90,
                     boxstyle="round,pad=0,rounding_size=8",
                     facecolor="#e7eeeb", edgecolor="#b6cdca"))
        ax.text(x + width / 2, 65, title, ha="center", va="center",
                fontsize=17, weight="bold", color=TEAL)
        ax.text(x + width / 2, 100, detail, ha="center", va="center", fontsize=11)
        ax.annotate("", xy=(x + width + (42 if i == 3 else 28), 79), xytext=(x + width + 3, 79),
                    arrowprops=dict(arrowstyle="->", color=TEAL, lw=2))
        for j, label in enumerate(rejected):
            y = 200 + j * 85
            ax.annotate("", xy=(x + 32, y), xytext=(x + 16, 128),
                        arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.4,
                                        connectionstyle="angle,angleA=-90,angleB=0,rad=8"))
            ax.text(x + 38, y, label, fontsize=11.2, va="center", color=ORANGE)
    ax.text(1082, 68, "Selected", ha="center", va="center",
            fontsize=17, color=TEAL, weight="bold")
    ax.text(1082, 101, "program", ha="center", va="center", fontsize=12)
    ax.text(12, 373, "Teal: continue. Brown: drop this candidate; retain the qualified incumbent.",
            fontsize=13, weight="bold")
    ax.text(12, 416, "These checks explain selection. They do not explain where the selected program spends its time.",
            fontsize=11.5, color=MUTED)
    save(fig, "path-admission.svg")


def scaling():
    fig, axes = plt.subplots(1, 2, figsize=(11.56, 4.38))
    fig.subplots_adjust(left=.08, right=.98, top=.83, bottom=.25, wspace=.28)
    for ax, d in zip(axes, ["nvidia-h100", "amd-mi300"]):
        for phase, title, color, lift in [("inference", "Prefill", TEAL, -17), ("training", "Training", ORANGE, 8)]:
            rs = [DATA[d if size == "135M" else d + "-large", "SmolLM2-" + size, "strict", phase] for size in ["135M", "360M", "1.7B"]]
            vals = [float(r["ratio"]) for r in rs]
            ax.plot(range(3), vals, "o-", color=color, label=title, linewidth=2)
            for i, (r, v) in enumerate(zip(rs, vals)):
                spread = ast.literal_eval(r["paired_ratios"])
                ax.vlines(i, min(spread), max(spread), color=color, linewidth=3)
                ax.annotate(f"{v:.2f}", (i, v), xytext=(0, lift), textcoords="offset points", ha="center", color=color, fontsize=12)
        ax.set_title(DEVICES[d], loc="left", weight="bold")
        ax.set_xticks(range(3), ["135M", "360M", "1.7B"])
        ax.set_ylim(.9, 4.3)
        ax.axhline(1, color=MUTED, linestyle="--", linewidth=1)
        clean(ax)
    axes[0].set_ylabel("Strict time ratio · lower is better")
    axes[1].legend(frameon=False, loc="upper left")
    fig.text(.08, .07, "Dots: Meganeura time ÷ PyTorch time. Bars: range over the three process pairs.", fontsize=11)
    save(fig, "scaling.svg")




def preparation_models():
    rows = list(csv.DictReader((TABLES / "preparation.csv").open()))
    fig, ax = plt.subplots(figsize=(11.56, 4.38))
    fig.subplots_adjust(left=.20, right=.96, top=.88, bottom=.23)
    for i, model in enumerate(MODELS):
        devices = defaultdict(list)
        for row in rows:
            if row["model"] == model and row["precision"] == "strict" and row["reference_complete"] == "True":
                devices[row["archive"]].append(row)
        devices = {d: group for d, group in devices.items() if len(group) == 3}
        assert len(devices) == (5 if model in ("ResNet-50", "Whisper-tiny") else 6)
        assert all(len(group) == 3 for group in devices.values())
        values = [statistics.median(statistics.median(float(r[key]) for r in group)
                                   for group in devices.values())
                  for key in ("native_preparation_seconds", "reference_compile_seconds")]
        for offset, value, color in zip((-.18, .18), values, (TEAL, ORANGE)):
            ax.barh(i+offset, value, height=.31, color=color)
            ax.text(value+1, i+offset, f"{value:.1f}s", va="center", fontsize=12, color=color)
    ax.set_yticks(range(5), ["SmolLM2-135M", "SmolVLA", "SD U-Net", "ResNet-50", "Whisper-tiny"])
    ax.invert_yaxis(); ax.set_xlim(0, 80)
    ax.set_xlabel("Seconds per workload process · strict f32")
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(.20, .95, "Meganeura: all preparation, almost all of it search", color=TEAL, fontsize=13, weight="bold")
    fig.text(.20, .89, "PyTorch: compilation and first calls", color=ORANGE, fontsize=13, weight="bold")
    fig.text(.20, .065, "Median across GPUs of three-process medians. ResNet and Whisper: 5 GPUs with complete PyTorch results.", fontsize=10)
    fig.text(.20, .018, "The columns cover different work under different limits.", fontsize=10)
    save(fig, "preparation-models.svg")




def timing_spread():
    fig, ax = plt.subplots(figsize=(11.56, 4.38))
    fig.subplots_adjust(left=.09, right=.98, top=.94, bottom=.23)
    for engine, color, offset in [("native", TEAL, -.14), ("reference", ORANGE, .14)]:
        for i, d in enumerate(DEVICES):
            values = []
            for r in ROWS:
                if r["archive"] != d: continue
                ps = ast.literal_eval(r[engine + "_process_medians_ms"])
                if len(ps) == 3: values.append(100*(max(ps)-min(ps))/statistics.median(ps))
            ax.scatter([i+offset]*len(values), values, color=color, alpha=.65, s=18, label=engine if i == 0 else None)
    ax.set_xticks(range(len(DEVICES)), list(DEVICES.values()), rotation=10, ha="right", fontsize=11)
    ax.set_yscale("log");ax.set_ylim(.02, 250)
    ax.set_ylabel("Process spread, % (log scale)")
    ax.axhline(10, color="#637679", linestyle="--", linewidth=1)
    ax.legend(frameon=False, loc="upper left")
    clean(ax)
    fig.text(.09, .035, "Spread = (max − min) / median of 3 process medians. Qualified timed cells; all repeats retained.", fontsize=11)
    save(fig, "timing-spread.svg")




if __name__ == "__main__":
    for profile in ("strict", "accelerated"):
        ratio_panels([(profile, "inference", "Inference"), (profile, "training", "Training")],
                     f"{profile}-ratios.svg")
    egraph()
    search_loop()
    path_admission()
    scaling();preparation_models();timing_spread()
