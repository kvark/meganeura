# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib==3.11.2"]
# ///
"""GTC poster figures: NVIDIA rows of the P3HPC cohort."""
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
P3HPC = HERE.parent / "p3hpc"
OUTPUT = HERE / "figures"
INK, TEAL, ORANGE, CANVAS, MUTED, GRID = "#23343b", "#006b70", "#a44616", "#ffffff", "#637679", "#d1dad8"
DEVICES = {"nvidia-5070": "RTX 5070", "nvidia-h100": "H100 SXM"}
MODELS = ["SmolLM2-135M", "SmolVLA", "StableDiffusion", "ResNet-50", "Whisper-tiny"]
SHORT = ["SmolLM2-135M", "SmolVLA expert", "SD U-Net", "ResNet-50", "Whisper enc."]
PHASES = [("inference", "Prefill /\nbatch"), ("latency", "Minimal\nshape"), ("training", "Training\nstep")]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 22, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "figure.facecolor": CANVAS, "axes.facecolor": CANVAS,
    "svg.fonttype": "none", "svg.hashsalt": "meganeura-gtc27"})
ROWS = list(csv.DictReader((P3HPC / "tables" / "results.csv").open()))
DATA = {(r["archive"], r["model"], r["precision"], r["phase"]): r for r in ROWS}


def save(fig, name):
    OUTPUT.mkdir(exist_ok=True)
    fig.savefig(OUTPUT / name, metadata={"Date": None})
    plt.close(fig)


def clean(ax, axis="x"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=axis, color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def times(device, model, profile, phase):
    r = DATA[device, model, profile, phase]
    if not r["ratio"]:
        return None
    return float(r["native_ms"]), float(r["reference_ms"])


def shade(value):
    """Teal below parity, brown above, deeper with distance from 1 (capped at 4×)."""
    if value is None:
        return "#eceeec", MUTED
    depth = min(abs(math.log(value)) / math.log(4), 1)
    base = to_rgb(TEAL if value < 1 else ORANGE)
    fill = tuple(1 - (1 - c) * (.12 + .78 * depth) for c in base)
    return fill, ("white" if depth > .55 else INK)


def ratio_grid(rows, name, height):
    """One heat grid per GPU; rows are (label, model, archive suffix)."""
    fig, axes = plt.subplots(1, 2, figsize=(13.2, height), gridspec_kw={"wspace": .06})
    for ax, (device, label) in zip(axes, DEVICES.items()):
        for row, (_, model, suffix) in enumerate(rows):
            for col, (profile, phase) in enumerate((p, ph) for p in ("strict", "accelerated") for ph, _ in PHASES):
                x = col + (.25 if col >= 3 else 0)
                if (device + suffix, model, profile, phase) not in DATA:
                    ax.text(x + .5, row + .5, "–", ha="center", va="center", fontsize=21, color=MUTED)
                    continue
                pair = times(device + suffix, model, profile, phase)
                value = pair[0] / pair[1] if pair else None
                fill, ink = shade(value)
                ax.add_patch(Rectangle((x + .04, row + .05), .92, .9, facecolor=fill, linewidth=0))
                ax.text(x + .5, row + .5, f"{value:.2f}" if value else "n/q", ha="center", va="center",
                        fontsize=21, color=ink, weight="bold" if value and value < 1 else "normal")
        ax.set_xlim(0, 6.25)
        ax.set_ylim(len(rows), -1.15)
        ax.set_xticks([c + .5 + (.25 if c >= 3 else 0) for c in range(6)],
                      [name for _ in range(2) for _, name in PHASES], fontsize=15)
        ax.set_yticks([r + .5 for r in range(len(rows))], [r[0] for r in rows] if ax is axes[0] else [])
        ax.tick_params(length=0)
        ax.spines[:].set_visible(False)
        ax.text(3.125, -1.0, label, ha="center", va="center", fontsize=25, weight="bold", color=INK)
        ax.text(1.5, -.45, "strict f32", ha="center", va="center", fontsize=19, color=MUTED)
        ax.text(4.75, -.45, "reduced precision", ha="center", va="center", fontsize=19, color=MUTED)
    # Keep the cell size of the five-row grid: 8.6 in holds 6.15 grid units plus axis labels.
    fig.subplots_adjust(left=.19, right=.995, top=1 - .08 / height, bottom=1.12 / height)
    save(fig, name)


def geomean(values):
    return math.exp(sum(map(math.log, values)) / len(values))


def precision_gains():
    """Speedup each engine gets from its own reduced-precision path over strict f32, same conditions."""
    groups = []
    for device, label in DEVICES.items():
        for phase, name in PHASES:
            torch, native = [], []
            for model in MODELS:
                strict, accel = times(device, model, "strict", phase), times(device, model, "accelerated", phase)
                if strict and accel:
                    native.append(strict[0] / accel[0])
                    torch.append(strict[1] / accel[1])
            groups.append((f"{label}\n{name.replace(chr(10), ' ')}", geomean(torch), geomean(native), len(torch)))
    fig, ax = plt.subplots(figsize=(13.2, 6.6))
    height = .36
    for i, (name, torch, native, n) in enumerate(groups):
        y = i + (.5 if i >= 3 else 0)
        ax.barh(y - height / 2 - .02, torch, height, color=MUTED, left=0)
        ax.barh(y + height / 2 + .02, native, height, color=TEAL, left=0)
        ax.text(torch + .04, y - height / 2 - .02, f"{torch:.2f}×", va="center", fontsize=19, color=INK)
        ax.text(native + .04, y + height / 2 + .02, f"{native:.2f}×", va="center", fontsize=19, color=INK)
    ax.set_yticks([i + (.5 if i >= 3 else 0) for i in range(len(groups))], [g[0] for g in groups], fontsize=17)
    ax.invert_yaxis()
    ax.axvline(1, color=INK, linewidth=1.5)
    ax.set_xlim(0, 3.4)
    ax.set_xlabel("Speedup from reduced precision (geometric mean, same qualified models)", fontsize=18)
    clean(ax)
    ax.tick_params(axis="y", length=0)
    ax.legend(handles=[Rectangle((0, 0), 1, 1, color=MUTED), Rectangle((0, 0), 1, 1, color=TEAL)],
              labels=["PyTorch: autocast BF16", "Meganeura: searched precision policy"],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=17)
    fig.subplots_adjust(left=.2, right=.98, top=.93, bottom=.13)
    save(fig, "precision-gains.svg")


ratio_grid([(short, model, "") for short, model in zip(SHORT, MODELS)], "ratios.svg", 8.6)
ratio_grid([("SmolLM2-360M", "SmolLM2-360M", "-large"), ("SmolLM2-1.7B", "SmolLM2-1.7B", "-large")],
           "sizes.svg", 5.0)
precision_gains()
