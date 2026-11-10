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
INK, TEAL, BROWN, CANVAS, MUTED, GRID = "#23343b", "#006b70", "#a44616", "#ffffff", "#637679", "#d1dad8"
DEVICES = {"nvidia-3050": ("RTX 3050", "Windows"), "nvidia-5070": ("RTX 5070", "Linux"),
           "nvidia-h100": ("H100 SXM", "Linux")}
MODELS = ["SmolLM2-135M", "SmolVLA", "StableDiffusion", "ResNet-50", "Whisper-tiny"]
# Grid rows: label, model, archive suffix. The larger SmolLM2 sizes come from the "-large" archives.
ROWS = [("SmolLM2-135M", "SmolLM2-135M", ""), ("SmolLM2-360M", "SmolLM2-360M", "-large"),
        ("SmolLM2-1.7B", "SmolLM2-1.7B", "-large"), ("SmolVLA", "SmolVLA", ""),
        ("SD U-Net", "StableDiffusion", ""), ("ResNet-50", "ResNet-50", ""), ("Whisper", "Whisper-tiny", "")]
PROFILES = [("strict", "strict f32"), ("accelerated", "reduced")]
PHASES = ["inference", "training"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 22, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "figure.facecolor": CANVAS, "axes.facecolor": CANVAS,
    "svg.fonttype": "none", "svg.hashsalt": "meganeura-gtc27"})
DATA = {(r["archive"], r["model"], r["precision"], r["phase"]): r
        for r in csv.DictReader((P3HPC / "tables" / "results.csv").open())}


def save(fig, name):
    OUTPUT.mkdir(exist_ok=True)
    fig.savefig(OUTPUT / name, metadata={"Date": None})
    plt.close(fig)


def times(device, model, profile, phase):
    r = DATA[device, model, profile, phase]
    if not r["ratio"]:
        return None
    return float(r["native_ms"]), float(r["reference_ms"])


def geomean(values):
    return math.exp(sum(map(math.log, values)) / len(values))


def shade(value):
    """Teal below parity, brown above, deeper with distance from 1 (capped at 4×)."""
    if value is None:
        return "#eceeec", MUTED
    depth = min(abs(math.log(value)) / math.log(4), 1)
    base = to_rgb(TEAL if value < 1 else BROWN)
    fill = tuple(1 - (1 - c) * (.12 + .78 * depth) for c in base)
    return fill, ("white" if depth > .55 else INK)


def ratio_grid():
    """One strict-f32 heat grid per GPU: workloads by phase.

    Reduced precision is covered by the tiles and the gains chart, and minimal-shape ratios by the caption.
    """
    height = (len(ROWS) + 1.15) * .95 + .08  # 0.95 in per row, header included
    fig, axes = plt.subplots(1, len(DEVICES), figsize=(13.2, height), gridspec_kw={"wspace": .09})
    for ax, (device, (label, system)) in zip(axes, DEVICES.items()):
        for row, (_, model, suffix) in enumerate(ROWS):
            for x, phase in enumerate(PHASES):
                if (device + suffix, model, "strict", phase) not in DATA:
                    ax.text(x + .5, row + .5, "–", ha="center", va="center", fontsize=24, color=MUTED)
                    continue
                native, reference = times(device + suffix, model, "strict", phase)
                value = native / reference
                fill, ink = shade(value)
                ax.add_patch(Rectangle((x + .025, row + .05), .95, .9, facecolor=fill, linewidth=0))
                ax.text(x + .5, row + .5, f"{value:.2f}", ha="center", va="center", fontsize=26, color=ink,
                        weight="bold" if value < 1 else "normal")
        ax.set_xlim(0, len(PHASES))
        ax.set_ylim(len(ROWS), -1.15)
        ax.set_xticks([])
        ax.set_yticks([r + .5 for r in range(len(ROWS))], [r[0] for r in ROWS] if ax is axes[0] else [],
                      fontsize=21)
        ax.tick_params(length=0)
        ax.spines[:].set_visible(False)
        ax.text(.025, -.82, label, ha="left", va="center", fontsize=27, weight="bold", color=INK)
        ax.text(len(PHASES) - .025, -.82, system, ha="right", va="center", fontsize=19, color=MUTED)
        for x, phase in enumerate(PHASES):
            ax.text(x + .5, -.27, phase, ha="center", va="center", fontsize=19, color=MUTED)
    axes[0].text(-.04, -.27, "strict f32", transform=axes[0].get_yaxis_transform(), ha="right", va="center",
                 fontsize=19, weight="bold", color=INK)
    fig.subplots_adjust(left=.175, right=.997, top=1 - .04 / height, bottom=.04 / height)
    save(fig, "ratios.svg")


def precision_gains():
    """Speedup each engine gets from its own reduced-precision path over strict f32, same conditions."""
    groups = []
    for device, (label, _) in DEVICES.items():
        for phase in PHASES:
            torch, native = [], []
            for model in MODELS:
                strict, accel = times(device, model, "strict", phase), times(device, model, "accelerated", phase)
                if strict and accel:
                    native.append(strict[0] / accel[0])
                    torch.append(strict[1] / accel[1])
            groups.append((label, phase, geomean(torch), geomean(native)))
            print(f"{label:9s} {phase:9s} PyTorch {geomean(torch):.2f}  Meganeura {geomean(native):.2f}  n={len(torch)}")
    fig, ax = plt.subplots(figsize=(13.2, 5.5))
    bar, gap = .31, .055  # each bar sits `gap` away from the group's centre line
    positions = [i + .45 * (i // len(PHASES)) for i in range(len(groups))]
    for y, (_, _, torch, native) in zip(positions, groups):
        ax.barh(y - bar / 2 - gap, torch, bar, color=BROWN)
        ax.barh(y + bar / 2 + gap, native, bar, color=TEAL)
        ax.text(torch + .04, y - bar / 2 - gap, f"{torch:.2f}×", va="center", fontsize=17, color=INK)
        ax.text(native + .04, y + bar / 2 + gap, f"{native:.2f}×", va="center_baseline", fontsize=17, color=INK)
    ax.set_yticks(positions, [phase for _, phase, _, _ in groups], fontsize=18)
    for i, (label, _) in enumerate(DEVICES.values()):
        middle = (positions[i * len(PHASES)] + positions[(i + 1) * len(PHASES) - 1]) / 2
        ax.text(-.2, middle, label, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=20, weight="bold")
    ax.invert_yaxis()
    ax.axvline(1, color=INK, linewidth=1.5)
    ax.set_xlim(0, 3.4)
    ax.set_xlabel("Speedup over strict f32 (geometric mean, same qualified models)", fontsize=18)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=18)
    ax.legend(handles=[Rectangle((0, 0), 1, 1, color=BROWN), Rectangle((0, 0), 1, 1, color=TEAL)],
              labels=["PyTorch: autocast BF16", "Meganeura: searched precision policy"],
              loc="lower left", bbox_to_anchor=(-.02, 1.0), ncol=2, frameon=False, fontsize=18)
    fig.subplots_adjust(left=.27, right=.985, top=.91, bottom=.165)
    save(fig, "precision-gains.svg")


def summary():
    """Numbers quoted in poster.html and submission.md."""
    for device, (label, _) in DEVICES.items():
        cells = []
        for profile, _ in PROFILES:
            for phase in ("inference", "latency", "training"):
                pairs = [times(device, model, profile, phase) for model in MODELS]
                cells.append(f"{profile[:1]}/{phase[:3]} {geomean([a / b for a, b in filter(None, pairs)]):.2f}")
        qualified = sum(bool(r["ratio"]) for key, r in DATA.items() if key[0] in (device, device + "-large"))
        print(f"{label:9s} " + "  ".join(cells) + f"  ratios {qualified}")


if __name__ == "__main__":
    ratio_grid()
    precision_gains()
    summary()
