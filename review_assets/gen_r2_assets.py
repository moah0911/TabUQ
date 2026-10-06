"""Generate R2 visuals: detailed architecture (Fig 4.1) + proper DFD.

Only 2 figures are kept, plus copying the 2 real demo plots.
All other R2 slides are text-only (bullets).
ASCII-safe text only (no emoji, no unicode arrows).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shutil
from pathlib import Path

OUT = Path(__file__).parent

NAVY = "#1E2761"
TEAL = "#0B8A7C"
DARK = "#1A1A2E"
LIGHT_BLUE = "#E8EDF5"
LIGHT_TEAL = "#DFF3EF"
LIGHT_AMBER = "#FFF6E0"
GREY = "#6B7280"

plt.rcParams.update({"font.size": 10, "axes.edgecolor": GREY})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def title(ax, text):
    ax.text(0.5, 0.94, text, fontsize=15, fontweight="bold", color=NAVY,
            ha="center", va="center", transform=ax.transAxes, family="serif")


def box(ax, x0, y0, w, h, line1, line2, fc, ec, fs=9):
    ax.add_patch(mpatches.FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.05",
                                         facecolor=fc, edgecolor=ec, lw=1.5))
    ax.text(x0 + w / 2, y0 + h / 2 + 0.15, line1, fontsize=fs, ha="center",
            va="center", color=DARK, fontweight="bold")
    ax.text(x0 + w / 2, y0 + h / 2 - 0.26, line2, fontsize=fs - 0.5, ha="center",
            va="center", color=DARK)


def arrow(ax, x0, y0, x1, y1, label=None, color=NAVY, lw=1.5, ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw, linestyle=ls))
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.12, label, fontsize=8,
                ha="center", color=color, style="italic",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.5))


# ---------- Fig 4.1: detailed 6-stage architecture ----------
fig, ax = plt.subplots(figsize=(10, 5.6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis("off")
title(ax, "System Architecture: 6 Stages, Frozen-Model Reuse (Fig 4.1)")

row1 = [
    (0.25, "1. Dataset archive", "50 sets, 11 used", LIGHT_BLUE, NAVY),
    (3.53, "2. Data loader", "splits + encode cats", LIGHT_BLUE, NAVY),
    (6.81, "3. Trainer", "TabM 32/3/512, AdamW", LIGHT_BLUE, NAVY),
]
row2 = [
    (0.25, "4. Stored models", "33 models + pred. sets", LIGHT_AMBER, "#E6A817"),
    (3.53, "5. UQ + calibration", "H/MI/var; ECE15, NLL", LIGHT_TEAL, TEAL),
    (6.81, "6. OOD scorer", "7 shifts; AUROC/AUPR", LIGHT_TEAL, TEAL),
]
for x0, l1, l2, fc, ec in row1:
    box(ax, x0, 4.3, 2.94, 1.0, l1, l2, fc, ec)
for x0, l1, l2, fc, ec in row2:
    box(ax, x0, 2.75, 2.94, 1.0, l1, l2, fc, ec)
# row-1 flow
arrow(ax, 3.19, 4.8, 3.53, 4.8)
arrow(ax, 6.47, 4.8, 6.81, 4.8)
# stage 3 -> stage 4: elbow connector through the gap between rows
elbow_x, elbow_y = 8.28, 4.02
ax.plot([elbow_x, elbow_x, 1.72, 1.72], [4.3, elbow_y, elbow_y, 3.78],
        color=NAVY, lw=1.5)
ax.annotate("", xy=(1.72, 3.75), xytext=(1.72, 3.95),
            arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.5))
ax.text(5.0, 4.12, "33 trained models", fontsize=8, ha="center", color=DARK,
        style="italic")
# row-2 flow (4 -> 5 -> 6)
arrow(ax, 3.19, 3.25, 3.53, 3.25)
arrow(ax, 6.47, 3.25, 6.81, 3.25)
ax.text(5, 1.9, "33 uncertainty files + 48 plots + 245 OOD files (frozen models reused, zero retraining)",
        fontsize=9.5, ha="center", color=DARK,
        bbox=dict(facecolor="white", edgecolor=TEAL, boxstyle="round,pad=0.35"))
ax.text(5, 1.15, "Early stopping (patience 20) - batch 256-2048 - seeds 0,1,2 - CPU only",
        fontsize=9, ha="center", color=GREY)
ax.text(5, 0.5, "Binary indicator features of 2 sets excluded (identical features for ID and OOD)",
        fontsize=9, ha="center", color=GREY)
save(fig, "r2_arch.png")

# ---------- DFD L0/L1: processes (circles), stores, external entity ----------
fig, ax = plt.subplots(figsize=(10, 5.6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis("off")
title(ax, "Data-Flow Diagram, Levels 0-1 (Processes, Stores, Flows)")


def entity(ax, x0, y0, w, h, l1, l2):
    ax.add_patch(mpatches.FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.05",
                                         facecolor=LIGHT_BLUE, edgecolor=NAVY, lw=1.6))
    ax.text(x0 + w / 2, y0 + h / 2 + 0.15, l1, fontsize=9, ha="center",
            va="center", color=NAVY, fontweight="bold")
    ax.text(x0 + w / 2, y0 + h / 2 - 0.25, l2, fontsize=8.5, ha="center",
            va="center", color=DARK)


def process(ax, cx, cy, r, l1, l2):
    ax.add_patch(mpatches.Circle((cx, cy), r, facecolor="white", edgecolor=TEAL, lw=1.6))
    ax.text(cx, cy + 0.14, l1, fontsize=9, ha="center", va="center", color=DARK,
            fontweight="bold")
    ax.text(cx, cy - 0.24, l2, fontsize=8.5, ha="center", va="center", color=DARK)


def store(ax, x0, y0, w, h, l1, l2):
    ax.add_patch(mpatches.Rectangle((x0, y0), w, h, facecolor=LIGHT_AMBER,
                                    edgecolor="#E6A817", lw=1.4))
    ax.plot([x0 + 0.22, x0 + 0.22], [y0, y0 + h], color="#E6A817", lw=1.4)
    ax.text(x0 + w / 2 + 0.1, y0 + h / 2 + 0.15, l1, fontsize=8.5, ha="center",
            va="center", color=DARK, fontweight="bold")
    ax.text(x0 + w / 2 + 0.1, y0 + h / 2 - 0.25, l2, fontsize=8, ha="center",
            va="center", color=DARK)


entity(ax, 0.25, 3.9, 1.55, 1.0, "Dataset", "archive")
process(ax, 2.95, 4.4, 0.62, "P1", "Train x33")
store(ax, 4.25, 3.9, 1.75, 1.0, "D1 Checkpoints", "33 models")
process(ax, 1.7, 1.9, 0.62, "P2", "UQ+calib")
store(ax, 2.95, 1.4, 1.7, 1.0, "D2 UQ results", "33 files")
store(ax, 5.0, 1.4, 1.6, 1.0, "D3 Figures", "48 plots")
process(ax, 7.55, 1.9, 0.62, "P4", "OOD x7")
store(ax, 8.5, 1.4, 1.35, 1.0, "D4 OOD", "245 files")

arrow(ax, 1.8, 4.4, 2.33, 4.4, "splits", NAVY)
arrow(ax, 3.57, 4.4, 4.25, 4.4, "model", NAVY)
arrow(ax, 4.9, 3.9, 1.9, 2.52, "frozen model", GREY, ls="--")
arrow(ax, 1.35, 3.9, 1.6, 2.52, "test splits", GREY, ls="--")
arrow(ax, 2.32, 1.9, 2.95, 1.9, "scores", NAVY)
arrow(ax, 4.65, 1.62, 5.0, 1.62)
ax.text(4.82, 1.78, "plots", fontsize=8, ha="center", color=NAVY, style="italic",
        bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
arrow(ax, 5.6, 3.9, 7.3, 2.52, "model+shifts", GREY, ls="--")
arrow(ax, 8.17, 1.9, 8.5, 1.9)
ax.text(8.33, 2.12, "AUROC", fontsize=8, ha="center", color=NAVY, style="italic",
        bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
ax.text(5, 0.6, "No database: versioned file archive. Interface: train / evaluate-UQ / evaluate-OOD commands.",
        fontsize=9.5, ha="center", color=GREY)
save(fig, "r2_dfd.png")

# ---------- Demo figures: copy real result plots ----------
src_rel = Path(__file__).parent.parent / "results/figures/phoneme_seed0_reliability.png"
src_ent = Path(__file__).parent.parent / "results/figures/phoneme_seed0_entropy_hist.png"
if src_rel.exists():
    shutil.copy(src_rel, OUT / "r2_demo_reliability.png")
    print("copied r2_demo_reliability.png")
if src_ent.exists():
    shutil.copy(src_ent, OUT / "r2_demo_entropy.png")
    print("copied r2_demo_entropy.png")

print("R2 done:", sorted(p.name for p in OUT.glob("r2_*.png")))
