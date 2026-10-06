"""Generate R1 visuals: detailed solution diagram (Fig 1.1) + real Gantt (Fig A.1).

Only 2 figures are kept. All other R1 slides are text-only (bullets);
thesis tables are typed as Word tables, not images.
ASCII-safe text only (no emoji, no unicode arrows).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT = Path(__file__).parent
OUT.mkdir(parents=True, exist_ok=True)

NAVY = "#1E2761"
TEAL = "#0B8A7C"
DARK = "#1A1A2E"
LIGHT_BLUE = "#E8EDF5"
LIGHT_TEAL = "#DFF3EF"
LIGHT_AMBER = "#FFF6E0"
GREY = "#6B7280"
RED = "#C0392B"

plt.rcParams.update({"font.size": 10, "axes.edgecolor": GREY})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def title(ax, text):
    ax.text(0.5, 0.94, text, fontsize=15, fontweight="bold", color=NAVY,
            ha="center", va="center", transform=ax.transAxes, family="serif")


def box(ax, x0, y0, w, h, line1, line2, fc, ec, fs=9.5):
    ax.add_patch(mpatches.FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.06",
                                         facecolor=fc, edgecolor=ec, lw=1.5))
    ax.text(x0 + w / 2, y0 + h / 2 + 0.14, line1, fontsize=fs, ha="center",
            va="center", color=DARK, fontweight="bold")
    ax.text(x0 + w / 2, y0 + h / 2 - 0.24, line2, fontsize=fs - 0.5, ha="center",
            va="center", color=DARK)


def arrow(ax, x0, x1, y, color=NAVY, lw=1.5):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw))


# ---------- Fig 1.1: detailed UQ-layer pipeline ----------
fig, ax = plt.subplots(figsize=(10, 5.6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis("off")
title(ax, "Proposed Solution: Post-hoc UQ Layer on Frozen TabM (Fig 1.1)")

stages = [
    (0.25, "Tabular row", "8-50 features", LIGHT_BLUE, NAVY),
    (2.15, "TabM, k=32", "logits (B,32,1)", LIGHT_BLUE, NAVY),
    (4.05, "Sigmoid / softmax", "probs (B,32,C)", "white", GREY),
    (5.95, "Mean over k", "p-bar (B,C)", "white", GREY),
    (7.85, "UQ scores", "H / MI / var", LIGHT_TEAL, TEAL),
]
for x0, l1, l2, fc, ec in stages:
    box(ax, x0, 4.15, 1.9, 1.05, l1, l2, fc, ec)
for i in range(4):
    arrow(ax, stages[i][0] + 1.9, stages[i + 1][0], 4.67)

forms = [
    (0.25, "H = -sum p-bar log p-bar", "total uncertainty", LIGHT_AMBER),
    (3.52, "MI = H - mean(H_i)", "epistemic (model) part", LIGHT_AMBER),
    (6.79, "var = mean((y_i - y-bar)^2)", "regression uncertainty", LIGHT_AMBER),
]
for x0, l1, l2, fc in forms:
    box(ax, x0, 2.85, 2.96, 0.95, l1, l2, fc, "#E6A817", fs=9)

outs = [
    (0.25, "Prediction", "argmax of p-bar", LIGHT_BLUE, NAVY),
    (3.52, "Calibration", "ECE(15 bins), NLL", LIGHT_BLUE, NAVY),
    (6.79, "OOD score", "AUROC(id, shifted)", LIGHT_TEAL, TEAL),
]
for x0, l1, l2, fc, ec in outs:
    box(ax, x0, 1.55, 2.96, 0.95, l1, l2, fc, ec, fs=9)
# vertical connectors: scores row -> outputs row
for x0, _, _, _, _ in outs:
    ax.annotate("", xy=(x0 + 1.48, 2.5), xytext=(x0 + 1.48, 2.85),
                arrowprops=dict(arrowstyle="->", color=GREY, lw=1.2))
ax.text(5, 0.75, "Training: mean loss over k members (keeps members diverse)",
        fontsize=9.5, ha="center", color=DARK, style="italic",
        bbox=dict(facecolor="white", edgecolor=GREY, boxstyle="round,pad=0.3"))
ax.text(5, 0.15, "Post-hoc: model frozen, no retraining - scoring layer about 50 lines - CPU only",
        fontsize=9.5, ha="center", color=GREY)
save(fig, "r1_solution.png")

# ---------- Fig A.1: real Gantt chart ----------
tasks = [
    ("M1 pilot (small config)", 1.0, 1.0, "done", "1.5 h done"),
    ("M2 full-size + UQ", 2.0, 3.0, "done", "30 h, 33/33 runs"),
    ("M3 OOD, 7 shifts", 5.0, 1.0, "done", "245 result files"),
    ("Paper draft", 6.0, 1.0, "active", "workshop"),
    ("Thesis + viva", 7.0, 1.0, "todo", "Ch1-7"),
]
fig, ax = plt.subplots(figsize=(10, 5.6))
ypos = list(range(len(tasks)))[::-1]
ax.set_ylim(-0.9, 5.4)
cols = {"done": TEAL, "active": "#E6A817", "todo": "#9AA5C4"}
for (name, s, d, st, note), y in zip(tasks, ypos):
    ax.barh(y, d, left=s, height=0.55, color=cols[st], edgecolor=DARK, lw=1)
    if d < 1.6:
        ax.text(s + d + 0.08, y, note, fontsize=9, ha="left", va="center",
                color=DARK, fontweight="bold")
    else:
        ax.text(s + d / 2, y, note, fontsize=8.5, ha="center", va="center",
                color="white", fontweight="bold")
ax.set_yticks(ypos)
ax.set_yticklabels([t[0] for t in tasks], fontsize=10)
ax.set_xlabel("Project month (M1-M8)", fontsize=10)
ax.set_xlim(0.5, 9.3)
ax.set_xticks(range(1, 9))
ax.grid(axis="x", linestyle=":", color="#BBBBBB", lw=0.8)
for rv, rm in [("R1", 1.0), ("R2", 4.0), ("R3", 6.5), ("R4", 8.0)]:
    ax.axvline(rm, color=NAVY, linestyle="--", lw=1)
    ax.text(rm, 4.95, rv, fontsize=9, ha="center", color=NAVY, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor=NAVY, boxstyle="round,pad=0.2"))
ax.set_title("Project Plan and Timeline (Fig A.1)", fontsize=14, fontweight="bold",
             color=NAVY, family="serif", pad=10)
fig.text(0.5, 0.01, "External baselines deferred per single-model scope (workshop needs UQ+OOD only)",
         ha="center", fontsize=9, color=GREY)
fig.tight_layout(rect=[0, 0.05, 1, 0.95])
fig.savefig(OUT / "r1_gantt.png", dpi=200, facecolor="white", bbox_inches="tight")
plt.close(fig)
print("saved r1_gantt.png")

print("R1 done:", sorted(p.name for p in OUT.glob("r1_*.png")))
