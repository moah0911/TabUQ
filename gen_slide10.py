import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(10, 5.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 5.5)
ax.axis("off")

# ---------- colours ----------
BG        = "#f8f9fa"
BLUE_M    = "#1565C0"  # main blue
BLUE_L    = "#E3F2FD"  # light blue
TEAL_M    = "#00897B"
TEAL_L    = "#B2DFDB"
AMBER_L   = "#FFF8E1"
GREY_L    = "#EEEEEE"
GREY_TEXT = "#424242"

fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

# ---------- title ----------
ax.text(5, 5.15, "Team Roles & Report Organisation", fontsize=16, fontweight="bold",
        ha="center", va="center", color=BLUE_M)
ax.plot([2.5, 7.5], [4.95, 4.95], color=BLUE_M, lw=1.5)

# ===================== LEFT SIDE — Module Ownership =====================
left_x = 0.4
col_w = 5.6

table_box = mpatches.FancyBboxPatch(
    (left_x, 2.65), col_w, 2.2,
    boxstyle="round,pad=0.15", facecolor=BLUE_L, edgecolor=BLUE_M, lw=1.2
)
ax.add_patch(table_box)
ax.text(left_x + col_w / 2, 4.6, "Module Ownership", fontsize=11, fontweight="bold",
        ha="center", va="center", color=BLUE_M)

# module rows
modules = [
    ("AW-TabM Implementation", "UQ Scoring Pipeline", "OOD Detection Pipeline"),
    ("Baselines (MC Dropout, Deep Ensemble, LightGBM)", "Calibration Analysis", "Visualisation & Plots"),
]

row_y = 4.15
for i, row in enumerate(modules):
    for j, mod in enumerate(row):
        x = left_x + 0.25 + j * (col_w - 0.5) / 3
        cell = mpatches.FancyBboxPatch(
            (x - 0.75, row_y - 0.3), 1.45, 0.5,
            boxstyle="round,pad=0.08", facecolor="white", edgecolor=BLUE_M, lw=0.8
        )
        ax.add_patch(cell)
        ax.text(x + 0, row_y, mod, fontsize=6.5, ha="center", va="center",
                color=GREY_TEXT, fontweight="medium")
    row_y -= 0.65

# owner label
owner_box = mpatches.FancyBboxPatch(
    (left_x + 0.25, row_y - 0.15), col_w - 0.5, 0.45,
    boxstyle="round,pad=0.08", facecolor=TEAL_L, edgecolor=TEAL_M, lw=1
)
ax.add_patch(owner_box)
ax.text(left_x + col_w / 2, row_y + 0.08, "Owned by: [Your Name] (Solo Project)",
        fontsize=9, ha="center", va="center", color=TEAL_M, fontweight="bold")

# ===================== RIGHT SIDE — Report Outline =====================
right_x = 6.5

table_box2 = mpatches.FancyBboxPatch(
    (right_x, 0.2), 3.3, 4.55,
    boxstyle="round,pad=0.15", facecolor=AMBER_L, edgecolor="#FF8F00", lw=1.2
)
ax.add_patch(table_box2)
ax.text(right_x + 1.65, 4.5, "Thesis Chapter Outline", fontsize=11, fontweight="bold",
        ha="center", va="center", color="#E65100")

chapters = [
    ("Ch 1", "Introduction", "Problem, objectives, scope, approach"),
    ("Ch 2", "Literature Survey", "UQ, OOD, tabular DL, gaps"),
    ("Ch 3", "System Design", "AW-TabM architecture, modules"),
    ("Ch 4", "Implementation", "Code, training, UQ pipeline"),
    ("Ch 5", "Testing", "Unit, integration, system tests"),
    ("Ch 6", "Results & Analysis", "Metrics, comparisons, plots"),
    ("Ch 7", "Conclusion & Future Work", "Summary, limitations, extensions"),
]

cy = 4.05
for ch_no, ch_title, ch_desc in chapters:
    ch_box = mpatches.FancyBboxPatch(
        (right_x + 0.2, cy - 0.45), 2.9, 0.4,
        boxstyle="round,pad=0.05", facecolor="white", edgecolor="#FF8F00", lw=0.7
    )
    ax.add_patch(ch_box)
    ax.text(right_x + 0.35, cy - 0.22, f"{ch_no}: {ch_title}",
            fontsize=7.5, ha="left", va="center", color="#E65100", fontweight="bold")
    ax.text(right_x + 0.35, cy - 0.5, ch_desc,
            fontsize=6.5, ha="left", va="center", color=GREY_TEXT)
    cy -= 0.55

# ---------- footer ----------
ax.text(5, 0.05, "Phase 1 · Review 1 · Project Acceptance Review  —  MVGR College of Engineering  —  CSE Dept.",
        fontsize=7, ha="center", va="center", color="#9E9E9E", fontstyle="italic")

plt.tight_layout()
plt.savefig("slide10_team_roles.png", dpi=200, bbox_inches="tight", facecolor=BG)
print("Saved: slide10_team_roles.png")
