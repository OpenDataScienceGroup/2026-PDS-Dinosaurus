"""
Generates figures for the report:
  1. Colour feature distributions by class
  2. Hair coverage distribution by annotation label

Run from the project root:
    python generate_figures.py

Requires: features_cleaned.csv and annotations_combined.csv in data/
Saves figures to: results/figures/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pathlib import Path

DATA_PATH = Path("data")
FIGURES_PATH = Path("results/figures")
FIGURES_PATH.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", font_scale=1.1)
mpl.rcParams["figure.dpi"] = 150
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False

CANCER_COLOR = "#c0392b"
BENIGN_COLOR = "#2980b9"
LABEL_0_COLOR = "#2980b9"
LABEL_1_COLOR = "#e67e22"
LABEL_2_COLOR = "#27ae60"

# ---------------------------------------------------------------------------
# Figure 1: Colour feature distributions by class
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH / "features_cleaned.csv")

features_to_plot = [
    ("mean_h",          "Mean Hue (H)"),
    ("std_s",           "Std Saturation (S)"),
    ("color_entropy",   "Colour Entropy"),
    ("blue_veil_score", "Blue-White Veil Score"),
]

fig, axes = plt.subplots(4, 1, figsize=(6, 16))

for ax, (feat, title) in zip(axes, features_to_plot):
    for label, name, color in [
        (0, "Non-cancerous", BENIGN_COLOR),
        (1, "Cancerous",     CANCER_COLOR),
    ]:
        subset = df[df["label"] == label][feat].dropna()
        sns.kdeplot(subset, ax=ax, label=name, color=color,
                    fill=True, alpha=0.35, linewidth=2)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Feature value")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)

fig.suptitle("Colour feature distributions by class", fontsize=14,
             fontweight="bold", y=0.995)
plt.tight_layout()
plt.savefig(FIGURES_PATH / "colour_distributions.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved colour_distributions.png")

# ---------------------------------------------------------------------------
# Figure 2: Hair coverage by annotation label
# ---------------------------------------------------------------------------
annotations = pd.read_csv(DATA_PATH / "annotations_combined.csv")
features = pd.read_csv(DATA_PATH / "features_cleaned.csv")

hair_cols = ["hair_1", "hair_2", "hair_3", "hair_4", "hair_5"]

mode_vals = annotations[hair_cols].mode(axis=1)[0]
annotations = annotations[mode_vals.notna()].copy()
annotations["majority_hair"] = mode_vals[mode_vals.notna()].astype(int)

merged = annotations[["img_id", "majority_hair"]].merge(
    features[["img_id", "hair_coverage"]], on="img_id", how="inner"
)

fig, ax = plt.subplots(figsize=(8, 5))

label_styles = [
    (0, "0 — no or little hair", LABEL_0_COLOR),
    (1, "1 — some hair",         LABEL_1_COLOR),
    (2, "2 — heavy hair",        LABEL_2_COLOR),
]

for label, name, color in label_styles:
    subset = merged[merged["majority_hair"] == label]["hair_coverage"].dropna()
    sns.kdeplot(subset, ax=ax, label=name, color=color,
                fill=True, alpha=0.35, linewidth=2)

ax.set_xlabel("Hair coverage value", fontsize=12)
ax.set_ylabel("Density", fontsize=12)
ax.set_title("Hair coverage values by annotation label",
             fontsize=13, fontweight="bold")
ax.legend(frameon=False, title="Annotation label")
plt.tight_layout()
plt.savefig(FIGURES_PATH / "hair_coverage_distribution.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved hair_coverage_distribution.png")
