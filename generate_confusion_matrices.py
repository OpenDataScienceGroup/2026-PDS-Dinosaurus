"""
Generates confusion matrix figures for the report.
Reads predictions from results/predictions/ and saves
confusion matrix plots to results/figures/.

Run from the project root:
    python generate_confusion_matrices.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path

PREDICTIONS_DIR = Path("results/predictions")
FIGURES_PATH = Path("results/figures")
FIGURES_PATH.mkdir(parents=True, exist_ok=True)

mpl.rcParams["figure.dpi"] = 150

models = [
    ("predictions_baseline_LogisticRegression.csv", "cm_baseline.png"),
    ("predictions_extended_LogisticRegression.csv", "cm_extended.png"),
]

for filename, output in models:
    df = pd.read_csv(PREDICTIONS_DIR / filename)
    cm = confusion_matrix(df["label"], df["predicted_label"])
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Non-cancerous", "Cancerous"]
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(output.replace("cm_", "").replace(".png", "").capitalize()
                 + " model", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output}")
