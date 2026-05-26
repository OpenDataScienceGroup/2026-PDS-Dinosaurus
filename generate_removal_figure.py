"""
Generates a before/after removal example figure for the report.
Picks one image with hair and one with pen marks automatically.

Run from the project root:
    python generate_removal_figure.py

Saves to: results/figures/removal_examples.png
"""

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import sys

sys.path.insert(0, str(Path("src").resolve()))
from asymmetry_features import get_asymmetry
from hair_detection import hair_coverage, remove_hair
from penmark_detection import penmark_coverage, remove_penmarks

DATA_PATH = Path("data")
FIGURES_PATH = Path("results/figures")
FIGURES_PATH.mkdir(parents=True, exist_ok=True)

mpl.rcParams["figure.dpi"] = 150

df = pd.read_csv(DATA_PATH / "features_cleaned.csv")

# pick image with most hair coverage
hair_img_id = df.sort_values("hair_coverage", ascending=False).iloc[0]["img_id"]

# pick image with most pen mark coverage
pen_img_id = df.sort_values("penmark_coverage", ascending=False).iloc[0]["img_id"]

def load_rgb(img_id):
    path = DATA_PATH / "imgs" / img_id
    img_bgr = cv2.imread(str(path))
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def process(img_rgb):
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    hair_cov = hair_coverage(img_gray)
    pen_cov = penmark_coverage(img_rgb)
    processed = img_rgb.copy()
    if pen_cov > 0.005:
        _, processed = remove_penmarks(processed)
    proc_gray = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
    if hair_cov > 0.01:
        _, processed = remove_hair(processed, proc_gray, coverage=hair_cov)
    return processed

hair_orig = load_rgb(hair_img_id)
hair_clean = process(hair_orig)
pen_orig = load_rgb(pen_img_id)
pen_clean = process(pen_orig)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].imshow(hair_orig)
axes[0, 0].set_title("Hair — original", fontweight="bold")
axes[0, 0].axis("off")

axes[0, 1].imshow(hair_clean)
axes[0, 1].set_title("Hair — after removal", fontweight="bold")
axes[0, 1].axis("off")

axes[1, 0].imshow(pen_orig)
axes[1, 0].set_title("Pen marks — original", fontweight="bold")
axes[1, 0].axis("off")

axes[1, 1].imshow(pen_clean)
axes[1, 1].set_title("Pen marks — after removal", fontweight="bold")
axes[1, 1].axis("off")

plt.tight_layout()
plt.savefig(FIGURES_PATH / "removal_examples.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved removal_examples.png")
