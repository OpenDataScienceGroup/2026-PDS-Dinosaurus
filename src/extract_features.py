"""
Feature extraction pipeline for PAD-UFES-20 skin lesion dataset.

Reads all images + masks listed in metadata.csv, extracts features for each,
and writes data/features.csv.

Usage:
    python src/extract_features.py

Paths can be overridden via environment variables:
    DATA_PATH   (default: ./data/)
    OUTPUT_CSV  (default: ./data/features.csv)
"""

import os
import sys
import numpy as np
import pandas as pd
import cv2
from pathlib import Path

# allow running from project root or from src/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from asymetry_A import get_asymmetry
from color_features import get_color_features
from border_features import get_border_features
from texture_features import get_texture_features
from hair_detection import hair_coverage, remove_hair
from penmark_detection import penmark_coverage, remove_penmarks

# ---------------------------------------------------------------------------
# Paths (override with env vars if needed)
# ---------------------------------------------------------------------------
DATA_PATH = Path(os.environ.get("DATA_PATH", Path(__file__).resolve().parent.parent / "data"))
OUTPUT_CSV = Path(os.environ.get("OUTPUT_CSV", DATA_PATH / "features.csv"))

IMG_DIR = DATA_PATH / "imgs"
MASK_DIR = DATA_PATH / "masks"
METADATA_CSV = DATA_PATH / "metadata.csv"

# Diagnostic → binary label mapping
# Malignant: BCC, SCC, MEL  |  Benign/pre-malignant: NEV, SEK, ACK
MALIGNANT_CLASSES = {"BCC", "SCC", "MEL"}


def load_image_and_mask(img_id):
    """
    Load image (RGB uint8) and segmentation mask (grayscale uint8).
    Returns (image, mask) or (None, None) if either file is missing.
    Images without a segmentation mask are skipped — lesion-specific
    features cannot be reliably extracted without one.
    """
    img_path = IMG_DIR / img_id
    mask_path = MASK_DIR / img_id.replace(".png", "_mask.png")

    if not img_path.exists() or not mask_path.exists():
        return None, None

    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    # Some masks are off by 1 pixel — resize to match the image exactly
    if mask.shape != img_rgb.shape[:2]:
        mask = cv2.resize(mask, (img_rgb.shape[1], img_rgb.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    return img_rgb, mask


def extract_features_for_row(row):
    """
    Given a metadata row, load image+mask and extract all features.
    Returns a flat dict of feature values.
    """
    img_id = row["img_id"]
    diagnostic = row["diagnostic"]
    patient_id = row["patient_id"]
    label = 1 if diagnostic in MALIGNANT_CLASSES else 0

    img_rgb, mask = load_image_and_mask(img_id)
    if img_rgb is None:
        return None

    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # ------------------------------------------------------------------
    # Shortcut detection (before any preprocessing)
    # ------------------------------------------------------------------
    hair_cov = hair_coverage(img_gray)
    pen_cov = penmark_coverage(img_rgb)

    # ------------------------------------------------------------------
    # Preprocessing: remove shortcuts before extracting lesion features
    # ------------------------------------------------------------------
    processed_img = img_rgb.copy()

    if pen_cov > 0.005:
        _, processed_img = remove_penmarks(processed_img)

    proc_gray = cv2.cvtColor(processed_img, cv2.COLOR_RGB2GRAY)
    if hair_cov > 0.01:
        _, processed_img = remove_hair(processed_img, proc_gray, coverage=hair_cov)

    # ------------------------------------------------------------------
    # Feature extraction on the cleaned image
    # ------------------------------------------------------------------
    asymmetry = get_asymmetry(mask)
    border_feats = get_border_features(mask)
    color_feats = get_color_features(processed_img, mask)
    texture_feats = get_texture_features(processed_img, mask)

    feats = {
        "img_id": img_id,
        "patient_id": patient_id,
        "diagnostic": diagnostic,
        "label": label,          # 0 = benign, 1 = malignant
        # shape
        "asymmetry": asymmetry,
        **border_feats,
        # colour
        **color_feats,
        # texture
        **texture_feats,
        # shortcut features (used in the extended classifier)
        "hair_coverage": hair_cov,
        "penmark_coverage": pen_cov,
    }

    return feats


def main():
    print(f"Loading metadata from {METADATA_CSV}")
    df = pd.read_csv(METADATA_CSV)

    print(f"Extracting features for {len(df)} images …")
    records = []
    failed = 0

    for i, row in df.iterrows():
        try:
            feats = extract_features_for_row(row)
            if feats is None:
                failed += 1
                continue
            records.append(feats)
        except Exception as e:
            print(f"  [WARN] {row['img_id']}: {e}")
            failed += 1

        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(df)} done …")

    features_df = pd.DataFrame(records)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nDone. {len(features_df)} rows saved to {OUTPUT_CSV}")
    print(f"Failed / missing: {failed}")
    print("\nClass distribution:")
    print(features_df["diagnostic"].value_counts())
    print("\nLabel distribution (0=benign, 1=malignant):")
    print(features_df["label"].value_counts())


if __name__ == "__main__":
    main()
