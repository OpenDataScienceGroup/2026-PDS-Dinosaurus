"""
Computes Fleiss' Kappa inter-rater agreement for the course annotation study.
Reads data/annotations_combined.csv and prints the kappa value for hair ratings.

Usage:
    python compute_kappa.py
"""
import pandas as pd
import numpy as np
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters

DATA_PATH = "data/annotations_combined.csv"
HAIR_COLS = ["hair_1", "hair_2", "hair_3", "hair_4", "hair_5"]

def main():
    df = pd.read_csv(DATA_PATH)

    # keep only rows where at least 4 raters gave a score
    mask = df[HAIR_COLS].notna().sum(axis=1) >= 4
    ratings = df.loc[mask, HAIR_COLS].values.astype(float)

    # fill missing 5th rater with rounded row mean
    for i in range(len(ratings)):
        row = ratings[i]
        if np.isnan(row).any():
            row[np.isnan(row)] = round(np.nanmean(row))
        ratings[i] = row

    # clip to valid label range 0-2
    ratings = np.clip(ratings.astype(int), 0, 2)

    table, _ = aggregate_raters(ratings, n_cat=3)
    kappa = fleiss_kappa(table)

    print(f"Images used:        {len(ratings)}")
    print(f"Fleiss Kappa (hair): {round(kappa, 3)}")

    if kappa < 0.20:
        level = "slight"
    elif kappa < 0.40:
        level = "fair"
    elif kappa < 0.60:
        level = "moderate"
    elif kappa < 0.80:
        level = "substantial"
    else:
        level = "almost perfect"
    print(f"Interpretation:     {level} agreement")

if __name__ == "__main__":
    main()
