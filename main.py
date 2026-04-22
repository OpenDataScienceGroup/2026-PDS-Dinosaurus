"""
Main classification pipeline for PAD-UFES-20 skin lesion dataset.

Steps:
  1. Load features.csv
  2. Train/test split (group-aware: same patient never in both splits)
  3. Cross-validation on dev set to compare classifiers
  4. Retrain best models on full dev set
  5. Evaluate on held-out test set
  6. Save models and predictions

Run:
    python main.py

Set load_model=True (or pass --load) to skip training and load saved models.
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
FEATURES_CSV = PROJECT_ROOT / "data" / "features.csv"
MODELS_DIR = PROJECT_ROOT / "results" / "models"
PREDICTIONS_DIR = PROJECT_ROOT / "results" / "predictions"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------
BASELINE_FEATURES = [
    "asymmetry",
    "area", "perimeter", "compactness", "solidity", "eccentricity",
    "mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b",
    "mean_h", "mean_s", "mean_v", "std_h", "std_s", "std_v",
    "color_entropy",
] + [f"lbp_{i}" for i in range(16)]

EXTENDED_FEATURES = BASELINE_FEATURES + ["hair_coverage", "penmark_coverage"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_features(features_path):
    df = pd.read_csv(features_path)
    df = df.dropna(subset=["label"])
    return df


def prepare_splits(df, feature_cols, label_col="label", group_col="patient_id",
                   test_size=0.2, random_state=42):
    """
    Split into dev / test using GroupShuffleSplit so the same patient is never
    in both splits (important for medical imaging).
    """
    from sklearn.model_selection import GroupShuffleSplit

    groups = df[group_col].values
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    dev_idx, test_idx = next(gss.split(df, df[label_col], groups=groups))

    dev_df = df.iloc[dev_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    X_dev = dev_df[feature_cols].values
    y_dev = dev_df[label_col].values
    g_dev = dev_df[group_col].values

    X_test = test_df[feature_cols].values
    y_test = test_df[label_col].values

    return X_dev, y_dev, g_dev, X_test, y_test, dev_df, test_df


def cross_validate(X, y, groups, classifiers, n_splits=5):
    """
    GroupKFold cross-validation. Returns a results DataFrame with
    mean/std AUC and accuracy per classifier.
    """
    gkf = GroupKFold(n_splits=n_splits)
    results = []

    for name, clf_factory in classifiers.items():
        aucs, accs = [], []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            # scale per fold (fit on train only)
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_val_s = scaler.transform(X_val)

            # handle NaN (mean imputation on training fold)
            col_means = np.nanmean(X_tr_s, axis=0)
            for col_i in range(X_tr_s.shape[1]):
                mask = np.isnan(X_tr_s[:, col_i])
                X_tr_s[mask, col_i] = col_means[col_i]
                mask_v = np.isnan(X_val_s[:, col_i])
                X_val_s[mask_v, col_i] = col_means[col_i]

            clf = clf_factory()
            clf.fit(X_tr_s, y_tr)

            if hasattr(clf, "predict_proba"):
                val_prob = clf.predict_proba(X_val_s)[:, 1]
            else:
                val_prob = clf.decision_function(X_val_s)

            val_pred = clf.predict(X_val_s)

            try:
                auc = roc_auc_score(y_val, val_prob)
            except ValueError:
                auc = np.nan
            acc = accuracy_score(y_val, val_pred)
            aucs.append(auc)
            accs.append(acc)

        results.append({
            "classifier": name,
            "mean_auc": np.nanmean(aucs),
            "std_auc": np.nanstd(aucs),
            "mean_acc": np.nanmean(accs),
            "std_acc": np.nanstd(accs),
        })

    return pd.DataFrame(results).sort_values("mean_auc", ascending=False)


def train_final(X_dev, y_dev, clf_factory, model_name):
    """Fit scaler + classifier on full dev set, return (scaler, clf, col_means)."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_dev)

    col_means = np.nanmean(X_s, axis=0)
    for col_i in range(X_s.shape[1]):
        mask = np.isnan(X_s[:, col_i])
        X_s[mask, col_i] = col_means[col_i]

    clf = clf_factory()
    clf.fit(X_s, y_dev)
    return scaler, clf, col_means


def evaluate_on_test(scaler, clf, col_means, X_test, y_test, feature_cols,
                     test_df, model_name, label="baseline"):
    """Scale, impute, predict, print metrics, save predictions."""
    X_s = scaler.transform(X_test)
    for col_i in range(X_s.shape[1]):
        mask = np.isnan(X_s[:, col_i])
        X_s[mask, col_i] = col_means[col_i]

    y_pred = clf.predict(X_s)
    y_prob = clf.predict_proba(X_s)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_s)

    print(f"\n{'='*60}")
    print(f"  {label} — {model_name}  (test set)")
    print(f"{'='*60}")
    print(f"  AUC  : {roc_auc_score(y_test, y_prob):.4f}")
    print(f"  Acc  : {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=["Benign", "Malignant"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # save predictions
    pred_df = test_df[["img_id", "patient_id", "diagnostic", "label"]].copy()
    pred_df["predicted_label"] = y_pred
    pred_df["malignant_prob"] = y_prob
    out_path = PREDICTIONS_DIR / f"predictions_{label}_{model_name}.csv"
    pred_df.to_csv(out_path, index=False)
    print(f"  Predictions saved → {out_path}")


def save_model(scaler, clf, col_means, feature_cols, model_name, label):
    bundle = {
        "scaler": scaler,
        "clf": clf,
        "col_means": col_means,
        "feature_cols": feature_cols,
    }
    path = MODELS_DIR / f"model_{label}_{model_name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"  Model saved → {path}")
    return path


def load_model_bundle(model_path):
    with open(model_path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Classifier factories  (keep as callables so cross_validate can re-instantiate)
# ---------------------------------------------------------------------------
CLASSIFIERS = {
    "RandomForest":     lambda: RandomForestClassifier(n_estimators=200, max_depth=10,
                                                       class_weight="balanced", random_state=42),
    "LogisticRegression": lambda: LogisticRegression(max_iter=1000, class_weight="balanced",
                                                     random_state=42, C=1.0),
    "KNN_k7":           lambda: KNeighborsClassifier(n_neighbors=7),
}

# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main(features_path=FEATURES_CSV,
         prediction_results_path=None,
         model_path=None,
         load_model=False):
    """
    Parameters
    ----------
    features_path            : path to features.csv
    prediction_results_path  : (optional) override for prediction output directory
    model_path               : path to a saved .pkl bundle (used when load_model=True)
    load_model               : if True, skip training and load from model_path
    """
    print(f"Loading features from {features_path}")
    df = load_features(features_path)
    print(f"  {len(df)} samples, {df['label'].sum()} malignant, "
          f"{(df['label']==0).sum()} benign")

    # -----------------------------------------------------------------------
    # LOAD MODEL MODE
    # -----------------------------------------------------------------------
    if load_model:
        if model_path is None:
            # default: load the best extended RandomForest
            model_path = MODELS_DIR / "model_extended_RandomForest.pkl"
        print(f"\nLoading model from {model_path}")
        bundle = load_model_bundle(model_path)
        scaler = bundle["scaler"]
        clf = bundle["clf"]
        col_means = bundle["col_means"]
        feature_cols = bundle["feature_cols"]

        X_all = df[feature_cols].values
        y_all = df["label"].values
        evaluate_on_test(scaler, clf, col_means, X_all, y_all, feature_cols,
                         df, type(clf).__name__, label="loaded")
        return

    # -----------------------------------------------------------------------
    # BASELINE pipeline
    # -----------------------------------------------------------------------
    baseline_cols = [c for c in BASELINE_FEATURES if c in df.columns]
    print(f"\n--- Baseline features ({len(baseline_cols)}) ---")

    (X_dev_b, y_dev, g_dev, X_test_b, y_test,
     dev_df, test_df) = prepare_splits(df, baseline_cols)

    print("Cross-validating baseline classifiers …")
    cv_results_b = cross_validate(X_dev_b, y_dev, g_dev, CLASSIFIERS)
    print("\nBaseline CV results:")
    print(cv_results_b.to_string(index=False))

    # -----------------------------------------------------------------------
    # EXTENDED pipeline (adds shortcut features)
    # -----------------------------------------------------------------------
    extended_cols = [c for c in EXTENDED_FEATURES if c in df.columns]
    print(f"\n--- Extended features ({len(extended_cols)}) ---")

    X_dev_e = dev_df[extended_cols].values
    X_test_e = test_df[extended_cols].values

    print("Cross-validating extended classifiers …")
    cv_results_e = cross_validate(X_dev_e, y_dev, g_dev, CLASSIFIERS)
    print("\nExtended CV results:")
    print(cv_results_e.to_string(index=False))

    # -----------------------------------------------------------------------
    # Retrain best classifiers on full dev set and evaluate on test
    # -----------------------------------------------------------------------
    best_clf_name = cv_results_e.iloc[0]["classifier"]
    best_factory = CLASSIFIERS[best_clf_name]
    print(f"\nBest classifier: {best_clf_name}")

    # --- Baseline final model ---
    print("\nTraining final BASELINE model …")
    scaler_b, clf_b, means_b = train_final(X_dev_b, y_dev, CLASSIFIERS["RandomForest"], "RandomForest")
    evaluate_on_test(scaler_b, clf_b, means_b, X_test_b, y_test, baseline_cols,
                     test_df, "RandomForest", label="baseline")
    save_model(scaler_b, clf_b, means_b, baseline_cols, "RandomForest", "baseline")

    # --- Extended final model ---
    print("\nTraining final EXTENDED model …")
    scaler_e, clf_e, means_e = train_final(X_dev_e, y_dev, best_factory, best_clf_name)
    evaluate_on_test(scaler_e, clf_e, means_e, X_test_e, y_test, extended_cols,
                     test_df, best_clf_name, label="extended")
    save_model(scaler_e, clf_e, means_e, extended_cols, best_clf_name, "extended")

    # Also train LogReg extended for comparison
    if best_clf_name != "LogisticRegression":
        scaler_lr, clf_lr, means_lr = train_final(X_dev_e, y_dev,
                                                   CLASSIFIERS["LogisticRegression"],
                                                   "LogisticRegression")
        evaluate_on_test(scaler_lr, clf_lr, means_lr, X_test_e, y_test, extended_cols,
                         test_df, "LogisticRegression", label="extended")
        save_model(scaler_lr, clf_lr, means_lr, extended_cols, "LogisticRegression", "extended")

    # Save CV summaries
    cv_path_b = PREDICTIONS_DIR / "cv_results_baseline.csv"
    cv_path_e = PREDICTIONS_DIR / "cv_results_extended.csv"
    cv_results_b.to_csv(cv_path_b, index=False)
    cv_results_e.to_csv(cv_path_e, index=False)
    print(f"\nCV summaries saved to {PREDICTIONS_DIR}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PAD-UFES-20 classifier pipeline")
    parser.add_argument("--features", default=str(FEATURES_CSV),
                        help="Path to features.csv")
    parser.add_argument("--model", default=None,
                        help="Path to saved model .pkl (used with --load)")
    parser.add_argument("--load", action="store_true",
                        help="Load saved model instead of training")
    args = parser.parse_args()

    main(
        features_path=args.features,
        model_path=args.model,
        load_model=args.load,
    )
