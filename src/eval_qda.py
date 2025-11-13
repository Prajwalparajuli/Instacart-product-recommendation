"""
- Loads saved artifacts from QDA model.
- Replays preprocessing steps (log1p, imputation, scaling)
- Scores a given CSV
- Plots/metrics for analysis
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap

from sklearn.metrics import (roc_auc_score, RocCurveDisplay,
                             precision_recall_curve, average_precision_score,
                             classification_report, confusion_matrix, roc_curve)

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Dict

import numpy as np
import pandas as pd
import lightgbm as lgb

# Setups
DATA_PATH = "data/processed/train_candidates.csv"
MODEL_PATH = "models/qda_model.joblib"
TARGET_NAME = None # Use target saved in artifacts

THRESHOLD = 0.5  # Classification threshold
SHOW_PR = True  # Whether to show precision-recall curve
SHOW_CAL = True  # Whether to show calibration plot

SHAP_BG = 200 # Background samples for SHAP
SHAP_EX = 100 # Explanation samples for SHAP

# Load Artifacts
art = joblib.load(MODEL_PATH)
qda = art["model"]
imputer = art["imputer"]
scaler = art["scaler"]
numeric_cols = art["numeric_cols"]
skewed_cols = art["skewed_cols"]
no_log_cols = art["no_log_cols"]
reg_param = art.get("reg_param", None)
cv_auc = art.get("cv_auc", None)

target_name = art.get("target", TARGET_NAME)
if target_name is None:
    raise ValueError("Target name not specified in artifacts or script.")
print(f"Loaded model artifacts from {MODEL_PATH}")

# Load Data
data = pd.read_csv(DATA_PATH)
has_label = target_name in data.columns

X_raw = data[numeric_cols].copy()

# Apply log1p transformation
for c in skewed_cols:
    if c in X_raw.columns:
        X_raw[c] = np.log1p(np.clip(X_raw[c].astype(float), 0.0, None))

# Impute + Scale
X_imp = imputer.transform(X_raw)
X = scaler.transform(X_imp)

# Predict Probabilities
probs = qda.predict_proba(X)[:, 1]
print(f"Scored {len(probs)} samples. | reg_param={reg_param} | cv_auc={cv_auc}")

# If labels exist, evaluate
if has_label:
    y = data[target_name].astype(int).values
else:
    print("No labels found in data; skipping evaluation.")

# ROC Curve
if has_label:
    auc = roc_auc_score(y, probs)
    RocCurveDisplay.from_predictions(y, probs)
    plt.title(f"ROC Curve (AUC = {auc:.4f})")
    plt.show()
    print(f"ROC AUC: {auc:.4f}")

# Precision-Recall Curve
if has_label and SHOW_PR:
    precision, recall, _ = precision_recall_curve(y, probs)
    ap = average_precision_score(y, probs)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - AP: {ap:.4f}")
    plt.grid()
    plt.legend()
    plt.show()
    print(f"Average Precision: {ap:.4f}")

# Classification Report
if has_label:
    thr = float(THRESHOLD)
    yhat = (probs >= thr).astype(int)
    print(f"Classification Report (Threshold = {thr:.2f}):")
    print(classification_report(y, yhat, digits=4))
    cm = confusion_matrix(y, yhat)
    print("Confusion Matrix:")
    print(cm)

# KS Statistic
if has_label:
    fpr, tpr, _ = roc_curve(y, probs)
    ks_stat = np.max(tpr - fpr)
    print(f"KS Statistic: {ks_stat:.4f}")

# Calibration Plot
if has_label and SHOW_CAL:
    prob_true, prob_pred = calibration_curve(y, probs, n_bins=10, strategy='quantile')
    plt.figure()
    plt.plot(prob_pred, prob_true, marker='o')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("Predicted Probability")
    plt.ylabel("True Probability")
    plt.title("Calibration Plot")
    plt.grid()
    plt.show()

# SHAP Analysis
rng = np.random.default_rng(42)
n = X.shape[0]
bg_indices = rng.choice(n, size=min(SHAP_BG, n), replace=False)
expl_indices = rng.choice(n, size=min(SHAP_EX, n), replace=False)
X_bg = X[bg_indices]
X_expl = X[expl_indices]

explainer = shap.KernelExplainer(lambda x: qda.predict_proba(x)[:, 1], X_bg, link = "logit")
print("Computing SHAP values (this may take a while)...")
shap_values = explainer.shap_values(X_expl, nsamples = 100)

# SHAP Analysis
rng = np.random.default_rng(42)
n = X.shape[0]
bg_indices = rng.choice(n, size=min(SHAP_BG, n), replace=False)
expl_indices = rng.choice(n, size=min(SHAP_EX, n), replace=False)
X_bg = X[bg_indices]
X_expl = X[expl_indices]

explainer = shap.KernelExplainer(lambda x: qda.predict_proba(x)[:, 1], X_bg, link = "logit")
print("Computing SHAP values (this may take a while)...")
shap_values = explainer.shap_values(X_expl, nsamples = 100)

feature_names = numeric_cols

# Summary plot with adjusted figure size to show full labels (bar plot)
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, features = X_expl, feature_names = feature_names, plot_type="bar", show = False)
plt.tight_layout()
plt.show()

# Beeswarm plot for detailed view
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, features = X_expl, feature_names = feature_names, show = False)
plt.tight_layout()
plt.show()

# Waterfall plots: one for label=1 and one for label=0
if has_label:
    # Get indices of samples in explanation set
    y_expl = y[expl_indices]
    probs_expl = probs[expl_indices]
    
    # Find a sample with actual label = 1
    label_1_indices = np.where(y_expl == 1)[0]
    if len(label_1_indices) > 0:
        # Get the one with highest probability among label=1 samples
        i_label_1 = label_1_indices[np.argmax(probs_expl[label_1_indices])]
    else:
        i_label_1 = None
    
    # Find a sample with actual label = 0
    label_0_indices = np.where(y_expl == 0)[0]
    if len(label_0_indices) > 0:
        # Get the one with lowest probability among label=0 samples (most confident negative)
        i_label_0 = label_0_indices[np.argmin(probs_expl[label_0_indices])]
    else:
        i_label_0 = None
    
    # Waterfall plot 1: Sample with actual label = 1
    if i_label_1 is not None:
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(shap.Explanation(values=shap_values[i_label_1],
                                             base_values=explainer.expected_value,
                                             data=X_expl[i_label_1],
                                             feature_names=numeric_cols),
                            show=False)
        plt.title(f"SHAP Waterfall: Label=1 Sample (pred={probs_expl[i_label_1]:.4f}, true={y_expl[i_label_1]})")
        plt.tight_layout()
        plt.show()
    else:
        print("No samples with label=1 found in explanation set.")
    
    # Waterfall plot 2: Sample with actual label = 0
    if i_label_0 is not None:
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(shap.Explanation(values=shap_values[i_label_0],
                                             base_values=explainer.expected_value,
                                             data=X_expl[i_label_0],
                                             feature_names=numeric_cols),
                            show=False)
        plt.title(f"SHAP Waterfall: Label=0 Sample (pred={probs_expl[i_label_0]:.4f}, true={y_expl[i_label_0]})")
        plt.tight_layout()
        plt.show()
    else:
        print("No samples with label=0 found in explanation set.")
else:
    # No labels available, just show first sample
    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(shap.Explanation(values=shap_values[0],
                                         base_values=explainer.expected_value,
                                         data=X_expl[0],
                                         feature_names=numeric_cols),
                        show=False)
    plt.title(f"SHAP Waterfall: First Sample (pred={probs[expl_indices[0]]:.4f})")
    plt.tight_layout()
    plt.show()

print("SHAP analysis completed.")

# --- LightGBM Ranker Evaluation ---

def ndcg_at_k(true_relevance: np.ndarray, pred_scores: np.ndarray, k: int, true_sorted: bool = False) -> float:
    """Normalized Discounted Cumulative Gain."""
    best = dcg_at_k(true_relevance, true_relevance, k)
    actual = dcg_at_k(true_relevance, pred_scores, k)
    return actual / best if best > 0 else 0.0


def dcg_at_k(true_relevance: np.ndarray, pred_scores: np.ndarray, k: int) -> float:
    """Discounted Cumulative Gain."""
    order = np.argsort(pred_scores)[::-1]
    true_relevance = np.take(true_relevance, order[:k])
    gains = 2**true_relevance - 1
    discounts = np.log2(np.arange(len(true_relevance)) + 2)
    return np.sum(gains / discounts)


def recall_at_k(true_relevance: np.ndarray, pred_scores: np.ndarray, k: int) -> float:
    """Recall."""
    order = np.argsort(pred_scores)[::-1]
    true_relevance = np.take(true_relevance, order[:k])
    total_relevant = np.sum(true_relevance)
    if total_relevant == 0:
        return 0.0
    return np.sum(true_relevance) / total_relevant


def map_at_k(true_relevance: np.ndarray, pred_scores: np.ndarray, k: int) -> float:
    """Mean Average Precision."""
    order = np.argsort(pred_scores)[::-1][:k]
    true_relevance = np.take(true_relevance, order)
    
    if np.sum(true_relevance) == 0:
        return 0.0
    
    p_at_i = np.array([
        np.mean(true_relevance[:i+1]) for i in range(len(true_relevance)) if true_relevance[i]
    ])
    return np.mean(p_at_i) if len(p_at_i) > 0 else 0.0


def hitrate_at_k(true_relevance: np.ndarray, pred_scores: np.ndarray, k: int) -> float:
    """Hit Rate."""
    order = np.argsort(pred_scores)[::-1][:k]
    true_relevance = np.take(true_relevance, order)
    return float(np.sum(true_relevance) > 0)


def eval_one_group(
    df: pd.DataFrame, label_col: str, ks: Iterable[int]
) -> pd.Series:
    """Evaluate ranking metrics for a single group."""
    true_relevance = df[label_col].to_numpy()
    pred_scores = df["score"].to_numpy()
    
    metrics = {}
    for k in ks:
        metrics[f"ndcg@{k}"] = ndcg_at_k(true_relevance, pred_scores, k)
        metrics[f"map@{k}"] = map_at_k(true_relevance, pred_scores, k)
        metrics[f"recall@{k}"] = recall_at_k(true_relevance, pred_scores, k)
        metrics[f"hitrate@{k}"] = hitrate_at_k(true_relevance, pred_scores, k)
    return pd.Series(metrics)


def load_candidates(paths: str | Path | List[str | Path]) -> pd.DataFrame:
    """Load one or more candidate files."""
    if not isinstance(paths, list):
        paths = [paths]
    
    dfs = []
    for p in paths:
        p = Path(p)
        if p.suffix == ".parquet":
            dfs.append(pd.read_parquet(p))
        elif p.suffix == ".csv":
            dfs.append(pd.read_csv(p))
        else:
            raise ValueError(f"Unsupported file type for {p} (use .parquet or .csv).")
    return pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    

# --- Configuration for LightGBM Evaluation (Hardcoded) ---
LGBM_MODEL_PATH = "models/lightgbm_ranker/lgbm_lambdarank.txt"
LGBM_FEATURES_PATH = "models/lightgbm_ranker/feature_cols.json"
LGBM_DATA_PATH = "data/processed/valid_candidates.csv" # Using validation set for evaluation
LGBM_OUT_DIR = "reports/lgbm_eval"
LGBM_GROUP_KEY = "order_id"
LGBM_LABEL_COL = "label"
LGBM_ID_COLS = ["user_id", "product_id", "order_id"]
LGBM_KS = [5, 10, 20]

print("\n--- Starting LightGBM Ranker Evaluation ---")
out_dir = Path(LGBM_OUT_DIR)
out_dir.mkdir(parents=True, exist_ok=True)

# Load artifacts
booster = lgb.Booster(model_file=LGBM_MODEL_PATH)
with open(LGBM_FEATURES_PATH, "r") as f:
    feature_cols: List[str] = json.load(f)
print(f"Loaded LightGBM model from: {LGBM_MODEL_PATH}")

# Load data
df = load_candidates(LGBM_DATA_PATH)
print(f"Loaded data for evaluation from: {LGBM_DATA_PATH} ({len(df):,} rows)")

# Sanity checks
required_cols = set(feature_cols + [LGBM_GROUP_KEY, LGBM_LABEL_COL] + LGBM_ID_COLS)
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing columns in data: {sorted(list(missing))[:20]}")

# Score
df = df.sort_values(LGBM_GROUP_KEY).reset_index(drop=True)
num_iter = booster.current_iteration() if booster.current_iteration() > 0 else None
df["score"] = booster.predict(df[feature_cols], num_iteration=num_iter)
print("Scored candidates with LightGBM model.")

# Evaluate per group, then average
ks = tuple(LGBM_KS)
per_group = (
    df.groupby(LGBM_GROUP_KEY, sort=False)
      .apply(eval_one_group, label_col=LGBM_LABEL_COL, ks=ks)
      .reset_index()
      .rename(columns={LGBM_GROUP_KEY: "group"})
)

# Calculate mean metrics
metrics = {}
for k in ks:
    metrics[f"ndcg@{k}"] = float(per_group[f"ndcg@{k}"].mean())
    metrics[f"map@{k}"] = float(per_group[f"map@{k}"].mean())
    metrics[f"recall@{k}"] = float(per_group[f"recall@{k}"].mean())
    metrics[f"hitrate@{k}"] = float(per_group[f"hitrate@{k}"].mean())

# Save outputs
per_group_path = out_dir / "per_group_metrics.csv"
per_group.to_csv(per_group_path, index=False)

summary = {
    "data": str(Path(LGBM_DATA_PATH).resolve()),
    "model": str(Path(LGBM_MODEL_PATH).resolve()),
    "features": str(Path(LGBM_FEATURES_PATH).resolve()),
    "group_key": LGBM_GROUP_KEY,
    "label_col": LGBM_LABEL_COL,
    "ks": list(ks),
    "overall": metrics,
    "groups": int(per_group.shape[0]),
    "rows_scored": int(df.shape[0]),
}
with open(out_dir / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# Pretty print
print("\n=== Overall metrics (mean across groups) ===")
for k in ks:
    print(
        f"NDCG@{k}: {metrics[f'ndcg@{k}']:.4f} | "
        f"MAP@{k}: {metrics[f'map@{k}']:.4f} | "
        f"Recall@{k}: {metrics[f'recall@{k}']:.4f} | "
        f"HitRate@{k}: {metrics[f'hitrate@{k}']:.4f}"
    )

print(f"\nSaved per-group metrics → {per_group_path}")
print(f"Saved summary           → {out_dir / 'summary.json'}")
