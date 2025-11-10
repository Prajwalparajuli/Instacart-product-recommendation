import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.impute import SimpleImputer
import joblib
from pathlib import Path

# ==== Configuration ==== #
TRAIN_PATH = "data/processed/train_candidates.csv"
TEST_PATH = "data/processed/test_candidates.csv"
OUT_TRAIN_SCORES = "data/processed/qda_scores_train.csv"
OUT_TEST_SCORES = "data/processed/qda_scores_test.csv"
OUT_MODEL = "models/qda_model.joblib"
TARGET = "label"
SKEW_THRESHOLD = 1.0

# ==== load data ==== #
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# ==== preprocess ==== #
id_cols = [c for c in ["user_id", "product_id", "order_id"] if c in train_df.columns]
numeric_cols = [c for c in train_df.columns if c not in id_cols + [TARGET] and 
                pd.api.types.is_numeric_dtype(train_df[c])]

# Keep exact same feature order in test
Xtr_raw = train_df[numeric_cols].copy()
ytr = train_df[TARGET].astype(int).values
Xte_raw = test_df[numeric_cols].copy()

# Selective log1p on Train (record columns with high skewness), then apply to Test
no_log_cols = [c for c in [
    "als_score", "sim_mean", "sim_topk_mean_10", "sim_max",
    "sim_nonzero_cnt", "u_reorder_ratio", "p_reorder_prob", "up_rate_in_user_history"]
    if c in numeric_cols]

log_candidates = [c for c in numeric_cols if c not in no_log_cols]

skewed_cols = []
for c in log_candidates:
    s = Xtr_raw[c].astype(float)
    if not np.isfinite(s).all() or s.std(ddof = 0) == 0:
        continue
    if s.skew() > SKEW_THRESHOLD:
        Xtr_raw[c] = np.log1p(np.clip(s, 0.0, None))
        skewed_cols.append(c)

# Apply same log1p transformation to Test set
for c in skewed_cols:
    if c in Xte_raw.columns:
        Xte_raw[c] = np.log1p(np.clip(Xte_raw[c].astype(float), 0.0, None))


# Impute missing values
imputer = SimpleImputer(strategy = "median")
scaler = StandardScaler()

Xtr_imp = imputer.fit_transform(Xtr_raw)
Xtr = scaler.fit_transform(Xtr_imp)

Xte_imp = imputer.transform(Xte_raw)
Xte = scaler.transform(Xte_imp)

# CV tuning for regularization parameter
print("Tuning regularization parameter with Stratified 5-Fold CV...")
coarse = np.round(np.arange(0.10, 0.41, 0.05), 2)
cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)

best_auc, best_reg = -np.inf, None
for reg in coarse:
    fold_aucs = []
    for tr, va in cv.split(Xtr, ytr):
        model = QDA(reg_param = reg, store_covariance = True)
        model.fit(Xtr[tr], ytr[tr])
        va_probs = model.predict_proba(Xtr[va])[:, 1]
        fold_aucs.append(roc_auc_score(ytr[va], va_probs))
    mean_auc = float(np.mean(fold_aucs))
    print(f"reg_param = {reg:.2f} --> CV AUC = {mean_auc:.4f}")
    if mean_auc > best_auc:
        best_auc = mean_auc
        best_reg = reg

low = max(best_reg - 0.05, 0.05)
high = min(best_reg + 0.05, 0.50)
fine = np.round(np.arange(low, high + 1e-9, 0.01), 2)
print(f"Refining between {low:.2f} and {high:.2f}: {fine.tolist()}")

best_fine_auc, best_fine_reg = -np.inf, None
for reg in fine:
    fold_aucs = []
    for tr, va in cv.split(Xtr, ytr):
        model = QDA(reg_param = reg, store_covariance = True)
        model.fit(Xtr[tr], ytr[tr])
        va_probs = model.predict_proba(Xtr[va])[:, 1]
        fold_aucs.append(roc_auc_score(ytr[va], va_probs))
    mean_auc = float(np.mean(fold_aucs))
    print(f"reg_param = {reg:.2f} --> CV AUC = {mean_auc:.4f}")
    if mean_auc > best_fine_auc:
        best_fine_auc = mean_auc
        best_fine_reg = reg

print(f"\n[Best] reg_param = {best_fine_reg:.3f} (CV AUC = {best_fine_auc:.4f})")

# Train final model
qda = QDA(reg_param = best_fine_reg, store_covariance = True)
qda.fit(Xtr, ytr)

# Sanity metrics on Train
ptr = qda.predict_proba(Xtr)[:, 1]
auc = roc_auc_score(ytr, ptr)
ll = log_loss(ytr, ptr)
print(f"Train AUC = {auc:.4f}, LogLoss = {ll:.4f}")

# Add QDA scores as new columns to original dataframes
train_df["qda_p_score"] = ptr
train_df["qda_logit"] = np.log(np.clip(ptr, 1e-8, 1 - ptr))

test_probs = qda.predict_proba(Xte)[:, 1]
test_df["qda_p_score"] = test_probs
test_df["qda_logit"] = np.log(np.clip(test_probs, 1e-8, 1 - test_probs))

# Save updated candidate files
Path(TRAIN_PATH).parent.mkdir(parents = True, exist_ok = True)
train_df.to_csv(TRAIN_PATH, index = False)
test_df.to_csv(TEST_PATH, index = False)

print(f"Added qda_p_score and qda_logit columns to {TRAIN_PATH}")
print(f"Added qda_p_score and qda_logit columns to {TEST_PATH}")

# Also save standalone score files for backward compatibility
train_scores = train_df[id_cols + ["qda_p_score", "qda_logit"]].copy()
test_scores = test_df[[c for c in id_cols if c in test_df.columns] + ["qda_p_score", "qda_logit"]].copy()

Path(OUT_TRAIN_SCORES).parent.mkdir(parents = True, exist_ok = True)
Path(OUT_TEST_SCORES).parent.mkdir(parents = True, exist_ok = True)
train_scores.to_csv(OUT_TRAIN_SCORES, index = False)
test_scores.to_csv(OUT_TEST_SCORES, index = False)

print(f"Also saved standalone scores to {OUT_TRAIN_SCORES} and {OUT_TEST_SCORES}")

# Save minimal artifact
artifacts = {
    "model": qda,
    "imputer": imputer,
    "scaler": scaler,
    "numeric_cols": numeric_cols,
    "skewed_cols": skewed_cols,
    "no_log_cols": no_log_cols,
    "target": TARGET,
    "reg_param": best_fine_reg,
    "cv_auc": best_fine_auc
}

out_model = Path(OUT_MODEL)
out_model.parent.mkdir(parents = True, exist_ok = True)
joblib.dump(artifacts, out_model)
print(f"Saved model artifacts to {OUT_MODEL}")