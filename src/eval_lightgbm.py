"""
LightGBM Model Evaluation and Visualization Script

This script generates comprehensive visualizations for the LightGBM LambdaRank model:
1. Feature importance plot (gain-based)
2. SHAP summary bar plot (global feature importance)
3. SHAP beeswarm plot (feature value effects)
4. SHAP waterfall plot (individual prediction explanation)

Required data:
- Trained LightGBM model: models/lightgbm_ranker/lgbm_lambdarank.txt
- Feature columns: models/lightgbm_ranker/feature_cols.json
- Validation data: data/processed/valid_candidates.csv
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import json
from pathlib import Path

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Define paths
MODEL_DIR = Path("models/lightgbm_ranker")
DATA_DIR = Path("data/processed")
PLOTS_DIR = Path("reports/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("LightGBM Model Evaluation - Generating Visualizations")
print("="*80)

# ============================================================================
# 1. Load Model and Data
# ============================================================================

print("\n[1/5] Loading model and data...")

# Load the trained model
model_path = MODEL_DIR / "lgbm_lambdarank.txt"
if not model_path.exists():
    raise FileNotFoundError(f"Model file not found: {model_path}")

model = lgb.Booster(model_file=str(model_path))
print(f"✓ Loaded model from {model_path}")

# Load feature columns
feature_cols_path = MODEL_DIR / "feature_cols.json"
with open(feature_cols_path, 'r') as f:
    feature_cols = json.load(f)
print(f"✓ Loaded {len(feature_cols)} feature columns")

# Load validation data for SHAP analysis
valid_path = DATA_DIR / "valid_candidates.csv"
if not valid_path.exists():
    raise FileNotFoundError(f"Validation data not found: {valid_path}")

valid_df = pd.read_csv(valid_path)
print(f"✓ Loaded validation data: {valid_df.shape[0]:,} rows")

# Prepare feature matrix
X_valid = valid_df[feature_cols]

# ============================================================================
# 2. Feature Importance Plot (Gain-based)
# ============================================================================

print("\n[2/5] Generating feature importance plot (gain)...")

# Get feature importance
importance_gain = model.feature_importance(importance_type='gain')
feature_names = model.feature_name()

# Create DataFrame for plotting
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importance_gain
}).sort_values('importance', ascending=True)

# Take top 10 features for better visualization
top_n = min(10, len(importance_df))
importance_df_top = importance_df.tail(top_n)

# Plot
fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.5)))
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(importance_df_top)))
bars = ax.barh(importance_df_top['feature'], importance_df_top['importance'], color=colors)

ax.set_xlabel('Importance (Gain)', fontsize=12, fontweight='bold')
ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
ax.set_title('LightGBM Feature Importance (Top 10 by Gain)', 
             fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, importance_df_top['importance'])):
    ax.text(val, bar.get_y() + bar.get_height()/2, f'{val:,.0f}', 
            va='center', ha='left', fontsize=8, color='black', 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))

plt.tight_layout()
output_path = PLOTS_DIR / "lgbm_feature_importance_gain.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {output_path}")

# ============================================================================
# 3. SHAP Analysis Setup
# ============================================================================

print("\n[3/5] Computing SHAP values (this may take a few minutes)...")

# Sample data for SHAP analysis (use a subset to speed up computation)
# For ranking models, we should sample across different orders/groups
sample_size = min(5000, len(X_valid))
np.random.seed(42)
sample_indices = np.random.choice(len(X_valid), size=sample_size, replace=False)
X_sample = X_valid.iloc[sample_indices]

print(f"✓ Using {sample_size:,} samples for SHAP analysis")

# Create SHAP explainer
# For LightGBM, TreeExplainer is most efficient
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_sample)

print(f"✓ Computed SHAP values: {shap_values.values.shape}")

# ============================================================================
# 4. SHAP Summary Bar Plot (Global Feature Importance)
# ============================================================================

print("\n[4/5] Generating SHAP summary bar plot...")

fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_values, X_sample, plot_type="bar", 
                  max_display=10, show=False)
plt.title('SHAP Feature Importance (Mean |SHAP value|)', 
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Mean |SHAP value| (average impact on model output)', fontsize=11)
plt.tight_layout()

output_path = PLOTS_DIR / "lgbm_shap_summary_bar.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {output_path}")

# ============================================================================
# 5. SHAP Beeswarm Plot (Feature Value Effects)
# ============================================================================

print("\n[5/5] Generating SHAP beeswarm plot...")

fig, ax = plt.subplots(figsize=(12, 8))
shap.summary_plot(shap_values, X_sample, max_display=10, show=False)
plt.title('SHAP Summary Plot (Beeswarm)', 
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel('SHAP value (impact on model output)', fontsize=11)
plt.tight_layout()

output_path = PLOTS_DIR / "lgbm_shap_beeswarm.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {output_path}")

# ============================================================================
# 6. SHAP Waterfall Plot (Individual Prediction)
# ============================================================================

print("\n[Bonus] Generating SHAP waterfall plot for a sample prediction...")

# Select an interesting sample (one with high predicted score)
scores = model.predict(X_sample)
high_score_idx = np.argmax(scores)

fig, ax = plt.subplots(figsize=(10, 6))
shap.waterfall_plot(shap_values[high_score_idx], max_display=10, show=False)
plt.title('SHAP Waterfall Plot (Single Prediction Explanation)', 
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()

output_path = PLOTS_DIR / "lgbm_shap_waterfall.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {output_path}")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*80)
print("✓ All visualizations generated successfully!")
print("="*80)
print("\nGenerated plots:")
print(f"  1. {PLOTS_DIR / 'lgbm_feature_importance_gain.png'}")
print(f"  2. {PLOTS_DIR / 'lgbm_shap_summary_bar.png'}")
print(f"  3. {PLOTS_DIR / 'lgbm_shap_beeswarm.png'}")
print(f"  4. {PLOTS_DIR / 'lgbm_shap_waterfall.png'}")
print("\n" + "="*80)
