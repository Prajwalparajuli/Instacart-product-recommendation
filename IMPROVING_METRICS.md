# How to Improve LightGBM Ranking Metrics

Current Performance: **NDCG@20 = 0.640** (Validation)

##  Strategies Ranked by Impact

---

## 1. **Feature Engineering** (Expected Gain: +3-5% NDCG)

### Add Time-Based Features:
```python
# Day-of-week interaction
train['dow_product_interaction'] = train['order_dow'].astype(str) + '_' + train['product_id'].astype(str)

# Time since last purchase (if not already included)
train['days_since_last_purchase'] = train['days_since_prior_order']

# Purchase hour patterns
train['is_morning'] = (train['order_hour_of_day'] < 12).astype(int)
train['is_evening'] = (train['order_hour_of_day'] >= 18).astype(int)
```

### Add Product Category Features:
```python
# Merge with products, aisles, departments
# Add categorical encodings for department, aisle popularity
# Product reorder rate within department/aisle
```

### Add User-Product Interaction Features:
```python
# Purchase frequency rank (is this user's #1, #2, #3 most bought product?)
train['up_purchase_rank'] = train.groupby('user_id')['up_times_bought'].rank(ascending=False)

# Ratio of this product vs user's total purchases
train['up_purchase_ratio'] = train['up_times_bought'] / train['user_total_bought']

# Time momentum (recently trending up or down?)
train['up_recent_vs_old'] = train['up_last_order_number'] - train['up_first_order_number']
```

---

## 2. **Hyperparameter Tuning** (Expected Gain: +1-2% NDCG)

### Current Parameters Review:
```python
params = {
    "learning_rate": 0.03,      # Try: 0.01, 0.02, 0.05
    "num_leaves": 63,           # Try: 31, 127, 255
    "max_depth": 12,            # Try: 8, 10, 15
    "min_data_in_leaf": 200,    # Try: 100, 300, 500
    "feature_fraction": 0.8,    # Try: 0.7, 0.9
    "bagging_fraction": 0.8,    # Try: 0.7, 0.9
    "lambda_l1": 0.5,           # Try: 0.1, 1.0, 2.0
    "lambda_l2": 3.0,           # Try: 1.0, 5.0, 10.0
}
```

### Recommended Tuning Approach:
```python
# Try these combinations:

# More regularization (reduce overfitting):
"lambda_l1": 1.0, "lambda_l2": 5.0, "min_data_in_leaf": 300

# Larger trees (more capacity):
"num_leaves": 127, "max_depth": 15

# Slower learning (better convergence):
"learning_rate": 0.01, "num_boost_round": 5000
```

---

## 3. **Data Preprocessing** (Expected Gain: +0.5-1% NDCG)

### Handle Missing Values Better:
```python
# Instead of filling with 0, use strategic fills
train['days_since_prior_order'].fillna(train['days_since_prior_order'].median(), inplace=True)

# Create "is_missing" indicator features
train['days_missing'] = train['days_since_prior_order'].isna().astype(int)
```

### Feature Scaling/Normalization:
```python
from sklearn.preprocessing import StandardScaler

# Normalize continuous features (especially ALS scores and similarity)
scaler_cols = ['als_score', 'sim_max', 'sim_mean', 'u_mean_days_between']
scaler = StandardScaler()
train[scaler_cols] = scaler.fit_transform(train[scaler_cols])
```

### Remove Redundant Features:
```python
# Check feature importance and remove low-importance features
# This reduces noise and can improve generalization
```

---

## 4. **Ensemble Methods** (Expected Gain: +2-3% NDCG)

### Combine Multiple Models:
```python
# Train 3 models with different seeds
models = []
for seed in [42, 123, 999]:
    params['random_state'] = seed
    model = lgb.train(params, dtrain, valid_sets=[dvalid])
    models.append(model)

# Average predictions
ensemble_preds = np.mean([m.predict(X_test) for m in models], axis=0)
```

### Combine Different Model Types:
```python
# 1. LightGBM LambdaRank (current)
# 2. XGBoost Ranker
# 3. ALS scores (already included as features)
# 4. Item-item similarity (already included)

# Weighted average:
final_score = 0.5 * lgb_score + 0.3 * xgb_score + 0.2 * als_score
```

---

## 5. **Advanced Ranking Techniques** (Expected Gain: +1-2% NDCG)

### Try Different NDCG Cutoffs:
```python
# Optimize for your actual use case
params['ndcg_eval_at'] = [3, 5, 10]  # If users only see top 3-5 items
```

### Add Pairwise Ranking Features:
```python
# Create features that compare products within the same order
train['score_vs_median'] = train.groupby('order_id')['als_score'].transform(
    lambda x: x - x.median()
)
```

### Optimize for Precision@K:
```python
# Sometimes precision is better than NDCG for recommendations
params['metric'] = ['ndcg', 'map']  # Mean Average Precision
```

---

## 6. **Data Augmentation** (Expected Gain: +1-2% NDCG)

### Add Negative Sampling:
```python
# Currently you might have imbalanced data
# Add more negative examples (products NOT purchased)
# This helps the model learn what NOT to recommend
```

### Cross-Validation:
```python
# Instead of single train/valid split, use K-fold CV
# This gives more robust performance estimates
```

---

##  Quick Wins to Try First:

### 1. Lower Learning Rate + More Iterations:
```python
params['learning_rate'] = 0.01
params['num_boost_round'] = 5000
# Expected: +0.5-1% NDCG
```

### 2. Increase Regularization:
```python
params['lambda_l1'] = 1.0
params['lambda_l2'] = 5.0
params['min_data_in_leaf'] = 300
# Expected: +0.5-1% NDCG (reduces overfitting)
```

### 3. Add Purchase Rank Feature:
```python
train['up_purchase_rank'] = train.groupby('user_id')['up_times_bought'].rank(ascending=False, method='dense')
# Expected: +1-2% NDCG
```

---

##  How to Measure Improvement:

### 1. Track Multiple Metrics:
```python
params['metric'] = ['ndcg', 'map']
params['ndcg_eval_at'] = [5, 10, 20]
```

### 2. Look at the Gap:
- **Train NDCG@20: 0.657**
- **Valid NDCG@20: 0.640**
- Gap = 0.017 (1.7%) ← Small gap = good generalization
- If gap increases, add more regularization

### 3. Business Metrics:
```python
# Top-1 Accuracy: What % of users have their purchased item at rank 1?
# Top-5 Hit Rate: What % have it in top 5?
# Mean Reciprocal Rank (MRR)
```

---

##  Realistic Target:

Current: **0.640 NDCG@20**
Achievable with improvements: **0.67-0.70 NDCG@20** (+3-6%)

Beyond 0.70 requires:
- Deep learning models (transformers, neural collaborative filtering)
- More sophisticated feature engineering
- Larger datasets
- Domain-specific knowledge

---

## 💡 Next Steps:

1. **Start with Quick Wins** (lower LR, add purchase rank feature)
2. **Check Feature Importance** (remove low-value features)
3. **Hyperparameter Grid Search** (use Optuna or similar)
4. **Add More Features** (time-based, category-based)
5. **Ensemble** (combine 3 models with different seeds)

Would you like me to implement any of these strategies in your code?
