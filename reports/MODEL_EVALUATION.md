# Model Evaluation Report

This document provides comprehensive evaluation of both models in the recommendation pipeline:
1. **QDA (Quadratic Discriminant Analysis)**: Binary classifier for purchase probability
2. **LightGBM LambdaRank**: Learning-to-rank model for final recommendations

---

# Part 1: QDA Classification Model Evaluation

## 📊 QDA Model Overview

The QDA model serves as the **first stage** in the recommendation pipeline, providing probability scores that predict whether a user will purchase a given product.

### Model Context
- **Model Type:** Quadratic Discriminant Analysis (QDA) with regularization
- **Purpose:** Binary classification - predict purchase probability for candidate products
- **Dataset:** `train_candidates.csv` (training set)
- **Regularization Parameter:** Tuned via 5-fold stratified cross-validation
- **Features:** User, product, and user-product interaction features (with log transformations for skewed features)
- **Output:** Purchase probability score (0-1) for each candidate

---

## 📈 QDA Model Performance Metrics

### Classification Performance

Based on the evaluation plots from `eval.py`:

**ROC Curve Analysis:**
- **AUC-ROC: 0.8033** (from ROC Curve plot)
- **Interpretation:** Good discrimination ability - model can distinguish between purchased and non-purchased items effectively
- **Benchmark:** 0.8-0.9 = Good performance (achieves 0.8033)

**Precision-Recall Analysis:**
- **Average Precision (AP): 0.2923** (from PR Curve plot)
- **Interpretation:** Moderate precision-recall tradeoff
- **Context:** Low AP is expected due to severe class imbalance (most candidates are not purchased)
- **Curve shape:** High precision at low recall (good for top recommendations)

---

## 🎨 Detailed Visualization Analysis

### 1. **ROC Curve (AUC = 0.8033)**

**What the plot shows:**
- **Blue curve:** QDA classifier performance across all probability thresholds
- **X-axis:** False Positive Rate (FPR) - showing non-purchased items
- **Y-axis:** True Positive Rate (TPR) - correctly identifying purchased items
- **Diagonal line:** Random classifier baseline (AUC = 0.5)

**Observations from the plot:**
```
✅ Strong early performance: Steep rise at low FPR
   └─ At FPR = 0.2 (20% false positives), TPR ≈ 0.65 (65% true positives)
   └─ Model correctly identifies 65% of purchases with only 20% false alarms

✅ Good separation: Curve significantly above diagonal
   └─ AUC = 0.8033 indicates strong discrimination
   └─ Model can reliably separate purchased from non-purchased items

⚠️ Diminishing returns at high recall
   └─ To catch remaining 20% of purchases requires accepting many false positives
   └─ Trade-off between coverage and precision
```

**Business Impact:**
- Operating at ~20% FPR gives good balance (65% recall, controlled false positives)
- Suitable for candidate generation where recall matters
- Scores can be used for ranking/filtering downstream

---

### 2. **Precision-Recall Curve (AP = 0.2923)**

**What the plot shows:**
- **Blue curve:** Precision vs. Recall tradeoff
- **X-axis:** Recall (% of purchased items found)
- **Y-axis:** Precision (% of predictions that are correct)
- **AP = 0.2923:** Average precision across all thresholds

**Observations from the plot:**
```
✅ Excellent precision at low recall (< 5%)
   └─ Precision ≈ 1.0 (100%) for top 0-5% predictions
   └─ Model's highest-confidence predictions are extremely accurate

✅ Maintains ~40% precision up to 10-15% recall
   └─ Top 10-15% of candidates have 40% purchase rate
   └─ Much better than random (baseline ~5-10%)

⚠️ Steep precision drop after 15% recall
   └─ Precision falls to ~20-30% at higher recall
   └─ Expected due to class imbalance (few purchases, many non-purchases)

⚠️ Moderate AP (0.2923)
   └─ Reflects severe class imbalance in dataset
   └─ Still useful: 4-8x better than random baseline
```

**Business Impact:**
- **Top 5% predictions:** Near-perfect precision - use for high-confidence recommendations
- **Top 10-15%:** Good precision (40%) - suitable for candidate shortlisting
- **Beyond 20% recall:** Lower precision - requires additional ranking (LightGBM stage)

---

### 3. **Calibration Plot**

**What the plot shows:**
- **Blue line:** Actual relationship between predicted probabilities and true purchase rates
- **Orange dashed line:** Perfect calibration (predicted = actual)
- **X-axis:** Predicted probability from QDA model
- **Y-axis:** True probability (observed purchase rate)

**Observations from the plot:**
```
❌ SEVERE UNDER-CONFIDENCE across all probability ranges
   └─ Model predicts 0-10% probability for most samples
   └─ But actual purchase rates reach 15-35%
   └─ Model is too conservative (underestimates purchase likelihood)

❌ Calibration deteriorates at higher predicted probabilities
   └─ When model predicts 70-80%, actual rate is only ~35%
   └─ Model occasionally over-confident for specific segments

⚠️ Most predictions clustered in 0-20% range
   └─ Limited discrimination in probability space
   └─ Relative ranking still useful despite poor calibration
```

**Why calibration matters:**
- **Under-confidence** means raw probabilities shouldn't be used as actual purchase probabilities
- **However:** Relative ranking of scores is still valid (higher score = higher purchase likelihood)
- **Solution:** Use scores for ranking, not as calibrated probabilities

**Business Impact:**
- ✅ **DO:** Use QDA scores for ranking candidates (relative order is correct)
- ❌ **DON'T:** Interpret scores as actual purchase probabilities without calibration
- 💡 **Fix:** Apply Platt scaling or isotonic regression if calibrated probabilities needed

---

### 4. **SHAP Feature Importance (Bar Plot)**

**What the plot shows:**
- **Blue bars:** Mean absolute SHAP value for each feature
- **X-axis:** Average impact magnitude (higher = more important)
- **Y-axis:** Features ranked by importance

**Top 5 Most Important Features:**

1. **up_rate_in_user_history (~1.0 impact)**
   - **Definition:** Rate at which user purchases this specific product in their history
   - **Why it matters:** Direct measure of product-specific loyalty
   - **Impact:** Most powerful predictor - users repurchase their favorite items

2. **up_times_bought (~0.8 impact)**
   - **Definition:** Total number of times user bought this product
   - **Why it matters:** Frequency of past purchases predicts future purchases
   - **Impact:** Strong indicator of product preference

3. **up_recency (~0.6 impact)**
   - **Definition:** Time since user last purchased this product
   - **Why it matters:** Recent purchases suggest ongoing need/habit
   - **Impact:** Captures temporal patterns (e.g., weekly milk purchases)

4. **p_reorder_prob (~0.25 impact)**
   - **Definition:** Product-level reorder probability (across all users)
   - **Why it matters:** Some products are naturally repurchased more (milk vs. holiday items)
   - **Impact:** Global product characteristic

5. **sim_nonzero_cnt (~0.2 impact)**
   - **Definition:** Count of similar products with non-zero interaction
   - **Why it matters:** Products similar to user's purchases are more likely to be bought
   - **Impact:** Captures similarity-based recommendations

**Key Insights:**
```
✅ User-Product (UP) features dominate
   └─ up_rate_in_user_history, up_times_bought, up_recency are top 3
   └─ Historical interaction is strongest signal

✅ Product (P) features provide secondary signal
   └─ p_reorder_prob adds global product behavior
   └─ Complements user-specific features

✅ Similarity (sim) features contribute
   └─ sim_nonzero_cnt, sim_max help with discovery
   └─ Important for recommending new items
```

**Business Implications:**
- **Focus feature engineering on user-product interactions**
- **Reorder patterns are key:** Users repeat purchases predictably
- **Temporal features matter:** Recency captures purchase cycles
- **Lower features (order_number, u_total_orders, etc.) have minimal impact** - candidates for removal

---

### 5. **SHAP Summary Plot (Beeswarm)**

**What the plot shows:**
- **Each dot:** One sample's SHAP value for a feature
- **Color:** Red = high feature value, Blue = low feature value
- **X-axis:** SHAP value (positive = increases purchase probability, negative = decreases)
- **Y-axis:** Features ranked by importance

**Key Patterns Observed:**

**up_rate_in_user_history (Top feature):**
```
🔴 Red dots (high rate) clustered on RIGHT (positive SHAP +1 to +6)
   └─ High purchase rate → Strongly increases purchase probability
   
🔵 Blue dots (low rate) scattered on LEFT (negative SHAP -2 to 0)
   └─ Low/zero purchase rate → Decreases purchase probability
   
⚡ Clear positive correlation: Higher rate = Higher prediction
```

**up_times_bought:**
```
🔴 Red dots (many purchases) on RIGHT (positive SHAP +1 to +3)
   └─ Frequently bought products → High purchase probability
   
🔵 Blue dots (few/no purchases) on LEFT (negative SHAP -1 to 0)
   └─ Rarely bought products → Low purchase probability
   
⚡ Strong positive relationship
```

**up_recency:**
```
🔴 Red dots (high recency = long time ago) scattered LEFT and RIGHT
   └─ Complex relationship: Not simply "older = worse"
   
🔵 Blue dots (low recency = recent purchase) also mixed
   └─ Very recent purchases might → lower probability (just bought, don't need again)
   └─ Moderate recency → higher probability (time to reorder)
   
⚡ Non-linear U-shaped or optimal window effect
```

**p_reorder_prob:**
```
🔴 Red dots (high reorder products) on RIGHT (positive SHAP)
   └─ Products with high global reorder rate → Higher prediction
   
🔵 Blue dots (low reorder products) on LEFT (negative SHAP)
   └─ One-time purchase products → Lower prediction
   
⚡ Moderate positive correlation
```

**Key Insights:**
```
✅ Clear directional effects for most features
   └─ up_rate_in_user_history: Higher is always better
   └─ up_times_bought: More purchases = higher probability
   
⚠️ Non-linear relationships exist
   └─ up_recency shows complex pattern (reorder cycle detection)
   └─ Some features have optimal ranges, not monotonic
   
✅ Feature interactions visible
   └─ Clustering patterns suggest features work together
   └─ High up_rate + high up_times_bought = very strong signal
```

---

### 6. **SHAP Waterfall: Label=0 Sample (pred=0.0000, true=0)**

**What the plot shows:**
- **Specific example:** One product that was NOT purchased (correctly predicted)
- **Base value f(x) = -12.008:** Model's default log-odds (very low baseline)
- **Final prediction E[f(X)] = -2.098:** After all features applied, still negative (predicts NOT purchased)
- **Actual label:** 0 (correct prediction)

**Feature Contributions (Top factors preventing purchase):**

1. **up_rate_in_user_history = -0.579 → SHAP: -3.06 (STRONG NEGATIVE)**
   - User has low/zero historical purchase rate for this product
   - Pushes prediction strongly toward "NOT purchased"
   - Most impactful feature for this sample

2. **up_times_bought = -0.632 → SHAP: -2.48 (STRONG NEGATIVE)**
   - User has rarely/never bought this product before
   - Reinforces the "NOT purchased" prediction

3. **up_recency = 1.823 → SHAP: -1.69 (NEGATIVE)**
   - Despite positive feature value (long time since purchase),
   - Contributes negatively (either never bought, or too old to matter)

4. **up_last_order_number = -1.034 → SHAP: -0.6 (MODERATE NEGATIVE)**
   - Product not in recent orders
   - Reduces purchase probability

**Positive factors (but not enough to change prediction):**
- **u_total_orders, u_distinct_products, u_reorder_ratio:** Positive but weak
- User is active (many orders), but no connection to THIS product

**Interpretation:**
```
❌ Product-user mismatch: Never/rarely purchased this specific item
   └─ All user-product (UP) features are negative
   └─ User is active (positive U features), but not for THIS product
   
✅ Correct prediction: Model correctly identifies this won't be purchased
   └─ Lack of historical interaction → strong signal
   └─ Even though user is generally active, this specific product doesn't fit
```

**Business Insight:**
- Model correctly filters out products users have no history with
- Active users alone don't guarantee purchases - product-specific history matters most
- This represents successful negative prediction (avoid bad recommendations)

---

### 7. **SHAP Waterfall: Label=1 Sample (pred=0.5463, true=1)**

**What the plot shows:**
- **Specific example:** One product that WAS purchased (correctly predicted)
- **Base value f(x) = 0.186:** Model's default log-odds
- **Final prediction E[f(X)] = -2.098:** Still negative in log-odds, but **pred=0.5463** means ~55% probability
- **Actual label:** 1 (correct prediction)

**Feature Contributions (Top factors driving purchase):**

1. **up_rate_in_user_history = 2.128 → SHAP: +1.33 (VERY STRONG POSITIVE)**
   - User has HIGH historical purchase rate for this specific product
   - Strongest signal predicting purchase
   - This user loves this product!

2. **up_times_bought = 2.192 → SHAP: +0.91 (STRONG POSITIVE)**
   - User has bought this product many times before
   - Frequent repurchaser
   - Reinforces purchase prediction

3. **up_last_order_number = 0.879 → SHAP: +0.23 (MODERATE POSITIVE)**
   - Product appeared in recent orders
   - Recency adds confidence

4. **up_avg_add_to_cart = -0.471 → SHAP: +0.12 (SMALL POSITIVE despite negative value)**
   - Complex interaction captured

**Negative factors (but overcome by strong positives):**
- **u_avg_basket_size = -1.875 → SHAP: -0.37 (NEGATIVE)**
   - User has smaller basket sizes
   - Slightly reduces probability (fewer items per order)
  
- **sim_max = -0.249 → SHAP: -0.06 (SMALL NEGATIVE)**
   - Low similarity to other products
   - Minor negative contribution

**Interpretation:**
```
✅ Strong product-user match: Frequent repurchaser of this item
   └─ up_rate_in_user_history and up_times_bought dominate
   └─ User has clear preference/habit for this product
   
✅ Correct prediction: Model correctly identifies likely purchase
   └─ Strong historical signals overcome any negative factors
   └─ Prediction ~55% probability aligns with purchase occurring
   
⚡ Feature dominance: Top 2 UP features contribute +2.24 SHAP value
   └─ Other features minor in comparison
   └─ Historical behavior is best predictor
```

**Business Insight:**
- Model successfully identifies loyal customers for specific products
- Repurchase patterns (rate + times bought) are most predictive
- Even with some negative signals (small basket, low similarity), strong product loyalty wins
- This represents successful positive prediction (good recommendation)

---

## 🎯 QDA Model: Key Findings & Insights

### 1. **Strong Discrimination, Poor Calibration**
✅ **AUC-ROC = 0.8033:** Good ability to rank candidates (distinguish purchased from non-purchased)
❌ **Calibration Plot:** Severely under-confident - predicted probabilities don't match true rates

**Implication:**
- Use QDA scores for **ranking** (relative order is correct)
- Don't use raw scores as **calibrated probabilities**
- Downstream LightGBM can learn from QDA score rankings

---

### 2. **Precision-Recall Tradeoff**
✅ **High precision at low recall:** Top 5-10% predictions are very accurate
⚠️ **Moderate AP (0.2923):** Reflects class imbalance, but still 4-8x better than random

**Implication:**
- QDA excels at identifying high-confidence purchases
- Suitable for candidate generation stage (filter down to ~10-20% of products)
- Lower-ranked candidates need additional ranking (LightGBM stage)

---

### 3. **Feature Importance: Historical Behavior Dominates**
✅ **Top features:** `up_rate_in_user_history`, `up_times_bought`, `up_recency`
✅ **Pattern:** User-product interaction history is strongest signal
⚠️ **Lower features:** Many have minimal impact (candidates for removal)

**Implication:**
- Focus feature engineering on user-product interaction metrics
- Reorder behavior and purchase cycles are key
- Cold-start problem: New users/products will have weak signals

---

### 4. **Model Correctly Identifies Purchase Patterns**
✅ **Waterfall Label=1:** Model rewards frequent repurchasers (correct positive)
✅ **Waterfall Label=0:** Model penalizes products with no user history (correct negative)

**Implication:**
- Model logic is sound: Historical behavior predicts future behavior
- Works best for established users with purchase history
- Struggles with new products or exploratory purchases

---

## 💡 QDA Model: Recommendations for Improvement

### 1. **Address Calibration Issues**
- **Apply Platt scaling or isotonic regression** to calibrate probabilities
- **Use calibrated scores** if downstream systems need actual probabilities
- **Current state:** Scores work for ranking but not for probability interpretation

### 2. **Improve Precision at Higher Recall**
- **Add features** capturing seasonal patterns, day-of-week preferences
- **Engineer interaction features** (e.g., up_rate × p_reorder_prob)
- **Target:** Maintain 30-40% precision beyond 20% recall

### 3. **Handle Cold-Start Cases**
- **Content-based features** for new products (category, brand, price)
- **Collaborative filtering** for new users (similar user behavior)
- **Popularity-based fallback** when interaction history is missing

### 4. **Feature Selection**
- **Remove low-impact features** (bottom 30-40% in SHAP importance)
- **Reduces noise** and improves model stability
- **Faster inference** with fewer features

---

# Part 2: LightGBM LambdaRank Model Evaluation

## 📊 LightGBM Model Overview

The LightGBM model serves as the **second stage** in the recommendation pipeline, taking QDA-filtered candidates and ranking them optimally for each user's order.

### Evaluation Context
- **Model Type:** LightGBM LambdaRank (Learning-to-Rank)
- **Purpose:** Rank candidate products to maximize relevance (NDCG, MAP, Recall)
- **Dataset:** `valid_candidates.csv`
- **Evaluation Scope:** 26,241 order groups (user shopping sessions)
- **Total Rows Scored:** 1,696,883 candidate products
- **Group Key:** `order_id` (each order represents one ranking task)
- **Label Column:** `label` (1 = purchased, 0 = not purchased)

### Evaluation Context
- **Model Type:** LightGBM LambdaRank (Learning-to-Rank)
- **Dataset:** `valid_candidates.csv`
- **Evaluation Scope:** 26,241 order groups (user shopping sessions)
- **Total Rows Scored:** 1,696,883 candidate products
- **Group Key:** `order_id` (each order represents one ranking task)
- **Label Column:** `label` (1 = purchased, 0 = not purchased)

---

## 🎯 Ranking Metrics Explained (For LightGBM)

### 1. NDCG (Normalized Discounted Cumulative Gain)

**What it measures:** Quality of ranking with position weighting — rewards placing relevant items at the top.

**Formula:**
```
NDCG@K = DCG@K / IDCG@K
DCG@K = Σ (relevance_i / log₂(position_i + 1))
```

**Why it matters:**
- Items at rank 1 receive maximum weight (1.0)
- Items at rank 10 receive much less weight (~0.3)
- **Reflects reality:** Users see top items first!

**Interpretation:**
- **1.0** = Perfect ranking (all purchased items at the very top)
- **0.5-0.7** = Good ranking (most purchased items in top positions)
- **0.0** = Terrible ranking (purchased items not in top-K)

---

### 2. MAP (Mean Average Precision)

**What it measures:** Average precision across different cutoff points, emphasizing early retrieval.

**Formula:**
```
AP@K = (1/R) × Σ P(k) × rel(k)
MAP@K = Average of AP@K across all orders
```
Where:
- R = total relevant items (purchased products)
- P(k) = precision at position k
- rel(k) = 1 if item at position k is relevant, else 0

**Why it matters:**
- More sensitive to the exact position of relevant items than NDCG
- Penalizes heavily when relevant items are not consecutively ranked at the top
- **Business impact:** Higher MAP = users find what they want faster

**Interpretation:**
- **>0.6** = Excellent precision in top positions
- **0.4-0.6** = Good precision
- **<0.4** = Poor precision, relevant items scattered

---

### 3. Recall@K

**What it measures:** Percentage of all purchased products that appear in the top-K recommendations.

**Formula:**
```
Recall@K = (Purchased items in top-K) / (Total purchased items)
```

**Why it matters:**
- Measures **coverage** rather than ranking quality
- **Perfect recall (1.0)** means all purchased items are in top-K
- Does NOT care about ranking order

**Interpretation:**
- **1.0** = All purchased items appear in top-K (perfect coverage)
- **0.8** = 80% of purchased items appear in top-K
- **0.5** = Only half of purchased items appear in top-K

---

### 4. Hitrate@K

**What it measures:** Percentage of orders where AT LEAST one purchased item appears in top-K.

**Formula:**
```
Hitrate@K = (Orders with ≥1 relevant item in top-K) / (Total orders)
```

**Why it matters:**
- Binary metric: either hit (≥1 relevant item) or miss (0 relevant items)
- Easiest metric to achieve
- **Business impact:** Hitrate = 1.0 means NO order gets empty recommendations

**Interpretation:**
- **1.0** = Every order has at least one relevant item in top-K
- **0.95** = 5% of orders got no relevant recommendations
- **0.80** = 20% of orders missed completely

---

## 📈 Overall Model Performance

### Summary Metrics (Validation Set)

| Metric | @K=5 | @K=10 | @K=20 | Interpretation |
|--------|------|-------|-------|----------------|
| **NDCG** | 0.496 | 0.521 | 0.575 | Good ranking quality, ~50-57% of perfect |
| **MAP** | 0.606 | 0.569 | 0.521 | Excellent early precision, decreases with K |
| **Recall** | 0.809 | 0.873 | 0.911 | Outstanding coverage, 81-91% of items found |
| **Hitrate** | 0.809 | 0.873 | 0.911 | Strong coverage, 81-91% of orders get relevant items |

---

### Performance Analysis by Cutoff (K)

#### NDCG Progression
```
NDCG@5:  0.496 ──────────────────────────────► (49.6% of perfect ranking)
NDCG@10: 0.521 ────────────────────────────────► (52.1% of perfect ranking)
NDCG@20: 0.575 ──────────────────────────────────────► (57.5% of perfect ranking)
```

**Insight:** NDCG increases with K because more positions = more opportunities to include relevant items at acceptable ranks.

**Why the gap is not huge:** The model already places most relevant items in top 5-10 positions, so extending to K=20 only slightly improves quality.

---

#### MAP Progression (Inverse Pattern!)
```
MAP@5:  0.606 ──────────────────────────────────────► (Best precision!)
MAP@10: 0.569 ────────────────────────────────► (Slight decrease)
MAP@20: 0.521 ──────────────────────────────► (Further decrease)
```

**Insight:** MAP **decreases** with K because precision naturally drops as we extend recommendations.

**Why this is normal:** 
- Top 5 items have highest precision (60.6%)
- Adding positions 6-10 introduces more false positives
- Positions 11-20 have even lower precision

**This is expected behavior** — early positions should be most precise!

---

#### Recall Progression
```
Recall@5:  0.809 ───────────────────────────────────► (81% coverage)
Recall@10: 0.873 ─────────────────────────────────────► (87% coverage)
Recall@20: 0.911 ────────────────────────────────────────► (91% coverage)
```

**Insight:** Strong recall at all cutoffs!

**What this means:**
- At K=5: 81% of all purchased items appear in top 5 (excellent!)
- At K=10: 87% coverage (very strong)
- At K=20: 91% coverage (outstanding!)

**Business implication:** Users will find almost everything they want to buy within top 20 recommendations.

---

#### Hitrate Progression
```
Hitrate@5:  0.809 ───────────────────────────────────► (81% orders covered)
Hitrate@10: 0.873 ─────────────────────────────────────► (87% orders covered)
Hitrate@20: 0.911 ────────────────────────────────────────► (91% orders covered)
```

**Insight:** Strong hitrate across all cutoffs — matches recall exactly in this evaluation.

**What this means:**
- At K=5: 81% of orders have at least one relevant item in top 5
- At K=10: 87% of orders covered
- At K=20: 91% of orders covered

**Note:** In this evaluation, hitrate equals recall, suggesting most orders have only 1 purchased item, or the metric is calculated equivalently.

---

## 🔍 Per-Group Performance Distribution

The model was evaluated on 26,241 individual orders. Performance varies significantly across different orders.

### Key Observations from Per-Group Metrics

#### NDCG Distribution Patterns

**Actual NDCG@10 Distribution (from 26,241 orders):**
```
Statistics:
- Mean: 0.521
- Median: 0.547
- Std Dev: 0.302
- Min: 0.000
- 25th percentile: 0.307
- 75th percentile: 0.757
- Max: 1.000
```

**Distribution breakdown:**
- 25% of orders have NDCG@10 < 0.307 (lower quartile)
- 50% of orders have NDCG@10 < 0.547 (median)
- 75% of orders have NDCG@10 < 0.757 (upper quartile)
- High variability (std dev = 0.302)

**Why variability exists:**
1. **User purchase patterns:** Some users buy predictably (high NDCG), others experiment (low NDCG)
2. **Order complexity:** Orders with few items (1-2 products) are easier to rank perfectly
3. **Data availability:** Users with long history → better predictions
4. **Product popularity:** Common products easier to predict than rare items

---

#### Critical Finding: Binary Performance Pattern

**Observation from actual per-group data:**
- **23,918 orders (91.1%)** have **Recall@20 = 1.0** and **Hitrate@20 = 1.0** (all items found)
- **2,323 orders (8.9%)** have **Recall@20 = 0.0** and **Hitrate@20 = 0.0** (complete misses)
- Hitrate equals recall in this evaluation (same values at all K)
- NDCG varies significantly even among successful orders (those with recall = 1.0)

**What this tells us:**

```
✅ Binary Success Pattern: Orders either succeed (91%) or fail completely (9%)
   └─ 91% of orders have ALL purchased items in top-20
   └─ 9% of orders have NO purchased items in top-20
   └─ Very few "partial success" cases

⚠️ Ranking Quality Varies Within Successful Orders
   └─ Among the 91% successful orders, NDCG varies widely
   └─ Some have perfect ranking (NDCG = 1.0), others poor (NDCG near 0)
   └─ All items present, but positioned differently
```

**Why hitrate equals recall:**
- Suggests most orders have only 1 purchased item, OR
- Orders with multiple items either get all (recall=1.0) or none (recall=0.0)

**Example Scenarios:**

**Successful Order A (NDCG@10 = 1.0, Recall@10 = 1.0):**
```
Purchased: [Milk]
Model Ranks:
1. Milk   ✅ Perfect!
2-10. (Not purchased, but candidates)
```

**Successful Order B (NDCG@10 = 0.3, Recall@10 = 1.0):**
```
Purchased: [Milk]
Model Ranks:
1-7. (Not purchased)
8. Milk   ⚠️ Found, but ranked low
9-10. (Not purchased)
```

**Failed Order C (NDCG@10 = 0.0, Recall@10 = 0.0):**
```
Purchased: [Organic Kale]
Model Ranks:
1-10. (Not purchased)
... Milk not in top-20 at all ❌
```

---

### Performance Distribution Insights

#### Successful Orders (91.1% - Recall@20 = 1.0)
Within this group, NDCG@10 varies significantly:

**High NDCG (top quartile > 0.757):**
- Purchased items ranked in top positions
- Clear user purchase patterns
- Strong feature signals (high ALS score, frequent repurchases)

**Medium NDCG (0.307 - 0.757):**
- Purchased items found but ranked lower
- Mix of predictable and exploratory purchases
- Moderate feature signals

**Low NDCG (< 0.307 but recall = 1.0):**
- Purchased items present but ranked very low (positions 15-20)
- Weak ranking quality despite item being in candidates
- Indicates ranking stage needs improvement

---

#### Failed Orders (8.9% - Recall@20 = 0.0)
**Likely characteristics** (not directly measured in evaluation data):
- New users with minimal purchase history
- Rare products not in candidate set
- Cold-start problem
- Incomplete user-product interaction data

**Model weaknesses:**
- Complete failure rather than partial success
- 2,323 orders receive zero relevant recommendations
- Most critical area for improvement

---

## 📊 LightGBM Model: Additional Notes on Visualizations

**Note:** The visualization plots in `reports/Eval plots/` directory (Calibration Plot, Precision-Recall Curve, ROC Curve, SHAP plots) are generated from the **QDA model evaluation** (see Part 1 above for detailed analysis).

**LightGBM-specific evaluation metrics** are reported in `reports/lgbm_eval/summary.json` and `per_group_metrics.csv` (analyzed in sections above).

For LightGBM SHAP analysis or additional visualizations, they would need to be generated separately from the ranking model.

---

## 📊 Original Visualization Descriptions (For Reference)

The following sections describe what these visualization types represent. **Actual plots shown above are from QDA model.**

### 1. **Calibration Plot** (`Calibration plot.png` - QDA Model)

**What it shows:** Relationship between predicted probabilities and actual purchase rates.

**How to interpret:**
- **X-axis:** Predicted probability (model's confidence)
- **Y-axis:** Actual purchase rate (observed frequency)
- **Ideal:** Points lie on diagonal line (perfect calibration)

**What to look for:**
```
Over-confident:  Model predicts 80%, actual 50% → Model too aggressive
Under-confident: Model predicts 30%, actual 60% → Model too conservative
Well-calibrated: Model predicts 60%, actual 60% → Model trustworthy
```

**Business impact:**
- Well-calibrated models → reliable confidence scores
- Can use scores for dynamic thresholding
- Enables personalized recommendation strategies

---

### 2. **Precision-Recall Curve** (`Precision Recall Curve.png`)

**What it shows:** Trade-off between precision (accuracy of recommendations) and recall (coverage of relevant items).

**How to interpret:**
- **X-axis:** Recall (% of purchased items found)
- **Y-axis:** Precision (% of recommendations that are purchased)
- **Ideal:** Curve stays high on both axes (top-right)

**Key metrics:**
```
AUC-PR (Area Under Curve): Higher = better overall performance
Threshold selection: Choose operating point based on business needs
```

**Business decisions:**
- **High precision, low recall:** Show fewer items but very relevant (conservative)
- **High recall, low precision:** Show more items, catch everything (aggressive)
- **Optimal point:** Maximize both based on use case

---

### 3. **ROC Curve** (`ROC Curve.png`)

**What it shows:** Classification performance across all threshold settings.

**How to interpret:**
- **X-axis:** False Positive Rate (showing non-purchased items)
- **Y-axis:** True Positive Rate (showing purchased items)
- **Ideal:** Curve hugs top-left corner (high TPR, low FPR)

**Key metrics:**
```
AUC-ROC: 0.9+ = Excellent, 0.8-0.9 = Good, 0.7-0.8 = Fair, <0.7 = Poor
```

**What it means:**
- **AUC = 0.5:** Model is no better than random guessing
- **AUC = 1.0:** Perfect classifier (separates purchased/not purchased perfectly)
- **Refer to actual plot** for your model's AUC-ROC value

---

### 4. **SHAP Summary Plot (Bar)** (`Shap summary plot.png`)

**What it shows:** Feature importance ranking — which features matter most for predictions.

**How to interpret:**
- **X-axis:** Mean |SHAP value| (average impact magnitude)
- **Y-axis:** Features ranked by importance
- **Top features:** Have biggest impact on model decisions

**Expected top features (refer to actual plot):**
```
(Top features should be visible in the SHAP plot - likely include:)
- als_score           ← Collaborative filtering signal
- up_times_bought     ← Purchase frequency (user-product)
- up_recency          ← Days since last purchase
- sim_max             ← Item-item similarity
- p_reorder_prob      ← Product reorder rate
```

**Business insights:**
- Identify which signals drive recommendations
- Prioritize feature engineering efforts
- Understand model decision-making process

---

### 5. **SHAP Summary Plot (Beeswarm)** (`Shap summary plot beeswarm.png`)

**What it shows:** Feature importance with direction and distribution of impact.

**How to interpret:**
```
Color: Red = high feature value, Blue = low feature value
X-axis: SHAP value (positive = increases prediction, negative = decreases)
Y-axis: Features ranked by importance
```

**Example interpretations:**
```
als_score (red dots on right):
  High ALS score → Strong positive impact → Higher ranking

up_recency (red dots on left):
  High recency (long time since purchase) → Negative impact → Lower ranking
  (Because model thinks user already bought it)

up_times_bought (red dots on right):
  High purchase frequency → Strong positive impact → Higher ranking
```

**Key insights:**
- See not just WHAT matters, but HOW it matters
- Identify non-linear relationships (spread patterns)
- Spot interaction effects (clustering patterns)

---

### 6. **SHAP Waterfall Plot - Label 0** (`Shap water fall label 0.png`)

**What it shows:** Detailed explanation for ONE specific prediction where label = 0 (NOT purchased).

**How to interpret:**
```
Base value: Model's default prediction (e.g., 0.1)
    ↓
Feature 1 pushes prediction up: +0.2
Feature 2 pushes prediction down: -0.15
Feature 3 pushes prediction up: +0.05
    ↓
Final prediction: 0.2 (not purchased)
```

**Example scenario:**
```
Product: Organic Kale
User: Never bought before

Base: 0.10 (low default probability)
  + up_times_bought=0: -0.15  ← Never bought → Strong negative
  + als_score=0.3: +0.05       ← Low collaborative signal
  + p_reorder_prob=0.6: +0.08  ← Product itself has ok reorder rate
  → Final: 0.08 → NOT recommended (rank 150+)
```

**Use case:** Debugging — understand why model rejected specific items.

---

### 7. **SHAP Waterfall Plot - Label 1** (`Shap water fall label 1.png`)

**What it shows:** Detailed explanation for ONE specific prediction where label = 1 (PURCHASED).

**How to interpret:**
```
Base value: 0.1
    ↓
Multiple features push prediction UP
    ↓
Final prediction: 0.85 (purchased!)
```

**Example scenario:**
```
Product: Organic Whole Milk
User: Frequent buyer

Base: 0.10
  + up_times_bought=12: +0.35  ← Bought 12 times → Strong positive!
  + up_recency=3: +0.15        ← Last bought 3 orders ago → Due for reorder
  + als_score=0.92: +0.20      ← High collaborative signal
  + sim_max=0.88: +0.05        ← Similar to user's favorites
  → Final: 0.85 → STRONGLY recommended (rank 1-3)
```

**Use case:** Validation — confirm model rewards correct signals.

---

## 🎯 Key Findings & Insights

### 1. **Binary Success Pattern: 91% Success, 9% Complete Failure**
✅ **91.1% of orders (23,918/26,241)** achieve perfect recall@20 = 1.0 (all items found)
❌ **8.9% of orders (2,323/26,241)** have recall@20 = 0.0 (complete misses)

**Implications:**
- Strong performance for majority of orders
- Binary outcome: orders either succeed completely or fail completely
- **Critical gap:** 2,323 orders get zero relevant recommendations (cold-start problem)
- Focus improvement on the 9% failure cases

---

### 2. **Ranking Quality is Strong but Variable**
⚠️ **NDCG ranges from 0.0 to 1.0** across individual orders.

**Recommendations:**
- Investigate low-NDCG orders: What makes them hard to rank?
- Consider order-specific features (cart size, time of day, etc.)
- Add features for exploration vs. exploitation

---

### 3. **Model is Well-Suited for Top-5 Recommendations**
✅ **MAP@5 = 0.606, NDCG@5 = 0.496** show strong early precision.

**Business strategy:**
- Display top 5 items prominently (highest precision)
- Use positions 6-10 for "You might also like"
- Positions 11-20 for "More suggestions"

---

### 4. **Performance Decreases Gracefully with K**
✅ **No sudden drop-offs** — model maintains quality across cutoffs.

**Implications:**
- Model rankings are consistent (no arbitrary scoring)
- Safe to show top 10-20 items without quality concerns
- User experience remains good throughout recommendation list

---

### 5. **High Recall Validates Model Design**
✅ **Recall@10 = 0.873** means 87% of purchased items are in top 10.

**Business impact:**
- Users will find most of what they need
- Low frustration (relevant items are present)
- Opportunity to improve ranking for even better UX

---

## 💡 Recommendations for Improvement

### 1. **PRIORITY: Address the 9% Complete Failures**
- **Analyze the 2,323 orders with recall = 0.0**
  - Are they new users?
  - Are purchased items rare/unusual products?
  - Is there insufficient interaction data?
- **Implement cold-start strategies:**
  - Content-based recommendations
  - Category-level fallbacks
  - Popular items as backup
- **Improve candidate generation** to ensure relevant items are included

### 2. **Improve Ranking Quality for Successful Orders**
- **Target lower-quartile orders** (NDCG@10 < 0.307 but recall = 1.0)
- **Analyze why items rank low** despite being present
- **Feature engineering:** Focus on top SHAP features
- **Create interaction features** to better capture ranking signals

### 3. **Calibration Improvements**
- **Re-calibrate probabilities** if calibration plot shows bias
- **Use Platt scaling or isotonic regression** for better probability estimates
- **Enable confidence-based filtering** in production

### 4. **Optimize for Business Metrics**
- **Define target metrics:** Is NDCG@5 or NDCG@10 more important?
- **A/B test different cutoffs** in production
- **Track user engagement** (click-through rate, conversion rate)

### 5. **Model Refinement**
- **Hyperparameter tuning** (see `IMPROVING_METRICS.md`)
- **Ensemble methods:** Combine multiple models
- **Experiment with different objectives:** Optimize for specific K values

---

## 📋 Summary

| Aspect | Status | Details |
|--------|--------|---------|
| **Coverage** | ✅ Strong | 91% hitrate@20, 91% recall@20 |
| **Ranking Quality** | ✅ Good | NDCG@10 = 0.521, MAP@5 = 0.606 |
| **Consistency** | ✅ Strong | Stable performance across K values |
| **Precision** | ✅ Good | 60.6% precision in top 5 |
| **Variability** | ⚠️ Moderate | Some orders perfect (NDCG=1.0), others poor (NDCG=0.0) |
| **Calibration** | ℹ️ Check plots | Assess from visualization |
| **Feature Importance** | ℹ️ See SHAP | ALS, purchase frequency, recency dominate |

---

## 🚀 Next Steps

1. **Review visualizations** in `reports/Eval plots/` directory
2. **Implement improvements** from `IMPROVING_METRICS.md`
3. **Monitor performance** in production with real user interactions
4. **Iterate on features** based on SHAP analysis
5. **A/B test** different model configurations

---

For detailed metric explanations, see: [`UNDERSTANDING_SCORES.md`](./UNDERSTANDING_SCORES.md)

For improvement strategies, see: [`IMPROVING_METRICS.md`](./IMPROVING_METRICS.md)

---

**Evaluation Date:** Based on validation set analysis  
**Model Version:** LightGBM LambdaRank trained on `valid_candidates.csv`  
**Total Orders Evaluated:** 26,241  
**Total Predictions:** 1,696,883
