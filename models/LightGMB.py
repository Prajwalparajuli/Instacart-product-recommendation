"""


Introduction to the LightGBM LambdaRank Model

We are building a model that recommends products to users based on past behavior. 
For this, we use LightGBM with LambdaRank, which is specially designed for ranking items. 

Here's what that means:

1. Ranking Objective

Instead of just predicting “buy or not buy” for each product, the model learns to rank all products in an order.
Goal: Put the most likely-to-be-bought products at the top of the list.
This is different from a regular classifier, which only predicts 0 or 1.

Example:
If a user has 10 products in their cart, the model will score all 10 and rank them from most likely to buy to least likely.

2. Group Learning

Each order is treated as a group. The model only compares products within the same order.
This ensures the ranking is relevant for each individual order, not globally across all users.

Example:
For order #123, the model ranks only products in that order, ignoring products from other orders.

3. Evaluation Metric – NDCG

We use NDCG (Normalized Discounted Cumulative Gain) to measure ranking quality.
Focuses on top of the list, because the top recommendations matter most.
Higher NDCG → top products are correctly ranked.

Example:
If the model predicts the products the user actually buys at the top 5, NDCG score will be high.

4. Features Used

The model only looks at behavioral information, such as:

- How often a user reorders products
- Average ratings, purchase frequency, etc.
- IDs are excluded (user_id, product_id, order_id) to prevent the model from “cheating” by memorizing IDs instead of learning patterns.

5. Result

The model outputs top-K recommendations for each order. These recommendations tell the user what they are most likely to buy next.

Example:
For a given order, the model might rank products like this:

Product A → most likely to buy
Product B → second most likely
Product C → third most likely

This helps in personalized recommendations for e-commerce platforms like Instacart.
"""


# LightGBM (LambdaRank) Model for Ranking Products

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

import pandas as pd
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation

# pandas → read and handle data.
# lightgbm → the machine learning model.
# early_stopping → stop training if validation doesn’t improve.
# log_evaluation → print training progress.

#-------------
# 1. Load Data

train = pd.read_csv("Data/processed/train_candidates.csv")

test = pd.read_csv("Data/processed/test_candidates.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)

# Training data: has features + label (whether a product was reordered).
# Test data: only features (we want to predict rankings).
# The code prints the shape of the data to check the number of rows and columns.

    # Why it matters: You always want to know how big your dataset is before modeling.

#---------------------------------------------
# 2. Prepare Data for what goes into the model

ID_COLS = ["user_id", "order_id", "product_id"]
LABEL_COL = "label"
GROUP_KEY = "order_id"

feature_cols = [c for c in train.columns if c not in ID_COLS + [LABEL_COL]]

# Sanity check to fails fast if data is not shaped as expected
required = set(ID_COLS + [LABEL_COL])
missing = required - set(train.columns)

if missing:
    raise ValueError(f"Missing columns in train data: {missing}")

if not feature_cols:
    raise ValueError("No feature columns found for modeling.")
print(f"Using {len(feature_cols)} features for modeling.")

# Remove identifiers like user_id or product_id—they don’t help the model learn patterns.
# Keep behavioral features like how often a user orders, average reorder rate, days since last order, etc.

    # Why it matters: Machine learning models learn from patterns, not IDs.


#-----------------------------------------------
# 3. Split train into train/validation by orders

# This takes the last order of each user as validation 
last_order = train.groupby("user_id", as_index = False)["order_number"].max().rename(columns = {"order_number":"last_order_id"})

# Merge onto the train set to identify last orders
train_merged = train.merge(last_order, on = "user_id", how = "left")

# Validation set: last orders of each user
valid = train_merged[train_merged['order_id'] == train_merged['last_order_id']].copy()

# Training set: all other orders
train_for_model = train_merged[train_merged['order_id'] != train_merged['last_order_id']].copy()

print("Train orders:", len(train_for_model), "Validation orders:", len(valid))

# Split last order of each user for validation.
# Ensures the model is tested on unseen orders.

    # Why it matters: Prevents overfitting and ensures that model performance is realistic.


#-----------------------------------
# 4. Build datasets grouped by order

def make_dataset(df):
    df = df.sort_values(GROUP_KEY).reset_index(drop=True)
    X = df[feature_cols]
    y = df[LABEL_COL].astype(int)
    group_sizes = df.groupby(GROUP_KEY, sort=False).size().to_numpy()
    ds = lgb.Dataset(X, label=y, group=group_sizes,
                     feature_name=feature_cols, free_raw_data=False)
    return ds, df

dtrain, train_sorted = make_dataset(train_for_model)
dvalid, valid_sorted = make_dataset(valid)

# Each order is treated as a group.
# LightGBM learns to rank products within each order.
# group_sizes tells the model how many products are in each order.

    # Analogy: Imagine each order as a “mini leaderboard” of products. LambdaRank teaches the model to put the 
    # most likely bought products at the top.
 
#-----------------------
# 4. Model configuration

# We tell LightGBM to use LambdaRank (ranking objective):

params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [5, 10, 20],
    "learning_rate": 0.03,
    "num_leaves": 63,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 2,
    "min_data_in_leaf": 200,
    "random_state": 42,
    "max_depth": 12,
    "lambda_l1": 0.5,
    "lambda_l2": 3.0
}

# objective = lambdarank: learn ranking, not classification.
# metric = ndcg: Normalized Discounted Cumulative Gain, measures ranking quality (higher for correctly 
# ranked top items).
# learning_rate, num_leaves, feature_fraction → control learning speed, tree complexity, and overfitting.

    # Why it matters: Proper parameters make the model accurate and stable.

#---------------
# 5. Train Model

model = lgb.train(
    params=params,
    train_set=dtrain,
    valid_sets=[dtrain, dvalid],
    valid_names=["train", "valid"],
    num_boost_round = 3000,
    callbacks=[early_stopping(stopping_rounds = 150), log_evaluation(50)],
)

# Train the model on grouped orders.
# early_stopping → stop if validation doesn’t improve for 150 rounds.
# log_evaluation → prints progress every 50 rounds.

    # Why it matters: Training efficiently and preventing overfitting is crucial with millions of rows.

#------------------------------
# 6. Ranking on new test orders

def rank_topk(df, model, k=10):
    # Sanity check to fails fast if data is not shaped as expected
    for c in ID_COLS:
        if c not in df.columns:
            raise ValueError(f"Missing column in test data: {c}")

    df = df.sort_values(GROUP_KEY).reset_index(drop=True)
    X = df[feature_cols]
    scores = model.predict(X, num_iteration=model.best_iteration)
    out = df[ID_COLS].copy()
    out["score"] = scores
    out["rank"] = out.groupby(GROUP_KEY)["score"].rank(ascending=False, method="first")
    topk = out[out["rank"] <= k].copy()
    return topk.sort_values([GROUP_KEY, "rank"]) 

# Predict a score for each product in the test set.
# Rank products within each order.
# Keep only top K products (e.g., top 10).

    # Why it matters: This creates the final recommendation list for each order.

#------------------------------------
#7. Predict & rank top-K for test set 
topk_recs = rank_topk(test, model, k=10)
topk_recs.to_csv("lightgbm_lambdamart_top10.csv", index=False)
print("\nSaved top-K recommendations to lightgbm_lambdamart_top10.csv")

# Save top-ranked products for each order to a CSV file.
# Can be used directly in an application or for evaluation.