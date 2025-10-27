###Create user, item, and user x item features (CSV outputs in data/ processed/)###
# Purpose:
# Build final training and testing feature sets for the QDA and ranker stage.
# This script depends on ingest.load_all() to load the raw CSVs.        
# This script only handles feature merging, candidate generation and aggregate feature generation
# Using vectorized logic for building train/test candidates.

import os
import pandas as pd
import numpy as np
from pathlib import Path
from ingest import load_all


# Define paths
PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents = True, exist_ok = True)

ALS_SIM_PATH = r"data/processed/user_product_features.csv"
als_sim_features = pd.read_csv(ALS_SIM_PATH)

print("Loaded ALS + similarity features")

# Ensure (user_id, product_id) uniqueness and dtype alignment before merging 
id_cols = ["user_id", "product_id"]
for c in id_cols:
    if c not in als_sim_features.columns:
        als_sim_features[c] = als_sim_features[c].astype("int64")

if als_sim_features.duplicated(subset=id_cols).any():
    # Collapse duplicates conservatively: adjust aggs if your column names differ
    als_sim_features = (als_sim_features.groupby(id_cols, observed=True)
                        .agg({"als_score": "mean",
                        "sim_max": "max",
                        "sim_mean": "mean",
                        "sim_topk_mean_10": "mean",
                        "sim_nonzero_cnt": "sum"}))

# Load csv via ingest.py
dfs = load_all()
orders = dfs["orders"]
op_prior = dfs["op_prior"]
op_train = dfs["op_train"]
products = dfs["products"]
aisles = dfs["aisles"]
departments = dfs["departments"]

print("Loaded from ingest.load_all()")

# Normalize key dtypes to avoid silent joint mismatches
orders["user_id"] = orders["user_id"].astype("int64")
orders["order_id"] = orders["order_id"].astype("int64")

op_prior["order_id"] = op_prior["order_id"].astype("int64")
op_train["order_id"] = op_train["order_id"].astype("int64")

op_prior["product_id"] = op_prior["product_id"].astype("int64")
op_train["product_id"] = op_train["product_id"].astype("int64")

# Derive splits of prior, train, test from orders
orders_prior = orders.loc[orders.eval_set == "prior"].copy()
orders_train = orders.loc[orders.eval_set == "train"].copy()
orders_test = orders.loc[orders.eval_set == "test"].copy()

print("Derived order splits")

prior = (op_prior.merge(orders_prior[["order_id", "user_id", "order_number", "order_dow",
                                      "order_hour_of_day", "days_since_prior_order"]],
                       on="order_id", how="left"))
print("Merged prior order products with orders")

op_train = (op_train.merge(orders_train[["order_id", "user_id", "order_number", "order_dow",
                                        "order_hour_of_day", "days_since_prior_order"]],
                                        on="order_id", how="left"))

print("Merged train order products with orders")

# Sanity check
_needed = {"user_id", "product_id"}
missing = _needed - set(op_train.columns)
if missing:
    raise ValueError(f"Missing columns in op_train after merge: {missing}")     

# Collect unique prior products per user (user history set)
print("Building user prior products...")
user_prior_products = (prior.loc[:, ["user_id", "product_id"]].
                       dropna(subset=["user_id", "product_id"]).
                       drop_duplicates().groupby("user_id", observed=True)["product_id"]
                       .apply(list).rename("prior_products")).reset_index()

print("Collected unique prior products per user")

# Build user features from prior only
print("Building user features...")
user_grp = prior.groupby("user_id")
user_features = pd.DataFrame({
    # User ID
    "user_id": user_grp["user_id"].first(),
    # Total number of orders for each user
    "u_total_orders": user_grp["order_number"].max(),
    # Total number of items ordered by each user
    "u_total_items": user_grp["product_id"].count(),
    # Number of distinct products ordered by each user
    "u_distinct_products": user_grp["product_id"].nunique(),
    # Average basket size: items per unique order
    "u_avg_basket_size": user_grp["order_id"].count() / user_grp["order_id"].nunique(),
    # Reorder ratio: mean of reordered flag (0 if not present)
    "u_reorder_ratio": user_grp["reordered"].mean().fillna(0),
    # Mean days between orders (ignoring NaNs)
    "u_mean_days_between": user_grp["days_since_prior_order"].mean(),
    # Std dev of days between orders (ignoring NaNs)
    "u_std_days_between": user_grp["days_since_prior_order"].std()
    })

user_features = user_features.reset_index(drop=True)
print(f"User features shape: {user_features.shape}")

# Build product features from prior only
print("Building product features...")
product_grp = prior.groupby("product_id")
product_features = pd.DataFrame({
    "product_id": product_grp["product_id"].first(),
    "p_total_purchases": product_grp["order_id"].count(),
    "p_distinct_users": product_grp["user_id"].nunique(),
    "p_avg_add_to_cart": product_grp["add_to_cart_order"].mean(),
    "p_reorder_prob": product_grp["reordered"].mean().fillna(0)
    })

# reset index so that product_id is column, not index
product_features = product_features.reset_index(drop=True)
print(f"Product features shape: {product_features.shape}")

# Build user-product features from prior only
print("Building user-product features...")
up_grp = prior.groupby(["user_id", "product_id"])
up_features = pd.DataFrame({
    "user_id": up_grp["user_id"].first(),
    "product_id": up_grp["product_id"].first(),
    "up_times_bought": up_grp["order_id"].count(),
    "up_last_order_number": up_grp["order_number"].max(),
    "up_first_order_number": up_grp["order_number"].min(),
    "up_avg_add_to_cart": up_grp["add_to_cart_order"].mean()
}).reset_index(drop=True)

# Compute recency
u_last_prior_order = prior.groupby("user_id")["order_number"].max()

# map() to look my max order number for every user_id in up_features
up_features["u_last_prior_order"] = up_features["user_id"].map(u_last_prior_order)

# calculate recency using column subtraction
up_features["up_recency"] = up_features["u_last_prior_order"] - up_features["up_last_order_number"]

# Compute rate in user history
#calculate total items bought per customer
user_total_bought_series = up_features.groupby("user_id")["up_times_bought"].sum()

# map the total sum back onto the up_features table using user_id
up_features["user_total_bought"] = up_features["user_id"].map(user_total_bought_series)

# final rate calculation
up_features["up_rate_in_user_history"] = (
    up_features["up_times_bought"] / up_features["user_total_bought"]).replace({0: np.nan}).fillna(0)

print(f"User-product features shape: {up_features.shape}")

# Build train candidates (vectorized) + attach labels
print("Building train candidates...")
train_base = orders_train[["order_id", "user_id", "order_number", "order_dow",
                           "order_hour_of_day", "days_since_prior_order"]].copy()

# Expand each train order by user's prior product history
print("Expanding train candidates with prior products...")
train_candidates = (train_base.merge(user_prior_products, 
                                     on="user_id", how="left").
                                     explode("prior_products", ignore_index=True).
                                     rename(columns={"prior_products":"product_id"}))

print(f"Train candidates after expansion: {len(train_candidates):,} rows")
#8,474,611 rows Training size  

# Create label: 1 if product actually appeared in train order, else 0
labels = (op_train[["order_id", "user_id", "product_id"]].drop_duplicates().
                assign(label=1))

train_candidates = (train_candidates.merge(labels,
                                           on=["order_id", "user_id", "product_id"],
                                                 how="left"))
train_candidates["label"] = train_candidates["label"].fillna(0).astype("int8")

print("Built train candidates with labels")

# Merge ALS + similarity features into train candidates
required_cols = {"user_id", "product_id"}
missing_cols = required_cols - set(als_sim_features.columns)
if missing_cols:
    raise ValueError(f"Missing columns in ALS/similarity features: {missing_cols}")

dup_check_cols = ['user_id', 'order_id' ,'product_id']

# Merge onto candidate pairs
train_candidates = (train_candidates.merge(als_sim_features,
                                           on=["user_id", "product_id"],
                                           how="left"))
if train_candidates.duplicated(subset=dup_check_cols).any():
    raise ValueError("Duplicates found in train_candidates after merging ALS/similarity features, ensure uniqueness of (user_id, order_id, product_id)")

# Fill missing numeric values with 0 so the model interprets then as no signal
for col in ["als_score", "sim_max", "sim_mean", "sim_topk_mean_10"]:
    if col in train_candidates.columns:
        train_candidates[col] = train_candidates[col].fillna(0.0).astype("float32")

if "sim_nonzero_cnt" in train_candidates.columns:
    train_candidates["sim_nonzero_cnt"] = (train_candidates["sim_nonzero_cnt"].fillna(0).astype("int16"))

print("Merged ALS + similarity features into train candidates")

# Build test candidates (vectorized, no labels)
print("Building test candidates...")
test_base = orders_test[["order_id", "user_id", "order_number", "order_dow",
                         "order_hour_of_day", "days_since_prior_order"]].copy()

# Expand each test order by user's prior product purchase list
print("Expanding test candidates with prior products...")
test_candidates = (test_base.merge(user_prior_products, 
                                   on="user_id", how="left").
                                   explode("prior_products", ignore_index=True).
                                   rename(columns={"prior_products":"product_id"}))

print(f"Test candidates after expansion: {len(test_candidates):,} rows")
# testing size 4,833,292 rows

# Merge ALS + similarity onto candidate pairs
test_candidates = (test_candidates.merge(als_sim_features,
                                         on=["user_id", "product_id"],
                                         how="left"))

if test_candidates.duplicated(subset=dup_check_cols).any():
    raise ValueError("Duplicates found in test_candidates after expansion, ensure uniqueness of (user_id, product_id)")


# Fill missing numeric values with 0 so the model interprets then as no signal
for col in ["als_score", "sim_max", "sim_mean", "sim_topk_mean_10"]:
    if col in test_candidates.columns:
        test_candidates[col] = test_candidates[col].fillna(0.0).astype("float32")

if "sim_nonzero_cnt" in test_candidates.columns:
    test_candidates["sim_nonzero_cnt"] = (test_candidates["sim_nonzero_cnt"].fillna(0).astype("int16"))

print("Merged ALS + similarity features into test candidates: ", 
      f"rows = {len(test_candidates)}, columns = {len(test_candidates.columns)}")

# Merge remaining aggregate features and save outputs
print("Merging additional features into train candidates...")
train_candidates = (train_candidates.merge(user_features, on="user_id", how="left").
                                     merge(product_features, on="product_id", how="left").
                                     merge(up_features, on=["user_id", "product_id"], how="left"))

print("Merging additional features into test candidates...")
test_candidates = (test_candidates.merge(user_features, on="user_id", how="left").
                                     merge(product_features, on="product_id", how="left").
                                     merge(up_features, on=["user_id", "product_id"], how="left"))

# Fill missing numeric feature columns only (do not overwrite IDs or labels)
print("Filling missing values...")

def fill_numeric_features(df, exclude = ("order_id", "user_id", "product_id", "label")):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in exclude]
    if num_cols:
        df[num_cols] = df[num_cols].fillna(0)
    return df

train_candidates = fill_numeric_features(train_candidates)
test_candidates = fill_numeric_features(test_candidates)

print("Train dataset ready for training:", 
      f"rows = {len(train_candidates):,}, columns = {len(train_candidates.columns)}")

# rows = 8,474,611, columns =  32

print("Test dataset ready for inference", 
      f"rows = {len(test_candidates):,}, columns = {len(test_candidates.columns)}")

# rows = 4,833,292, columns =  31

# Final Sanity checks
bad_users = (train_candidates.groupby("user_id")["order_id"].nunique() > 1)
if bad_users.any():
    n_bad = int(bad_users.sum())
    raise AssertionError(f"{n_bad} users have multiple order_ids in train_candidates, expected uniqueness per (user_id, order_id)")

allowed = set(train_candidates["label"].unique())
if not allowed <= {0, 1}:
    raise AssertionError(f"Unexpected label values in train_candidates: {allowed}, expected only 0 and 1")

# Sort train rows by users (within user by order/product)
train_candidates = train_candidates.sort_values(by=["user_id", "order_id", "product_id"]).reset_index(drop=True)

# Sort test rows by users (within user by order/product)
test_candidates = test_candidates.sort_values(by=["user_id", "order_id", "product_id"]).reset_index(drop=True)

# Save outputs
train_out = PROCESSED / "train_candidates.csv"
test_out = PROCESSED / "test_candidates.csv"

print("Saving train candidates...")
train_candidates.to_csv(train_out, index=False)
print("Saving test candidates...")
test_candidates.to_csv(test_out, index=False)

group_sizes = train_candidates.groupby("user_id", sort = False).size().tolist()

# Group size for LambdaRank
group_out = PROCESSED / "train_groups_lambdarank.txt"
with open(group_out, "w") as f:
    f.write('\n'.join(str(int(x)) for x in group_sizes))


print("Saved train candidates to:", train_out)
print("Saved test candidates to:", test_out)

