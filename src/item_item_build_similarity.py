import os
import pandas as pd
from collections import defaultdict
from itertools import combinations
from math import sqrt

# Import load_all from ingest.py
from ingest import load_all

# Define paths
OUT_DIR = r"data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

# Load csv via ingest.py
dfs = load_all()
orders = dfs["orders"]
prior = dfs["op_prior"]

# Filter to prior orders only and attach user_id
orders_prior = orders.loc[orders["eval_set"].astype(str) == str.lower("prior"), ["order_id", "user_id"]]
baskets = prior.merge(orders_prior, on="order_id", how="inner")[["order_id", "user_id", "product_id"]]

# Product counts
item_counts = (baskets.groupby("product_id")
               .size().reset_index(name="count")
               .sort_values("count", ascending=False))

# Export item counts
baskets_out = os.path.join(OUT_DIR, "item_item_basket.csv")
counts_out = os.path.join(OUT_DIR, "item_item_product_counts.csv")

baskets.to_csv(baskets_out, index=False)
item_counts.to_csv(counts_out, index=False)

# Stats
print("Item-Item preprocessing using ingest.load_all()")
print(f"Orders: {baskets['order_id'].nunique()}")
print(f" Users: {baskets['user_id'].nunique() }")
print(f" Products: {baskets['product_id'].nunique() }")
print(f"Rows: {len(baskets) }")
print("Files written:")
print(f" {baskets_out}")
print(f" {counts_out}")