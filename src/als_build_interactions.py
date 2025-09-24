import os
import pandas as pd
from ingest import load_all

# Define paths
OUT_DIR = r"data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

# Load csv via ingest.py
dfs = load_all()
orders = dfs["orders"]
prior = dfs["op_prior"]
products = dfs["products"]

# Filter to prior orders only
order_prior = orders.loc[orders["eval_set"].astype(str) == str.lower("prior"), ["order_id", "user_id"]]

# Joint to get (user_id, product_id) pairs for all prior orders
up = prior.merge(order_prior, on="order_id", how="inner")[["user_id", "product_id"]]    

# Build implicit feedback 
interactions = (up.groupby(["user_id", "product_id"], as_index=False)
                .size()
                .rename(columns={"size":"count"})) 

# Create index mappings
unique_users = pd.DataFrame({"user_id": interactions['user_id'].drop_duplicates().sort_values()})
unique_users['user_index'] = range(len(unique_users))

unique_prods = pd.DataFrame({"product_id": interactions['product_id'].drop_duplicates().sort_values()})
unique_prods['product_index'] = range(len(unique_prods))

# Map indices onto interactions
inter_idx = (interactions.merge(unique_users, on="user_id", how="left")
             .merge(unique_prods, on="product_id", how="left")
                [['user_id', 'product_id', 'user_index', 'product_index', 'count']]).sort_values(
                    ['user_index', 'count'], ascending= [True, False]).reset_index(drop=True)

# Export to CSV
interactions_out = os.path.join(OUT_DIR, "als_interaction_user_product_counts.csv")
user_map_out = os.path.join(OUT_DIR, "als_user_index_map.csv")
prod_map_out = os.path.join(OUT_DIR, "als_product_index_map.csv")
coo_out = os.path.join(OUT_DIR, "als_coo.csv")

inter_idx.to_csv(interactions_out, index=False)
unique_users.to_csv(user_map_out, index=False)  
unique_prods.to_csv(prod_map_out, index=False)

coo = inter_idx[['user_index', 'product_index', 'count']] \
    .rename(columns={"user_index":"row", "product_index":"col", "count":"val"})
coo.to_csv(coo_out, index=False)

# STATS
print("ALS interaction built using ingest.load_all()")
print(f" Users: {len(unique_users) }")
print(f" Products: {len(unique_prods) }")
print(f"Pairs: {len(inter_idx) }")
print("Files written:")
print(f" {interactions_out}")
print(f" {user_map_out}")
print(f" {prod_map_out}")
print(f" {coo_out}")
