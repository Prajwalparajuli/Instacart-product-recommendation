###train QDA model for reorder probability###

"""Load the six Instacart CSVs from data/raw/ and perform Step 1 merge"""
import pandas as pd
from pathlib import Path


RAW = Path("data/raw")
PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents = True, exist_ok = True)

def split_orders_by_eval_set(orders):
    """
    Splits the orders DataFrame into three separate DataFrames based on the 'eval_set' column:
    - orders_prior: all rows where eval_set == 'prior'
    - orders_train: all rows where eval_set == 'train'
    - orders_test: all rows where eval_set == 'test'
    Only the key columns needed for analysis are kept in each output DataFrame.
    Returns:
        orders_prior, orders_train, orders_test (DataFrames)
    """
        # Define the columns to keep for each split DataFrame
    columns = [
        "order_id", "user_id", "order_number",
        "order_dow", "order_hour_of_day", "days_since_prior_order"
    ]
        # Create 3 separate DataFrames for prior, train, and test sets
    orders_prior = orders[orders["eval_set"] == "prior"][columns]
    orders_train = orders[orders["eval_set"] == "train"][columns]
    orders_test = orders[orders["eval_set"] == "test"][columns]
    return orders_prior, orders_train, orders_test

def load_all():
    """
    Loads all six Instacart CSVs from the data/raw/ directory into pandas DataFrames.
    Performs initial safety checks and prints summary statistics for the orders DataFrame.
    Returns:
        A dictionary containing all loaded DataFrames: orders, op_prior, op_train, products, aisles, departments.
    """
    # Read the orders CSV with memory-efficient dtypes for each column
    orders = pd.read_csv(RAW / "orders.csv",
        dtype = {"order_id":"int32",
            "user_id":"int32",
            "order_number":"int16",
            "order_dow":"int8",
            "order_hour_of_day":"int8"
        },
        keep_default_na=True,
    )
    print("orders loaded. Shape:", orders.shape) 
    print("orders columns:", orders.columns.tolist()) # k is key, v is DataFrame
    orders["eval_set"] = orders["eval_set"].astype(str).str.lower() # set eval_set to lowercase str

    # keep days_since_prior_order as float32: 
    # float32 is because we have a lot of data points/calculations for each model, each one will have a value for each feature. float 32 (others 
    # are float64, each computational number will be stored more efficiently (type of scientific notation)saves memory. Also maintinas accuracy.  
    if "days_since_prior_order" in orders.columns:
        orders["days_since_prior_order"] = pd.to_numeric(
            orders["days_since_prior_order"], errors = "coerce").astype("float32")

    # double checking
    print("\nMissing values in orders:")
    print(orders.isnull().sum()) # how many missing values in each column
    print("Duplicate rows in orders:", orders.duplicated().sum()) # check for duplicate rows
    print("orders dtypes:") # checking type of data in each column
    print(orders.dtypes)
    print("Unique order_id count:", orders['order_id'].nunique()) # how many unique order_ids
    print("Total rows in orders:", len(orders))
    print("Min order_number:", orders['order_number'].min()) # how many orders there are 
    print("Max order_number:", orders['order_number'].max())

# load order_products_prior and order_products_train
    op_prior = pd.read_csv(RAW / "order_products__prior.csv",
                  dtype = {"order_id":"int32",
                      "product_id":"int32",
                      "add_to_cart_order":"int16",
                      "reordered":"int8"},
                  )
    print("op_prior loaded. Shape:", op_prior.shape)
    print("op_prior columns:", op_prior.columns.tolist())

    # Read the train order-products table (line-item level)
    op_train = pd.read_csv(RAW / "order_products__train.csv",
        dtype = {"order_id":"int32",
            "product_id":"int32",
            "add_to_cart_order":"int16",
            "reordered":"int8"},
    )
    print("op_train loaded. Shape:", op_train.shape)
    print("op_train columns:", op_train.columns.tolist())

    products = pd.read_csv(RAW / "products.csv") # load products
    print("products loaded. Shape:", products.shape)
    print("products columns:", products.columns.tolist())

    aisles = pd.read_csv(RAW / "aisles.csv") # load aisles
    print("aisles loaded. Shape:", aisles.shape)
    print("aisles columns:", aisles.columns.tolist())

    departments = pd.read_csv(RAW / "departments.csv") # load departments
    print("departments loaded. Shape:", departments.shape)
    print("departments columns:", departments.columns.tolist())

    return {"orders": orders,
        "op_prior": op_prior,
        "op_train": op_train,
        "products": products,
        "aisles": aisles,
        "departments": departments}

def build_prior_line_items(dfs):
    """
    Splits the orders DataFrame into prior, train, and test sets, then merges prior orders with order_products__prior.csv
    to create a line-item table (prior) with user and order context for each product in prior orders.
    Also prints safety checks and previews for each split DataFrame and the merged result.
    Args:
        dfs: Dictionary of loaded DataFrames (from load_all)
    Returns:
        prior: DataFrame of merged prior line-items
        orders_prior, orders_train, orders_test: split DataFrames for further analysis
    """
    orders = dfs["orders"]
    op_prior = dfs["op_prior"]

    # Split the orders DataFrame into prior, train, and test sets using the helper function
    orders_prior, orders_train, orders_test = split_orders_by_eval_set(orders)

    # Now that orders_prior is defined, print safety checks and previews
    print("\nMissing values in orders_prior:")
    print(orders_prior.isnull().sum())
    print("Duplicate rows in orders_prior:", orders_prior.duplicated().sum())
    print("orders_prior dtypes:")
    print(orders_prior.dtypes)
    print("Unique order_id count in orders_prior:", orders_prior['order_id'].nunique())
    print("Total rows in orders_prior:", len(orders_prior))
    print("Min order_number in orders_prior:", orders_prior['order_number'].min())
    print("Max order_number in orders_prior:", orders_prior['order_number'].max())

    # Print shapes and preview the first few rows of each split DataFrame for verification
    print("orders_prior shape:", orders_prior.shape)
    print(orders_prior.head())
    print("orders_train shape:", orders_train.shape)
    print(orders_train.head())
    print("orders_test shape:", orders_test.shape)
    print(orders_test.head())

    # Merge the prior orders with the prior order-products table to create a line-item table
    # This adds user and order context to each product in prior orders
    prior = op_prior.merge(
        orders_prior,
        on = "order_id",
        how = "inner"
    )
    print("prior shape:", prior.shape)
    print(prior.head())
    print("prior columns:", prior.columns.tolist())

    return prior, orders_prior, orders_train, orders_test

# load all CSVs
def main(save_processed = False):
    # op_train contains all products actually bought in train orders
    # We merge with orders_train to get user_id for each order
    train_labels = op_train.merge(
        orders_train[["order_id", "user_id"]],
        on="order_id",
        how="left"
    )
    # Assign label=1 for these actual purchases
    train_labels["label"] = 1
    # Only keep necessary columns for labeling
    train_labels = train_labels[["order_id", "user_id", "product_id", "label"]]

    # For each train order, we want to predict which products the user will buy
    # To do this, we consider all products the user has bought in prior orders as candidates
    op_train = dfs["op_train"]
    # user_prior_products: dict mapping user_id to array of product_ids they've bought before
    user_prior_products = prior.groupby("user_id")["product_id"].unique()
    candidate_rows = []
    # Loop through each train order
    for _, order in orders_train.iterrows():
        user_id = order["user_id"]
        order_id = order["order_id"]
        # Context features for this order (day of week, hour, days since prior)
        context = order[["order_dow", "order_hour_of_day", "days_since_prior_order"]]
        # Get all prior products for this user
        products = user_prior_products.get(user_id, [])
        # For each prior product, create a candidate row
        for product_id in products:
            row = {
                "order_id": order_id,
                "user_id": user_id,
                "product_id": product_id,
                "order_dow": context["order_dow"],
                "order_hour_of_day": context["order_hour_of_day"],
                "days_since_prior_order": context["days_since_prior_order"]
            }
            candidate_rows.append(row)
    # All candidate rows for all train orders
    candidates = pd.DataFrame(candidate_rows)

    # Merge with train_labels: if candidate matches an actual purchase, label=1, else label=0
    candidates = candidates.merge(
        train_labels,
        on=["order_id", "user_id", "product_id"],
        how="left"
    )
    # Fill missing labels (not actual purchase) with 0
    candidates["label"] = candidates["label"].fillna(0).astype("int8")

    # Add user-level features
    candidates = candidates.merge(user_features, on="user_id", how="left")
    # Add product-level features
    candidates = candidates.merge(product_features, on="product_id", how="left")
    # Add user-product interaction features
    candidates = candidates.merge(up_features, on=["user_id", "product_id"], how="left")

    # Print shape and first few rows for inspection
    print("Train candidates shape:", candidates.shape)
    print(candidates.head())
    # Build product features from prior only
    product_grp = prior.groupby("product_id")
    product_features = pd.DataFrame({
        "product_id": product_grp["product_id"].first(),
        "p_total_purchases": product_grp["order_id"].count(),
        "p_distinct_users": product_grp["user_id"].nunique(),
        "p_avg_add_to_cart": product_grp["add_to_cart_order"].mean(),
        "p_reorder_prob": product_grp["reordered"].mean().fillna(0)
    })
    product_features.to_csv(PROCESSED / "ql_product_features.csv", index=False)
    print("Saved product features to data/processed/ql_product_features.csv")
    print(product_features.head())

    # Build user-product features from prior only
    up_grp = prior.groupby(["user_id", "product_id"])
    up_features = pd.DataFrame({
        "user_id": up_grp["user_id"].first(),
        "product_id": up_grp["product_id"].first(),
        "up_times_bought": up_grp["order_id"].count(),
        "up_last_order_number": up_grp["order_number"].max(),
        "up_first_order_number": up_grp["order_number"].min(),
        "up_avg_add_to_cart": up_grp["add_to_cart_order"].mean()
    })
    # Compute recency
    u_last_prior_order = prior.groupby("user_id")["order_number"].max()
    up_features["up_recency"] = up_features.apply(
        lambda row: u_last_prior_order[row["user_id"]] - row["up_last_order_number"], axis=1
    )
    # Compute rate in user history
    user_total_bought = up_features.groupby("user_id")["up_times_bought"].transform("sum")
    up_features["up_rate_in_user_history"] = up_features["up_times_bought"] / user_total_bought
    up_features.to_csv(PROCESSED / "ql_user_product_features_prior.csv", index=False)
    print("Saved user-product features to data/processed/ql_user_product_features_prior.csv")
    print(up_features.head())
    """
    Main workflow for the script. Loads all data, prints summary, builds core tables, and optionally saves processed results.
    Args:
        save_processed (bool): If True, saves the processed DataFrames to disk.
    """
    dfs = load_all()
    print("\nSummary of loaded DataFrames:")
    for k, v in dfs.items():
        print(f"{k}: shape = {v.shape}, columns = {v.columns.tolist()}") # k is key, v is DataFrame
    # merge prior orders with prior products

    prior, orders_prior, orders_train, orders_test = build_prior_line_items(dfs)

    # Build user features from prior only
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
    user_features.to_csv(PROCESSED / "ql_user_features.csv", index=False)
    print("Saved user features to data/processed/ql_user_features.csv")
    print(user_features.head())

    if save_processed:
        prior.to_csv(PROCESSED / "prior_line_items.csv", index = False)
        orders_prior.to_csv(PROCESSED / "orders_prior.csv", index = False)
        orders_train.to_csv(PROCESSED / "orders_train.csv", index = False)
        orders_test.to_csv(PROCESSED / "orders_test.csv", index = False)
        print(f"Saved processed prior line items to {PROCESSED / 'prior_line_items.csv'}")
        print("Saved orders_prior.csv, orders_train.csv, orders_test.csv to processed directory.")
            
#control when your code runs, only when you run the file itself, not when other code is borrowing it
if __name__ == "__main__":
    main(save_processed = False)