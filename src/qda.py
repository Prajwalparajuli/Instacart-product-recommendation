###train QDA model for reorder probability###

"""Load the six Instacart CSVs from data/raw/ and perform Step 1 merge"""
import pandas as pd
from pathlib import Path


RAW = Path("data/raw")
INTERIM = Path("data/interim"); INTERIM.mkdir(parents=True, exist_ok=True)

def load_all():
    """Load all six Instacart CSVs with memory-friendly dtypes."""
    orders = pd.read_csv(RAW / "orders.csv",
                         dtype={"order_id":"int32","user_id":"int32","order_number":"int16",
                                "order_dow":"int8","order_hour_of_day":"int8"},
                         keep_default_na=True)
    op_prior = pd.read_csv(RAW / "order_products__prior.csv",
                           dtype={"order_id":"int32","product_id":"int32",
                                  "add_to_cart_order":"int16","reordered":"int8"})
    op_train = pd.read_csv(RAW / "order_products__train.csv",
                           dtype={"order_id":"int32","product_id":"int32",
                                  "add_to_cart_order":"int16","reordered":"int8"})
    products = pd.read_csv(RAW / "products.csv")
    aisles = pd.read_csv(RAW / "aisles.csv")
    departments = pd.read_csv(RAW / "departments.csv")
    return {"orders":orders,"op_prior":op_prior,"op_train":op_train,
            "products":products,"aisles":aisles,"departments":departments}

def first_merge_prior(dfs):
    """Merge orders (eval_set = 'prior') with order_products_prior. """
    orders = dfs["orders"]
    op_prior = dfs["op_prior"]
    
    # filter orders to only 'prior'
    eval_col = orders["eval_set"] 
    prior_only = eval_col == "prior"
    orders_prior = orders[prior_only]
    
    # merges orders with porducts details from order_products_prior.csv
    prior_merged = op_prior.merge(
        orders_prior,
        on = "order_id",
        how = "inner"
    )
    return prior_merged

# load all CSVs
def main(save_interim = False):
    dfs = load_all()
    print("Shape of raw DataFrames:", {k: v.shape for k, v in dfs.items()})
    # merge prior orders with prior products
    prior_merged = first_merge_prior(dfs)
    print("prior_merged_shapes:", prior_merged.shape)
    print(prior_merged.head())
    
    # combined data set to use for ML
    if save_interim:
        for name, df in dfs.items():
            df.to_csv(INTERIM / f"{name}.csv", index = False)
            prior_merged.to_csv(INTERIM / "prior_merged.csv", index=False)
            print("Saved interim copies inclduing prior_merged.csv")
            
if __name__ == "__main__":
    main(save_interim = False)