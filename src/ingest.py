### Load raw CSVs into pandas ###

"""Load the six Instacart CSVs from data/raw/"""
import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
INTERIM = Path("data/interim"); INTERIM.mkdir(parents=True, exist_ok=True)

def load_all():
    # Minimal dtypes to keep memory reasonable
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

def main(save_interim=False):
    dfs = load_all()
    print({k: v.shape for k, v in dfs.items()})
    if save_interim:
        for name, df in dfs.items():
            df.to_csv(INTERIM / f"{name}.csv", index=False)
        print("Saved cleaned copies to data/interim/")

if __name__ == "__main__":
    main(save_interim=False)  # change to True if you want interim CSVs
