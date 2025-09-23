"""Basic integrity checks on raw CSVs."""
import pandas as pd
from pathlib import Path

RAW = Path("data/raw")

def check_foreign_keys():
    orders = pd.read_csv(RAW / "orders.csv", usecols=["order_id","user_id","order_number","days_since_prior_order"])
    products = pd.read_csv(RAW / "products.csv", usecols=["product_id"])
    op_prior = pd.read_csv(RAW / "order_products__prior.csv", usecols=["order_id","product_id"])
    op_train = pd.read_csv(RAW / "order_products__train.csv", usecols=["order_id","product_id"])

    order_ids = set(orders.order_id.unique())
    product_ids = set(products.product_id.unique())

    missing_prior_orders = (~op_prior.order_id.isin(order_ids)).sum()
    missing_train_orders = (~op_train.order_id.isin(order_ids)).sum()
    missing_prior_products = (~op_prior.product_id.isin(product_ids)).sum()
    missing_train_products = (~op_train.product_id.isin(product_ids)).sum()

    assert missing_prior_orders == 0, f"op_prior has {missing_prior_orders} order_id not in orders"
    assert missing_train_orders == 0, f"op_train has {missing_train_orders} order_id not in orders"
    assert missing_prior_products == 0, f"op_prior has {missing_prior_products} product_id not in products"
    assert missing_train_products == 0, f"op_train has {missing_train_products} product_id not in products"
    print("Foreign key checks passed ✔")

def check_days_since_prior_order_rule():
    orders = pd.read_csv(RAW / "orders.csv", usecols=["order_number","days_since_prior_order"])
    bad = orders.loc[orders["order_number"] == 1, "days_since_prior_order"].notna().sum()
    assert bad == 0, "days_since_prior_order must be NaN for order_number == 1"
    print("days_since_prior_order rule passed ✔")

if __name__ == "__main__":
    check_foreign_keys()
    check_days_since_prior_order_rule()
    print("Validation passed ✅")
