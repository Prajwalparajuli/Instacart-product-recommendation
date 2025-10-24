### Load raw CSVs into pandas ###

"""Load the six Instacart CSVs from data/raw/"""
import pandas as pd
from pathlib import Path
from utlis import load_config, get_paths, get_data_files, get_dtypes, ensure_dir

# Load configuration
config = load_config()
paths = get_paths(config)
data_files = get_data_files(config)
dtypes = get_dtypes(config)

RAW = Path(paths['raw'])
INTERIM = Path(paths['interim']); ensure_dir(INTERIM)

def load_all():
    # Load data with configured dtypes
    orders = pd.read_csv(RAW / data_files['orders'],
                         dtype=dtypes,
                         keep_default_na=True)
    op_prior = pd.read_csv(RAW / data_files['order_products_prior'],
                           dtype=dtypes)
    op_train = pd.read_csv(RAW / data_files['order_products_train'],
                           dtype=dtypes)
    products = pd.read_csv(RAW / data_files['products'])
    aisles = pd.read_csv(RAW / data_files['aisles'])
    departments = pd.read_csv(RAW / data_files['departments'])
    return {"orders":orders,"op_prior":op_prior,"op_train":op_train,
            "products":products,"aisles":aisles,"departments":departments}

def main(save_interim=False):
    dfs = load_all()
    print({k: v.shape for k, v in dfs.items()})
    if save_interim:
        for name, df in dfs.items():
            output_path = INTERIM / f"{name}.csv"
            df.to_csv(output_path, index=False)
        print(f"Saved cleaned copies to {INTERIM}/")

if __name__ == "__main__":
    main(save_interim=False)  # change to True if you want interim CSVs


