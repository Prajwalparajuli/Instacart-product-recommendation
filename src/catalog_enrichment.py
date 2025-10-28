import pandas as pd
import json
from pathlib import Path

# Load base files
products = pd.read_csv(Path("data/raw/products.csv"))
departments = pd.read_csv(Path("data/raw/departments.csv"))
index = json.load(open("assets/index.json",  "r", encoding="utf-8"))

# Merge products with departments to get department names
df = products.merge(departments, on = "department_id", how = "left")

# Attach image paths
def get_img(dept):
    dept = (dept or "missing").lower().strip()
    return index.get(dept, index["missing"])

df["dept_thumbnail_url"] = df["department"].apply(get_img)

# Save processed file
Path("data/processed").mkdir(parents=True, exist_ok=True)
df.to_parquet("data/processed/catalog_with_dept_images.parquet", index = False)

print("Catalog_with_dept_images.parquet created successfully with", len(df), "records.")