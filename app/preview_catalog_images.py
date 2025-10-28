import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Dept Image Preview", layout="wide")

DF_PATH = "data/processed/catalog_with_dept_images.parquet"
FALLBACK_IMG = "assets/thumbnails/missing.jpg"   # ensure this exists

df = pd.read_parquet(DF_PATH)

# Pick a sensible title column
TITLE_CANDIDATES = ["name", "product_name", "primary", "title"]
TITLE_COL = next((c for c in TITLE_CANDIDATES if c in df.columns), None)
if TITLE_COL is None:
    # last resort: synthesize a name
    df["__title__"] = df.get("product_id", pd.Series(range(len(df)))).map(lambda x: f"Product {x}")
    TITLE_COL = "__title__"

# Normalize department text a bit for display
if "department" in df.columns:
    df["department"] = df["department"].astype(str).str.strip()

# Ensure path column exists and uses forward slashes
IMG_COL = "dept_thumbnail_url" if "dept_thumbnail_url" in df.columns else None
if IMG_COL is None:
    st.error("Column 'dept_thumbnail_url' not found. Rebuild parquet after mapping department images.")
    st.stop()
df[IMG_COL] = df[IMG_COL].astype(str).str.replace("\\", "/", regex=False)

st.title("Department Images • Preview")

# Sidebar filters
departments = ["All"] + sorted(df["department"].dropna().unique().tolist()) if "department" in df.columns else ["All"]
selected = st.sidebar.selectbox("Department", departments)
n = st.sidebar.slider("How many products?", 12, 120, 36, 12)

view = df if selected == "All" else df[df["department"] == selected]

cols = st.columns(6)
for i, (_, row) in enumerate(view.head(n).iterrows()):
    path = row[IMG_COL]
    if not path or not Path(path).exists():
        path = FALLBACK_IMG
    with cols[i % 6]:
        st.image(path)
        st.markdown(f"**{row[TITLE_COL]}**")
        if "department" in row:
            st.caption(row["department"])
