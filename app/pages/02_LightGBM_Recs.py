# Streamlit page for LightGBM LambdaRank Top-K recommendations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import streamlit as st

# === Page config and title ===
st.set_page_config(page_title = "LightGBM Insights Dashboard",
                   layout = "wide")

# Custom CSS for modern look
st.markdown("""
<style>
    .model-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    .stat-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="model-card"><h2>🎯 LightGBM Insights Dashboard</h2><p>Explore personalized product rankings powered by LambdaRank</p></div>', unsafe_allow_html=True)

# === Paths ===
ARTIFACT_DIR = Path("models/lightgbm_ranker")
DATA_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")

# === Cached Loaders ===
def load_model_and_meta():
    model_path = ARTIFACT_DIR / "lgbm_lambdarank.txt"
    if not model_path.exists():
        st.error(f"Missing Model file!")
        st.stop()
    booster = lgb.Booster(model_file = str(model_path))
    fcols_path = ARTIFACT_DIR/ "feature_cols.json"
    if not fcols_path.exists():
        st.error(f"Missing feature column list at {fcols_path}. Please save feature cols after training.")
        st.stop()
    with open(fcols_path) as f:
        feature_cols = json.load(f)
    schema_path = ARTIFACT_DIR / "schema.json"
    if not schema_path.exists():
        # If schema does not exist, fallback to default values
        schema = {"group_key" : "order_id",
                  "id_cols" : ["user_id", "order_id", "product_id"],
                  "k" : 10}
    else: 
        with open(schema_path) as f:
            schema = json.load(f)
    
    return booster, feature_cols, schema

@st.cache_data
def load_candidates():
    pq = DATA_DIR / "test_candidates.parquet"
    csv = DATA_DIR / "test_candidates.csv"
    if pq.exists():
        df = pd.read_parquet(pq)
    elif csv.exists():
        df = pd.read_csv(csv)
    else:
        st.error("No candidate data found. " \
        "Expected data/processed/test_candidates.parquet " \
        "or " \
        "data/processed/test_candidates.csv")
        st.stop()

    # Ensure required ID columns are present
    required_id_cols = {"user_id", "order_id", "product_id"}
    missing = required_id_cols - set(df.columns)
    if missing:
        st.error(f"Candidates are missing required ID columns: {missing}")
        st.stop()
    
    return df

@st.cache_data
def load_catalog():
    # Merge product name, aisle and department
    product_path = RAW_DIR / "products.csv"
    aisles_path = RAW_DIR / "aisles.csv"
    department_path = RAW_DIR / "departments.csv"

    if not product_path.exists():
        st.error("Product data not found!")
        st.stop()
    products = pd.read_csv(product_path, dtype = {"product_id" : np.int64})

    if aisles_path.exists():
        aisles = pd.read_csv(aisles_path, dtype = {"aisle_id" : np.int64})
        products = pd.merge(products, aisles, on = "aisle_id", how = "left")

    if department_path.exists():
        departments = pd.read_csv(department_path, dtype = {"department_id" : np.int64})
        products = pd.merge(products, departments, on = "department_id", how = "left")

    # Standardize the column names 
    if "product_name" not in products.columns:
        for cand in ["name", "product", "title"]:
            if cand in products.columns:
                products = products.rename(columns = {cand : "product_name"})
                break

    return products

# Scoring helpers
def score_df(df: pd.DataFrame, booster: lgb.Booster, feature_cols, group_key:str):
    # Ensure the feature columns exists
    missing_feats = [c for c in feature_cols if c not in df.columns]
    if missing_feats:
        st.error(f"Candidate dataframe missing feature columns: {missing_feats[:20]}...")
        st.stop()

    # Predict using best iteration if available
    num_iter = booster.current_iteration() if booster.current_iteration() > 0 else None

    scores = booster.predict(df[feature_cols], num_iteration = num_iter)
    out = df[["user_id", "order_id", "product_id"]].copy()
    out["score"] = scores
    out["rank"] = out.groupby(group_key)["score"].rank(ascending = False, method = "first")

    return out
    
def topk_for_user_and_order(candidates: pd.DataFrame, 
                            booster: lgb.Booster, 
                            feature_cols, 
                            group_key:str, user_id:int, order_id:int, k:int):
    sub = candidates[(candidates["user_id"] == user_id) &
                     (candidates[group_key] == order_id)].copy()
    if sub.empty:
        return pd.DataFrame(columns = ["user_id", "order_id", "product_id", "score", "rank"])
    ranked = score_df(sub, booster, feature_cols, group_key)
    return ranked[ranked["rank"] <= k].sort_values([group_key, "rank"]).reset_index(drop = True)

def attach_catalog(df, catalog):
    if catalog is None or df.empty:
        return df
    cols = ["product_id"]
    if "product_name" in catalog.columns:
        cols.append("product_name")
    for cand in ["aisle", "aisle_id"]:
        if cand in catalog.columns:
            cols.append(cand)
            break
    for cand in ["department", "department_id"]:
        if cand in catalog.columns:
            cols.append(cand)
            break
    return df.merge(catalog[cols].drop_duplicates("product_id"), on = "product_id", how = "left")


# === Load artifacts and data ======
booster, feature_cols, schema = load_model_and_meta()
candidates = load_candidates()
catalog = load_catalog()

# Merge catalog info (department, aisle, product_name) into candidates
candidates = attach_catalog(candidates, catalog)

group_key = schema.get("group_key", "order_id")
id_cols = schema.get("id_cols", ["user_id", "order_id", "product_id"])
default_k = int(schema.get("k", 10))

# Get cart from query params if available
qp_cart = st.query_params.get("cart")
cart_items = []
if qp_cart:
    try:
        cart_items = [int(x) for x in qp_cart.split(",") if x.strip().isdigit()]
    except:
        pass

# === TWO-COLUMN LAYOUT ===
left_col, right_col = st.columns([1, 2])

with left_col:
    st.markdown("### ⚙️ Controls")
    
    # Query param handoff: preselect a user via user_id
    qp_user = st.query_params.get("user_id")
    users = candidates["user_id"].drop_duplicates().sort_values().tolist()
    
    if qp_user and str(qp_user).isdigit() and int(qp_user) in users:
        default_user_index = users.index(int(qp_user))
    else:
        default_user_index = 0 if users else 0
    
    user_id = st.selectbox("👤 User ID", users, index = default_user_index)
    
    # Let user choose which order to rank
    orders_for_user = (candidates.loc[candidates["user_id"] == user_id, group_key]
                       .drop_duplicates()
                       .sort_values(ascending=False)
                       .tolist())
    if orders_for_user:
        default_order_index = len(orders_for_user) - 1
    else:
        default_order_index = 0
    
    order_id = st.selectbox("📦 Order", orders_for_user, index = default_order_index)
    
    k = st.slider("📊 Top k", min_value = 5, max_value = 30, value = default_k, step = 1)
    
    # Hide items in cart toggle
    hide_cart = st.checkbox("🛒 Hide items in cart", value=False)
    
    # Department filter
    dept_filter_col = None
    for cand in ["department", "department_id","dept_id"]:
        if cand in catalog.columns:
            dept_filter_col = cand
            break
    
    if dept_filter_col:
        depts = candidates.loc[candidates["user_id"] == user_id, dept_filter_col].dropna().astype(str).unique().tolist()
        depts = sorted(depts)
        selected_dept = st.selectbox("🏷️ Department", depts, index=0)
    else:
        selected_dept = None
    
    # Aisle filter
    aisle_filter_col = None
    for cand in ["aisle", "aisle_id"]:
        if cand in catalog.columns:
            aisle_filter_col = cand
            break
    
    if aisle_filter_col:
        aisles = candidates.loc[candidates["user_id"] == user_id, aisle_filter_col].dropna().astype(str).unique().tolist()
        aisles = sorted(aisles)
        selected_aisle = st.multiselect("🏪 Aisles", aisles, default = [])
    else:
        selected_aisle = []
    
    # Back to shop button
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🛍️ Back to Shop", use_container_width=True, type="primary"):
        st.switch_page("app_streamlit.py")

with right_col:
    st.markdown("### 📈 Model Snapshot")
    st.markdown('<div class="stat-box">', unsafe_allow_html=True)
    
    snap_cols = st.columns(3)
    with snap_cols[0]:
        st.metric("Candidates Scored", len(candidates[(candidates["user_id"] == user_id) & (candidates[group_key] == order_id)]))
    with snap_cols[1]:
        st.metric("Features Used", len(feature_cols))
    with snap_cols[2]:
        st.metric("Best Iteration", booster.current_iteration())
    
    st.markdown('</div>', unsafe_allow_html=True)

# Apply optional filtering before scoring
user_order_mask = (candidates["user_id"] == user_id) & (candidates[group_key] == order_id)
sub = candidates[user_order_mask].copy()

# Filter out cart items if toggle is on
if hide_cart and cart_items:
    sub = sub[~sub["product_id"].isin(cart_items)]

if selected_dept and dept_filter_col and dept_filter_col in sub.columns:
    sub = sub[sub[dept_filter_col].astype(str) == selected_dept]

if selected_aisle and aisle_filter_col and aisle_filter_col in sub.columns:
    sub = sub[sub[aisle_filter_col].astype(str).isin(selected_aisle)]

# === Rank and display ===
if sub.empty:
    st.info("No candidates available for this selection.")
    st.stop()

ranked = score_df(sub, booster, feature_cols, group_key)
topk = ranked[ranked["rank"] <= k].sort_values([group_key, "rank"]).reset_index(drop=True)
topk_named = attach_catalog(topk, catalog)

st.markdown("---")
st.subheader(f"🎯 Top {k} Recommendations for User {user_id}")
st.info("ℹ️ **About Scores:** LightGBM LambdaRank produces raw scores (can be negative). Only the **relative ranking** matters - higher scores are better predictions, regardless of absolute values.")

# Display as table
st.dataframe(topk_named[["rank", "product_id"] + ([c for c in ["product_name"] if c in topk_named.columns]) + 
                        ([c for c in ["department", "department_id"] if c in topk_named.columns]) +
                        ([c for c in ["aisle", "aisle_id"] if c in topk_named.columns]) +
                        ["score"]], use_container_width=True, hide_index=True)

# Download CSV
st.download_button(label = "📥 Download as CSV", 
                   data = topk_named.to_csv(index = False).encode("utf-8"),
                   file_name = f"user_{user_id}_order_{order_id}_top_{k}.csv",
                   mime = "text/csv")

# Feature Importance Chart
with st.expander("🔍 Why these picks? (Feature Importance)"):
    st.markdown("**Top features driving model predictions:**")
    
    try:
        importance = booster.feature_importance(importance_type='gain')
        if len(importance) == len(feature_cols):
            importance_df = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': importance
            }).sort_values('Importance', ascending=False).head(20)
            
            st.bar_chart(importance_df.set_index('Feature'))
            st.caption("📊 Showing top 20 features by gain (information gain from splits)")
        else:
            st.warning("Feature importance dimensions mismatch")
    except Exception as e:
        st.warning(f"Could not generate feature importance: {e}")

# Diagnostics
with st.expander("🔧 Diagnostics"):
    st.write("**Model Info:**")
    st.write(f"- Best iteration: {booster.current_iteration()}")
    st.write(f"- Candidates scored: {len(sub)}")
    st.write(f"- Features used: {len(feature_cols)}")
    st.write(f"- First 10 features: {feature_cols[:10]}")
    
    missing_feats = [c for c in feature_cols if c not in sub.columns]
    if missing_feats:
        st.warning(f"Missing features in data: {missing_feats[:20]}")
    
    if cart_items:
        st.write(f"**Cart items:** {len(cart_items)} products")
        if hide_cart:
            st.caption("✅ Cart items hidden from recommendations")


