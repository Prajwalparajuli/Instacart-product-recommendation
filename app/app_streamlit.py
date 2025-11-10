import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image
from typing import List
import lightgbm as lgb
import json
import numpy as np


st.set_page_config(page_title="InstaRec - Your Grocery Store", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for ultra-modern, creative styling with purchase history badges
st.markdown("""
<style>
    /* Main app background with animated gradient */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8eef5 50%, #f0f4f8 100%);
        background-size: 200% 200%;
        animation: gradient 15s ease infinite;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Reduce main padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }
    
    /* Ultra-modern header with gradient, shadow, and shine effect */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #667eea 100%);
        background-size: 200% auto;
        padding: 0.7rem 1.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.4), 0 2px 8px rgba(0, 0, 0, 0.1);
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
        transform: rotate(45deg);
        animation: shine 3s infinite;
    }
    @keyframes shine {
        0% { transform: translateX(-100%) rotate(45deg); }
        100% { transform: translateX(100%) rotate(45deg); }
    }
    .main-header h3 {
        margin: 0;
        font-size: 1.6rem;
        position: relative;
        z-index: 1;
    }
    
    /* Purchase history badges - Creative overlay indicators */
    .purchase-badge {
        position: absolute;
        top: 8px;
        right: 8px;
        z-index: 10;
        display: flex;
        gap: 4px;
        flex-direction: column;
        align-items: flex-end;
    }
    
    .badge-bought {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 700;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.4);
        animation: pulse 2s infinite;
        display: flex;
        align-items: center;
        gap: 4px;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    .badge-reorder {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 3px 8px;
        border-radius: 10px;
        font-size: 0.65rem;
        font-weight: 600;
        box-shadow: 0 2px 6px rgba(245, 158, 11, 0.4);
    }
    
    .badge-frequent {
        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
        color: white;
        padding: 3px 8px;
        border-radius: 10px;
        font-size: 0.65rem;
        font-weight: 600;
        box-shadow: 0 2px 6px rgba(139, 92, 246, 0.4);
        animation: glow 2s infinite alternate;
    }
    
    @keyframes glow {
        0% { box-shadow: 0 2px 6px rgba(139, 92, 246, 0.4); }
        100% { box-shadow: 0 2px 12px rgba(139, 92, 246, 0.8); }
    }
    
    /* Product Grid - Perfect Alignment */
    .product-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
        gap: 1.5rem;
        margin: 1.5rem 0;
    }
    
    /* Ultra-modern product card */
    .product-card {
        border: 2px solid rgba(255, 255, 255, 0.6);
        border-radius: 20px;
        padding: 0;
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        height: 100%;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08), 0 1px 3px rgba(0, 0, 0, 0.05);
        position: relative;
        overflow: hidden;
        display: flex;
        flex-direction: column;
    }
    
    .product-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2, #667eea);
        background-size: 200% auto;
        animation: borderGlow 3s linear infinite;
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .product-card:hover::before {
        opacity: 1;
    }
    
    @keyframes borderGlow {
        0% { background-position: 0% center; }
        100% { background-position: 200% center; }
    }
    
    .product-card:hover {
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.3), 0 4px 12px rgba(0, 0, 0, 0.1);
        transform: translateY(-8px) scale(1.02);
        border-color: rgba(102, 126, 234, 0.5);
    }
    
    .product-image-container {
        position: relative;
        padding: 1rem;
        background: linear-gradient(135deg, #fafbfc 0%, #f0f2f5 100%);
        border-radius: 18px 18px 0 0;
    }
    
    .product-details {
        padding: 1rem;
        flex: 1;
        display: flex;
        flex-direction: column;
    }
    
    .product-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 0.5rem;
        line-height: 1.4;
        min-height: 2.8rem;
    }
    
    .product-dept {
        font-size: 0.8rem;
        color: #6b7280;
        padding: 0.25rem 0.6rem;
        background: rgba(102, 126, 234, 0.1);
        border-radius: 8px;
        display: inline-block;
        margin-bottom: 0.8rem;
    }
    
    /* Cart item with creative design */
    .cart-item {
        border: 3px solid transparent;
        border-radius: 20px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        background: linear-gradient(white, white) padding-box,
                    linear-gradient(135deg, #667eea, #764ba2, #667eea) border-box;
        background-size: 200% auto;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }
    
    .cart-item::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(102, 126, 234, 0.1), transparent 70%);
        animation: rotate 10s linear infinite;
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .cart-item:hover::before {
        opacity: 1;
    }
    
    @keyframes rotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .cart-item:hover {
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.25);
        transform: translateY(-4px);
        background-position: 100% center;
    }
    
    /* Recommendation section with creative styling */
    .rec-section {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
        padding: 1.5rem;
        border-radius: 20px;
        border-left: 6px solid #667eea;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.06);
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    
    .rec-section::after {
        content: '';
        position: absolute;
        right: -50px;
        bottom: -50px;
        width: 150px;
        height: 150px;
        background: radial-gradient(circle, rgba(102, 126, 234, 0.1), transparent);
        border-radius: 50%;
    }
    
    /* Section headers with creative effects */
    .section-header {
        color: #1f2937;
        font-size: 1.8rem;
        font-weight: 800;
        margin: 2rem 0 1.2rem 0;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        position: relative;
        padding-bottom: 0.8rem;
    }
    
    .section-header::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 60px;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 2px;
    }
    
    /* Quick picks with ultra-modern design */
    .quick-picks-container {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(255, 255, 255, 0.7) 100%);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 2rem;
        margin-bottom: 2.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1), 0 2px 8px rgba(0, 0, 0, 0.05);
        border: 2px solid rgba(255, 255, 255, 0.8);
        position: relative;
    }
    
    .quick-picks-container::before {
        content: '✨';
        position: absolute;
        top: 1.5rem;
        right: 2rem;
        font-size: 2rem;
        opacity: 0.3;
        animation: float 3s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    /* Stats box with modern gradient */
    .stat-box {
        background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);
        padding: 1rem;
        border-radius: 16px;
        text-align: center;
        border: 2px solid rgba(14, 165, 233, 0.2);
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.15);
        transition: all 0.3s ease;
    }
    
    .stat-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(14, 165, 233, 0.25);
    }
    
    /* Enhanced buttons */
    .stButton button {
        border-radius: 14px;
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        font-weight: 600;
    }
    
    .stButton button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
    }
    
    /* Loading animation */
    .spinner {
        border: 4px solid rgba(102, 126, 234, 0.1);
        border-top: 4px solid #667eea;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        animation: spin 0.8s linear infinite;
        margin: 3rem auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Ensure equal heights in Streamlit columns */
    [data-testid="column"] > div {
        height: 100%;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

CATALOG_PATH = Path("data/processed/catalog_with_dept_images.parquet")
THUMBNAILS_DIR = Path("assets/thumbnails")
FALLBACK_IMG = THUMBNAILS_DIR / "missing.jpg"
USER_IDS = [101, 318, 451, 999, 3678, 5890, 45678, 56789, 67890, 89012]

# LightGBM Model paths
ARTIFACT_DIR = Path("models/lightgbm_ranker")
MODEL_FILE = ARTIFACT_DIR / "lgbm_lambdarank.txt"
FEATURE_COLS_FILE = ARTIFACT_DIR / "feature_cols.json"
# Use demo dataset for deployment (switch to test_candidates.csv for full dataset)
CANDIDATES_PATH = Path("data/processed/test_candidates_demo.csv")

# Purchase history paths
ORDERS_PATH = Path("data/raw/orders.csv")
ORDER_PRODUCTS_PATH = Path("data/raw/order_products__prior.csv")
PRODUCTS_PATH = Path("data/raw/products.csv")
DEPARTMENTS_PATH = Path("data/raw/departments.csv")

@st.cache_data(show_spinner=False)
def load_catalog():
    if not Path(CATALOG_PATH).exists():
        st.error(f"Catalog file not found at {CATALOG_PATH}")
        st.stop()
    df = pd.read_parquet(CATALOG_PATH)
    title_col = "name" if "name" in df.columns else "product_name"
    if title_col in df.columns:
        df = df.rename(columns={title_col: "title"})
    else:
        df["title"] = df["product_id"].apply(lambda x: f"Product {x}")
    if "department" in df.columns:
        df["department"] = df["department"].astype(str).str.strip().str.lower()
    else:
        st.error("Catalog missing 'department' column")
        st.stop()
    keep_cols = ["product_id", "title", "department"]
    df = df[[c for c in keep_cols if c in df.columns]]
    return df.dropna(subset=["product_id", "title", "department"])

@st.cache_data(show_spinner=False)
def load_purchase_history():
    """Load user purchase history with product details"""
    try:
        # Load orders and filter for prior orders
        orders = pd.read_csv(ORDERS_PATH)
        orders_prior = orders[orders['eval_set'] == 'prior'][['order_id', 'user_id']]
        
        # Load order products
        order_products = pd.read_csv(ORDER_PRODUCTS_PATH)
        
        # Merge to get user-product relationships
        user_products = order_products.merge(orders_prior, on='order_id')
        
        # Calculate statistics per user-product
        stats = user_products.groupby(['user_id', 'product_id']).agg({
            'order_id': 'count',  # Number of times ordered
            'reordered': 'sum'    # Number of reorders
        }).rename(columns={'order_id': 'purchase_count'}).reset_index()
        
        # Load product names
        products = pd.read_csv(PRODUCTS_PATH)[['product_id', 'product_name']]
        stats = stats.merge(products, on='product_id', how='left')
        
        return stats
    except Exception as e:
        st.warning(f"Could not load purchase history: {e}")
        return pd.DataFrame()

DF = load_catalog()
PURCHASE_HISTORY = load_purchase_history()

@st.cache_resource
def load_lgbm_model():
    """Load LightGBM model and feature columns at startup"""
    if not MODEL_FILE.exists():
        st.warning("LightGBM model not found. Using fallback recommendations.")
        return None, None, None
    
    try:
        booster = lgb.Booster(model_file=str(MODEL_FILE))
        
        if FEATURE_COLS_FILE.exists():
            with open(FEATURE_COLS_FILE) as f:
                feature_cols = json.load(f)
        else:
            st.warning("Feature columns file not found.")
            return None, None, None
        
        # Load candidates data
        if CANDIDATES_PATH.exists():
            candidates = pd.read_csv(CANDIDATES_PATH)
        else:
            st.warning("Candidates data not found.")
            return None, None, None
            
        return booster, feature_cols, candidates
    except Exception as e:
        st.warning(f"Error loading model: {e}")
        return None, None, None

# Load model at startup
LGBM_MODEL, FEATURE_COLS, CANDIDATES_DF = load_lgbm_model()

@st.cache_resource
def load_item_item_similarity():
    """Load Item-Item similarity matrix"""
    # Try different possible file names
    possible_paths = [
        Path("data/processed/item_item_similarity_score.csv"),
        Path("data/processed/item_item_similarity.csv"),
        Path("data/processed/item_item_similarity.parquet"),
    ]
    
    try:
        df = None
        for path in possible_paths:
            if path.exists():
                if path.suffix == '.parquet':
                    df = pd.read_parquet(path)
                else:
                    df = pd.read_csv(path)
                break
        
        if df is None:
            return {}
        
        # Build lookup: product_id -> [(neighbor_id, sim), ...]
        similarity_index = {}
        
        # Handle different column naming conventions
        pid_col = 'product_id'
        nid_col = 'neighbor_product_id' if 'neighbor_product_id' in df.columns else 'neighbor_id' if 'neighbor_id' in df.columns else 'product_id_2'
        sim_col = 'score' if 'score' in df.columns else 'similarity_score' if 'similarity_score' in df.columns else 'sim'
        
        for _, row in df.iterrows():
            pid = int(row[pid_col])
            nid = int(row[nid_col])
            sim = float(row[sim_col])
            if pid not in similarity_index:
                similarity_index[pid] = []
            similarity_index[pid].append((nid, sim))
        
        # Sort by similarity descending
        for pid in similarity_index:
            similarity_index[pid].sort(key=lambda x: x[1], reverse=True)
        
        return similarity_index
    except Exception as e:
        return {}

@st.cache_resource
def load_als_artifacts():
    """Load ALS user/item factors and product mapping"""
    user_factors_path = Path("data/processed/als_user_factors.npy")
    item_factors_path = Path("data/processed/als_product_factors.npy")
    user_map_path = Path("data/processed/als_user_index_map.csv")
    product_map_path = Path("data/processed/als_product_index_map.csv")
    
    try:
        if not all([user_factors_path.exists(), item_factors_path.exists()]):
            return None, None, None, None, {}
        
        # Load factors - NOTE: files may have swapped dimensions
        U_file = np.load(user_factors_path)
        V_file = np.load(item_factors_path)
        
        # Check if dimensions are swapped by comparing with mapping sizes
        user_map = pd.read_csv(user_map_path) if user_map_path.exists() else None
        product_map = pd.read_csv(product_map_path) if product_map_path.exists() else None
        
        if user_map is not None and product_map is not None:
            n_users = len(user_map)
            n_products = len(product_map)
            
            # If user_factors.npy has product count rows, dimensions are swapped
            if U_file.shape[0] == n_products and V_file.shape[0] == n_users:
                # Swap them
                U = V_file
                V = U_file
            else:
                U = U_file
                V = V_file
        else:
            U = U_file
            V = V_file
        
        # Load product mapping
        pid_to_idx = {}
        idx_to_pid = np.array([])
        
        if product_map is not None:
            # Handle different column names
            if 'product_id' in product_map.columns and 'product_index' in product_map.columns:
                pid_to_idx = dict(zip(product_map['product_id'], product_map['product_index']))
                idx_to_pid = product_map.set_index('product_index')['product_id'].values
            elif 'product_id' in product_map.columns and 'index' in product_map.columns:
                pid_to_idx = dict(zip(product_map['product_id'], product_map['index']))
                idx_to_pid = product_map.set_index('index')['product_id'].values
            elif 'product_id' in product_map.columns and 'idx' in product_map.columns:
                pid_to_idx = dict(zip(product_map['product_id'], product_map['idx']))
                idx_to_pid = product_map.set_index('idx')['product_id'].values
        
        # Load user mapping if available
        user_to_idx = {}
        if user_map is not None:
            if 'user_id' in user_map.columns and 'user_index' in user_map.columns:
                user_to_idx = dict(zip(user_map['user_id'], user_map['user_index']))
            elif 'user_id' in user_map.columns and 'index' in user_map.columns:
                user_to_idx = dict(zip(user_map['user_id'], user_map['index']))
            elif 'user_id' in user_map.columns and 'idx' in user_map.columns:
                user_to_idx = dict(zip(user_map['user_id'], user_map['idx']))
        
        return U, V, pid_to_idx, idx_to_pid, user_to_idx
    except Exception as e:
        return None, None, None, None, {}

# Load additional models
ITEM_ITEM_SIM = load_item_item_similarity()
als_result = load_als_artifacts()
if als_result and len(als_result) == 5:
    ALS_U, ALS_V, ALS_PID_TO_IDX, ALS_IDX_TO_PID, ALS_USER_TO_IDX = als_result
else:
    ALS_U, ALS_V, ALS_PID_TO_IDX, ALS_IDX_TO_PID, ALS_USER_TO_IDX = None, None, None, None, {}

@st.cache_resource
def get_dept_image_map():
    special_cases = {"dairy eggs": "dairy egg.jpg"}
    mapping = {}
    
    # Ensure thumbnails directory exists
    if not THUMBNAILS_DIR.exists():
        st.warning(f"Thumbnails directory not found: {THUMBNAILS_DIR}")
        return {}
    
    for dept in DF["department"].unique():
        if dept in special_cases:
            img_path = THUMBNAILS_DIR / special_cases[dept]
            if img_path.exists():
                mapping[dept] = img_path
                continue
        img_path = THUMBNAILS_DIR / f"{dept}.jpg"
        if img_path.exists():
            mapping[dept] = img_path
        else:
            # Try with fallback
            if FALLBACK_IMG.exists():
                mapping[dept] = FALLBACK_IMG
    return mapping

DEPT_IMG_MAP = get_dept_image_map()

@st.cache_data(show_spinner=False)
def load_product_image(department: str):
    """Load product image with caching for better performance"""
    img_path = DEPT_IMG_MAP.get(department.lower() if department else "", FALLBACK_IMG)
    try:
        if img_path and Path(img_path).exists():
            return Image.open(img_path)
        elif FALLBACK_IMG.exists():
            return Image.open(FALLBACK_IMG)
        else:
            # Return a blank image if nothing exists
            from PIL import Image as PILImage
            return PILImage.new('RGB', (200, 200), color='lightgray')
    except Exception as e:
        # Return a blank image
        from PIL import Image as PILImage
        return PILImage.new('RGB', (200, 200), color='lightgray')

if "user_id" not in st.session_state:
    st.session_state.user_id = USER_IDS[0]
if "cart" not in st.session_state:
    st.session_state.cart = []
if "search" not in st.session_state:
    st.session_state.search = ""
if "dep_filter" not in st.session_state:
    st.session_state.dep_filter = "all"
if "current_page" not in st.session_state:
    st.session_state.current_page = "shop"
if "active_rec_tab" not in st.session_state:
    st.session_state.active_rec_tab = "all"
if "shop_page_num" not in st.session_state:
    st.session_state.shop_page_num = 0

def add_to_cart(pid: int):
    if pid not in st.session_state.cart:
        st.session_state.cart.append(pid)

def remove_from_cart(pid: int):
    st.session_state.cart = [x for x in st.session_state.cart if x != pid]

def clear_cart():
    st.session_state.cart = []

def go_to_cart():
    st.session_state.current_page = "cart"

def go_to_shop():
    st.session_state.current_page = "shop"
    st.session_state.shop_page_num = 0  # Reset to first page

def open_lgbm_page():
    st.query_params.update({"user_id": st.session_state.user_id, "cart": ",".join(map(str, st.session_state.cart))})
    st.switch_page("pages/02_LightGBM_Recs.py")

# ==================== RECOMMENDATION HELPERS ====================

def get_item_item_for_product(pid: int, top_k: int = 8) -> pd.DataFrame:
    """Get Item-Item collaborative filtering recommendations for a product"""
    if not ITEM_ITEM_SIM or pid not in ITEM_ITEM_SIM:
        return pd.DataFrame()
    
    neighbors = ITEM_ITEM_SIM[pid][:top_k * 2]  # Get more to filter
    neighbor_ids = [nid for nid, sim in neighbors if nid not in st.session_state.cart][:top_k]
    
    if not neighbor_ids:
        return pd.DataFrame()
    
    result = DF[DF["product_id"].isin(neighbor_ids)].copy()
    return result.head(top_k)

def get_item_item_for_cart(cart_pids: List[int], top_k_per_item: int = 4, cap: int = 16) -> pd.DataFrame:
    """Get Item-Item recommendations for entire cart"""
    if not ITEM_ITEM_SIM or not cart_pids:
        return pd.DataFrame()
    
    # Aggregate similarities across all cart items
    all_neighbors = {}
    for pid in cart_pids:
        if pid in ITEM_ITEM_SIM:
            for nid, sim in ITEM_ITEM_SIM[pid][:top_k_per_item * 3]:
                if nid not in cart_pids and nid not in st.session_state.cart:
                    if nid not in all_neighbors:
                        all_neighbors[nid] = 0.0
                    all_neighbors[nid] = max(all_neighbors[nid], sim)  # Max similarity
    
    if not all_neighbors:
        return pd.DataFrame()
    
    # Sort by aggregated score and take top cap
    top_neighbors = sorted(all_neighbors.items(), key=lambda x: x[1], reverse=True)[:cap]
    neighbor_ids = [nid for nid, _ in top_neighbors]
    
    result = DF[DF["product_id"].isin(neighbor_ids)].copy()
    return result.head(cap)

def als_recommend_for_user(user_id: int, N: int = 12) -> pd.DataFrame:
    """Get ALS collaborative filtering recommendations"""
    if ALS_U is None or ALS_V is None or ALS_PID_TO_IDX is None:
        return pd.DataFrame()
    
    try:
        # Find user index using mapping if available
        if ALS_USER_TO_IDX and user_id in ALS_USER_TO_IDX:
            user_idx = ALS_USER_TO_IDX[user_id]
        else:
            # Fallback: use modulo if no direct mapping
            user_idx = user_id % len(ALS_U)
        
        if user_idx >= len(ALS_U):
            return pd.DataFrame()
        
        # Compute scores: U[user] @ V.T
        scores = ALS_U[user_idx] @ ALS_V.T
        
        # Get product indices for items NOT in cart
        valid_indices = []
        cart_set = set(st.session_state.cart)
        
        for idx in range(len(scores)):
            if idx < len(ALS_IDX_TO_PID):
                pid = int(ALS_IDX_TO_PID[idx])
                if pid not in cart_set:
                    valid_indices.append((idx, scores[idx], pid))
        
        # Sort by score and take top N
        valid_indices.sort(key=lambda x: x[1], reverse=True)
        top_pids = [pid for _, _, pid in valid_indices[:N]]
        
        if not top_pids:
            return pd.DataFrame()
        
        result = DF[DF["product_id"].isin(top_pids)].copy()
        return result.head(N)
    except Exception as e:
        return pd.DataFrame()

def lgbm_recommend_for_user(user_id: int, top_k: int = 12, exclude: List[int] = None) -> pd.DataFrame:
    """Get LightGBM recommendations, excluding specified products"""
    if LGBM_MODEL is None or CANDIDATES_DF is None:
        return pd.DataFrame()
    
    # Get candidates for this user
    user_candidates = CANDIDATES_DF[CANDIDATES_DF["user_id"] == user_id].copy()
    if user_candidates.empty:
        return pd.DataFrame()
    
    # Check if feature columns exist
    missing_feats = [c for c in FEATURE_COLS if c not in user_candidates.columns]
    if missing_feats:
        return pd.DataFrame()
    
    # Predict scores
    try:
        scores = LGBM_MODEL.predict(user_candidates[FEATURE_COLS])
        user_candidates["score"] = scores
        
        # Exclude cart and additional items
        exclude_set = set(st.session_state.cart)
        if exclude:
            exclude_set.update(exclude)
        
        user_candidates = user_candidates[~user_candidates["product_id"].isin(exclude_set)]
        top_products = user_candidates.nlargest(top_k, "score")
        
        # Merge with catalog
        result = top_products[["product_id", "score"]].merge(
            DF, on="product_id", how="left"
        )
        return result
    except Exception as e:
        return pd.DataFrame()

def get_model_recommendations_for_user(user_id: int, top_k: int = 10):
    """Legacy wrapper - Get LightGBM model predictions for a user"""
    return lgbm_recommend_for_user(user_id, top_k, exclude=None)

def get_item_recommendations(product_id: int, top_k: int = 4):
    """Get model-based recommendations for items in cart"""
    if LGBM_MODEL is None or CANDIDATES_DF is None:
        # Fallback to department-based
        product = DF[DF["product_id"] == product_id]
        if product.empty:
            return pd.DataFrame()
        dept = product.iloc[0]["department"]
        candidates = DF[(DF["department"] == dept) & (DF["product_id"] != product_id) & (~DF["product_id"].isin(st.session_state.cart))]
        return candidates.head(top_k)
    
    # Use model predictions for current user
    user_id = st.session_state.user_id
    recs = get_model_recommendations_for_user(user_id, top_k=50)
    
    if recs.empty:
        # Fallback
        product = DF[DF["product_id"] == product_id]
        if product.empty:
            return pd.DataFrame()
        dept = product.iloc[0]["department"]
        candidates = DF[(DF["department"] == dept) & (DF["product_id"] != product_id) & (~DF["product_id"].isin(st.session_state.cart))]
        return candidates.head(top_k)
    
    # Filter out items already in cart
    recs = recs[~recs["product_id"].isin(st.session_state.cart)]
    return recs.head(top_k)

def get_cart_recommendations(cart_pids: List[int], top_k: int = 8):
    """Get model-based recommendations for the entire cart"""
    if LGBM_MODEL is None or CANDIDATES_DF is None:
        # Fallback to department-based
        if not cart_pids:
            return DF.head(top_k)
        cart_depts = set(DF[DF["product_id"].isin(cart_pids)]["department"].tolist())
        candidates = DF[~DF["product_id"].isin(cart_pids) & DF["department"].isin(cart_depts)]
        return candidates.head(top_k)
    
    # Use model predictions - cart filtering already done in get_model_recommendations_for_user
    user_id = st.session_state.user_id
    recs = get_model_recommendations_for_user(user_id, top_k=top_k)
    
    if recs.empty:
        # Fallback
        if not cart_pids:
            return DF.head(top_k)
        cart_depts = set(DF[DF["product_id"].isin(cart_pids)]["department"].tolist())
        candidates = DF[~DF["product_id"].isin(cart_pids) & DF["department"].isin(cart_depts)]
        return candidates.head(top_k)
    
    return recs

def get_user_recommendations(user_id: int, top_k: int = 12):
    """Get personalized recommendations for user using LightGBM model"""
    recs = get_model_recommendations_for_user(user_id, top_k=top_k)
    
    if recs.empty:
        # Fallback to popular items
        dept_counts = DF["department"].value_counts()
        dept_rank = {d: i for i, d in enumerate(dept_counts.index)}
        candidates = DF[~DF["product_id"].isin(st.session_state.cart)].assign(_rank=DF["department"].map(dept_rank).fillna(9999))
        return candidates.sort_values("_rank").drop(columns="_rank").head(top_k)
    
    return recs

def get_purchase_stats(user_id: int, product_id: int):
    """Get purchase statistics for a product and user"""
    if PURCHASE_HISTORY.empty:
        return None
    
    stats = PURCHASE_HISTORY[
        (PURCHASE_HISTORY['user_id'] == user_id) & 
        (PURCHASE_HISTORY['product_id'] == product_id)
    ]
    
    if stats.empty:
        return None
    
    row = stats.iloc[0]
    return {
        'purchased': True,
        'count': int(row['purchase_count']),
        'reordered': int(row['reordered'])
    }

def render_product_card(row, key_prefix, show_history=True):
    """Render a single product card with optional purchase history"""
    product_id = int(row["product_id"])
    title = row["title"]
    dept = row["department"]
    
    # Get purchase stats
    purchase_stats = None
    if show_history:
        purchase_stats = get_purchase_stats(st.session_state.user_id, product_id)
    
    # Render card
    with st.container():
        # Product image
        img = load_product_image(dept)
        st.image(img, use_column_width=True)
        
        # Product details
        st.markdown(f"**{title[:45]}{'...' if len(title) > 45 else ''}**")
        st.caption(f"📦 {dept.title()}")
        
        # Purchase history indicator (compact)
        if purchase_stats:
            count = purchase_stats['count']
            if count >= 3:
                st.caption("⭐ Bought " + f"{count}x before")
            elif count > 1:
                st.caption(f"✓ Bought {count}x before")
            else:
                st.caption("✓ Bought before")
        
        # Action button
        if product_id in st.session_state.cart:
            st.success("✓ In Cart", icon="✅")
        else:
            if st.button("➕ Add", key=f"{key_prefix}-{product_id}"):
                add_to_cart(product_id)
                st.rerun()

# ==================== HEADER ====================
header_col1, header_col2, header_col3, header_col4 = st.columns([1.5, 3.5, 1.5, 1])

with header_col1:
    st.markdown('<div class="main-header"><h3>🛒 InstaRec</h3></div>', unsafe_allow_html=True)

with header_col2:
    st.session_state.search = st.text_input("🔍 Search", st.session_state.search, placeholder="Search for products...", label_visibility="collapsed")

with header_col3:
    st.markdown("<div style='margin-top: 5px;'>", unsafe_allow_html=True)
    selected_user = st.selectbox("👤", USER_IDS, index=USER_IDS.index(st.session_state.user_id), label_visibility="collapsed", key="user_selector")
    if selected_user != st.session_state.user_id:
        st.session_state.user_id = selected_user
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

with header_col4:
    st.markdown("<div style='margin-top: 5px;'>", unsafe_allow_html=True)
    if st.button(f"🛒 ({len(st.session_state.cart)})", type="primary"):
        go_to_cart()
    st.markdown("</div>", unsafe_allow_html=True)

# ==================== FILTERS (Sidebar) ====================
with st.sidebar:
    st.markdown(f"### 👤 User Profile")
    st.markdown(f"**User ID:** {st.session_state.user_id}")
    
    st.divider()
    st.markdown("### 🎯 Filters")
    departments = ["all"] + sorted(DF["department"].unique().tolist())
    st.session_state.dep_filter = st.selectbox("Department", departments, index=departments.index(st.session_state.dep_filter) if st.session_state.dep_filter in departments else 0)
    
    st.divider()
    st.markdown("### 📊 Cart Stats")
    cart_df = DF[DF["product_id"].isin(st.session_state.cart)]
    if len(cart_df) > 0:
        st.metric("Items", len(st.session_state.cart))
        dept_breakdown = cart_df["department"].value_counts()
        st.write("**By Department:**")
        for dept, count in dept_breakdown.head(3).items():
            st.caption(f"• {dept.title()}: {count}")
    else:
        st.info("Cart is empty")
    
    if len(st.session_state.cart) > 0:
        if st.button("🗑️ Clear Cart", key="sidebar_clear_cart"):
            clear_cart()
            st.rerun()
    
    # Model Status
    st.divider()
    st.markdown("### 🤖 Recommendation Sources")
    
    models_active = []
    if ITEM_ITEM_SIM:
        st.success("✅ Item-Item CF", icon="🔗")
        models_active.append("Item-Item")
    else:
        st.warning("⚠️ Item-Item (fallback)", icon="🔗")
    
    if ALS_U is not None:
        st.success("✅ ALS CF", icon="👥")
        models_active.append("ALS")
    else:
        st.warning("⚠️ ALS (fallback)", icon="👥")
    
    if LGBM_MODEL is not None:
        st.success("✅ LightGBM Ranker", icon="🎯")
        models_active.append("LightGBM")
        if CANDIDATES_DF is not None:
            user_cands = len(CANDIDATES_DF[CANDIDATES_DF["user_id"] == st.session_state.user_id])
            st.caption(f"📊 {user_cands:,} candidates")
    else:
        st.warning("⚠️ LightGBM (fallback)", icon="🎯")
    
    if models_active:
        st.caption(f"🚀 **{len(models_active)}/3** models active")

# ==================== SHOP PAGE ====================
if st.session_state.current_page == "shop":
    col_left, col_right = st.columns([3, 1])
    
    with col_left:
        st.markdown('<p class="section-header">🛍️ Browse Products</p>', unsafe_allow_html=True)
    
    # Apply filters
    browse = DF.copy()
    if st.session_state.search:
        browse = browse[browse["title"].str.contains(st.session_state.search, case=False, na=False)]
    if st.session_state.dep_filter != "all":
        browse = browse[browse["department"] == st.session_state.dep_filter]
    
    if len(browse) == 0:
        st.warning("🔍 No products found. Try adjusting your filters.")
    else:
        # Pagination settings
        items_per_page = 50
        total_pages = (len(browse) + items_per_page - 1) // items_per_page
        
        # Ensure page number is valid
        if st.session_state.shop_page_num >= total_pages:
            st.session_state.shop_page_num = 0
        
        start_idx = st.session_state.shop_page_num * items_per_page
        end_idx = start_idx + items_per_page
        
        # Pagination info
        col_info, col_pages = st.columns([2, 1])
        with col_info:
            st.caption(f"Showing {start_idx + 1}-{min(end_idx, len(browse))} of {len(browse)} products")
        with col_pages:
            st.caption(f"Page {st.session_state.shop_page_num + 1} of {total_pages}")
        
        # Display products in grid (no purchase history to avoid glitches)
        page_items = browse.iloc[start_idx:end_idx]
        cols = st.columns(5)
        for i, (_, row) in enumerate(page_items.iterrows()):
            with cols[i % 5]:
                render_product_card(row, f"browse-p{st.session_state.shop_page_num}-{i}", show_history=False)
        
        # Pagination controls
        st.markdown("---")
        pagecol1, pagecol2, pagecol3, pagecol4, pagecol5 = st.columns([1, 1, 2, 1, 1])
        
        with pagecol1:
            if st.button("⏮️ First", disabled=(st.session_state.shop_page_num == 0), key="first_page"):
                st.session_state.shop_page_num = 0
                st.rerun()
        
        with pagecol2:
            if st.button("◀️ Previous", disabled=(st.session_state.shop_page_num == 0), key="prev_page"):
                st.session_state.shop_page_num -= 1
                st.rerun()
        
        with pagecol3:
            st.markdown(f"<center>Page {st.session_state.shop_page_num + 1} of {total_pages}</center>", unsafe_allow_html=True)
        
        with pagecol4:
            if st.button("Next ▶️", disabled=(st.session_state.shop_page_num >= total_pages - 1), key="next_page"):
                st.session_state.shop_page_num += 1
                st.rerun()
        
        with pagecol5:
            if st.button("Last ⏭️", disabled=(st.session_state.shop_page_num >= total_pages - 1), key="last_page"):
                st.session_state.shop_page_num = total_pages - 1
                st.rerun()

# ==================== CART PAGE ====================
elif st.session_state.current_page == "cart":
    col_back, col_title = st.columns([1, 5])
    
    with col_back:
        if st.button("← Shop"):
            go_to_shop()
    
    with col_title:
        st.markdown(f'<p class="section-header">🛒 Shopping Cart ({len(st.session_state.cart)} items)</p>', unsafe_allow_html=True)
    
    if not st.session_state.cart:
        st.info("🛒 Your cart is empty. Start shopping to see personalized recommendations!")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🛍️ Start Shopping", type="primary"):
                go_to_shop()
    else:
        cart_df = DF[DF["product_id"].isin(st.session_state.cart)]
        
        # ==================== CART ITEMS SECTION ====================
        st.markdown("### 🛍️ Items in Your Cart")
        st.markdown("---")
        
        # Display cart items in grid (4 columns for consistent alignment)
        num_cols = 4
        cart_items_list = list(cart_df.iterrows())
        
        # Process items in rows to maintain alignment
        for row_start in range(0, len(cart_items_list), num_cols):
            row_items = cart_items_list[row_start:row_start + num_cols]
            
            # Always create 4 columns to maintain consistent sizing
            cart_cols = st.columns(num_cols, gap="medium")
            
            for col_idx in range(num_cols):
                with cart_cols[col_idx]:
                    # Only render if we have an item for this column
                    if col_idx < len(row_items):
                        _, cart_item = row_items[col_idx]
                        
                        # Wrap in a div with styling
                        st.markdown('<div class="cart-item">', unsafe_allow_html=True)
                        
                        # Product image
                        img = load_product_image(cart_item["department"])
                        st.image(img, use_column_width=True)
                        
                        # Product details with consistent formatting
                        title_display = cart_item['title'][:45] + ('...' if len(cart_item['title']) > 45 else '')
                        st.markdown(f"**{title_display}**")
                        st.caption(f"📦 {cart_item['department'].title()}")
                        st.caption(f"🆔 {cart_item['product_id']}")
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Remove button
                        if st.button("🗑️ Remove", key=f"rm-{cart_item['product_id']}", type="secondary"):
                            remove_from_cart(int(cart_item["product_id"]))
                            st.rerun()
            
            # Add spacing between rows
            st.markdown("<div style='margin-bottom: 15px;'></div>", unsafe_allow_html=True)
        
        # ==================== RECOMMENDATION TABS ====================
        st.markdown("---")
        st.markdown("## � Recommendations")
        
        # Tab pills for filtering
        tab_cols = st.columns(4)
        with tab_cols[0]:
            if st.button("🌟 All Picks", type="primary" if st.session_state.active_rec_tab == "all" else "secondary"):
                st.session_state.active_rec_tab = "all"
                st.rerun()
        with tab_cols[1]:
            if st.button("🔗 Similar to Cart", type="primary" if st.session_state.active_rec_tab == "similar" else "secondary"):
                st.session_state.active_rec_tab = "similar"
                st.rerun()
        with tab_cols[2]:
            if st.button("👥 Users Like You", type="primary" if st.session_state.active_rec_tab == "als" else "secondary"):
                st.session_state.active_rec_tab = "als"
                st.rerun()
        with tab_cols[3]:
            if st.button("🎯 For You", type="primary" if st.session_state.active_rec_tab == "lgbm" else "secondary"):
                st.session_state.active_rec_tab = "lgbm"
                st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Initialize recommendation dataframes
        item_item_recs = pd.DataFrame()
        als_recs = pd.DataFrame()
        lgbm_recs = pd.DataFrame()
        
        # Section 1: Item-Item (Because these are in your cart)
        if st.session_state.active_rec_tab in ["all", "similar"]:
            st.markdown('<div class="rec-section">', unsafe_allow_html=True)
            st.markdown("### 🔗 Because these are in your cart")
            st.caption("💡 Items frequently bought together (Item-Item Collaborative Filtering)")
            
            with st.spinner("Finding items bought together..."):
                item_item_recs = get_item_item_for_cart(st.session_state.cart, cap=12)
            
            if not item_item_recs.empty:
                ii_cols = st.columns(4)
                for i, (_, rec) in enumerate(item_item_recs.iterrows()):
                    with ii_cols[i % 4]:
                        render_product_card(rec, "ii", show_history=True)
            else:
                st.info("🔗 Item-Item recommendations unavailable. Add more items to your cart!")
                st.caption("Fallback: showing popular items from your cart's departments")
                if st.session_state.cart:
                    cart_depts = set(DF[DF["product_id"].isin(st.session_state.cart)]["department"].tolist())
                    fallback = DF[~DF["product_id"].isin(st.session_state.cart) & DF["department"].isin(cart_depts)].head(8)
                if not fallback.empty:
                    fb_cols = st.columns(4)
                    for i, (_, rec) in enumerate(fallback.iterrows()):
                        with fb_cols[i % 4]:
                            render_product_card(rec, "fb-ii", show_history=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
        
        # Section 2: ALS (Users like you also bought)
        if st.session_state.active_rec_tab in ["all", "als"]:
            st.markdown('<div class="rec-section">', unsafe_allow_html=True)
            st.markdown("### 👥 Users like you also bought")
            st.caption("🤝 Collaborative filtering based on similar user preferences (ALS)")
            
            with st.spinner("Finding what similar users love..."):
                als_recs = als_recommend_for_user(st.session_state.user_id, N=12)
            
            if not als_recs.empty:
                als_cols = st.columns(4)
                for i, (_, rec) in enumerate(als_recs.iterrows()):
                    with als_cols[i % 4]:
                        render_product_card(rec, "als", show_history=True)
            else:
                st.info("👥 ALS recommendations unavailable. Using popularity-based fallback.")
                st.caption("Fallback: showing popular products")
                dept_counts = DF["department"].value_counts()
                dept_rank = {d: i for i, d in enumerate(dept_counts.index)}
                fallback = DF[~DF["product_id"].isin(st.session_state.cart)].assign(_rank=DF["department"].map(dept_rank).fillna(9999)).sort_values("_rank").head(8)
                if not fallback.empty:
                    fb_cols = st.columns(4)
                    for i, (_, rec) in enumerate(fallback.iterrows()):
                        with fb_cols[i % 4]:
                            render_product_card(rec, "fb-als", show_history=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
        
        # Section 3: LightGBM (Recommended for you)
        if st.session_state.active_rec_tab in ["all", "lgbm"]:
            st.markdown('<div class="rec-section">', unsafe_allow_html=True)
            st.markdown("### 🎯 Recommended for you")
            st.caption("✨ Personalized ranking model (LightGBM LambdaRank)")
            
            with st.spinner("Ranking products just for you..."):
                # Get products already shown to avoid duplicates
                shown_pids = []
                if st.session_state.active_rec_tab == "all":
                    if not item_item_recs.empty:
                        shown_pids.extend(item_item_recs["product_id"].tolist())
                    if not als_recs.empty:
                        shown_pids.extend(als_recs["product_id"].tolist())
                
                lgbm_recs = lgbm_recommend_for_user(st.session_state.user_id, top_k=12, exclude=shown_pids)
            
            if not lgbm_recs.empty:
                lgbm_cols = st.columns(4)
                for i, (_, rec) in enumerate(lgbm_recs.iterrows()):
                    with lgbm_cols[i % 4]:
                        render_product_card(rec, "lgbm", show_history=True)
            else:
                st.info("🎯 LightGBM recommendations unavailable. Using department-based fallback.")
                st.caption("Fallback: showing items from your cart's departments")
                if st.session_state.cart:
                    cart_depts = set(DF[DF["product_id"].isin(st.session_state.cart)]["department"].tolist())
                    fallback = DF[~DF["product_id"].isin(st.session_state.cart) & DF["department"].isin(cart_depts)].head(8)
                else:
                    fallback = DF[~DF["product_id"].isin(st.session_state.cart)].head(8)
                
                if not fallback.empty:
                    fb_cols = st.columns(4)
                    for i, (_, rec) in enumerate(fallback.iterrows()):
                        with fb_cols[i % 4]:
                            render_product_card(rec, "fb-lgbm", show_history=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # ==================== ADDITIONAL ACTIONS ====================
        st.markdown("---")
        col_shop, col_lgbm, col_clear = st.columns([1, 1, 1])
        
        with col_shop:
            if st.button("🛍️ Continue Shopping", type="primary"):
                go_to_shop()
        
        with col_lgbm:
            if st.button("📊 View LightGBM Insights"):
                open_lgbm_page()
        
        with col_clear:
            if st.button("🗑️ Clear Cart", key="cart_bottom_clear", type="secondary"):
                clear_cart()
                st.rerun()

st.markdown("---")
st.caption(f"🤖 Powered by ALS, Item-Item CF, and LightGBM Ranker | User: {st.session_state.user_id} | Cart: {len(st.session_state.cart)} items")
