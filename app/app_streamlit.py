import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image
from typing import List

st.set_page_config(page_title="InstaRec - Your Grocery Store", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for Amazon/Instacart-like styling
st.markdown("""
<style>
    /* Reduce main padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
    }
    /* Header styling - smaller */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0.5rem 1.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        color: white;
    }
    .main-header h3 {
        margin: 0;
        font-size: 1.5rem;
    }
    .cart-badge {
        background: #ff6b6b;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        cursor: pointer;
    }
    /* Product card styling - smaller */
    .product-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 0.8rem;
        background: white;
        transition: all 0.3s;
        height: 100%;
    }
    .product-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }
    /* Product image container */
    [data-testid="stImage"] {
        min-height: 180px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    [data-testid="stImage"] img {
        min-height: 180px;
        object-fit: cover;
        border-radius: 8px;
    }
    /* Cart item styling - smaller */
    .cart-item {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        background: #fafafa;
    }
    .cart-item img {
        max-height: 150px;
        object-fit: contain;
    }
    /* Recommendation section */
    .rec-section {
        background: #f8f9fa;
        padding: 0.8rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    /* Price tag styling */
    .price-tag {
        color: #e74c3c;
        font-size: 1.2rem;
        font-weight: bold;
    }
    /* Section headers - smaller */
    .section-header {
        color: #2c3e50;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 1rem 0 0.8rem 0;
        border-bottom: 2px solid #667eea;
        padding-bottom: 0.4rem;
    }
    /* User badge */
    .user-badge {
        background: #f0f0f0;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.9rem;
        color: #555;
    }
    /* Stats */
    .stat-box {
        background: #e8f4f8;
        padding: 0.5rem;
        border-radius: 6px;
        text-align: center;
        border: 1px solid #b8dce8;
    }
</style>
""", unsafe_allow_html=True)

CATALOG_PATH = "data/processed/catalog_with_dept_images.parquet"
THUMBNAILS_DIR = Path("assets/thumbnails")
FALLBACK_IMG = THUMBNAILS_DIR / "missing.jpg"
USER_IDS = [101, 212, 318, 451, 612, 777, 999]

@st.cache_data(show_spinner=False)
def load_catalog():
    df = pd.read_parquet(CATALOG_PATH)
    title_col = "name" if "name" in df.columns else "product_name"
    if title_col in df.columns:
        df = df.rename(columns={title_col: "title"})
    else:
        df["title"] = df["product_id"].apply(lambda x: f"Product {x}")
    if "department" in df.columns:
        df["department"] = df["department"].astype(str).str.strip().str.lower()
    keep_cols = ["product_id", "title", "department"]
    df = df[[c for c in keep_cols if c in df.columns]]
    return df.dropna(subset=["product_id", "title", "department"])

DF = load_catalog()

@st.cache_resource
def get_dept_image_map():
    special_cases = {"dairy eggs": "dairy egg.jpg"}
    mapping = {}
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
            mapping[dept] = FALLBACK_IMG
    return mapping

DEPT_IMG_MAP = get_dept_image_map()

def load_product_image(department: str):
    img_path = DEPT_IMG_MAP.get(department, FALLBACK_IMG)
    try:
        return Image.open(img_path)
    except Exception:
        return Image.open(FALLBACK_IMG)

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

def get_item_recommendations(product_id: int, top_k: int = 4):
    """Get recommendations for a specific product"""
    product = DF[DF["product_id"] == product_id]
    if product.empty:
        return pd.DataFrame()
    dept = product.iloc[0]["department"]
    candidates = DF[(DF["department"] == dept) & (DF["product_id"] != product_id) & (~DF["product_id"].isin(st.session_state.cart))]
    return candidates.head(top_k)

def get_cart_recommendations(cart_pids: List[int], top_k: int = 8):
    """Get general recommendations based on entire cart"""
    if not cart_pids:
        dept_counts = DF["department"].value_counts()
        dept_rank = {d: i for i, d in enumerate(dept_counts.index)}
        candidates = DF.assign(_rank=DF["department"].map(dept_rank).fillna(9999))
        return candidates.sort_values("_rank").drop(columns="_rank").head(top_k)
    cart_depts = set(DF[DF["product_id"].isin(cart_pids)]["department"].tolist())
    candidates = DF[~DF["product_id"].isin(cart_pids) & DF["department"].isin(cart_depts)]
    if len(candidates) < top_k:
        backfill = DF[~DF["product_id"].isin(cart_pids)]
        candidates = pd.concat([candidates, backfill]).drop_duplicates("product_id")
    return candidates.head(top_k)

def get_user_recommendations(user_id: int, top_k: int = 12):
    """Get personalized recommendations for user - placeholder for your ML models"""
    # TODO: Replace with actual model predictions (ALS, Item-Item CF, LightGBM)
    # For now, use popular items from frequently purchased departments
    dept_counts = DF["department"].value_counts()
    dept_rank = {d: i for i, d in enumerate(dept_counts.index)}
    candidates = DF[~DF["product_id"].isin(st.session_state.cart)].assign(_rank=DF["department"].map(dept_rank).fillna(9999))
    return candidates.sort_values("_rank").drop(columns="_rank").head(top_k)

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
    if st.button(f"🛒 ({len(st.session_state.cart)})", use_container_width=True, type="primary"):
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
        if st.button("🗑️ Clear Cart", use_container_width=True):
            clear_cart()
            st.rerun()

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
        display_limit = 100
        st.caption(f"Showing {min(display_limit, len(browse))} of {len(browse)} products")
        
        # Display products in grid
        cols = st.columns(5)
        for i, (_, row) in enumerate(browse.head(display_limit).iterrows()):
            with cols[i % 5]:
                with st.container():
                    img = load_product_image(row["department"])
                    st.image(img, use_container_width=True)
                    st.markdown(f"**{row['title'][:45]}{'...' if len(row['title']) > 45 else ''}**")
                    st.caption(f"📦 {row['department'].title()}")
                    
                    # Show if already in cart
                    if int(row["product_id"]) in st.session_state.cart:
                        st.success("✓ In Cart")
                    else:
                        if st.button("➕ Add", key=f"add-{row['product_id']}", use_container_width=True):
                            add_to_cart(int(row["product_id"]))
                            st.rerun()

# ==================== CART PAGE ====================
elif st.session_state.current_page == "cart":
    col_back, col_title = st.columns([1, 5])
    
    with col_back:
        if st.button("← Shop", use_container_width=True):
            go_to_shop()
    
    with col_title:
        st.markdown(f'<p class="section-header">🛒 Shopping Cart ({len(st.session_state.cart)} items)</p>', unsafe_allow_html=True)
    
    if not st.session_state.cart:
        st.info("🛒 Your cart is empty. Start shopping to see personalized recommendations!")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🛍️ Start Shopping", type="primary", use_container_width=True):
                go_to_shop()
    else:
        cart_df = DF[DF["product_id"].isin(st.session_state.cart)]
        
        # Display each cart item with recommendations beside it
        for idx, (_, cart_item) in enumerate(cart_df.iterrows()):
            if idx > 0:
                st.markdown("---")
            
            # Cart item on left, recommendations on right
            item_col, rec_col = st.columns([1, 2])
            
            with item_col:
                st.markdown('<div class="cart-item">', unsafe_allow_html=True)
                img = load_product_image(cart_item["department"])
                st.image(img, width=150)
                st.markdown(f"**{cart_item['title']}**")
                st.caption(f"📦 {cart_item['department'].title()}")
                st.caption(f"🆔 Product ID: {cart_item['product_id']}")
                
                col_rm, col_qty = st.columns([1, 1])
                with col_rm:
                    if st.button("🗑️ Remove", key=f"rm-{cart_item['product_id']}", use_container_width=True, type="secondary"):
                        remove_from_cart(int(cart_item["product_id"]))
                        st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
            
            with rec_col:
                st.markdown('<div class="rec-section">', unsafe_allow_html=True)
                st.markdown(f"**💡 Recommended with** *{cart_item['title'][:35]}...*")
                
                # Get recommendations for this specific item
                item_recs = get_item_recommendations(int(cart_item["product_id"]), top_k=4)
                
                if len(item_recs) > 0:
                    rec_cols = st.columns(4)
                    for i, (_, rec) in enumerate(item_recs.iterrows()):
                        with rec_cols[i % 4]:
                            rec_img = load_product_image(rec["department"])
                            st.image(rec_img, use_container_width=True)
                            st.caption(f"{rec['title'][:28]}...")
                            if int(rec["product_id"]) in st.session_state.cart:
                                st.caption("✓ In Cart")
                            else:
                                if st.button("➕", key=f"add-rec-{cart_item['product_id']}-{rec['product_id']}", use_container_width=True):
                                    add_to_cart(int(rec["product_id"]))
                                    st.rerun()
                else:
                    st.info("No similar items available")
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Overall recommendations at the bottom
        st.markdown("---")
        st.markdown('<p class="section-header">🌟 You May Also Like</p>', unsafe_allow_html=True)
        st.caption(f"Personalized for User {st.session_state.user_id} based on your cart")
        
        overall_recs = get_cart_recommendations(st.session_state.cart, top_k=10)
        rec_cols = st.columns(5)
        
        for i, (_, rec) in enumerate(overall_recs.iterrows()):
            with rec_cols[i % 5]:
                rec_img = load_product_image(rec["department"])
                st.image(rec_img, use_container_width=True)
                st.markdown(f"**{rec['title'][:35]}...**")
                st.caption(rec["department"].title())
                if int(rec["product_id"]) in st.session_state.cart:
                    st.caption("✓ In Cart")
                else:
                    if st.button("➕ Add", key=f"add-overall-{rec['product_id']}", use_container_width=True):
                        add_to_cart(int(rec["product_id"]))
                        st.rerun()

st.markdown("---")
st.caption(f"🤖 Powered by ALS, Item-Item CF, and LightGBM Ranker | User: {st.session_state.user_id} | Cart: {len(st.session_state.cart)} items")
