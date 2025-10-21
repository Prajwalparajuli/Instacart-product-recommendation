### Compute item-item similarity for candidates (minimal changes; counts file removed) ###

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import warnings
from pathlib import Path
import time

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)

# --- 1. Data Loading and Initial Preparation ---
print("Step 1: Loading and preparing data...")
try:
    # CHANGED: use repo-relative processed path; remove hard-coded Windows path
    BASKETS_CSV = Path("data/processed/item_item_basket.csv")
    if not BASKETS_CSV.exists():
        raise FileNotFoundError(f"{BASKETS_CSV} not found")

    # Load the datasets
    df_baskets = pd.read_csv(BASKETS_CSV)

    print("Data loaded successfully.")
    
    # Ensure data types are correct for memory efficiency
    df_baskets['order_id'] = df_baskets['order_id'].astype('category')
    df_baskets['product_id'] = df_baskets['product_id'].astype('category')

    # Get all unique products and map them to a contiguous index
    product_ids = df_baskets['product_id'].cat.categories
    product_to_index = {product: i for i, product in enumerate(product_ids)}
    index_to_product = {i: product for product, i in product_to_index.items()}
    
except FileNotFoundError as e:
    print(f"Error: {e}. Make sure the CSV files are in the expected directory.")
    raise SystemExit(1)

# --- 2. Create the Item-Order Sparse Matrix ---
print("\nStep 2: Creating the item-order sparse matrix...")
rows = df_baskets['product_id'].cat.codes.to_numpy()
cols = df_baskets['order_id'].cat.codes.to_numpy()
data = np.ones(len(df_baskets), dtype=np.uint8)
item_order_matrix = csr_matrix(
    (data, (rows, cols)),
    shape=(len(product_ids), len(df_baskets['order_id'].cat.categories))
)

print("Shape of the sparse item-order matrix:", item_order_matrix.shape)

# (CHANGED) Derive product frequencies internally (no external counts file)
# Each row sum = number of orders containing that product
freq = np.asarray(item_order_matrix.sum(axis=1)).ravel().astype(np.float32)
inv_sqrt = 1.0 / np.sqrt(np.clip(freq, 1e-12, None))
# Pre-normalize rows so that Xn @ Xn.T yields cosine-like similarity
Xn = item_order_matrix.multiply(inv_sqrt[:, None]).tocsr()


# --- 3. Calculate Item-Item Similarity (Mini-batch; no full matrix) ---
print("\nStep 3: Calculating item-item similarity matrix (mini-batch, Top-K only)...")

K = 10  # keep original K default
BATCH_SIZE = 512  # (ADDED) mini-batch size for faster sparse BLAS
n_products = len(product_ids)
top_k_similar_items = {}

def _topk_from_sparse_row(indptr, indices, data, local_row_idx, global_row_idx, K):
    """
    Extract Top-K for one row in a CSR block.
    Deterministic ordering: score desc, index asc. Excludes self.
    """
    start, end = indptr[local_row_idx], indptr[local_row_idx + 1]
    js = indices[start:end]
    vals = data[start:end].astype(np.float32)

    # Exclude self
    mask = (js != global_row_idx)
    js = js[mask]
    vals = vals[mask]
    if js.size == 0:
        return []

    keep = min(K, js.size)
    sel = np.argpartition(vals, -keep)[-keep:]
    js_top = js[sel]
    vals_top = vals[sel]

    # Deterministic order: score desc, then neighbor index asc
    order = np.lexsort((js_top, -vals_top))
    js_top = js_top[order]
    vals_top = vals_top[order]

    return [{'product_id': index_to_product[j], 'score': float(v)} for j, v in zip(js_top, vals_top)]

t0 = time.time()
for start in range(0, n_products, BATCH_SIZE):
    end = min(start + BATCH_SIZE, n_products)
    # (CHANGED) Batch multiply against all items; stays sparse
    S_chunk = Xn[start:end, :].dot(Xn.T).tocsr()
    indptr, indices, dataS = S_chunk.indptr, S_chunk.indices, S_chunk.data

    for local_i in range(end - start):
        global_i = start + local_i
        pid = product_ids[global_i]
        top_k_similar_items[pid] = _topk_from_sparse_row(indptr, indices, dataS, local_i, global_i, K)

    if (end % 2000) == 0 or end == n_products:
        print(f"  processed {end}/{n_products} items in {time.time()-t0:.1f}s")

print("\nStep 4: Finding Top-K similar items for each product...")
# NOTE: Already done per row during Step 3 (Top-K extracted from each batch)

print("\nTop 3 Similar Items for a sample product (ID: 24852):")
if 24852 in top_k_similar_items:
    top_3 = top_k_similar_items[24852][:3]
    for item in top_3:
        item['score'] = round(item['score'], 5)
    print(top_3 if len(top_3) > 0 else "[no neighbors found]")
else:
    print("Sample product ID 24852 not found in the sample data.")

# --- 5. Generate Outputs (Example Usage) ---
print("\nStep 5: Generating candidate items for a user...")

# (CHANGED) Aggregate by SUM (better multi-anchor signal). 'max' still supported.
def get_candidate_items(user_recent_purchases, num_recommendations=5, agg="sum"):
    candidate_items = {}
    recent_set = set(user_recent_purchases)
    for product_id in user_recent_purchases:
        if product_id in top_k_similar_items:
            for item in top_k_similar_items[product_id]:
                candidate_product = item['product_id']
                score = float(item['score'])
                if candidate_product in recent_set:
                    continue
                if agg == "sum":
                    candidate_items[candidate_product] = candidate_items.get(candidate_product, 0.0) + score
                else:  # 'max'
                    if (candidate_product not in candidate_items) or (score > candidate_items[candidate_product]):
                        candidate_items[candidate_product] = score

    sorted_candidates = sorted(candidate_items.items(), key=lambda x: (-x[1], str(x[0])))
    return [item[0] for item in sorted_candidates[:num_recommendations]]

# Using product IDs known to be popular from your counts file (example values kept)
user_purchases = [13176, 21137] 
recommended_candidates = get_candidate_items(user_purchases, num_recommendations=5, agg="sum")

print(f"\nCandidate items for a user who recently bought products {user_purchases}:")
print(recommended_candidates)

# (CHANGED) Save compact artifact for downstream models (CSV only; mirrors ALS style)
OUT_CSV = Path("data/processed/item_item_similarity_score.csv")
rows_out = []
for pid, neighbors in top_k_similar_items.items():
    for rank, n in enumerate(neighbors, start=1):
        rows_out.append([pid, n["product_id"], float(n["score"]), rank])

df_out = pd.DataFrame(rows_out, columns=["product_id", "neighbor_product_id", "score", "rank"])
df_out["score"] = df_out["score"].astype("float32")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df_out.to_csv(OUT_CSV, index=False)

print("\nSaved item-item similarity scores to:", OUT_CSV.as_posix())
print(f"Total elapsed: {time.time()-t0:.1f}s")
print("\nProcess finished successfully.")