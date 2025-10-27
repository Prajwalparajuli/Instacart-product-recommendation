###Train ALS (implicit matrix factorization) to produce candidate items.###

# Build the confidence weighted user-product matrix
""" This step takes the processed Coordinate sparse matrix (COO) file (row = user_index, col=product_index, val=count)
    and transform it into the Compressed Sparse Row (CSR) format required by the implicit library."""

# Each entry is confidence-weighted using:
#  confidence = 1 + alpha * count
#  where alpha is a hyperparameter (default=40) and count is the number of times a user purchased a product.

import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix


# File paths for ALS COO files, User_index, and product_index maps files
COO_FILE = r"data/processed/als_coo.csv"
USER_MAP_FILE = r"data/processed/als_user_index_map.csv"
PROD_MAP_FILE = r"data/processed/als_product_index_map.csv"

# Load processed files
interaction_df = pd.read_csv(COO_FILE)
user_index_map = pd.read_csv(USER_MAP_FILE) # Column = user_id, user_index
prod_index_map = pd.read_csv(PROD_MAP_FILE) # Column = product_id, product_index

# Matrix dimensions
num_users = int(interaction_df['row'].max()) + 1
num_products = int(interaction_df['col'].max()) + 1

# Confidence weighting 
alpha = 40
purchase_counts = interaction_df['val'].astype(np.float32)
confidence_values = (1 + alpha * purchase_counts).astype(np.float32)

# Build sparse matrix (CSR format)
confidence_matrix: csr_matrix = coo_matrix(
    (confidence_values,
     (interaction_df['row'].to_numpy(), interaction_df['col'].to_numpy())),
      shape = (num_users, num_products),
       dtype=np.float32).tocsr()

print("Confidence matrix shape (user x product):", confidence_matrix.shape)
print("Non-zero entries in confidence matrix:", confidence_matrix.nnz)
print("Alpha =", alpha, "| first 5 confidence values:", confidence_values.iloc[:5].tolist())


""" 
Train ALS on the confidence-weighted user-product matrix.
"""
# Import the ALS model from the implicit library

from implicit.als import AlternatingLeastSquares

# Starting hyperparameters
num_factors = 64        # Number of latent factors to use (k) We will try 32, 64, 128
regularization = 0.1    # lambda: L2 regularization parameter; penalty to prevent overfitting
num_iterations = 15     # Number of alternating updates (user - item and item - user)
random_state = 42       # Random seed for reproducibility
use_gpu = False         # Set True for compatible GPU, else False for CPU

# Build ITEM x USER matrix for implicit library
# Confidence matrix is USER x ITEM (rows = users, cols = items) and implicit expects the transpose
item_user_confidence = confidence_matrix.T.tocsr()

# Initialize the ALS model
als_model = AlternatingLeastSquares(factors = num_factors,
                                    regularization = regularization,
                                    iterations = num_iterations,
                                    random_state = random_state,
                                    use_gpu = use_gpu)

# FIt the model to the item-user confidence matrix
als_model.fit(item_user_confidence)

# Confimation prints
print("ALS model trained")
print(f"factors = {num_factors} | regularization = {regularization} | iterations = {num_iterations}")
print("User {cols}:", item_user_confidence.shape[1], "| Items {rows}:", item_user_confidence.shape[0])

# Convergence is indicated by a decreasing loss function printed at each iteration
# If the loss is increasing, consider increasing the regularization parameter
# The final loss is the sum of squared differences between the confidence values and the predicted values
# A lower loss indicates a better fit to the data, but does not necessarily mean better recommendations
# Convergence check: if the loss is not decreasing, consider increasing the regularization parameter

user_factors = np.asarray(als_model.user_factors, dtype=np.float32)
item_factors = np.asarray(als_model.item_factors, dtype=np.float32)

# Quick drift check against a few extra iterations
als_model_more = AlternatingLeastSquares(factors = 64,
                                        regularization = 0.1,
                                        iterations = als_model.iterations + 5,
                                        random_state = 42)
als_model_more.fit(item_user_confidence)

user_factors_more = np.asarray(als_model_more.user_factors, dtype=np.float32)
item_factors_more = np.asarray(als_model_more.item_factors, dtype=np.float32)

# Cosine similarity between original and more-trained factors
def avg_row_cosine_similarity(A, B):
    """Compute the average cosine similarity between corresponding rows of two matrices."""
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return float(np.mean(np.sum(A * B, axis=1)))

print("Avg user-factors cosine similarity (original vs more-trained):",
      avg_row_cosine_similarity(user_factors, user_factors_more))
print("Avg product-factors cosine similarity (original vs more-trained):",
      avg_row_cosine_similarity(item_factors, item_factors_more))

# The cosine similarity simulataions ~0.992 for users and ~0.994 for products mean the factors are barely changed
# So the corrent parameters effectively converge the model, so no need to tune iterations further

""""
Extract user/item factors and define a scorer
"""
# Saving the factors to data/processed so we can reuse without retraining
# This provides a fast scorer for (user_index, product_index) pairs

from pathlib import Path

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Get factors from implicit ALS model
user_factors_np = np.asarray(als_model.user_factors, dtype=np.float32)
item_factors_np = np.asarray(als_model.item_factors, dtype=np.float32)

print("User factors shape:", user_factors_np.shape)   # (num_users, num_factors(k))
print("Item factors shape:", item_factors_np.shape)   # (num_items, num_factors(k))

# Save factors for reuse
np.save(PROCESSED_DIR / "als_user_factors.npy", user_factors_np)
np.save(PROCESSED_DIR / "als_product_factors.npy", item_factors_np)
print("User and product factors saved to data/processed")

# Load ID index maps
user_map = pd.read_csv(PROCESSED_DIR / "als_user_index_map.csv")       # Columns: user_id, user_index
product_map = pd.read_csv(PROCESSED_DIR / "als_product_index_map.csv") # Columns: product_id, product_index

# Reverse lookup Series: index -> ID
idx_to_user_id = pd.Series(user_map["user_id"].values, 
                           index=user_map["user_index"].values)
idx_to_product_id = pd.Series(product_map["product_id"].values, 
                              index=product_map["product_index"].values) 

# Define a scorer function using the dot product of user and item factors
def als_score_pairs(user_idx_array: np.ndarray, 
                    prod_idx_array: np.ndarray) -> np.ndarray:
    """
    Compute ALS scores for given (user_index, product_index).
    Returns: 1D float32 numpy array of dot product scores
    """
    U = user_factors_np[user_idx_array]      # shape :(n, k)
    P = item_factors_np[prod_idx_array]      # shape: (n, k)
    return np.einsum("ik,ik->i", U, P, optimize=True).astype(np.float32)

# Correct shape unpacking
num_users, num_factors = user_factors_np.shape
num_products, _ = item_factors_np.shape

# Load PRIOR pairs (als_coo.csv should already be loaded as interaction_df)
user_idx_all = interaction_df['row'].to_numpy(dtype = np.int32)
prod_idx_all = interaction_df['col'].to_numpy(dtype = np.int32)

# Validate indices
valid_mask = (user_idx_all >= 0) & (user_idx_all < num_users) & \
             (prod_idx_all >= 0) & (prod_idx_all < num_products)

user_idx = user_idx_all[valid_mask]
product_idx = prod_idx_all[valid_mask]

print(f"Valid pairs: {len(user_idx):,} / {len(interaction_df):,}")

# Score in batches
batch_size = 500_000
scores = np.zeros(len(user_idx), dtype=np.float32)

for start in range(0, len(user_idx), batch_size):
    end = min(start + batch_size, len(user_idx))
    scores[start:end] = als_score_pairs(user_idx[start:end], product_idx[start:end])
    print(f"Scored rows {start} to {end} / {len(user_idx)}")

# Assemble (user_id, product_id, score) DataFrame and save to CSV
als_scores = pd.DataFrame({
    "user_id": idx_to_user_id.reindex(user_idx).to_numpy(),
    "product_id": idx_to_product_id.reindex(product_idx).to_numpy(),
    "als_score": scores
})

out_path = PROCESSED_DIR / "als_scores_prior_pairs.csv"
als_scores.to_csv(out_path, index=False)
print(f"ALS scores for PRIOR pairs saved to {out_path}")
print(als_scores.head())
