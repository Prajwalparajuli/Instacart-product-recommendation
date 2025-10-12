# src/similarity_agg_als_score.py
# ------------------------------------------------------------------------------
# PURPOSE
#   Build ONE handoff features file for downstream models by combining:
#     1) ALS user–product affinity scores, and
#     2) Item–item similarity aggregates computed against each user’s history.
#
#   Output schema (one row per (user_id, product_id) candidate):
#     user_id, product_id, als_score,
#     sim_max, sim_mean, sim_topk_mean_10, sim_nonzero_cnt
#
# WHY THIS FILE EXISTS
#   • Keeps the pipeline modular but easy to consume: QDA/Ranker gets one CSV.
#   • Matches our project report: top-10 neighbor logic for item–item features.
#
# INPUT FILES (must already exist in data/processed):
#   1) als_scores.csv
#        Columns: user_id, product_id, als_score
#        Meaning: ALS-predicted affinity for each (user, candidate product).
#
#   2) als_interaction_user_product_counts.csv
#        Columns: user_id, product_id, count (we only need user_id, product_id)
#        Meaning: historical purchases (prior orders) that define each user’s
#                 “history items”. Duplicates may exist; we de-duplicate.
#
#   3) item_item_similarity_score.csv
#        Columns: product_id_1, product_id_2, similarity
#        Meaning: similarity edge list. We interpret product_id_1 as a
#                 history item and product_id_2 as a candidate product.
#
# OUTPUT FILE (written here):
#   data/processed/user_product_features.csv
#
# KEY IDEAS
#   • For each (user, candidate) pair from ALS, look up all the user’s history
#     items and gather similarities between (history item ↔ candidate).
#   • Aggregate those similarities to produce robust features:
#       - sim_max:       strongest single match to the user’s history
#       - sim_mean:      average signal over all available history matches
#       - sim_topk_mean_10: average of the top-10 strongest matches
#       - sim_nonzero_cnt: how many history links existed (coverage signal)
#
#   • Missing similarities (no overlap or edge not present) are treated as NaN
#     during aggregation and then filled with 0.0 in the final output so the
#     downstream model can handle “cold-ish” user–candidate pairs.
#
# ASSUMPTIONS / CONTRACTS
#   • Product IDs in ALS and the similarity file share the SAME id space.
#   • item_item_similarity_score.csv already reflects your chosen metric
#     (cosine/Jaccard/etc.) and has been pruned to top-10 neighbors per item.
#   • TOPK=10 is the project standard (per our first report).
#
# PERFORMANCE NOTES
#   • This script uses pandas merges + groupby; on full Instacart scale it’s fine
#     on a modern laptop. If memory is tight, batch by user_id or switch to
#     DuckDB/Polars with the same join logic.
#
# VERSIONING / AUDITABILITY
#   • Keep this script name stable: similarity_agg_als_score.py (canonical).
#   • Inputs are read-only; output is reproducible given the same inputs.
# ------------------------------------------------------------------------------

from pathlib import Path
import pandas as pd
import numpy as np

# ----------------------------- CONFIG -----------------------------------------
PROCESSED = Path("data/processed")

# If you use als_scores_prior_pairs.csv instead, change ALS_PATH below.
ALS_PATH  = PROCESSED / "als_scores_prior_pairs.csv"
HIST_PATH = PROCESSED / "als_interaction_user_product_counts.csv"
SIM_PATH  = PROCESSED / "item_item_similarity_score.csv"

OUT_PATH  = PROCESSED / "user_product_features.csv"

TOPK = 10          # Project-standard: we aggregate over top-10 neighbors.
FILL_VALUE = 0.0   # Final fill for missing sim aggregates (no overlap/edges).
# ------------------------------------------------------------------------------


def load_als() -> pd.DataFrame:
    """
    Load ALS predictions for user–product candidates.

    Expected columns:
      • user_id (int)
      • product_id (int)
      • als_score (float)

    Why strict usecols?
      • Fails fast if the schema drifts.
      • Avoids silently pulling extra/incorrect columns.
    """
    df = pd.read_csv(ALS_PATH, usecols=["user_id", "product_id", "als_score"])
    # Enforce dtypes for smaller memory + deterministic merges
    df["user_id"] = df["user_id"].astype(np.int64)
    df["product_id"] = df["product_id"].astype(np.int64)
    df["als_score"] = df["als_score"].astype(np.float32)
    # Optional sanity: check for duplicates in the (user, product) key
    # Duplicates would make "one_to_one" validations fail later.
    return df


def load_history() -> pd.DataFrame:
    """
    Load user purchase history derived from ALS interactions.

    We only need (user_id, product_id) to define "history items".
    Any 'count' column is ignored. We drop duplicates to prevent double-counting
    the same history item in aggregates.

    Output columns:
      • user_id (int)
      • hist_product_id (int)
    """
    hist = pd.read_csv(HIST_PATH, usecols=["user_id", "product_id"]).drop_duplicates()
    hist["user_id"] = hist["user_id"].astype(np.int64)
    hist["product_id"] = hist["product_id"].astype(np.int64)
    return hist.rename(columns={"product_id": "hist_product_id"})


def load_similarity() -> pd.DataFrame:
    """
    Load item–item similarity edges.

    Input columns:
      • product_id_1: treated as history item id
      • product_id_2: treated as candidate product id
      • similarity:   float similarity score (cosine/Jaccard/etc.)

    Output columns:
      • hist_product_id
      • candidate_product_id
      • similarity
    """
    sim = pd.read_csv(SIM_PATH, usecols=["product_id", "neighbor_product_id", "score", "rank"])
    sim = sim.rename(columns={
        "product_id": "hist_product_id",
        "neighbor_product_id": "candidate_product_id",
        "score": "similarity"
    })
    sim["hist_product_id"] = sim["hist_product_id"].astype(np.int64)
    sim["candidate_product_id"] = sim["candidate_product_id"].astype(np.int64)
    sim["similarity"] = sim["similarity"].astype(np.float32)
    return sim


def topk_mean(s: pd.Series, k: int) -> float:
    """
    Compute the mean of the top-k values in a Series, ignoring NaNs.

    Why partition instead of full sort?
      • np.partition costs O(n) average vs O(n log n) for sort;
        faster when lists are long.

    Edge cases:
      • If all values are NaN or the series is empty => return NaN (handled later).
      • If len(values) < k => use all available values.
    """
    vals = s.dropna().values
    if vals.size == 0:
        return np.nan
    k = min(k, vals.size)
    # np.partition places the k largest elements at the end, unordered among themselves
    thresholded = np.partition(vals, -k)[-k:]
    return float(thresholded.mean())


def build_features(als: pd.DataFrame, hist: pd.DataFrame, sim: pd.DataFrame) -> pd.DataFrame:
    """
    Core transformation:
      1) Expand ALS (user, candidate) by user history -> (user, candidate, hist_prod).
      2) Join similarity on (hist_prod, candidate).
      3) Groupby (user, candidate) and aggregate.

    Why left joins?
      • We keep every ALS candidate even if the user has zero history
        or there are no similarity edges — downstream models still want ALS.
    """
    # Rename to "candidate_product_id" to make join keys explicit and unambiguous.
    als_cand = als.rename(columns={"product_id": "candidate_product_id"})

    # Step 1: Cross each (user, candidate) with that user's history items.
    # This creates the rows along which we can fetch item–item similarities.
    als_expanded = als_cand.merge(hist, on="user_id", how="left")
    # If a user has no history, hist_product_id will be NaN; that's OK.

    # Step 2: Attach similarity edges for (hist_product_id, candidate_product_id).
    merged = als_expanded.merge(
        sim,
        on=["hist_product_id", "candidate_product_id"],
        how="left"  # keep rows even if no similarity edge exists
    )

    # Step 3: Aggregate similarities by (user, candidate).
    grouped = merged.groupby(["user_id", "candidate_product_id"], sort=False)

    agg = grouped["similarity"].agg(
        # Max similarity: "closest neighbor" strength
        sim_max="max",
        # Mean similarity: "average affinity" over all available history items
        sim_mean="mean",
        # Count of non-null similarities: coverage / evidence size
        sim_nonzero_cnt=lambda x: x.notna().sum()
    ).reset_index()

    # Top-k mean (k=10 per project spec). We compute it separately to keep the
    # code readable and to avoid writing a custom agg with closures.
    topk = grouped["similarity"].apply(lambda x: topk_mean(x, TOPK)) \
                                .reset_index(name=f"sim_topk_mean_{TOPK}")

    # Merge aggregates back to the ALS candidates; restore original column name.
    out = (
        als_cand
        .merge(agg,  on=["user_id", "candidate_product_id"], how="left")
        .merge(topk, on=["user_id", "candidate_product_id"], how="left")
        .rename(columns={"candidate_product_id": "product_id"})
    )

    # Fill missing aggregate values:
    #   • Users with no history or no edges produce NaNs; models typically
    #     prefer an explicit 0.0 (meaning "no similarity signal observed").
    for c in ["sim_max", "sim_mean", f"sim_topk_mean_{TOPK}"]:
        out[c] = out[c].fillna(FILL_VALUE).astype(np.float32)
    out["sim_nonzero_cnt"] = out["sim_nonzero_cnt"].fillna(0).astype(np.int32)

    # Final, stable column order for downstream merges/model inputs.
    cols = [
        "user_id",
        "product_id",
        "als_score",
        "sim_max",
        "sim_mean",
        f"sim_topk_mean_{TOPK}",
        "sim_nonzero_cnt",
    ]
    return out[cols]


def main():
    # Ensure the processed directory exists (idempotent)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    # Load inputs with strict schemas. If any file is missing columns,
    # this will fail early with a clear error.
    als  = load_als()
    hist = load_history()
    sim  = load_similarity()

    # Fast sanity: there should be some overlap between ALS product ids and
    # the candidate product ids in the similarity graph. If not, you likely
    # have an ID mapping issue (e.g., reindexing done in one step but not the other).
    overlap = np.intersect1d(als["product_id"].unique(), sim["candidate_product_id"].unique())
    if overlap.size == 0:
        raise ValueError(
            "No overlap between ALS product_id and similarity candidate_product_id. "
            "Check that both use the same product index mapping."
        )

    # Compute features and write output.
    features = build_features(als, hist, sim)
    features.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}  rows={len(features):,}")


if __name__ == "__main__":
    main()
