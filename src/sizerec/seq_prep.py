"""
Sequence preparation for fit_outcome prediction.

What this does:
- Join product attributes onto transactions.
- Encode categorical features to integer IDs using vocabs.
- Build next-step examples: for each consumer at step t>0, input = previous steps (up to max_len), target = fit_outcome at t.
- Make consumer-based train/val/test splits.
- Save encoded splits to CSV; list columns are serialized as JSON strings.

Assumptions:
- Required columns exist in the provided DataFrames (from datagen).
- Vocabs are dicts: {token -> id} for each feature (built via sizerec.vocab).
- label_map is {fit_label -> class_id} using the fixed order from config.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sizerec.vocab import _normalize_shoe_str

# ---------------------------
# 1) Join product attributes
# ---------------------------
def join_product_attrs(transactions_df: pd.DataFrame, products_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join product_type, section, material onto transactions by product_id."""
    cols = ["product_id", "product_type", "section", "material"]
    prod = products_df[cols].copy()
    # available_countries not needed for modeling inputs here
    df = transactions_df.merge(prod, on="product_id", how="left")
    return df


# ---------------------------
# 2) Encode features to IDs
# ---------------------------
def _bin_ages(consumers_df: pd.DataFrame, age_bins: List[int]) -> pd.Series:
    """Return string bin labels for age using inclusive-left bins and a closed last bin."""
    labels = []
    for i in range(len(age_bins) - 2):
        labels.append(f"[{age_bins[i]},{age_bins[i+1]})")
    labels.append(f"[{age_bins[-2]},{age_bins[-1]}]")
    # right=False makes it [left,right)
    bins = pd.cut(consumers_df["age"], bins=age_bins, right=False, include_lowest=True)
    # map Interval -> label
    label_map = {}
    for i, lab in enumerate(labels):
        label_map[pd.Interval(left=age_bins[i], right=age_bins[i+1], closed="left")] = lab
    # The final catch-all interval is closed='left' except pandas represents last with right edge
    # pd.cut with right=False already treats last as [left, right)
    # We'll handle any nulls by assigning the last label
    out = bins.map(label_map)
    out = out.fillna(labels[-1])
    return out


def encode_features(
    consumers_df: pd.DataFrame,
    products_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    *,
    vocabs: Dict[str, Dict[str, int]],
    size_vocab: Dict[str, int],
    label_map: Dict[str, int],
    age_bins: List[int],
    use_section: bool,
    use_country: bool,
) -> pd.DataFrame:
    """Produce an encoded, time-sorted transactions table with integer IDs only."""
    # 1) Join product attrs onto transactions
    df = join_product_attrs(transactions_df, products_df)

    # 2) Basic sorting
    df = df.sort_values(["consumer_id", "transaction_date"]).reset_index(drop=True)

    # 3) Compute consumer-level age_bin and encode gender/country once
    consumers = consumers_df[["consumer_id", "gender", "country", "age"]].copy()
    consumers["age_bin_lbl"] = _bin_ages(consumers, age_bins)
    # Encode static fields
    consumers["gender_id"] = consumers["gender"].astype(str).map(vocabs["gender"])
    consumers["age_bin_id"] = consumers["age_bin_lbl"].astype(str).map(vocabs["age_bin"])
    if use_country:
        consumers["country_id"] = consumers["country"].astype(str).map(vocabs["country"])

    # 4) Attach encoded static fields to each transaction row
    keep_cols = ["consumer_id", "gender_id", "age_bin_id"] + (["country_id"] if use_country else [])
    df = df.merge(consumers[keep_cols], on="consumer_id", how="left")

    # 5) Encode step-wise fields
    df["product_type_id"] = df["product_type"].astype(str).map(vocabs["product_type"])
    df["material_id"] = df["material"].astype(str).map(vocabs["material"])
    if use_section:
        df["section_id"] = df["section"].astype(str).map(vocabs["section"])

    def _to_size_token(x):
        # normalize numbers to the same string form the vocab used ("10" not "10.0")
        try:
            return _normalize_shoe_str(float(x))
        except Exception:
            return str(x)
    
    size_str = df["purchased_size"].apply(_to_size_token)
    df["size_id"] = size_str.map(size_vocab)

    # label
    df["label_id"] = df["fit_outcome"].astype(str).map(label_map)

    return df


# ---------------------------
# 3) Build next-step examples
# ---------------------------
def build_examples(
    encoded_df: pd.DataFrame,
    *,
    max_len: int,
    use_section: bool,
    use_country: bool,
) -> pd.DataFrame:
    """
    For each consumer and each t>0:
      - context = rows < t (most recent up to max_len)
      - target  = label_id at t

    Returns a flat DataFrame with list columns serialized as JSON strings to keep it simple.
    Columns:
      consumer_id, seq_len, product_type_ids, material_ids, size_ids,
      (section_ids?), gender_id, (country_id?), age_bin_id, label_id,
      product_type_id_t (optional feature for analysis), transaction_date_t
    """
    rows = []
    group_cols = ["consumer_id"]
    step_cols = ["product_type_id", "material_id", "size_id"] + (["section_id"] if use_section else [])
    static_cols = ["gender_id", "age_bin_id"] + (["country_id"] if use_country else [])

    for cid, grp in encoded_df.groupby("consumer_id", sort=False):
        grp = grp.sort_values("transaction_date")
        n = len(grp)
        if n < 2:
            continue

        # Extract arrays once
        step_vals = {col: grp[col].tolist() for col in step_cols}
        labels = grp["label_id"].tolist()
        dates = grp["transaction_date"].tolist()

        # Static (same for all examples of this consumer)
        static_vals = {col: int(grp[col].iloc[0]) for col in static_cols}

        for t in range(1, n):
            # context indices are [max(0, t-max_len) .. t-1]
            start = max(0, t - max_len)
            ctx_len = t - start

            # Build context lists (slice and keep order)
            rec = {
                "consumer_id": cid,
                "seq_len": ctx_len,
                "label_id": int(labels[t]),
                "transaction_date_t": dates[t],
                "product_type_id_t": int(grp["product_type_id"].iloc[t]),
                "material_id_t": int(grp["material_id"].iloc[t]),
                "size_id_t": int(grp["size_id"].iloc[t]),
            }

            if use_section:
                rec["section_id_t"] = int(grp["section_id"].iloc[t])

            # add step-wise lists (JSON strings)
            for col in step_cols:
                rec[f"{col}s"] = json.dumps(step_vals[col][start:t])

            # add statics
            for col, val in static_vals.items():
                rec[col] = val

            rows.append(rec)

    out = pd.DataFrame(rows)

    # Keep a stable column order
    ordered = ["consumer_id", "seq_len"] \
          + [f"{c}s" for c in step_cols] \
          + static_cols \
          + ["label_id",
             "product_type_id_t", "material_id_t", "size_id_t"] \
          + (["section_id_t"] if use_section else []) \
          + ["transaction_date_t"]
    return out[ordered]


# ---------------------------
# 4) Consumer-based splits
# ---------------------------
def split_by_consumer(
    examples_df: pd.DataFrame,
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by distinct consumers into train/val/test (80/10/10 by default)."""
    rng = np.random.RandomState(seed)
    consumers = examples_df["consumer_id"].unique().tolist()
    rng.shuffle(consumers)

    n = len(consumers)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_ids = set(consumers[:n_train])
    val_ids = set(consumers[n_train : n_train + n_val])
    test_ids = set(consumers[n_train + n_val :])

    train_df = examples_df[examples_df["consumer_id"].isin(train_ids)].reset_index(drop=True)
    val_df   = examples_df[examples_df["consumer_id"].isin(val_ids)].reset_index(drop=True)
    test_df  = examples_df[examples_df["consumer_id"].isin(test_ids)].reset_index(drop=True)
    return train_df, val_df, test_df


# ---------------------------
# 5) Save processed splits
# ---------------------------
def save_processed_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: Path) -> None:
    """Save train/val/test as CSVs with list columns already JSON-encoded."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train.csv", index=False, encoding="utf-8")
    val_df.to_csv(out_dir / "val.csv", index=False, encoding="utf-8")
    test_df.to_csv(out_dir / "test.csv", index=False, encoding="utf-8")
