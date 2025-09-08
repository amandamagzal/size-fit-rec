"""
Vocabulary builders for fit_outcome modeling.

Creates integer ID mappings for:
- fit_outcome labels (from config order)
- categorical inputs: product_type, material, gender, (optional) section, (optional) country, age_bin
- size tokens for purchased_size (apparel sizes + shoe sizes as strings)

Assumptions:
- Required columns exist in the provided DataFrames.
- Age is binned using the provided edges; last edge is a catch-all.
"""

from pathlib import Path
import json
import pandas as pd

# Use your canonical apparel order from datagen (fallback included)
try:
    from datagen.constants import SIZES as CANON_APPAREL_SIZES
except Exception:
    CANON_APPAREL_SIZES = ["2XS", "XS", "S", "M", "L", "XL", "2XL"]


# ------------------------
# Label mapping
# ------------------------
def build_fit_label_map(label_order):
    """Map fit_outcome strings to class IDs based on the given order."""
    return {lab: i for i, lab in enumerate(label_order)}


# ------------------------
# Size vocab (inputs)
# ------------------------
def _normalize_shoe_str(x):
    """Turn a numeric-like shoe size into a compact string ('10.0' -> '10')."""
    try:
        s = f"{float(x)}"
        return s[:-2] if s.endswith(".0") else s
    except Exception:
        return str(x)

def infer_size_strings(transactions_df):
    """Return ordered size tokens (strings): apparel (canonical order) + shoes (numeric order)."""
    vals = transactions_df["purchased_size"].dropna().unique().tolist()

    apparel = set()
    shoes = []

    canon = set(CANON_APPAREL_SIZES)
    for v in vals:
        if isinstance(v, str) and v in canon:
            apparel.add(v)
        else:
            # try numeric shoe
            try:
                shoes.append(float(v))
            except Exception:
                apparel.add(str(v))  # unknown string size

    # apparel: canonical first, then any extras sorted
    extras = sorted([s for s in apparel if s not in CANON_APPAREL_SIZES])
    apparel_ordered = [s for s in CANON_APPAREL_SIZES if s in apparel] + extras

    # shoes: numeric sort, then stringify
    shoes_ordered = [_normalize_shoe_str(x) for x in sorted(set(shoes))]

    return apparel_ordered + shoes_ordered

def build_size_vocab(transactions_df):
    """Map size token (string) -> id."""
    ordered = infer_size_strings(transactions_df)
    return {tok: i+1 for i, tok in enumerate(ordered)}


# ------------------------
# Categorical vocabs (inputs)
# ------------------------
def _age_bin_labels(edges):
    """Make simple bin labels like '[18,25)' ... '[66,200]' from edges."""
    labels = []
    for i in range(len(edges) - 2):
        labels.append(f"[{edges[i]},{edges[i+1]})")
    labels.append(f"[{edges[-2]},{edges[-1]}]")
    return labels

def build_categorical_vocabs(
    consumers_df,
    products_df,
    transactions_df,  # kept for symmetry; not used here
    *,
    use_section: bool,
    use_country: bool,
    age_bins,
):
    """
    Build minimal vocabs for categorical inputs.

    Returns a dict:
      {
        "product_type": {...},
        "material": {...},
        "gender": {...},
        "age_bin": {...},
        ("section": {...})?,
        ("country": {...})?
      }
    """
    vocabs = {}

    # From products
    vocabs["product_type"] = {tok: i+1 for i, tok in enumerate(sorted(products_df["product_type"].astype(str).unique()))}
    vocabs["material"]     = {tok: i+1 for i, tok in enumerate(sorted(products_df["material"].astype(str).unique()))}

    # From consumers
    vocabs["gender"]       = {tok: i for i, tok in enumerate(sorted(consumers_df["gender"].astype(str).unique()))}

    if use_section:
        vocabs["section"] = {tok: i+1 for i, tok in enumerate(sorted(products_df["section"].astype(str).unique()))}

    if use_country:
        vocabs["country"] = {tok: i for i, tok in enumerate(sorted(consumers_df["country"].astype(str).unique()))}

    # Age bins (labels reflect edges order)
    labels = _age_bin_labels(age_bins)
    vocabs["age_bin"] = {lab: i for i, lab in enumerate(labels)}

    return vocabs


# ------------------------
# Save / Load
# ------------------------
def save_vocabs(vocabs, out_dir: Path):
    """Save all vocabs to <out_dir>/vocabs.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "vocabs.json").open("w", encoding="utf-8") as f:
        json.dump(vocabs, f, ensure_ascii=False, indent=2)

def load_vocabs(out_dir: Path):
    """Load vocabs from <out_dir>/vocabs.json."""
    with (Path(out_dir) / "vocabs.json").open("r", encoding="utf-8") as f:
        return json.load(f)
