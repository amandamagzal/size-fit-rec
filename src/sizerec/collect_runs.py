"""
Consolidate results from artifacts/runs/<run_id>/ into a single CSV/JSON for analysis.

This includes:
- Candidate-size expansion (train/val/test processed CSVs can be very large)
- Cross-entropy-only training (no MSE)
- Key metric for size recommendation: size_accuracy (from metrics_*.json)
- Efficiency: params, epoch time mean, peak GPU memory, full fit time
- Dataset scale: n_consumers/n_products (from config) + raw transactions count (from csv_dir)
- Processed scale: train/val/test processed row counts (optional; can be slow for huge CSVs)

Usage:
  python -m sizerec.collect_runs
  python collect_runs.py

Outputs:
  artifacts/summary/runs_summary.csv
  artifacts/summary/runs_summary.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List

import pandas as pd

from sizerec.paths import RUNS_DIR, ARTIFACTS_DIR, ensure_dir


# ---------------------------
# Helpers
# ---------------------------
def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_get(d: Dict[str, Any], path: str, default=None):
    """
    Safe nested getter with dot-path keys.
    Example: _safe_get(cfg, "data.max_len")
    """
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _count_csv_rows(csv_path: Path) -> Optional[int]:
    """
    Count number of rows in a CSV quickly by counting lines (minus header).
    This is faster than pandas.read_csv for very large files.
    Returns None if file doesn't exist.
    """
    if not csv_path.exists():
        return None
    # Count lines; subtract 1 for header
    with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
        n = sum(1 for _ in f)
    return max(n - 1, 0)


def _infer_dataset_label(csv_dir: Optional[str]) -> Optional[str]:
    """
    Optional convenience: map csv_dir naming conventions to small/medium/large labels.
    """
    if not csv_dir:
        return None
    s = str(csv_dir).lower()
    if "c1k" in s or "small" in s:
        return "small"
    if "c50k" in s or "medium" in s:
        return "medium"
    if "c250k" in s or "large" in s:
        return "large"
    return None


def _extract_run_id(run_dir: Path) -> str:
    return run_dir.name


# ---------------------------
# Main collection
# ---------------------------
def collect_runs(
    runs_root: Path = RUNS_DIR,
    out_dir: Path = ARTIFACTS_DIR / "summary",
    *,
    # If True, count processed train/val/test rows from run_dir/processed/*.csv
    # (may be slow for huge expanded CSVs)
    count_processed_rows: bool = True,
    # If True, count raw transactions rows from csv_dir/transactions.csv
    count_raw_transactions: bool = True,
) -> pd.DataFrame:
    ensure_dir(out_dir)

    rows: List[Dict[str, Any]] = []

    run_dirs = sorted([p for p in Path(runs_root).iterdir() if p.is_dir()])

    for rd in run_dirs:
        cfg = _load_json(rd / "config_resolved.json")
        info = _load_json(rd / "run_info.json")
        m_val = _load_json(rd / "metrics_val.json")
        m_test = _load_json(rd / "metrics_test.json")

        run_id = _extract_run_id(rd)

        # --- config fields (data) ---
        csv_dir = _safe_get(cfg, "data.csv_dir")
        dataset_label = _infer_dataset_label(csv_dir)

        max_len = _safe_get(cfg, "data.max_len")
        batch_size = _safe_get(cfg, "data.batch_size")
        use_section = _safe_get(cfg, "data.use_section")
        use_country = _safe_get(cfg, "data.use_country")
        age_bins = _safe_get(cfg, "data.age_bins")

        label_order = _safe_get(cfg, "data.label_order")

        # data generation metadata (stored in config)
        gen_enabled = _safe_get(cfg, "data.gen.enable")
        gen_seed = _safe_get(cfg, "data.gen.seed")
        n_consumers = _safe_get(cfg, "data.gen.n_consumers")
        n_products = _safe_get(cfg, "data.gen.n_products")

        # --- config fields (model/train) ---
        model_type = (_safe_get(cfg, "model.type") or "").lower() or None
        d_model = _safe_get(cfg, "model.d_model")
        n_layers = _safe_get(cfg, "model.n_layers")
        n_heads = _safe_get(cfg, "model.n_heads")  # transformer/xlstm only
        dropout = _safe_get(cfg, "model.dropout")

        train_seed = _safe_get(cfg, "train.seed")
        lr = _safe_get(cfg, "train.lr")
        weight_decay = _safe_get(cfg, "train.weight_decay")
        epochs = _safe_get(cfg, "train.epochs")
        amp = _safe_get(cfg, "train.amp")

        tag = _safe_get(cfg, "logging.tag")

        # --- metrics ---
        # Core fit-outcome classification
        val_acc = _safe_get(m_val, "classification_on_true_rows.accuracy")
        test_acc = _safe_get(m_test, "classification_on_true_rows.accuracy")

        # Macro F1
        val_macro_f1 = _safe_get(m_val, "classification_on_true_rows.per_class.macro.f1")
        test_macro_f1 = _safe_get(m_test, "classification_on_true_rows.per_class.macro.f1")

        # Size recommendation
        val_size_acc = m_val.get("size_accuracy")
        test_size_acc = m_test.get("size_accuracy")

        # --- efficiency ---
        params_millions = info.get("params_millions")
        epoch_time_sec_mean = info.get("epoch_time_sec_mean")
        peak_cuda_mb = info.get("peak_cuda_mb")

        # total wall time for train_main call (runner adds this)
        fit_time_sec = info.get("fit_time_sec")

        # --- scale: raw transactions count ---
        n_transactions_raw = None
        if count_raw_transactions and csv_dir:
            tx_path = Path(csv_dir) / "transactions.csv"
            n_transactions_raw = _count_csv_rows(tx_path)

        # --- scale: processed train/val/test counts ---
        n_train_rows = n_val_rows = n_test_rows = None
        if count_processed_rows:
            processed_dir = rd / "processed"
            n_train_rows = _count_csv_rows(processed_dir / "train.csv")
            n_val_rows = _count_csv_rows(processed_dir / "val.csv")
            n_test_rows = _count_csv_rows(processed_dir / "test.csv")

        rows.append(
            {
                # identity
                "run_id": run_id,
                "tag": tag,
                "run_dir": str(rd),

                # model
                "model_type": model_type,
                "d_model": d_model,
                "n_layers": n_layers,
                "n_heads": n_heads,
                "dropout": dropout,

                # train
                "train_seed": train_seed,
                "lr": lr,
                "weight_decay": weight_decay,
                "epochs": epochs,
                "amp": amp,

                # data
                "csv_dir": csv_dir,
                "dataset_label": dataset_label,
                "max_len": max_len,
                "batch_size": batch_size,
                "use_section": use_section,
                "use_country": use_country,
                "age_bins": age_bins,
                "label_order": label_order,

                # data generation meta
                "gen_enabled": gen_enabled,
                "gen_seed": gen_seed,
                "n_consumers": n_consumers,
                "n_products": n_products,

                # scale
                "n_transactions_raw": n_transactions_raw,
                "n_train_rows_processed": n_train_rows,
                "n_val_rows_processed": n_val_rows,
                "n_test_rows_processed": n_test_rows,

                # predictive metrics
                "val_accuracy": val_acc,
                "val_macro_f1": val_macro_f1,
                "val_size_accuracy": val_size_acc,

                "test_accuracy": test_acc,
                "test_macro_f1": test_macro_f1,
                "test_size_accuracy": test_size_acc,

                # efficiency metrics
                "params_millions": params_millions,
                "epoch_time_sec_mean": epoch_time_sec_mean,
                "fit_time_sec": fit_time_sec,
                "peak_cuda_mb": peak_cuda_mb,
            }
        )

    df = pd.DataFrame(rows)

    # Helpful ordering for analysis
    sort_cols = [c for c in ["dataset_label", "model_type", "max_len", "train_seed", "tag"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    # Save outputs
    out_csv = out_dir / "runs_summary.csv"
    out_json = out_dir / "runs_summary.json"
    df.to_csv(out_csv, index=False, encoding="utf-8")
    out_json.write_text(df.to_json(orient="records", indent=2), encoding="utf-8")

    print(f"[collect_runs] Collected {len(df)} runs")
    print(f"[collect_runs] Wrote: {out_csv}")
    print(f"[collect_runs] Wrote: {out_json}")

    return df


if __name__ == "__main__":
    collect_runs(
        count_processed_rows=True,
        count_raw_transactions=True,
    )