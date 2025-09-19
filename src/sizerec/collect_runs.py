"""
Scan artifacts/runs/* and collect metrics + run info into one CSV.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from sizerec.paths import ARTIFACTS_DIR, RUNS_DIR, ensure_dir


def _safe_load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding = "utf-8"))
    except Exception:
        return {}

def collect(output_path: str | Path = ARTIFACTS_DIR / "summary" / "all_runs.csv") -> Path:
    rows: List[Dict[str, Any]] = []
    for run_dir in sorted(RUNS_DIR.glob("*")):
        if not run_dir.is_dir():
            continue
        cfg = _safe_load_json(run_dir / "config_resolved.json")
        info = _safe_load_json(run_dir / "run_info.json")
        met_test = _safe_load_json(run_dir / "metrics_test.json")
        met_val = _safe_load_json(run_dir / "metrics_val.json")

        row: Dict[str, Any] = {
            "run_dir": str(run_dir),
            "experiment_name": info.get("experiment_name"),
            "model_type": cfg.get("model", {}).get("type"),
            "max_len": cfg.get("data", {}).get("max_len"),
            "csv_dir": cfg.get("data", {}).get("csv_dir"),
            "seed": cfg.get("train", {}).get("seed"),
            "params_millions": info.get("params_millions"),
            "epoch_time_sec_mean": info.get("epoch_time_sec_mean"),
            "peak_cuda_mb": info.get("peak_cuda_mb"),
            "peak_cpu_mb": info.get("peak_cpu_mb"),
            "fit_time_sec": info.get("fit_time_sec"),
            # metrics
            "val_accuracy": met_val.get("accuracy"),
            "val_macro_f1": (met_val.get("per_class", {}).get("macro", {}) or {}).get("f1"),
            "test_accuracy": met_test.get("accuracy"),
            "test_macro_f1": (met_test.get("per_class", {}).get("macro", {}) or {}).get("f1"),
        }
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["model_type", "max_len", "seed", "experiment_name"], na_position = "last")
    out = Path(output_path)
    ensure_dir(out.parent)
    df.to_csv(out, index = False, encoding = "utf-8")
    print(f"Wrote: {out}  ({len(df)} rows)")
    return out

if __name__ == "__main__":
    collect()
