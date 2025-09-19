"""
Run a matrix of experiments and log timing/memory into each run folder.

Reads:  configs/experiments.yaml
For each experiment:
  - merge overrides onto a base YAML config
  - write a temp merged config under artifacts/tmp/
  - call sizerec.train.main(merged_cfg_path)
  - read/write run_info.json (params, times, memory, etc.)
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Dict

import yaml

from sizerec.paths import CONFIGS_DIR, ARTIFACTS_DIR, ensure_dir
from sizerec.train import main as train_main  # uses the existing training entrypoint


def _deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow+deep dict update using dot-path keys in overrides (e.g., 'data.max_len')."""
    out = dict(d)
    # expand dot-paths like {'data.max_len': 32} -> {'data': {'max_len': 32}}
    expanded: Dict[str, Any] = {}
    for k, v in u.items():
        if "." in k:
            head, *tail = k.split(".")
            cur = expanded.setdefault(head, {})
            for t in tail[:-1]:
                cur = cur.setdefault(t, {})
            cur[tail[-1]] = v
        else:
            expanded[k] = v

    def rec(base: Dict[str, Any], delta: Dict[str, Any]) -> Dict[str, Any]:
        for k, v in delta.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                base[k] = rec(dict(base[k]), v)
            else:
                base[k] = v
        return base

    return rec(out, expanded)


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _save_yaml(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def run_experiments(matrix_path: str | Path = CONFIGS_DIR / "experiments.yaml") -> None:
    matrix_path = Path(matrix_path)
    matrix = _load_yaml(matrix_path)
    exps = matrix.get("experiments", [])
    if not exps:
        print(f"No experiments found in {matrix_path}")
        return

    tmp_cfg_dir = ARTIFACTS_DIR / "tmp_configs"
    ensure_dir(tmp_cfg_dir)

    for exp in exps:
        name = exp["name"]
        base_cfg = Path(exp["base"])
        overrides = exp.get("overrides", {})

        print(f"\n=== Running experiment: {name} ===")
        cfg = _load_yaml(base_cfg)
        cfg = _deep_update(cfg, overrides)

        # Tag the run with the experiment name
        cfg.setdefault("logging", {}).setdefault("tag", name)

        merged_cfg_path = tmp_cfg_dir / f"{name}.yaml"
        _save_yaml(cfg, merged_cfg_path)

        # Time the full training call
        t0 = perf_counter()
        run_dir = train_main(str(merged_cfg_path))
        fit_time = perf_counter() - t0

        # Load or create run_info.json and augment with experiment metadata
        run_dir = Path(run_dir)
        run_info_path = run_dir / "run_info.json"
        info: Dict[str, Any] = {}
        if run_info_path.exists():
            info = json.loads(run_info_path.read_text(encoding = "utf-8"))

        # Ensure minimum fields
        info.update({
            "experiment_name": name,
            "merged_config_path": str(merged_cfg_path),
            "fit_time_sec": fit_time,
            "run_dir": str(run_dir),
        })
        run_info_path.write_text(json.dumps(info, ensure_ascii = False, indent = 2), encoding = "utf-8")

        print(f"Finished: {name}  →  {run_dir}\n")


if __name__ == "__main__":
    run_experiments()
