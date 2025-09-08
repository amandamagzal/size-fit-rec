from __future__ import annotations
import os
from pathlib import Path
from typing import Optional

def _default_repo_root() -> Path:
    # .../src/sizerec/paths.py -> parents[0]=sizerec, [1]=src, [2]=REPO
    return Path(__file__).resolve().parents[2]

def get_repo_root() -> Path:
    """Repo root can be overridden with env var SIZEREC_ROOT; else inferred."""
    env = os.getenv("SIZEREC_ROOT")
    if env:
        p = Path(env).expanduser().resolve()
        if p.exists():
            return p
    return _default_repo_root()

REPO_ROOT   = get_repo_root()
SRC_DIR     = REPO_ROOT / "src"
CONFIGS_DIR = REPO_ROOT / "configs"
DATA_DIR    = REPO_ROOT / "data"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
RUNS_DIR      = ARTIFACTS_DIR / "runs"

def ensure_dir(p: Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def config_path(name: str) -> Path:
    return CONFIGS_DIR / name

def data_path(name: Optional[str] = None) -> Path:
    return (DATA_DIR / name) if name else DATA_DIR

def run_dir(stamp: str) -> Path:
    """Compute run dir under artifacts/runs/<stamp> (does not create)."""
    return RUNS_DIR / stamp
