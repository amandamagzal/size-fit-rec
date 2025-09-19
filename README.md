# Sequence-Based Size & Fit Recommender

**Comparison of Transformer vs xLSTM encoders for next-step fit outcome prediction**

## What is this?

A minimal, reproducible pipeline that predicts the **fit outcome** of a customer’s next purchase (`too small / fit / too large / not applicable`) from their **purchase history** and the **current item** they’re buying.
We compare two sequence encoders—**Transformer** and **xLSTM**—with everything else held constant.

---

## Repo layout (essentials)

.
├─ configs/
│  ├─ transformer_base.yaml
│  └─ xlstm_base.yaml
├─ data/                          # generated CSVs
├─ notebooks/
│  ├─ data_exploration.ipynb
│  └─ main.ipynb
├─ src/
│  ├─ datagen/                    # synthetic data generation (no external inputs)
│  │  ├─ __init__.py
│  │  ├─ build_data.py           # generate_and_read_data(...) → writes CSVs
│  │  ├─ constants.py            # global constants & distributions
│  │  ├─ consumers.py            # generate_consumers(...)
│  │  ├─ products.py             # generate_products(...)
│  │  └─ transactions.py         # generate_transactions(...)
│  └─ sizerec/
│     ├─ paths.py                # central paths (DATA_DIR, CONFIGS_DIR, etc.)
│     ├─ vocab.py                # build & save/load vocabs
│     ├─ seq_prep.py             # join/encode/build examples + splits
│     ├─ data_module.py          # Dataset + collate (padding/masks)
│     ├─ metrics.py              # accuracy / PRF / confusion
│     ├─ models/
│     │  ├─ encoders.py          # TransformerEncoderBackbone / xLSTMEncoder
│     │  └─ seqrec.py            # shared wrapper (embeddings → encoder → head)
│     └─ train.py                # end-to-end train/eval
├─ README.md
├─ requirements.txt
└─ .gitignore

---

## Quick start

### 1) Environment

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> GPU (optional): install the CUDA-matched PyTorch wheel from pytorch.org. Otherwise it runs on CPU.

### 2) Generate synthetic data (no CLI needed)

From a notebook or Python REPL:

```python
from datagen.build_data import generate_and_read_data
from sizerec.paths import DATA_DIR
generate_and_read_data(DATA_DIR, n_consumers=1000, n_products=1000, seed=10)
```

This writes `data/consumers.csv`, `products.csv`, `transactions.csv`.

### 3) Train (Transformer)

```bash
python -m sizerec.train configs/transformer_base.yaml
```

### 4) Train (xLSTM)

```bash
python -m sizerec.train configs/xlstm_base.yaml
```

---

## What the configs control

`configs/*.yaml` set **data**, **model**, **train**, and **logging** options.
Key fields:

* `data.csv_dir`: where the 3 CSVs live (default `data/`).
* `data.max_len`: max history length per example.
* `model.type`: `transformer` or `xlstm`.
* `model.d_model, n_layers, n_heads, dropout`: backbone size (heads are ignored by xLSTM).
* `train.*`: epochs, lr, weight decay, AMP, early stopping.
* `logging.out_dir`: run artifacts root (default `artifacts/runs/`).

---

## What gets saved per run

Under `artifacts/runs/<timestamp>/`:

* `config_resolved.json` – the exact config used
* `vocabs/` – `vocabs.json`
* `processed/` – `train.csv`, `val.csv`, `test.csv` (next-step examples)
* `checkpoint.pt` – best model by val loss
* `metrics_val.json`, `metrics_test.json` – accuracy, per-class precision/recall/F1, confusion matrix
* `preds_val.csv`, `preds_test.csv` – predicted vs true labels

---

## How the model works (one paragraph)

We create **next-step** examples per consumer: inputs are the **history** of item tokens (product type, material, size, optional section) plus **static** user features (gender, age-bin, optional country), and **current purchase** tokens at time *t* (product type/material/size/(section), known at purchase time; no leakage).
These integers are embedded, summed per step, and passed to a **pluggable encoder** (Transformer or xLSTM). We pool the last valid history state, **fuse** it with the current-purchase embedding, and classify into the 4 outcomes.

---

## Notebooks

* `notebooks/main.ipynb` shows how to:

  * import paths (`from sizerec.paths import DATA_DIR, CONFIGS_DIR`)
  * generate data
  * run training with a config
* `notebooks/data_exploration.ipynb` provides quick sanity plots.

---
