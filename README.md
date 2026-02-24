# Sequence-Based Size & Fit Recommender

**Comparison of Transformer vs xLSTM encoders for next-step size selection**

## Description

A minimal, reproducible pipeline for **personalized next-step size recommendation** from purchase histories.

For each consumer and purchase event, the model scores candidate sizes using a three-class fit label
`{too small / fit / too large}` and selects the size with the highest predicted probability of `fit`.

We compare three sequence encoders:

* **Transformer (causal self-attention)**
* **xLSTM (modern recurrent backbone)**
* **LSTM (classical recurrent baseline)**

All other components (embeddings, pooling, classifier head, optimization, batching) are held constant to isolate backbone differences.

---

## Task formulation

This is a causal next-step size-selection problem.

For each purchase at time *t*:

1. We condition only on:

   * Past purchase history `(x₁,…,x_{t-1})`
   * Current item attributes at time *t* (product type, material, candidate size, optional section)
   * Static consumer features (gender, age-bin, optional country)

2. Each purchase event is expanded into multiple candidate-size rows:

   * One row per valid size in the appropriate size vocabulary
   * Labels assigned deterministically as:

     * `too small`
     * `fit`
     * `too large`

3. The final predicted size is:

   ```
   argmax P(fit | history, current item, candidate size)
   ```

---

## Repo layout

```
.
├─ configs/
│  ├─ transformer_base.yaml
│  ├─ xlstm_base.yaml
|  ├─ lstm_base.yaml
│  └─ experiments.yaml
├─ data/                          # generated CSVs
├─ notebooks/
│  ├─ data_exploration.ipynb
│  └─ eval_runs.ipynb
├─ src/
│  ├─ datagen/                    # synthetic data generation
│  │  ├─ build_data.py
│  │  ├─ consumers.py
│  │  ├─ products.py
│  │  └─ transactions.py
│  └─ sizerec/
│     ├─ paths.py
│     ├─ vocab.py
│     ├─ seq_prep.py
│     ├─ data_module.py
│     ├─ metrics.py
│     ├─ models/
│     │  ├─ encoders.py          # Transformer / xLSTM / LSTM backbones
│     │  └─ seqrec.py            # shared wrapper
│     ├─ train.py                # end-to-end training & evaluation
|     ├─ runner.py               # experiments wrapper
│     └─ collect_runs.py         # experiments consolidation
├─ README.md
├─ pyproject.toml
└─ .gitignore
```

---

## Quick start

### 1) Environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e .
```

> GPU is optional. If using CUDA, install the matching PyTorch wheel from pytorch.org.

---

### 2) Generate synthetic data

```python
from datagen.build_data import generate_and_read_data
from sizerec.paths import DATA_DIR

generate_and_read_data(DATA_DIR, n_consumers=1000, n_products=1000, seed=10)
```

This writes:

```
data/consumers.csv
data/products.csv
data/transactions.csv
```

Generation is idempotent — existing CSVs are reused.

---

### 3) Train (Transformer)

```bash
python -m sizerec.train configs/transformer_base.yaml
```

### 4) Train (xLSTM)

```bash
python -m sizerec.train configs/xlstm_base.yaml
```

### 5) Run multiple experiments

```bash
python -m sizerec.runner configs/experiments.yaml
```

---

## What the configs control

Key YAML fields:

### Data

* `data.csv_dir` – directory containing the 3 base CSVs
* `data.max_len` – maximum retained history length
* `data.use_section`, `data.use_country`

### Model

* `model.type` – `transformer`, `xlstm`, or `lstm`
* `model.d_model`
* `model.n_layers`
* `model.n_heads` (Transformer only)
* `model.dropout`

### Training

* `train.epochs`
* `train.lr`
* `train.weight_decay`
* `train.amp`
* `train.patience` (early stopping)

### Logging

* `logging.out_dir`

---

## What gets saved per run

Under:

```
artifacts/runs/<timestamp>/
```

You’ll find:

* `config_resolved.json`
* `vocabs/`
* `processed/train.csv`, `val.csv`, `test.csv`
* `checkpoint.pt` (best by validation size-loss)
* `metrics_val.json`
* `metrics_test.json`
* `preds_val.csv`
* `preds_test.csv`
* `run_info.json`

---

## Evaluation metrics

Primary metric:

### **Size-Accuracy**

Fraction of purchase events where:

```
argmax P(fit)  ==  true purchased size
```

Secondary metrics:

* Size-loss (cross-entropy on `fit` class for true row)
* Classification metrics computed only on true purchased rows

---

## How the model works

1. Step-wise tokens (product type, material, size, optional section) and static user features are embedded.
2. Context history is encoded with a **pluggable causal backbone**:
   * Transformer (masked self-attention)
   * xLSTM (modern recurrent)
   * LSTM (baseline)
3. We pool the last valid history state.
4. We fuse it with the current purchase candidate embedding.
5. A linear head outputs 3-class logits.
6. The recommended size is the candidate with highest `P(fit)`.

All non-backbone components are shared across models to ensure fair architectural comparison.

---

## Synthetic data

The dataset is fully synthetic and reproducible.

* Consumer sizes and tolerances are generated with controlled distributions.
* Products receive systematic fit offsets.
* Fit labels are deterministically assigned based on tolerance bands.
* Only realistic, decision-time features are used (no leakage from label-generating internals).

---

## Design principles

* Strict causality (masking or recurrence)
* No consumer ID embeddings (scalable design)
* No leakage from internal fit offsets
* Backbone comparison under matched capacity
* Fully reproducible runs via seeds + idempotent data generation

