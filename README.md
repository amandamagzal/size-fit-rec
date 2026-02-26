# Sequence-Based Size Recommendation

**Comparison of LSTM, Transformer, and xLSTM encoders for next-step size selection**

---

## Overview

This repository implements a **causal sequence model for size recommendation** using synthetic purchase histories.

We compare three sequence backbones under a shared architecture:

* **LSTM** (baseline)
* **Transformer** (causal self-attention)
* **xLSTM** (modern recurrent)

All other components (embeddings, pooling, head, optimizer, training schedule) are held constant to isolate backbone differences.

---

## Problem Setup

The core modeling task is:

> **Next-step fit outcome classification**

For each consumer and purchase step `t > 0`:

* **Input**: history up to `t-1`
* **Target**: fit outcome at `t`

  * `too small`
  * `fit`
  * `too large`

The model is strictly causal:

* History excludes the current step
* Transformer uses a causal mask
* LSTM/xLSTM are inherently causal

---

## From Fit Prediction to Size Recommendation

We convert fit classification into a size recommendation problem via **synthetic candidate expansion**.

### Training

1. Build next-step examples.
2. Keep only rows where the original purchased outcome was `"fit"`.
   (This ensures exactly one true “fit” candidate per purchase event.)
3. Expand each example into one row per valid size.
4. Assign labels deterministically:

```
candidate < purchased → too small
candidate > purchased → too large
candidate == purchased → fit
```

Labels during expansion are based purely on ordinal size comparison (not on internal synthetic fit offsets).

---

### Inference & Evaluation

For each purchase event:

```
predicted_size = argmax P(fit | history, current_item, candidate_size)
```

Candidates are grouped by:

```
(consumer_id, transaction_date_t)
```

### Primary Metric

**Test Size Accuracy**

Fraction of events where the selected size matches the true purchased size.

### Secondary Metrics

Computed **only on true purchased rows (one per event)** and reported for reference:

* Classification accuracy
* Per-class precision / recall / F1
* Macro-F1
* Confusion matrix

---

## Model Architecture

### Inputs (per history step)

* product_type
* material
* purchased size
* optional section

### Static Features (broadcast across time)

* gender
* age_bin
* optional country

All embeddings are summed with a learned positional embedding.

### Backbone Encoders

| Model       | Description                       |
| ----------- | --------------------------------- |
| LSTM        | Packed unidirectional LSTM        |
| Transformer | Causal masked self-attention      |
| xLSTM       | Block stack (sLSTM configuration) |

### Pooling & Prediction

1. Encode full history
2. Extract **last valid time step**
3. Add candidate embedding (current item + size)
4. Linear layer → 3-class logits

No CLS token is used in the current implementation.

---

## Synthetic Dataset

Data is fully synthetic and reproducible.

### Consumers

* Gender, country, age
* Upper/lower clothing size
* Shoe size
* Personal tolerance margins

### Products

* Section
* Product type
* Material
* Fit type
* Size accuracy
* Internal `fit_offset` (used only during data generation)

### Transactions

* Sequential purchase timestamps
* Purchased size sampled near true size
* Fit outcome computed using tolerance + fit_offset

**Important:**
The model never receives `fit_offset` as input.

---

## Experimental Setup

We evaluate:

* 3 models
* Multiple dataset scales
* Multiple history lengths
* 3 training seeds per configuration

Primary focus:

* **RQ1:** Performance comparison (Transformer vs xLSTM vs LSTM)
* **RQ2:** Effect of dataset scale and history length
* **RQ3:** Efficiency trade-offs (parameters, runtime, memory)

---

## Efficiency Metrics

Per run:

* Trainable parameters
* Mean epoch time
* Total training time
* Peak GPU memory

Results can be consolidated using:

```
python -m sizerec.collect_runs
```

---

## Repository Structure

```
.
├─ configs/
│  ├─ transformer_base.yaml
│  ├─ xlstm_base.yaml
│  ├─ lstm_base.yaml
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
│     ├─ runner.py               # experiments wrapper
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
source .venv/bin/activate            # Windows: .venv/Scripts/activate
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
* `model.n_heads` – number of heads (Transformer and xLSTM)
* `model.dropout`

### Training

* `train.epochs`
* `train.lr`
* `train.weight_decay`
* `train.amp`
* `train.early_stopping_patience`

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

## Design Principles

* Strict causality
* No consumer ID embeddings
* No fit_offset leakage
* Controlled backbone comparison
* Fully reproducible via configs and seeds

---
