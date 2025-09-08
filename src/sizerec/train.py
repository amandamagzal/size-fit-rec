"""
Training script for Transformer fit_outcome baseline.

Flow:
  1) Load YAML config.
  2) Build/load vocabs; encode tables; build examples; split & save processed CSVs (idempotent).
  3) Create DataLoaders (SequenceDataset + make_collate).
  4) Instantiate TransformerRec and train with AdamW (+AMP, early stopping).
  5) Evaluate on val/test; save metrics, preds, and best checkpoint.

Artifacts layout (under logging.out_dir / run_id):
  - config_resolved.json
  - metrics_val.json, metrics_test.json
  - preds_val.csv, preds_test.csv
  - checkpoint.pt
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
import yaml

from sizerec.vocab import (
    build_fit_label_map,
    build_size_vocab,
    build_categorical_vocabs,
    save_vocabs,
    load_vocabs,
)
from sizerec.seq_prep import (
    encode_features,
    build_examples,
    split_by_consumer,
    save_processed_splits,
    join_product_attrs,  # not used here directly but handy if you extend
)
from sizerec.data_module import SequenceDataset, make_collate
from sizerec.models.transformer import TransformerRec, count_params
from sizerec.metrics import accuracy, precision_recall_f1_per_class, confusion_matrix

from sizerec.paths import CONFIGS_DIR, DATA_DIR, RUNS_DIR, ensure_dir, run_dir


# ---------------------------
# Small helpers
# ---------------------------
def _seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True  # faster


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_json(obj: Dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------
# Main entry
# ---------------------------
def main(cfg_path: str | None = None) -> None:
    # 1) Load config
    cfg_path = str(CONFIGS_DIR / "transformer_base.yaml") if cfg_path is None else cfg_path

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]; model_cfg = cfg["model"]; train_cfg = cfg["train"]; log_cfg = cfg["logging"]

    _seed_everything(train_cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare run directory
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_root = run_dir(run_id)
    _ensure_dir(out_root)

    # Save resolved config for reproducibility
    _save_json(cfg, out_root / "config_resolved.json")

    # 2) Load raw CSVs
    csv_dir = DATA_DIR
    consumers = pd.read_csv(csv_dir / "consumers.csv", parse_dates=["start_date"])
    products = pd.read_csv(csv_dir / "products.csv", converters={"available_countries": json.loads})
    transactions = pd.read_csv(csv_dir / "transactions.csv", parse_dates=["transaction_date"])

    # 3) Build or load vocabs
    vocabs_dir = out_root / "vocabs"
    processed_dir = out_root / "processed"
    _ensure_dir(vocabs_dir); _ensure_dir(processed_dir)

    # Build label map from config order
    label_order = data_cfg["label_order"]
    label_map = build_fit_label_map(label_order)

    # Fresh vocabs for this run
    size_vocab = build_size_vocab(transactions)
    vocabs = build_categorical_vocabs(
        consumers, products, transactions,
        use_section=bool(data_cfg["use_section"]),
        use_country=bool(data_cfg["use_country"]),
        age_bins=list(data_cfg["age_bins"]),
    )
    save_vocabs({"size": size_vocab, **vocabs, "fit_label": label_map}, vocabs_dir)

    # 4) Encode, build examples, split, save
    encoded = encode_features(
        consumers, products, transactions,
        vocabs=vocabs,
        size_vocab=size_vocab,
        label_map=label_map,
        age_bins=list(data_cfg["age_bins"]),
        use_section=bool(data_cfg["use_section"]),
        use_country=bool(data_cfg["use_country"]),
    )
    examples = build_examples(
        encoded,
        max_len=int(data_cfg["max_len"]),
        use_section=bool(data_cfg["use_section"]),
        use_country=bool(data_cfg["use_country"]),
    )
    train_df, val_df, test_df = split_by_consumer(examples, seed=train_cfg.get("seed", 42))
    save_processed_splits(train_df, val_df, test_df, processed_dir)

    # 5) DataLoaders
    batch_size = int(data_cfg["batch_size"])
    num_workers = int(data_cfg.get("num_workers", 0))

    train_ds = SequenceDataset(processed_dir / "train.csv",
                               use_section=bool(data_cfg["use_section"]),
                               use_country=bool(data_cfg["use_country"]))
    val_ds   = SequenceDataset(processed_dir / "val.csv",
                               use_section=bool(data_cfg["use_section"]),
                               use_country=bool(data_cfg["use_country"]))
    test_ds  = SequenceDataset(processed_dir / "test.csv",
                               use_section=bool(data_cfg["use_section"]),
                               use_country=bool(data_cfg["use_country"]))

    collate_fn = make_collate(max_len=int(data_cfg["max_len"]))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                             num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                              num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)

    # 6) Model
    use_section = bool(data_cfg["use_section"]); use_country = bool(data_cfg["use_country"])
    model = TransformerRec(
        num_product_types=len(vocabs["product_type"]),
        num_materials=len(vocabs["material"]),
        num_sizes=len(size_vocab),
        num_sections=(len(vocabs["section"]) if use_section else None),
        num_genders=len(vocabs["gender"]),
        num_age_bins=len(vocabs["age_bin"]),
        num_countries=(len(vocabs["country"]) if use_country else None),
        d_model=int(model_cfg["d_model"]),
        n_layers=int(model_cfg["n_layers"]),
        n_heads=int(model_cfg["n_heads"]),
        dropout=float(model_cfg["dropout"]),
        max_len=int(data_cfg["max_len"]),
        num_classes=len(label_order),
        use_cls=bool(model_cfg["use_cls"]),
    ).to(device)

    print(f"Model params: {count_params(model):,}")

    # 7) Train
    class_weights = train_cfg.get("class_weights", None)
    if class_weights:
        cw = torch.tensor(class_weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=float(train_cfg.get("label_smoothing", 0.0)))
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=float(train_cfg.get("label_smoothing", 0.0)))

    optimizer = AdamW(model.parameters(),
                      lr=float(train_cfg["lr"]),
                      weight_decay=float(train_cfg["weight_decay"]),
                      betas=tuple(train_cfg.get("betas", [0.9, 0.999])))

    scaler = torch.cuda.amp.GradScaler(enabled=bool(train_cfg.get("amp", True)))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    epochs = int(train_cfg["epochs"])
    patience = int(train_cfg.get("early_stopping_patience", 5))

    best_val_loss = float("inf"); best_state = None; stall = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0; n_batches = 0
        for batch in train_loader:
            for k, v in batch.items():
                batch[k] = v.to(device) if torch.is_tensor(v) else v

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(train_cfg.get("amp", True))):
                logits = model(batch)          # [B, C]
                loss = criterion(logits, batch["label"])

            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running += loss.item(); n_batches += 1

        train_loss = running / max(n_batches, 1)

        # Validation
        model.eval()
        val_loss, ys, ps = 0.0, [], []
        with torch.no_grad():
            for batch in val_loader:
                for k, v in batch.items():
                    batch[k] = v.to(device) if torch.is_tensor(v) else v
                with torch.cuda.amp.autocast(enabled=False):
                    logits = model(batch)
                    loss = criterion(logits, batch["label"])
                val_loss += loss.item()
                ys.extend(batch["label"].detach().cpu().tolist())
                ps.extend(logits.argmax(dim=1).detach().cpu().tolist())
        val_loss /= max(len(val_loader), 1)
        y_true = np.asarray(ys, dtype=int) if ys else np.array([], dtype=int)
        y_pred = np.asarray(ps, dtype=int) if ps else np.array([], dtype=int)
        val_acc = accuracy(y_true, y_pred)

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        # Early stopping
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            stall = 0
        else:
            stall += 1
            if stall >= patience:
                print("Early stopping.")
                break

    # Save best checkpoint
    if best_state is not None:
        torch.save(best_state, out_root / "checkpoint.pt")
        model.load_state_dict(best_state)

    # 8) Final evaluation (val & test), write metrics + preds
    def _eval_and_write(split_name: str, loader: torch.utils.data.DataLoader):
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for batch in loader:
                for k, v in batch.items():
                    batch[k] = v.to(device) if torch.is_tensor(v) else v
                logits = model(batch)
                ys.extend(batch["label"].detach().cpu().tolist())
                ps.extend(logits.argmax(dim=1).detach().cpu().tolist())
        y_true = np.asarray(ys, dtype=int) if ys else np.array([], dtype=int)
        y_pred = np.asarray(ps, dtype=int) if ps else np.array([], dtype=int)

        # Metrics
        num_classes = len(label_order)
        acc = accuracy(y_true, y_pred)
        prf = precision_recall_f1_per_class(y_true, y_pred, num_classes)
        cm = confusion_matrix(y_true, y_pred, num_classes).tolist()

        metrics = {"accuracy": acc, "per_class": prf, "confusion_matrix": cm}
        _save_json(metrics, out_root / f"metrics_{split_name}.json")

        # Preds (id + true/pred)
        pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(
            out_root / f"preds_{split_name}.csv", index=False, encoding="utf-8"
        )

        print(f"{split_name}: acc={acc:.4f}, macroF1={prf['macro']['f1']:.4f}")

    if log_cfg.get("write_preds", True):
        _eval_and_write("val", val_loader)
        _eval_and_write("test", test_loader)

    print(f"Run artifacts saved to: {out_root}")


if __name__ == "__main__":
    main()
