"""
Training script for SeqRec encoders (Transformer/LSTM/xLSTM).

Flow:
  1) Load YAML config.
  2) Build/load vocabs; encode tables; build examples; split & save processed CSVs (idempotent).
  3) Create DataLoaders (SequenceDataset + make_collate).
  4) Instantiate encoder and train with AdamW (+AMP, early stopping).
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
from time import perf_counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
import yaml
import torch.nn.functional as F

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
    join_product_attrs,
)

from datagen.constants import SIZES as CLOTHING_SIZES
from datagen.build_data import generate_and_read_data

from sizerec.data_module import SequenceDataset, make_collate
from sizerec.models.encoders import TransformerEncoderBackbone, xLSTMEncoderBackbone, LSTMEncoderBackbone
from sizerec.models.seqrec import SeqRec
from sizerec.models.utils import count_params
from sizerec.metrics import accuracy, precision_recall_f1_per_class, confusion_matrix

from sizerec.paths import REPO_ROOT, CONFIGS_DIR, DATA_DIR, RUNS_DIR, ensure_dir, run_dir


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
    torch.backends.cudnn.benchmark = True


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents = True, exist_ok = True)


def _save_json(obj: Dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(obj, ensure_ascii = False, indent = 2), encoding = "utf-8")


# ---------------------------
# Main entry
# ---------------------------
def main(cfg_path: str | None = None) -> str:
    # 1) Load config
    cfg_path = str(CONFIGS_DIR / "transformer_base.yaml") if cfg_path is None else cfg_path

    with open(cfg_path, "r", encoding = "utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]; model_cfg = cfg["model"]; train_cfg = cfg["train"]; log_cfg = cfg["logging"]

    _seed_everything(train_cfg.get("seed", 42))

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    
    # Prepare run directory
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_root = run_dir(run_id)
    _ensure_dir(out_root)

    # Save resolved config for reproducibility
    _save_json(cfg, out_root / "config_resolved.json")

    # 2) Load raw CSVs
    csv_dir = Path(data_cfg["csv_dir"])

    use_preprocessed = True

    if use_preprocessed:
        # --- Load shared vocabs ---
        shared_vocabs_dir = Path("/content/drive/MyDrive/size-fit-rec-runs/data/vocabs")
        if not shared_vocabs_dir.is_absolute():
            shared_vocabs_dir = REPO_ROOT / shared_vocabs_dir

        loaded = load_vocabs(shared_vocabs_dir)
        # Expecting keys: size, product_type, material, gender, age_bin, (section?), (country?), fit_label
        size_vocab = loaded["size"]
        vocabs = {
            "product_type": loaded["product_type"],
            "material": loaded["material"],
            "gender": loaded["gender"],
            "age_bin": loaded["age_bin"],
        }
        if bool(data_cfg["use_section"]):
            vocabs["section"] = loaded["section"]
        if bool(data_cfg["use_country"]):
            vocabs["country"] = loaded["country"]

        # label_map: prefer config-defined order (for safety/clarity), but you *can* assert it matches saved
        # Build label map from config order
        valid_labels = ["too small", "fit", "too large"]
        label_order = data_cfg["label_order"]
        assert set(label_order) == set(valid_labels) and len(label_order) == 3
        label_map = {label: idx for idx, label in enumerate(label_order)}
        if "fit_label" in loaded:
            assert loaded["fit_label"] == label_map, (
                f"Label map mismatch between config and vocabs. "
                f"config={label_map}, vocabs={loaded['fit_label']}"
            )

        # --- Point processed_dir to seed-specific folder ---
        pre_root = Path("/content/drive/MyDrive/size-fit-rec-runs/data")
        if not pre_root.is_absolute():
            pre_root = REPO_ROOT / pre_root

        seed = int(train_cfg.get("seed", 0))
        seed_dir = pre_root / f"s{seed}"
        processed_dir = seed_dir
        try:
            save_vocabs({"size": size_vocab, **vocabs, "fit_label": label_map}, vocabs_dir)
        except Exception:
            pass

    else:
        gen_cfg = (data_cfg.get("gen") or {})
        if bool(gen_cfg.get("enable", False)):
            if gen_cfg.get("force", False) and csv_dir.exists():
                for p in csv_dir.glob("*.csv"): p.unlink()

            generate_and_read_data(
                out_dir = csv_dir,
                n_consumers = int(gen_cfg.get("n_consumers", 1000)),
                n_products = int(gen_cfg.get("n_products", 1000)),
                seed = int(gen_cfg.get("seed", 10)),
            )

        csv_dir = Path(data_cfg["csv_dir"])
        if not csv_dir.is_absolute():
            csv_dir = REPO_ROOT / csv_dir
        csv_dir.mkdir(parents=True, exist_ok=True)

        consumers = pd.read_csv(csv_dir / "consumers.csv", parse_dates = ["start_date"])
        products = pd.read_csv(csv_dir / "products.csv", converters = {"available_countries": json.loads})
        transactions = pd.read_csv(csv_dir / "transactions.csv", parse_dates = ["transaction_date"])

        # 3) Build or load vocabs
        vocabs_dir = out_root / "vocabs"
        processed_dir = out_root / "processed"
        _ensure_dir(vocabs_dir); _ensure_dir(processed_dir)

        # Build label map from config order
        valid_labels = ["too small", "fit", "too large"]
        label_order = data_cfg["label_order"]
        assert set(label_order) == set(valid_labels) and len(label_order) == 3
        label_map = {label: idx for idx, label in enumerate(label_order)}

        # Fresh vocabs for this run
        size_vocab = build_size_vocab(transactions)
        vocabs = build_categorical_vocabs(
            consumers, products, transactions,
            use_section = bool(data_cfg["use_section"]),
            use_country = bool(data_cfg["use_country"]),
            age_bins = list(data_cfg["age_bins"]),
        )
        save_vocabs({"size": size_vocab, **vocabs, "fit_label": label_map}, vocabs_dir)

        # 4) Encode, build examples, split, save
        encoded = encode_features(
            consumers, products, transactions,
            vocabs = vocabs,
            size_vocab = size_vocab,
            label_map = label_map,
            age_bins = list(data_cfg["age_bins"]),
            use_section = bool(data_cfg["use_section"]),
            use_country = bool(data_cfg["use_country"]),
        )

        encoded = encoded[encoded['label_id'].notna()].copy()

        examples = build_examples(
            encoded,
            max_len = int(data_cfg["max_len"]),
            use_section = bool(data_cfg["use_section"]),
            use_country = bool(data_cfg["use_country"]),
        )
        train_df, val_df, test_df = split_by_consumer(examples, seed = train_cfg.get("seed", 111))
        print("[TRAIN EXPAND] Expanding train_df only on FIT rows...")

        fit_id = label_map["fit"]
        inv_size_vocab = {v: k for k, v in size_vocab.items()}  # id → token

        # keep only real FIT rows
        fit_only = train_df[train_df["label_id"] == fit_id]
        print("[TRAIN EXPAND] FIT rows:", len(fit_only))

        expanded = []

        for _, r in fit_only.iterrows():
            purchased_id = int(r["size_id_t"])
            purchased_tok = inv_size_vocab[purchased_id]

            try:
                purchased_val = float(purchased_tok)
            except:
                purchased_val = CLOTHING_SIZES.index(purchased_tok)

            # valid candidate sizes
            purchased_tok = inv_size_vocab[purchased_id]
            is_numeric = purchased_tok.replace(".", "", 1).isdigit()

            if is_numeric:
                valid_tokens = [
                    tok for tok in inv_size_vocab.values()
                    if tok.replace(".", "", 1).isdigit()
                ]
            else:
                valid_tokens = [
                    tok for tok in inv_size_vocab.values()
                    if tok in CLOTHING_SIZES
                ]

            for tok in valid_tokens:
                cand_id = size_vocab[tok]
                try:
                    cand_val = float(tok)
                except:
                    cand_val = CLOTHING_SIZES.index(tok)

                if cand_id == purchased_id:
                    new_label = fit_id
                elif cand_val < purchased_val:
                    new_label = label_map["too small"]
                else:
                    new_label = label_map["too large"]

                new_row = r.copy()
                new_row["size_id_t"] = cand_id
                new_row["size_numeric_t"] = cand_val
                new_row["label_id"] = new_label
                expanded.append(new_row)

        train_df = pd.DataFrame(expanded).reset_index(drop=True)
        print("[TRAIN EXPAND] New train size:", len(train_df))
        
        # 4) a) Keep only "fit" events in val/test ===

        val_df = val_df[val_df["label_id"] == fit_id].reset_index(drop=True)
        test_df = test_df[test_df["label_id"] == fit_id].reset_index(drop=True)

        # 4) b): Synthetic size expansion for val/test ===
        # get inverse map: id -> size_token (string)
        inv_size_vocab = {v: k for k, v in size_vocab.items()}

        def expand_split(df):
            """
            Expand each validation/test example into multiple synthetic size-candidates.

            For each purchased item, we generate one row per possible size in the size vocabulary.
            Labels are automatically assigned as:
                - "too small"   if candidate size < purchased size
                - "too large"   if candidate size > purchased size
                - "fit"         otherwise

            This expansion is used only for evaluation, so the model can choose the best size.
            """
                
            rows = []
            for _, r in df.iterrows():
                purchased_size_id = int(r["size_id_t"])
                purchased_token = inv_size_vocab[purchased_size_id]

                # Select valid size candidates based on product type
                purchased_token = inv_size_vocab[purchased_size_id]
                is_numeric = purchased_token.replace(".", "", 1).isdigit()

                if is_numeric:
                    valid_tokens = [tok for tok in inv_size_vocab.values()
                                    if tok.replace(".", "", 1).isdigit()]
                else:
                    valid_tokens = [tok for tok in inv_size_vocab.values()
                                    if tok in CLOTHING_SIZES]  # Clothing only


                # Convert vocab IDs back to numeric (when possible)
                try:
                    # purchased_size_float = float(inv_size_vocab[purchased_size_id])
                    purchased_val = float(purchased_token)
                except:
                    # fallback for non-numeric sizes
                    # purchased_size_float = None
                    purchased_val = CLOTHING_SIZES.index(purchased_token) if purchased_token in CLOTHING_SIZES else None
                if purchased_val is None:
                    continue

                for token in valid_tokens:
                    # Convert candidate size
                    try:
                        cand_val = float(token)
                    except:
                        cand_val = CLOTHING_SIZES.index(token) if token in CLOTHING_SIZES else None

                    if cand_val is None:
                        continue
                    # Assign labels too small / too large / fit
                    if cand_val < purchased_val:
                        new_label = label_map["too small"]
                    elif cand_val > purchased_val:
                        new_label = label_map["too large"]
                    else:
                        new_label = label_map["fit"]

                    # clone example but override size and label
                    new_row = r.copy()
                    new_row["size_id_t"] = size_vocab[token]
                    new_row["size_numeric_t"] = cand_val  
                    new_row["label_id"] = new_label
                    rows.append(new_row)


            return pd.DataFrame(rows).reset_index(drop=True)

        val_df = expand_split(val_df)
        test_df = expand_split(test_df)
        save_processed_splits(train_df, val_df, test_df, processed_dir)


    # 5) DataLoaders
    batch_size = int(data_cfg["batch_size"])
    num_workers = int(data_cfg.get("num_workers", 0))

    train_ds = SequenceDataset(processed_dir / "train.csv",
                               use_section = bool(data_cfg["use_section"]),
                               use_country = bool(data_cfg["use_country"]))
    val_ds   = SequenceDataset(processed_dir / "val.csv",
                               use_section = bool(data_cfg["use_section"]),
                               use_country = bool(data_cfg["use_country"]))
    test_ds  = SequenceDataset(processed_dir / "test.csv",
                               use_section = bool(data_cfg["use_section"]),
                               use_country = bool(data_cfg["use_country"]))

    collate_fn = make_collate(max_len = int(data_cfg["max_len"]))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size = batch_size, shuffle = True,
                                               num_workers = num_workers, collate_fn = collate_fn, pin_memory = True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size = batch_size, shuffle = False,
                                             num_workers = num_workers, collate_fn = collate_fn, pin_memory = True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size = batch_size, shuffle = False,
                                              num_workers = num_workers, collate_fn = collate_fn, pin_memory = True)

    # 6) Model
    use_section = bool(data_cfg["use_section"]); use_country = bool(data_cfg["use_country"])
    model_type = model_cfg.get("type", "transformer").lower()

    if model_type == "transformer":
        encoder = TransformerEncoderBackbone(
            d_model = int(model_cfg["d_model"]),
            n_layers = int(model_cfg["n_layers"]),
            n_heads = int(model_cfg["n_heads"]),
            dropout = float(model_cfg["dropout"]),
        )
    elif model_type == "lstm":
        encoder = LSTMEncoderBackbone(
            d_model = int(model_cfg["d_model"]),
            n_layers = int(model_cfg["n_layers"]),
            dropout = float(model_cfg["dropout"]),
        )
    elif model_type == "xlstm":
        encoder = xLSTMEncoderBackbone(
            context_length = int(data_cfg["max_len"]),
            d_model = int(model_cfg["d_model"]),
            n_layers = int(model_cfg["n_layers"]),
            n_heads = int(model_cfg["n_heads"]),
            dropout = float(model_cfg["dropout"]),
            enable_mlstm = False,
            slstm_backend = "vanilla",
        )
    else:
        raise ValueError(f"Unknown model.type: {model_type}")

    model = SeqRec(
        num_product_types = len(vocabs["product_type"]),
        num_materials = len(vocabs["material"]),
        num_sizes = len(size_vocab),
        num_sections = (len(vocabs["section"]) if use_section else None),
        num_genders = len(vocabs["gender"]),
        num_age_bins = len(vocabs["age_bin"]),
        num_countries = (len(vocabs["country"]) if use_country else None),
        d_model = int(model_cfg["d_model"]),
        dropout = float(model_cfg["dropout"]),
        max_len = int(data_cfg["max_len"]),
        num_classes = len(label_order),
        encoder = encoder,
    ).to(device)

    epoch_times = []

    num_params = count_params(model)
    print(f"Model params: {num_params:,}")

    # 7) Train
    class_weights = train_cfg.get("class_weights", None)
    if class_weights:
        cw = torch.tensor(class_weights, dtype = torch.float32, device = device)
        criterion = nn.CrossEntropyLoss(weight = cw, label_smoothing = float(train_cfg.get("label_smoothing", 0.0)))
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing = float(train_cfg.get("label_smoothing", 0.0)))

    optimizer = AdamW(model.parameters(),
                      lr = float(train_cfg["lr"]),
                      weight_decay = float(train_cfg["weight_decay"]),
                      betas = tuple(train_cfg.get("betas", [0.9, 0.999])))

    amp_enabled = bool(train_cfg.get("amp", True)) and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled = amp_enabled)
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    epochs = int(train_cfg["epochs"])
    patience = int(train_cfg.get("early_stopping_patience", 5))

    best_val_loss = float("inf"); best_state = None; stall = 0

    for epoch in range(1, epochs + 1):
        t_epoch0 = perf_counter()
        model.train()
        running = 0.0; n_batches = 0
        for batch in train_loader:
            for k, v in batch.items():
                batch[k] = v.to(device) if torch.is_tensor(v) else v

            optimizer.zero_grad(set_to_none = True)
            with torch.cuda.amp.autocast(enabled = amp_enabled):
                class_logits = model(batch)
                loss = criterion(class_logits, batch["label"])

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm = grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running += loss.item(); n_batches += 1

        train_loss = running / max(n_batches, 1)

        # Validation
        model.eval()
        fit_class_idx = label_map["fit"]

        ys, ps = [], []
        fit_scores = []
        all_logits = []

        with torch.no_grad():
            for batch in val_loader:
                for k, v in batch.items():
                    batch[k] = v.to(device) if torch.is_tensor(v) else v

                class_logits = model(batch)

                # store only classification logits here (size preds not used in val CE)
                all_logits.append(class_logits.detach().cpu())

                ys.extend(batch["label"].detach().cpu().tolist())
                ps.extend(class_logits.argmax(dim=1).detach().cpu().tolist())
                # fit_scores.extend(class_logits[:, fit_class_idx].detach().cpu().tolist())
                probs = F.softmax(class_logits, dim=1)
                fit_scores.extend(probs[:, fit_class_idx].detach().cpu().tolist())

        # Full logits tensor, aligned with val.csv rows
        logits_val = torch.cat(all_logits, dim=0)   # [N_val, C]

        # -------------------------------
        # Load expanded val.csv
        # -------------------------------
        val_df = pd.read_csv(processed_dir / "val.csv").copy()

        # Sanity check
        if len(val_df) != len(logits_val):
            raise RuntimeError(
                f"Validation alignment error: val_df rows={len(val_df)} "
                f"but logits rows={len(logits_val)}"
            )

        val_df["fit_prob"] = fit_scores

        # -------------------------------
        # Compute size-accuracy + size-loss
        # -------------------------------
        groups = val_df.groupby(["consumer_id", "transaction_date_t"])
        correct = 0
        total = 0
        val_losses = []
        fit_id = label_map["fit"]

        criterion_no_smooth = nn.CrossEntropyLoss()  # for clean per-row CE without smoothing

        for (_, _), g in groups:
            true_rows = g[g["label_id"] == fit_id]
            if true_rows.empty:
                continue

            true_idx = true_rows.index[0]
            true_size = true_rows["size_id_t"].iloc[0]

            # Predicted size = candidate with max fit prob
            pred_idx = g["fit_prob"].idxmax()
            pred_size = val_df.loc[pred_idx, "size_id_t"]

            if pred_size == true_size:
                correct += 1
            total += 1

            # ---- Correct size-loss using full logits ----
            logit_row = logits_val[true_idx].unsqueeze(0).to(device)  # [1, C]
            label_tensor = torch.tensor([fit_id], dtype=torch.long).to(device)

            loss = criterion_no_smooth(logit_row, label_tensor)
            val_losses.append(loss.item())

        size_acc = correct / total if total > 0 else 0.0
        val_size_loss = float(np.mean(val_losses)) if val_losses else 0.0

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} "
            f"val_size_acc={size_acc:.4f}  val_size_loss={val_size_loss:.4f}"
        )

        # Early stopping
        if val_size_loss < best_val_loss - 1e-6:
            best_val_loss = val_size_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            stall = 0
            torch.save(best_state, out_root / "checkpoint.pt")
        else:
            stall += 1
            if stall >= patience:
                print("Early stopping.")
                break
        epoch_times.append(perf_counter() - t_epoch0)

    # Save best checkpoint
    if best_state is not None:
        torch.save(best_state, out_root / "checkpoint.pt")
        model.load_state_dict(best_state)

    # Compute simple runtime stats
    epoch_time_sec_mean = (sum(epoch_times) / len(epoch_times)) if epoch_times else None
    peak_cuda_mb = None
    if torch.cuda.is_available():
        peak_cuda_mb = int(torch.cuda.max_memory_allocated() / (1024 ** 2))
        torch.cuda.reset_peak_memory_stats()

    # Write run_info.json
    run_info = {
        "params_millions": round(num_params / 1e6, 3),
        "epoch_time_sec_mean": epoch_time_sec_mean,
        "peak_cuda_mb": peak_cuda_mb,
    }
    run_info.update({
        "dataset_dir": str(csv_dir),
        "data_gen": {
            "enabled": bool(gen_cfg.get("enable", False)),
            "n_consumers": gen_cfg.get("n_consumers"),
            "n_products": gen_cfg.get("n_products"),
            "seed": gen_cfg.get("seed"),
        }
    })
    (out_root / "run_info.json").write_text(json.dumps(run_info, ensure_ascii = False, indent = 2), encoding = "utf-8")

    # 8) Final evaluation (val & test), write metrics + preds
    def _eval_and_write(split_name: str, loader: torch.utils.data.DataLoader):
        """
        Run full evaluation on a split (val or test).

        Primary metric:
            • size-accuracy: pick the size candidate with highest P(fit) per (consumer_id, transaction_date_t)
            and compare to the true purchased size (the row labeled "fit").

        Also reports (for reference):
            • classification metrics computed ONLY on the true purchased rows (one per group),
            not on all expanded candidates.

        Writes:
            • metrics_<split>.json
            • preds_<split>.csv (includes fit_prob; row-aligned with processed <split>.csv)
        """
        model.eval()

        # Row-level outputs aligned with processed <split>.csv
        y_true_rows: list[int] = []
        y_pred_rows: list[int] = []
        fit_probs: list[float] = []

        fit_class_idx = label_map["fit"]

        with torch.no_grad():
            for batch in loader:
                for k, v in batch.items():
                    batch[k] = v.to(device) if torch.is_tensor(v) else v

                class_logits = model(batch)  # [B, C]
                probs = F.softmax(class_logits, dim=1)

                y_true_rows.extend(batch["label"].detach().cpu().tolist())
                y_pred_rows.extend(class_logits.argmax(dim=1).detach().cpu().tolist())
                fit_probs.extend(probs[:, fit_class_idx].detach().cpu().tolist())

        y_true = np.asarray(y_true_rows, dtype=int) if y_true_rows else np.array([], dtype=int)
        y_pred = np.asarray(y_pred_rows, dtype=int) if y_pred_rows else np.array([], dtype=int)

        metrics: dict = {}

        try:
            df_split = pd.read_csv(processed_dir / f"{split_name}.csv")

            # Require row alignment between CSV and model outputs
            required_cols = {"consumer_id", "transaction_date_t", "label_id", "size_id_t"}
            if len(df_split) != len(fit_probs) or not required_cols.issubset(df_split.columns):
                raise ValueError(
                    f"Cannot align preds with {split_name}.csv "
                    f"(rows: csv={len(df_split)} preds={len(fit_probs)}; "
                    f"missing_cols={sorted(required_cols - set(df_split.columns))})"
                )

            df_split = df_split.copy()
            df_split["fit_prob"] = fit_probs
            df_split["y_true_row"] = y_true
            df_split["y_pred_row"] = y_pred

            groups = df_split.groupby(["consumer_id", "transaction_date_t"], sort=False)

            # --- Size-accuracy (main metric) ---
            total = 0
            correct = 0

            # --- Classification metrics on TRUE rows only ---
            true_row_indices: list[int] = []

            fit_id = label_map["fit"]

            for _, g in groups:
                # true purchased candidate = the row labeled "fit"
                true_rows = g[g["label_id"] == fit_id]
                if true_rows.empty:
                    continue

                true_idx = int(true_rows.index[0])
                true_row_indices.append(true_idx)
                true_size = true_rows["size_id_t"].iloc[0]

                # predicted size = candidate with max P(fit)
                pred_row = g.loc[g["fit_prob"].idxmax()]
                pred_size = pred_row["size_id_t"]

                correct += int(pred_size == true_size)
                total += 1

            size_acc = correct / total if total > 0 else 0.0
            metrics["size_accuracy"] = float(size_acc)

            # --- Reference-only classification metrics (true rows only) ---
            if true_row_indices:
                y_true_true = df_split.loc[true_row_indices, "y_true_row"].to_numpy(dtype=int)
                y_pred_true = df_split.loc[true_row_indices, "y_pred_row"].to_numpy(dtype=int)

                num_classes = 3
                acc_true = accuracy(y_true_true, y_pred_true)
                prf_true = precision_recall_f1_per_class(y_true_true, y_pred_true, num_classes)
                cm_true = confusion_matrix(y_true_true, y_pred_true, num_classes).tolist()

                metrics["classification_on_true_rows"] = {
                    "accuracy": float(acc_true),
                    "per_class": prf_true,
                    "confusion_matrix": cm_true,
                    "n_true_rows": int(len(true_row_indices)),
                }

            # Save aligned preds (includes fit_prob)
            df_split.to_csv(out_root / f"preds_{split_name}.csv", index=False, encoding="utf-8")

            print(f"{split_name}: size-accuracy={size_acc:.4f}")

        except Exception as e:
            # Still save minimal outputs if alignment fails
            metrics["error"] = f"{type(e).__name__}: {e}"
            pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "fit_prob": fit_probs}).to_csv(
                out_root / f"preds_{split_name}.csv", index=False, encoding="utf-8"
            )
            print(f"[WARN] {split_name}: evaluation incomplete: {e}")

        _save_json(metrics, out_root / f"metrics_{split_name}.json")

    if log_cfg.get("write_preds", True):
        _eval_and_write("val", val_loader)
        _eval_and_write("test", test_loader)

    print(f"Run artifacts saved to: {out_root}")

    return str(out_root)


if __name__ == "__main__":
    main()
