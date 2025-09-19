"""
Dataset + collate for fit_outcome prediction (1-based step-wise IDs).

- Reads processed split CSVs produced by seq_prep.save_processed_splits().
- Decodes list columns (JSON) back to Python lists of ints.
- Pads batches with PAD_ID=0 (reserved), builds padding & causal masks.
- Returns integer tensors only (no strings).

Expected columns in each split CSV:
  consumer_id (str)
  seq_len (int)
  product_type_ids (JSON list[int])   # step-wise, IDs start at 1
  material_ids     (JSON list[int])   # step-wise, IDs start at 1
  size_ids         (JSON list[int])   # step-wise, IDs start at 1
  section_ids      (JSON list[int])   # step-wise, IDs start at 1  [optional if use_section=False]
  gender_id        (int)              # static, can be 0-based
  country_id       (int)              # static, can be 0-based      [optional if use_country=False]
  age_bin_id       (int)              # static, can be 0-based
  label_id         (int)              # target class in {0..3}
  product_type_id_t (int)             # for analysis; not used here
  transaction_date_t                  # for analysis; not used here
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

PAD_ID = 0  # reserved padding index for all step-wise token sequences


class SequenceDataset(Dataset):
    """
    Loads a single split CSV (train/val/test) and exposes integer features.

    Args:
        csv_path: Path to the split CSV.
        use_section: If True, expect and load `section_ids`.
        use_country: If True, expect and load `country_id`.

    __getitem__ returns a dict with:
        step-wise: "product_type", "material", "size", ("section")
        static:   "gender", "age_bin", ("country")
        target:   "label"
        meta:     "seq_len" (int)
    """

    def __init__(
        self,
        csv_path: str | Path,
        *,
        use_section: bool,
        use_country: bool,
    ) -> None:
        import pandas as pd  # local import to keep module lightweight

        self.path = Path(csv_path)
        self.use_section = use_section
        self.use_country = use_country

        df = pd.read_csv(self.path)

        # decode list columns (stored as JSON strings)
        def loads(x):
            return json.loads(x) if isinstance(x, str) else x

        self.product_type_ids: List[List[int]] = df["product_type_ids"].map(loads).tolist()
        self.material_ids: List[List[int]] = df["material_ids"].map(loads).tolist()
        self.size_ids: List[List[int]] = df["size_ids"].map(loads).tolist()
        if self.use_section:
            self.section_ids: Optional[List[List[int]]] = df["section_ids"].map(loads).tolist()
        else:
            self.section_ids = None

        # static per-example fields
        self.gender_id: List[int] = df["gender_id"].astype(int).tolist()
        self.age_bin_id: List[int] = df["age_bin_id"].astype(int).tolist()
        if self.use_country:
            self.country_id: Optional[List[int]] = df["country_id"].astype(int).tolist()
        else:
            self.country_id = None

        self.label_id: List[int] = df["label_id"].astype(int).tolist()
        self.seq_len: List[int] = df["seq_len"].astype(int).tolist()

        self.pt_t   = df["product_type_id_t"].astype(int).tolist()
        self.mat_t  = df["material_id_t"].astype(int).tolist()
        self.size_t = df["size_id_t"].astype(int).tolist()
        self.sec_t  = df["section_id_t"].astype(int).tolist() if self.use_section else None


    def __len__(self) -> int:
        return len(self.label_id)

    def __getitem__(self, idx: int) -> Dict:
        item = {
            "product_type": self.product_type_ids[idx],  # list[int], 1-based ids
            "material": self.material_ids[idx],          # list[int], 1-based ids
            "size": self.size_ids[idx],                  # list[int], 1-based ids
            "gender": int(self.gender_id[idx]),          # int
            "age_bin": int(self.age_bin_id[idx]),        # int
            "label": int(self.label_id[idx]),            # int (0..3)
            "seq_len": int(self.seq_len[idx]),           # int
            "pt_t": self.pt_t[idx],
            "mat_t": self.mat_t[idx],
            "size_t": self.size_t[idx],
        }
        if self.use_section and self.section_ids is not None:
            item["section"] = self.section_ids[idx]      # list[int], 1-based ids
            item["sec_t"] = self.sec_t[idx]
        if self.use_country and self.country_id is not None:
            item["country"] = int(self.country_id[idx])  # int
        return item


def _pad_1d(seq: List[int], T: int) -> List[int]:
    """Right-pad a list with PAD_ID to length T (truncate from the left if longer)."""
    if len(seq) >= T:
        return seq[-T:]
    return seq + [PAD_ID] * (T - len(seq))


def make_collate(max_len: Optional[int] = None):
    """
    Create a collate function for DataLoader.

    If max_len is provided, sequences are truncated to at most max_len here.
    Otherwise, batch max length is used.

    Returns a function(batch) -> dict of torch tensors:
      step-wise:
        - product_type [B, T] (long, 0=PAD)
        - material     [B, T]
        - size         [B, T]
        - section      [B, T] (optional)
      static:
        - gender       [B]
        - age_bin      [B]
        - country      [B]    (optional)
      targets:
        - label        [B]
      masks:
        - padding_mask [B, T] (bool; True=PAD)
        - causal_mask  [T, T] (bool; True=block upper triangle)
    """

    def collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        B = len(batch)
        lengths = [len(b["product_type"]) for b in batch]
        T = max(lengths) if max_len is None else min(max(lengths), max_len)

        # allocate tensors
        def zeros_bt():
            return torch.full((B, T), fill_value = PAD_ID, dtype = torch.long)

        pt = zeros_bt()
        mat = zeros_bt()
        siz = zeros_bt()
        sec = zeros_bt() if ("section" in batch[0]) else None

        gender = torch.empty(B, dtype = torch.long)
        age_bin = torch.empty(B, dtype = torch.long)
        country = torch.empty(B, dtype = torch.long) if ("country" in batch[0]) else None
        label = torch.empty(B, dtype = torch.long)

        pt_t   = torch.empty(B, dtype = torch.long)
        mat_t  = torch.empty(B, dtype = torch.long)
        size_t = torch.empty(B, dtype = torch.long)
        sec_t  = torch.empty(B, dtype = torch.long) if ("sec_t" in batch[0]) else None


        padding_mask = torch.ones((B, T), dtype = torch.bool)  # True=PAD

        for i, b in enumerate(batch):
            # step-wise ids are already 1-based on disk; just pad/truncate
            pt[i] = torch.tensor(_pad_1d(b["product_type"], T), dtype = torch.long)
            mat[i] = torch.tensor(_pad_1d(b["material"], T), dtype = torch.long)
            siz[i] = torch.tensor(_pad_1d(b["size"], T), dtype = torch.long)
            
            pt_t[i]   = b["pt_t"]
            mat_t[i]  = b["mat_t"]
            size_t[i] = b["size_t"]
            
            if sec is not None and "section" in b:
                sec[i] = torch.tensor(_pad_1d(b["section"], T), dtype = torch.long)
                sec_t[i] = b["sec_t"]

            L = min(b["seq_len"], T)
            padding_mask[i, :L] = False  # False=real token, True=PAD

            gender[i] = b["gender"]
            age_bin[i] = b["age_bin"]
            if country is not None and "country" in b:
                country[i] = b["country"]

            label[i] = b["label"]

        # causal mask [T, T]: block attention to future positions (upper triangle)
        causal_mask = torch.triu(torch.ones((T, T), dtype = torch.bool), diagonal = 1)

        out = {
            "product_type": pt,
            "material": mat,
            "size": siz,
            "gender": gender,
            "age_bin": age_bin,
            "label": label,
            "padding_mask": padding_mask,
            "causal_mask": causal_mask,
            "pt_t": pt_t, 
            "mat_t": mat_t, 
            "size_t": size_t
        }
        if sec is not None:
            out["section"] = sec
            out["sec_t"] = sec_t
        if country is not None:
            out["country"] = country
        return out

    return collate
