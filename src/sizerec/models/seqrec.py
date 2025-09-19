"""
SeqRec: shared model wrapper around a pluggable encoder (Transformer or xLSTM).

Inputs (from data_module.collate):
  step-wise: product_type [B,T], material [B,T], size [B,T], (section [B,T]?)
  static:   gender [B], age_bin [B], (country [B]?)
  target-t: pt_t [B], mat_t [B], size_t [B], (sec_t [B]?)
  masks:    padding_mask [B,T], causal_mask [T,T]

Outputs:
  logits [B, num_classes]
"""

from __future__ import annotations
from typing import Optional, Dict

import torch
import torch.nn as nn


class SeqRec(nn.Module):
    def __init__(
        self,
        *,
        # vocab sizes (step-wise 1-based; 0 is PAD)
        num_product_types: int,
        num_materials: int,
        num_sizes: int,
        num_sections: Optional[int],      # None to disable
        # static (0-based)
        num_genders: int,
        num_age_bins: int,
        num_countries: Optional[int],
        # model dims
        d_model: int = 256,
        dropout: float = 0.1,
        max_len: int = 512,
        num_classes: int = 4,
        # pluggable encoder module
        encoder: nn.Module = None,
    ) -> None:
        super().__init__()
        assert encoder is not None, "SeqRec requires an encoder module"
        self.encoder = encoder
        self.d_model = d_model
        self.num_classes = num_classes

        # ---- Embeddings: step-wise (padding_idx=0) ----
        self.emb_product_type = nn.Embedding(num_product_types + 1, d_model, padding_idx=0)
        self.emb_material     = nn.Embedding(num_materials + 1, d_model, padding_idx=0)
        self.emb_size         = nn.Embedding(num_sizes + 1, d_model, padding_idx=0)
        self.emb_section      = nn.Embedding(num_sections + 1, d_model, padding_idx=0) if num_sections else None

        # ---- Embeddings: static ----
        self.emb_gender    = nn.Embedding(num_genders, d_model)
        self.emb_age_bin   = nn.Embedding(num_age_bins, d_model)
        self.emb_country   = nn.Embedding(num_countries, d_model) if num_countries else None

        # Positional embeddings (harmless for LSTM, helpful for Transformer)
        self.positional = nn.Embedding(max_len, d_model)

        self.input_ln = nn.LayerNorm(d_model)
        self.input_do = nn.Dropout(dropout)

        self.head = nn.Linear(d_model, num_classes)
        nn.init.xavier_uniform_(self.head.weight); nn.init.zeros_(self.head.bias)

    def _sum_history_embeddings(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        pt, mat, siz = batch["product_type"], batch["material"], batch["size"]  # [B,T]
        x = self.emb_product_type(pt) + self.emb_material(mat) + self.emb_size(siz)
        if self.emb_section is not None and "section" in batch:
            x = x + self.emb_section(batch["section"])  # [B,T,d]

        # add static embeddings (broadcast) and position
        g = self.emb_gender(batch["gender"])    # [B,d]
        a = self.emb_age_bin(batch["age_bin"])  # [B,d]
        x = x + g.unsqueeze(1) + a.unsqueeze(1)
        if self.emb_country is not None and "country" in batch:
            x = x + self.emb_country(batch["country"]).unsqueeze(1)

        B, T = pt.shape
        pos_idx = torch.arange(T, device=pt.device).unsqueeze(0)
        x = x + self.positional(pos_idx)

        return self.input_do(self.input_ln(x))

    def _candidate_embedding(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # If current-purchase fields are missing, return zeros (keeps backward compatibility)
        if not all(k in batch for k in ("pt_t", "mat_t", "size_t")):
            return torch.zeros((batch["gender"].shape[0], self.d_model), device=batch["gender"].device)
        cand = self.emb_product_type(batch["pt_t"]) \
             + self.emb_material(batch["mat_t"]) \
             + self.emb_size(batch["size_t"])
        if self.emb_section is not None and "sec_t" in batch:
            cand = cand + self.emb_section(batch["sec_t"])
        return cand  # [B,d]

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        pad_mask = batch["padding_mask"]      # [B,T] True=PAD
        attn_mask = batch.get("causal_mask")  # [T,T]

        # 1) History → embeddings → encoder
        x = self._sum_history_embeddings(batch)       # [B,T,d]
        enc = self.encoder(x, padding_mask = pad_mask, causal_mask = attn_mask)  # [B,T,d]

        # 2) Pool last valid step
        lengths = (~pad_mask).sum(dim=1).clamp(min=1)             # [B]
        B = lengths.size(0)
        last_idx = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(B, 1, self.d_model)
        last_states = enc.gather(dim = 1, index = last_idx).squeeze(1)  # [B,d]

        # 3) Fuse with current-purchase embedding (if provided)
        cand = self._candidate_embedding(batch)  # [B,d]
        fused = last_states + cand

        # 4) Classify
        return self.head(fused)  # [B, num_classes]
