"""
Transformer for fit_outcome prediction (4 classes).

Inputs (from data_module.make_collate):
  - step-wise IDs (1-based, 0 is PAD):
      product_type [B,T], material [B,T], size [B,T], (section [B,T]? )
  - static IDs (per example):
      gender [B], age_bin [B], (country [B]? )
  - masks:
      padding_mask [B,T] (bool, True=PAD), causal_mask [T,T] (bool, True=block future)

Output:
  - logits [B, 4]  (class order must match your config’s label_order)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TransformerRec(nn.Module):
    def __init__(
        self,
        *,
        # vocab sizes (step-wise are 1-based with PAD=0, so embeddings need +1 rows for index 0)
        num_product_types: int,
        num_materials: int,
        num_sizes: int,
        num_sections: int | None,    # optional step-wise feature
        # static vocabs (0-based is fine)
        num_genders: int,
        num_age_bins: int,
        num_countries: int | None,   # optional static feature
        # model dims
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
        max_len: int = 512,
        num_classes: int = 4,
        use_cls: bool = False,       # keep False for minimal baseline (pool last step)
    ) -> None:
        super().__init__()
        self.use_cls = use_cls
        self.num_classes = num_classes
        self.d_model = d_model
        self.max_len = max_len

        # ---- Embeddings: step-wise (with padding_idx=0) ----
        self.emb_product_type = nn.Embedding(num_product_types + 1, d_model, padding_idx=0)
        self.emb_material     = nn.Embedding(num_materials + 1, d_model, padding_idx=0)
        self.emb_size         = nn.Embedding(num_sizes + 1, d_model, padding_idx=0)
        self.emb_section      = nn.Embedding(num_sections + 1, d_model, padding_idx=0) if num_sections else None

        # ---- Embeddings: static (no padding; broadcast over time) ----
        self.emb_gender    = nn.Embedding(num_genders, d_model)
        self.emb_age_bin   = nn.Embedding(num_age_bins, d_model)
        self.emb_country   = nn.Embedding(num_countries, d_model) if num_countries else None

        # ---- Positional embeddings ----
        self.positional = nn.Embedding(max_len, d_model)

        # Optional input norm/dropout
        self.input_ln = nn.LayerNorm(d_model)
        self.input_do = nn.Dropout(dropout)

        # ---- Transformer encoder ----
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # ---- Classification head ----
        self.head = nn.Linear(d_model, num_classes)

        # Init (lightweight Xavier for linear; embeddings default are fine)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            batch: dict from collate():
              - product_type/material/size/(section): [B,T] long
              - gender/age_bin/(country): [B] long
              - padding_mask: [B,T] bool (True=PAD)
              - causal_mask:  [T,T] bool (True=block)

        Returns:
            logits: [B, num_classes]
        """
        pt = batch["product_type"]   # [B,T]
        mat = batch["material"]      # [B,T]
        siz = batch["size"]          # [B,T]
        sec = batch.get("section", None)  # [B,T] or None

        gender  = batch["gender"]    # [B]
        age_bin = batch["age_bin"]   # [B]
        country = batch.get("country", None)  # [B] or None

        pad_mask = batch["padding_mask"]  # [B,T]
        attn_mask = batch["causal_mask"]  # [T,T]

        B, T = pt.shape
        device = pt.device

        # ---- Step-wise embeddings (sum) ----
        x = self.emb_product_type(pt) + self.emb_material(mat) + self.emb_size(siz)
        if self.emb_section is not None and sec is not None:
            x = x + self.emb_section(sec)  # [B,T,d]

        # ---- Broadcast static embeddings and add to every step ----
        g = self.emb_gender(gender)            # [B,d]
        a = self.emb_age_bin(age_bin)          # [B,d]
        x = x + g.unsqueeze(1) + a.unsqueeze(1)  # [B,1,d] broadcast across T
        if self.emb_country is not None and country is not None:
            c = self.emb_country(country)      # [B,d]
            x = x + c.unsqueeze(1)

        # ---- Positional embeddings ----
        pos_idx = torch.arange(T, device=device).unsqueeze(0)  # [1,T]
        x = x + self.positional(pos_idx)

        # ---- Norm + dropout before encoder ----
        x = self.input_do(self.input_ln(x))

        # ---- Transformer encoder ----
        # src_key_padding_mask: True at PAD positions (to ignore them)
        enc = self.encoder(x, mask=attn_mask, src_key_padding_mask=pad_mask)  # [B,T,d]

        # ---- Pool last valid step (based on padding mask) ----
        # lengths: count of non-pad tokens per example
        lengths = (~pad_mask).sum(dim=1).clamp(min=1)  # [B]
        last_idx = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(B, 1, self.d_model)  # [B,1,d]
        last_states = enc.gather(dim=1, index=last_idx).squeeze(1)  # [B,d]

        # ---- Head ----
        logits = self.head(last_states)  # [B,num_classes]
        return logits


def count_params(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
