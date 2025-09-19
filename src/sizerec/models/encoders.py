"""
Backbone encoders with a shared interface:
  forward(x: [B,T,d], padding_mask: [B,T] bool, causal_mask: [T,T] bool) -> [B,T,d]

- TransformerEncoderBackbone: wraps PyTorch TransformerEncoder (causal + padding masks).
- xLSTMEncoder: stacked uni-directional LSTM; handles padding via pack/pad. Causal by construction.
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TransformerEncoderBackbone(nn.Module):
    def __init__(self, d_model: int, n_layers: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model = d_model,
            nhead = n_heads,
            dim_feedforward = 4 * d_model,
            dropout = dropout,
            batch_first = True,
            norm_first = True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, 
            num_layers = n_layers,
            enable_nested_tensor = False)

    def forward(
        self,
        x: torch.Tensor,                # [B,T,d]
        padding_mask: torch.Tensor,     # [B,T] True=PAD
        causal_mask: Optional[torch.Tensor] = None,  # [T,T] True=block future
    ) -> torch.Tensor:
        return self.encoder(x, mask = causal_mask, src_key_padding_mask = padding_mask)


class xLSTMEncoder(nn.Module):
    """
    Minimal xLSTM-like recurrent encoder:
    - uni-directional LSTM stack (causal by design)
    - dropout between layers
    - optional final LayerNorm

    Note: This is not a faithful reimplementation of every xLSTM detail from the paper;
    it's a clean, efficient recurrent backbone that matches our interface for fair comparison.
    """
    def __init__(self, d_model: int, n_layers: int, dropout: float, layernorm: bool = True) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size = d_model,
            hidden_size = d_model,
            num_layers = n_layers,
            dropout = dropout if n_layers > 1 else 0.0,
            batch_first = True,
            bidirectional = False,
        )
        self.ln = nn.LayerNorm(d_model) if layernorm else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,                # [B,T,d]
        padding_mask: torch.Tensor,     # [B,T] True=PAD
        causal_mask: Optional[torch.Tensor] = None,  # ignored; recurrence is causal
    ) -> torch.Tensor:
        # lengths from padding mask
        lengths = (~padding_mask).sum(dim=1).cpu()  # [B]
        # sort by length desc for pack (PyTorch requirement)
        lengths_sorted, idx_sort = torch.sort(lengths, descending=True)
        x_sorted = x.index_select(0, idx_sort)

        packed = pack_padded_sequence(x_sorted, lengths_sorted, batch_first = True, enforce_sorted = True)
        packed_out, _ = self.lstm(packed)
        out_sorted, _ = pad_packed_sequence(packed_out, batch_first = True, total_length = x.size(1))
        # restore original order
        _, idx_unsort = torch.sort(idx_sort)
        out = out_sorted.index_select(0, idx_unsort)
        return self.ln(out)
