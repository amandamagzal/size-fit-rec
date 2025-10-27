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
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)


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


class LSTMEncoderBackbone(nn.Module):
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
        lengths = (~padding_mask).sum(dim = 1).clamp(min=1)  # [B]

        packed = pack_padded_sequence(x, lengths.detach().cpu(), batch_first = True, enforce_sorted = False)
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first = True, total_length = x.size(1))
        return self.ln(out)


class xLSTMEncoderBackbone(nn.Module):
    """
    A thin adapter around the official xLSTM Block Stack that matches our interface:
      forward(x: [B,T,d], padding_mask: [B,T] bool, causal_mask: [T,T] bool|None) -> [B,T,d]

    Defaults to sLSTM-only (native backends) so it runs everywhere without custom kernels.
    Enable mLSTM by setting enable_mlstm=True (optionally with mlstm_kernels installed).
    """
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
        context_length: int,
        *,
        enable_mlstm: bool = False,
        slstm_backend: str = "native",  # "native" | "cuda"
        conv1d_kernel_size: int = 4,
        ff_proj_factor: float = 4/3,    # matches paper’s typical gated-MLP sizing
        ff_activation: str = "gelu",
    ) -> None:
        super().__init__()

        # Configure sLSTM block (safe default: native backend, no custom kernels)
        slstm_block_cfg = sLSTMBlockConfig(
            slstm = sLSTMLayerConfig(
                backend = slstm_backend,             # "native" runs anywhere; "cuda" needs kernels
                num_heads = n_heads,
                conv1d_kernel_size = conv1d_kernel_size,
                bias_init = "powerlaw_blockdependent",
            ),
            feedforward = FeedForwardConfig(
                proj_factor = ff_proj_factor,
                act_fn = ff_activation,
                dropout = dropout,
            ),
        )

        # Optional mLSTM block: only attach when requested
        mlstm_block_cfg = None
        if enable_mlstm:
            mlstm_block_cfg = mLSTMBlockConfig(
                mlstm = mLSTMLayerConfig(
                    # keep defaults simple; you can tune these later
                    conv1d_kernel_size = conv1d_kernel_size,
                    qkv_proj_blocksize = 4,
                    num_heads = n_heads,
                )
            )

        stack_cfg = xLSTMBlockStackConfig(
            mlstm_block = mlstm_block_cfg,
            slstm_block = slstm_block_cfg,
            context_length = context_length,
            num_blocks = n_layers,
            embedding_dim = d_model,
            dropout = dropout,
            # Optionally place sLSTM blocks at specific layers, e.g. [1] to insert one sLSTM
            # slstm_at = [1],
        )

        self.backbone = xLSTMBlockStack(stack_cfg)

        self.out_ln = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,                # [B,T,d]
        padding_mask: torch.Tensor,     # [B,T] True=PAD
        causal_mask: Optional[torch.Tensor] = None,  # not needed (xLSTM is causal)
    ) -> torch.Tensor:
        # Zero-out padded positions to avoid positional embedding leakage into recurrence
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        # xLSTM block stack expects [B,T,d]; causal behavior is internal
        y = self.backbone(x)
        return self.out_ln(y)