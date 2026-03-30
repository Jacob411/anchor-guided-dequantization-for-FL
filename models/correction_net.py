"""
CorrectionNet: server-side MLP that learns to debias quantized weight deltas.

Trained each round using anchor (full-precision) updates as supervision.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Simple residual block for the correction network."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class CorrectionNet(nn.Module):
    """
    A residual MLP that maps quantized weight deltas → corrected (FP-like) deltas.

    Architecture:
        Linear(in_dim → hidden_dim)
        3 × ResidualBlock(hidden_dim)
        Linear(hidden_dim → in_dim)   # output residual correction

    The network learns an additive correction: output = input + Δcorrection,
    so it only needs to learn the *bias* introduced by quantization.

    Args:
        in_dim      : number of elements in the flattened weight delta
        hidden_dim  : width of the hidden residual blocks
    """

    def __init__(self, in_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.residual_blocks = nn.Sequential(
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
        )
        self.decoder = nn.Linear(hidden_dim, in_dim)

        # Zero-init decoder so the net starts as identity (no correction)
        nn.init.zeros_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: flattened quantized delta, shape (B, in_dim) or (in_dim,)
        Returns:
            corrected delta, same shape as x
        """
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)

        h = self.encoder(x)
        h = self.residual_blocks(h)
        correction = self.decoder(h)
        out = x + correction          # additive residual correction

        return out.squeeze(0) if squeeze else out
