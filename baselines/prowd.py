"""
ProWD baseline (simplified) — server-side per-layer dequantization MLP.

ProWD trains a small MLP per layer to reconstruct FP weights from quantized ones.
This is a simplified version: a single shared MLP operating on flattened deltas,
trained with the same anchor supervision as our method (for fair comparison).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from server.aggregator import fedavg, apply_delta


class _DequantizerMLP(nn.Module):
    """Simple 3-layer MLP dequantizer (ProWD-style)."""

    def __init__(self, dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
        )
        # Zero-init output layer
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)  # Residual


class ProWDStrategy:
    """
    Simplified ProWD: server-side MLP dequantizer trained each round
    using anchor updates as supervision; reconstructed deltas are then FedAvg'd.
    """

    name = "prowd"

    def __init__(
        self,
        delta_dim: int,
        hidden_dim: int = 256,
        lr: float = 1e-3,
        epochs: int = 5,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.epochs = epochs
        self.mlp = _DequantizerMLP(delta_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.mlp.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def _flatten(self, delta: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([t.float().flatten() for t in delta.values()])

    def _unflatten(
        self, flat: torch.Tensor, ref: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        result = {}
        offset = 0
        for k, v in ref.items():
            result[k] = flat[offset : offset + v.numel()].reshape(v.shape)
            offset += v.numel()
        return result

    def aggregate(
        self,
        all_deltas: List[Dict[str, torch.Tensor]],
        anchor_deltas: List[Dict[str, torch.Tensor]],
        quantized_deltas: List[Dict[str, torch.Tensor]],
        global_state: Dict[str, torch.Tensor],
        sample_counts: Optional[List[float]] = None,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], float]:
        if not anchor_deltas or not quantized_deltas:
            agg_delta = fedavg(all_deltas, sample_counts)
            return apply_delta(global_state, agg_delta), 0.0

        anchor_flat = torch.stack(
            [self._flatten(d) for d in anchor_deltas]
        ).mean(dim=0).to(self.device)

        q_batch = torch.stack(
            [self._flatten(d) for d in quantized_deltas]
        ).to(self.device)

        # Train dequantizer
        total_loss = 0.0
        self.mlp.train()
        target = anchor_flat.unsqueeze(0).expand_as(q_batch)
        for _ in range(self.epochs):
            self.optimizer.zero_grad()
            out = self.mlp(q_batch)
            loss = self.criterion(out, target)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        # Apply dequantizer to quantized deltas
        self.mlp.eval()
        reconstructed: List[Dict[str, torch.Tensor]] = []
        with torch.no_grad():
            for d_flat in q_batch:
                rec_flat = self.mlp(d_flat.unsqueeze(0)).squeeze(0).cpu()
                reconstructed.append(self._unflatten(rec_flat, quantized_deltas[0]))

        combined = anchor_deltas + reconstructed
        n_total = len(anchor_deltas) + len(reconstructed)
        counts = sample_counts if sample_counts else [1.0] * n_total

        agg_delta = fedavg(combined, counts)
        new_state = apply_delta(global_state, agg_delta)
        return new_state, total_loss / self.epochs
