"""
Abstract base class for all FL clients.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class BaseClient(ABC):
    """
    Abstract FL client.

    Subclasses must implement:
        - train_one_round() → returns (weight_delta, comm_bytes)
    """

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        lr: float = 0.01,
        local_epochs: int = 2,
    ) -> None:
        self.client_id    = client_id
        self.model        = model
        self.dataloader   = dataloader
        self.device       = device
        self.lr           = lr
        self.local_epochs = local_epochs

    # ------------------------------------------------------------------
    # Shared training logic
    # ------------------------------------------------------------------

    def _local_train(self) -> None:
        """Run local SGD for `local_epochs` over the client's data."""
        self.model.train()
        self.model.to(self.device)
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4
        )
        criterion = nn.CrossEntropyLoss()

        for _ in range(self.local_epochs):
            for images, labels in self.dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(images), labels)
                loss.backward()
                optimizer.step()

    def _compute_delta(
        self, global_state: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Return Δw = w_local − w_global (full precision, CPU tensors)."""
        local_state = self.model.state_dict()
        delta = {
            k: (local_state[k].float().cpu() - global_state[k].float().cpu())
            for k in global_state
        }
        return delta

    def load_global_model(self, global_state: Dict[str, torch.Tensor]) -> None:
        """Overwrite local model weights with the global model, respecting model device."""
        target_device = next(self.model.parameters()).device
        self.model.load_state_dict(
            {k: v.clone().to(target_device) for k, v in global_state.items()}
        )

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    @abstractmethod
    def train_one_round(
        self, global_state: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        """
        Perform local training and return:
            delta     : weight delta dict {layer_name: tensor}
            comm_bytes: estimated uplink communication cost in bytes
        """
        ...

    @property
    @abstractmethod
    def is_anchor(self) -> bool:
        """True if this client sends full-precision updates."""
        ...
