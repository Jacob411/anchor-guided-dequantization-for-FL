"""
Anchor client — sends full-precision weight deltas.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch

from clients.base_client import BaseClient


class AnchorClient(BaseClient):
    """
    Full-precision client. Trains locally and returns unquantized Δw.
    Used as the reference/supervision signal for the CorrectionNet.
    """

    @property
    def is_anchor(self) -> bool:
        return True

    def train_one_round(
        self, global_state: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        """
        Returns:
            delta      : {layer: FP32 tensor} — full-precision weight delta
            comm_bytes : cost of sending FP32 parameters (4 bytes/element)
        """
        self.load_global_model(global_state)
        self._local_train()

        delta = self._compute_delta(global_state)

        # Communication cost: FP32 = 4 bytes per element
        total_elements = sum(t.numel() for t in delta.values())
        comm_bytes = total_elements * 4

        return delta, comm_bytes
