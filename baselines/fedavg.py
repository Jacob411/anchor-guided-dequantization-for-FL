"""
FedAvg baseline — uniform aggregation, no quantization correction.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from server.aggregator import fedavg, apply_delta


class FedAvgStrategy:
    """
    Vanilla FedAvg: aggregate all deltas with uniform (or count-weighted) averaging.
    Does not distinguish between anchor and constrained clients.
    """

    name = "fedavg"

    def aggregate(
        self,
        all_deltas: List[Dict[str, torch.Tensor]],
        anchor_deltas: List[Dict[str, torch.Tensor]],
        quantized_deltas: List[Dict[str, torch.Tensor]],
        global_state: Dict[str, torch.Tensor],
        sample_counts: Optional[List[float]] = None,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], float]:
        """
        Returns:
            new_state    : updated global model state dict
            corr_loss    : 0.0 (no correction network)
        """
        agg_delta = fedavg(all_deltas, sample_counts)
        new_state = apply_delta(global_state, agg_delta)
        return new_state, 0.0
