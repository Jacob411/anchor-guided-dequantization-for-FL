"""
FedShift baseline — global mean-shift alignment before aggregation.

Aligns quantized updates to anchor distribution via:
    shifted_delta = quantized_delta + (mean_anchor - mean_quantized)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from server.aggregator import fedavg, apply_delta


class FedShiftStrategy:
    """
    FedShift: compute per-layer mean shift between anchor and quantized updates,
    then apply that shift to all quantized updates before FedAvg.
    """

    name = "fedshift"

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
            new_state    : updated global model state dict (after shift + FedAvg)
            corr_loss    : 0.0 (no learned correction)
        """
        if not anchor_deltas or not quantized_deltas:
            # Fallback: plain FedAvg
            agg_delta = fedavg(all_deltas, sample_counts)
            return apply_delta(global_state, agg_delta), 0.0

        # Per-layer mean of anchor and quantized deltas
        anchor_mean = {
            k: torch.stack([d[k].float() for d in anchor_deltas]).mean(dim=0)
            for k in anchor_deltas[0]
        }
        quant_mean = {
            k: torch.stack([d[k].float() for d in quantized_deltas]).mean(dim=0)
            for k in quantized_deltas[0]
        }

        # Global shift: add (anchor_mean - quant_mean) to every quantized delta
        shifted_quantized: List[Dict[str, torch.Tensor]] = []
        for d in quantized_deltas:
            shifted = {
                k: d[k].float() + (anchor_mean[k] - quant_mean[k])
                for k in d
            }
            shifted_quantized.append(shifted)

        # Aggregate anchors + shifted quantized
        combined = anchor_deltas + shifted_quantized
        n_anchor = len(anchor_deltas)
        n_quant = len(shifted_quantized)

        if sample_counts is not None:
            # sample_counts is assumed to be in order [anchors..., quantized...]
            combined_counts = sample_counts
        else:
            combined_counts = [1.0] * (n_anchor + n_quant)

        agg_delta = fedavg(combined, combined_counts)
        new_state = apply_delta(global_state, agg_delta)
        return new_state, 0.0
