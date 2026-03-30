"""
Aggregation strategies for the FL server.

Includes:
  - FedAvg (uniform weighted by dataset size)
  - Score-based selective aggregation (cosine similarity to anchor mean)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


def _flatten_delta(delta: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.float().flatten() for t in delta.values()])


# ---------------------------------------------------------------------------
# FedAvg
# ---------------------------------------------------------------------------

def fedavg(
    deltas: List[Dict[str, torch.Tensor]],
    weights: Optional[List[float]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Weighted average of weight deltas. If `weights` is None, uses uniform avg.

    Args:
        deltas  : list of per-client weight delta dicts
        weights : per-client sample counts (or any positive weights)
    Returns:
        aggregated delta dict
    """
    if not deltas:
        raise ValueError("Must provide at least one delta.")

    if weights is None:
        weights = [1.0] * len(deltas)

    total_weight = sum(weights)
    agg: Dict[str, torch.Tensor] = {}

    for key in deltas[0]:
        agg[key] = sum(
            w * d[key].float() for w, d in zip(weights, deltas)
        ) / total_weight  # type: ignore[assignment]

    return agg


# ---------------------------------------------------------------------------
# Score-based selective aggregation
# ---------------------------------------------------------------------------

def compute_similarity_scores(
    deltas: List[Dict[str, torch.Tensor]],
    reference_delta: Dict[str, torch.Tensor],
) -> List[float]:
    """
    Compute cosine similarity of each delta to a reference (e.g. mean anchor).

    Args:
        deltas          : list of client deltas
        reference_delta : reference direction (mean anchor delta)
    Returns:
        list of similarity scores in [0, 1], one per client
    """
    ref_flat = _flatten_delta(reference_delta)
    scores = []
    for d in deltas:
        d_flat = _flatten_delta(d)
        sim = F.cosine_similarity(d_flat.unsqueeze(0), ref_flat.unsqueeze(0)).item()
        # Map [-1, 1] → [0, 1]
        scores.append((sim + 1.0) / 2.0)
    return scores


def selective_aggregation(
    deltas: List[Dict[str, torch.Tensor]],
    anchor_deltas: List[Dict[str, torch.Tensor]],
    sample_counts: Optional[List[float]] = None,
    similarity_weight: float = 0.5,
) -> Tuple[Dict[str, torch.Tensor], List[float]]:
    """
    Aggregate `deltas` weighted by a combination of:
      - sample count (data quantity)
      - cosine similarity to the mean anchor delta (quality / direction)

    Final weight = (1 - sim_weight) * norm_count + sim_weight * sim_score

    Args:
        deltas            : all client deltas (anchors + corrected constrained)
        anchor_deltas     : anchor client deltas used to build reference direction
        sample_counts     : dataset sizes per client (same order as `deltas`)
        similarity_weight : how much to weight direction similarity vs. data size
    Returns:
        aggregated delta, list of final per-client weights
    """
    if not anchor_deltas:
        # No anchors available — fall back to pure FedAvg
        return fedavg(deltas, sample_counts), [1.0 / len(deltas)] * len(deltas)

    # Mean anchor delta as reference direction
    ref_delta = fedavg(anchor_deltas)

    # Normalised sample counts
    if sample_counts is None:
        sample_counts = [1.0] * len(deltas)
    total = sum(sample_counts)
    norm_counts = [s / total for s in sample_counts]

    # Cosine similarity scores
    sim_scores = compute_similarity_scores(deltas, ref_delta)

    # Combined weights
    final_weights = [
        (1 - similarity_weight) * nc + similarity_weight * ss
        for nc, ss in zip(norm_counts, sim_scores)
    ]

    # Re-normalise so they sum to 1 (for weighted average)
    weight_sum = sum(final_weights)
    final_weights = [w / weight_sum for w in final_weights]

    agg = fedavg(deltas, final_weights)
    return agg, final_weights


def apply_delta(
    global_state: Dict[str, torch.Tensor],
    delta: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Apply an aggregated delta to the global model state dict."""
    new_state = {}
    for k in global_state:
        new_state[k] = global_state[k].float() + delta[k].float().to(global_state[k].device)
        new_state[k] = new_state[k].to(global_state[k].dtype)
    return new_state
