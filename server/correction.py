"""
Server-side CorrectionNet training and application.

This module:
  1. Pools anchor (FP) and quantized deltas from each round
  2. Trains the CorrectionNet to predict FP deltas from quantized ones
  3. Applies the trained network to all quantized deltas before aggregation
"""
from __future__ import annotations

import copy
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.correction_net import CorrectionNet
from losses.losses import CombinedCorrectionLoss


def _flatten_delta(delta: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten all layer tensors in a delta dict into a single 1-D vector."""
    return torch.cat([t.float().flatten() for t in delta.values()])


def _unflatten_delta(
    flat: torch.Tensor, reference: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """Restore a flat vector back into a layer-keyed dict matching `reference` shapes."""
    result = {}
    offset = 0
    for k, v in reference.items():
        numel = v.numel()
        result[k] = flat[offset : offset + numel].reshape(v.shape)
        offset += numel
    return result


class ServerCorrection:
    """
    Manages the CorrectionNet lifecycle on the server.

    Args:
        delta_dim       : total number of parameters in the model (flat dim)
        hidden_dim      : CorrectionNet hidden width
        lr              : learning rate for CorrectionNet optimizer
        epochs          : inner training steps per FL round
        rec_weight      : weight for reconstruction loss term
        distill_weight  : weight for distillation loss term
        temperature     : KL distillation temperature
        device          : torch device
    """

    def __init__(
        self,
        delta_dim: int,
        hidden_dim: int = 256,
        lr: float = 1e-3,
        epochs: int = 5,
        rec_weight: float = 1.0,
        distill_weight: float = 0.5,
        temperature: float = 4.0,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.epochs = epochs

        self.net = CorrectionNet(delta_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.criterion = CombinedCorrectionLoss(rec_weight, distill_weight, temperature)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        quantized_deltas: List[Dict[str, torch.Tensor]],
        anchor_deltas: List[Dict[str, torch.Tensor]],
        global_model: nn.Module | None = None,
        val_loader: DataLoader | None = None,
    ) -> float:
        """
        Train the CorrectionNet for one FL round.

        We pair each quantized delta with the *mean* anchor delta as the
        FP target, since anchors and constrained clients may differ in count.

        Args:
            quantized_deltas : list of dequantized (lossy) deltas from constrained clients
            anchor_deltas    : list of FP deltas from anchor clients
            global_model     : current global model (for distillation; optional)
            val_loader       : server validation loader (for distillation; optional)
        Returns:
            mean training loss over correction epochs
        """
        if not anchor_deltas or not quantized_deltas:
            return 0.0

        # Build mean anchor target
        anchor_flat = torch.stack(
            [_flatten_delta(d) for d in anchor_deltas]
        ).mean(dim=0).to(self.device)

        # Build input batch from quantized deltas
        q_flat_batch = torch.stack(
            [_flatten_delta(d) for d in quantized_deltas]
        ).to(self.device)

        # Optionally precompute teacher logits on val set (distillation)
        teacher_logits_batch = None
        if global_model is not None and val_loader is not None:
            teacher_logits_batch = self._compute_teacher_logits(global_model, val_loader)

        total_loss = 0.0
        self.net.train()
        for _ in range(self.epochs):
            self.optimizer.zero_grad()
            corrected_batch = self.net(q_flat_batch)

            # Expand anchor target to match batch size
            target_batch = anchor_flat.unsqueeze(0).expand_as(corrected_batch)

            # Student logits: apply mean corrected delta to model and run val set
            student_logits_batch = None
            if global_model is not None and val_loader is not None and teacher_logits_batch is not None:
                mean_corrected = corrected_batch.mean(dim=0)
                student_model = self._apply_delta_to_model(
                    global_model,
                    _unflatten_delta(mean_corrected.detach(), next(iter(anchor_deltas)))
                )
                student_logits_batch = self._compute_teacher_logits(student_model, val_loader)

            loss = self.criterion(
                corrected_batch,
                target_batch,
                student_logits_batch,
                teacher_logits_batch,
            )
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / self.epochs

    # ------------------------------------------------------------------
    # Correction application
    # ------------------------------------------------------------------

    def apply(
        self,
        quantized_delta: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Pass a single quantized delta through CorrectionNet and reshape back.
        """
        self.net.eval()
        with torch.no_grad():
            flat = _flatten_delta(quantized_delta).to(self.device)
            corrected_flat = self.net(flat).cpu()
        return _unflatten_delta(corrected_flat, quantized_delta)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_teacher_logits(
        self, model: nn.Module, val_loader: DataLoader
    ) -> torch.Tensor:
        """Run model on validation set and collect logits."""
        model.eval()
        model.to(self.device)
        all_logits = []
        for images, _ in val_loader:
            all_logits.append(model(images.to(self.device)))
        return torch.cat(all_logits, dim=0)

    def _apply_delta_to_model(
        self, global_model: nn.Module, delta: Dict[str, torch.Tensor]
    ) -> nn.Module:
        """Return a copy of global_model with delta applied."""
        student = copy.deepcopy(global_model)
        state = student.state_dict()
        for k in delta:
            if k in state:
                state[k] = state[k].float() + delta[k].float().to(state[k].device)
        student.load_state_dict(state)
        return student
