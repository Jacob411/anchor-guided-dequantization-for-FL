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


# Keys to exclude from CorrectionNet: BN running stats are not learnable
# parameters and must not be perturbed by the correction network.
_BN_BUFFER_SUFFIXES = ("running_mean", "running_var", "num_batches_tracked")


def _is_learnable_key(key: str) -> bool:
    """Return True if this state-dict key should go through CorrectionNet."""
    return not any(key.endswith(s) for s in _BN_BUFFER_SUFFIXES)


def _flatten_delta_learnable(delta: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten only the *learnable* entries (excludes BN buffers)."""
    return torch.cat(
        [t.float().flatten() for k, t in delta.items() if _is_learnable_key(k)]
    )


def _unflatten_delta_learnable(
    flat: torch.Tensor, reference: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """Restore a flat learnable vector; BN buffer keys get their original values."""
    result = {}
    offset = 0
    for k, v in reference.items():
        if _is_learnable_key(k):
            numel = v.numel()
            result[k] = flat[offset : offset + numel].reshape(v.shape)
            offset += numel
        else:
            result[k] = v  # pass BN buffers through unchanged
    return result

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
        grad_clip: float = 1.0,
        buffer_size: int = 10,
        smart_match: bool = False,
        blend_factor: float = 1.0,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.epochs = epochs
        self.grad_clip = grad_clip
        self.smart_match = smart_match

        self.blend_factor = blend_factor  # 1.0 = full correction; <1 preserves original direction
        self.net = CorrectionNet(delta_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.criterion = CombinedCorrectionLoss(rec_weight, distill_weight, temperature)

        # Normalization stats stored from last training call (used in apply())
        self._norm_mean: torch.Tensor | None = None
        self._norm_std:  torch.Tensor | None = None

        # Replay buffer: list of (q_flat_batch, target_flat_batch) tensors from past rounds.
        # Capped at buffer_size rounds so memory stays bounded.
        self._buffer: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self._buffer_size = buffer_size

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

        # Build anchor targets and input batch — learnable parameters only
        anchor_flats = torch.stack(
            [_flatten_delta_learnable(d) for d in anchor_deltas]
        ).to(self.device)
        
        q_flat_batch = torch.stack(
            [_flatten_delta_learnable(d) for d in quantized_deltas]
        ).to(self.device)

        if self.smart_match:
            # Map each quantized delta to the most similar anchor
            norm_q = torch.nn.functional.normalize(q_flat_batch, p=2, dim=1)
            norm_a = torch.nn.functional.normalize(anchor_flats, p=2, dim=1)
            sim = torch.mm(norm_q, norm_a.t())
            best_anchor_idxs = torch.argmax(sim, dim=1)
            target_flat_batch = anchor_flats[best_anchor_idxs]
        else:
            # Map all to the mean anchor
            mean_anchor = anchor_flats.mean(dim=0)
            target_flat_batch = mean_anchor.unsqueeze(0).expand(q_flat_batch.shape[0], -1)

        # ---- Replay buffer ---------------------------------------------------
        # Accumulate (q_flat, target_flat) pairs across FL rounds so the net
        # sees increasing amounts of data and CorrLoss trends downward.
        self._buffer.append((q_flat_batch.cpu(), target_flat_batch.cpu()))
        if len(self._buffer) > self._buffer_size:
            self._buffer.pop(0)  # evict oldest round

        # Concatenate all buffered rounds into one training batch
        all_q      = torch.cat([b[0] for b in self._buffer], dim=0).to(self.device)
        all_anchor = torch.cat([b[1] for b in self._buffer], dim=0).to(self.device)
        # ----------------------------------------------------------------------

        # ---- Input normalisation (global scalar z-score) --------------------
        s_mean = all_q.mean()                    # scalar over entire buffer
        s_std  = all_q.std().clamp(min=1e-8)
        q_flat_norm = (all_q      - s_mean) / s_std
        anchor_norm = (all_anchor - s_mean) / s_std

        # Cache stats (scalars) so apply() uses the same scale
        self._norm_mean = s_mean.detach()
        self._norm_std  = s_std.detach()
        # ---------------------------------------------------------------------

        # Optionally precompute teacher logits on val set (distillation)
        teacher_logits_batch = None
        if global_model is not None and val_loader is not None:
            teacher_logits_batch = self._compute_teacher_logits(global_model, val_loader)

        total_loss = 0.0
        self.net.train()
        for _ in range(self.epochs):
            self.optimizer.zero_grad()
            corrected_norm_batch = self.net(q_flat_norm)

            # Targets already broadcast-matched via all_anchor expansion above
            target_norm_batch = anchor_norm

            # Student logits: apply mean corrected delta to model and run val set
            student_logits_batch = None
            if global_model is not None and val_loader is not None and teacher_logits_batch is not None:
                # Denormalise before applying to model weights (use current-round stats)
                mean_corrected_norm = corrected_norm_batch[-len(q_flat_batch):].mean(dim=0)
                mean_corrected = mean_corrected_norm.detach() * s_std + s_mean
                student_model = self._apply_delta_to_model(
                    global_model,
                    _unflatten_delta_learnable(mean_corrected, next(iter(anchor_deltas)))
                )
                student_logits_batch = self._compute_teacher_logits(student_model, val_loader)

            loss = self.criterion(
                corrected_norm_batch,
                target_norm_batch,
                student_logits_batch,
                teacher_logits_batch,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
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
        Uses the same input normalisation stats computed during train().
        BN running stats (running_mean/var, num_batches_tracked) bypass CorrectionNet.
        """
        self.net.eval()
        with torch.no_grad():
            flat = _flatten_delta_learnable(quantized_delta).to(self.device)
            # Apply same normalisation (scalar) used during training
            if self._norm_mean is not None and self._norm_std is not None:
                flat_norm = (flat - self._norm_mean) / self._norm_std
                corrected_norm = self.net(flat_norm)
                corrected_flat = corrected_norm * self._norm_std + self._norm_mean
            else:
                corrected_flat = self.net(flat)
            # Blend corrected output with original to preserve client-specific direction.
            # blend_factor=1.0 → full correction; blend_factor<1 preserves direction.
            if self.blend_factor < 1.0:
                corrected_flat = self.blend_factor * corrected_flat + (1.0 - self.blend_factor) * flat
            corrected_flat = corrected_flat.cpu()
        return _unflatten_delta_learnable(corrected_flat, quantized_delta)

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
