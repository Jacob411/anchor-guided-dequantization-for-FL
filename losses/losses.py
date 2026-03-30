"""
Reconstruction and knowledge-distillation losses used to train CorrectionNet.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconstructionLoss(nn.Module):
    """
    MSE between CorrectionNet output and true full-precision delta.

    L_rec = MSE( correction_net(Δw_q), Δw_fp )
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, corrected: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(corrected, target)


class DistillationLoss(nn.Module):
    """
    KL-divergence loss between logits of the corrected model and the
    anchor (full-precision) model on the server validation set.

    L_distill = KL( softmax(logits_fp / T) || log_softmax(logits_corrected / T) )

    Args:
        temperature: softmax temperature (T). Higher T → softer targets.
    """

    def __init__(self, temperature: float = 4.0) -> None:
        super().__init__()
        self.T = temperature

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        teacher_probs = F.softmax(teacher_logits / self.T, dim=-1)
        student_log_probs = F.log_softmax(student_logits / self.T, dim=-1)
        # KLDiv expects input=log-probabilities, target=probabilities
        loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
        return loss * (self.T ** 2)   # scale by T² to keep gradient magnitudes stable


class CombinedCorrectionLoss(nn.Module):
    """
    Weighted sum of reconstruction and distillation losses.

    L = w_rec * L_rec + w_distill * L_distill
    """

    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        distillation_weight: float = 0.5,
        temperature: float = 4.0,
    ) -> None:
        super().__init__()
        self.w_rec = reconstruction_weight
        self.w_distill = distillation_weight
        self.rec_loss = ReconstructionLoss()
        self.distill_loss = DistillationLoss(temperature)

    def forward(
        self,
        corrected_delta: torch.Tensor,
        target_delta: torch.Tensor,
        student_logits: torch.Tensor | None = None,
        teacher_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss = self.w_rec * self.rec_loss(corrected_delta, target_delta)
        if student_logits is not None and teacher_logits is not None:
            loss = loss + self.w_distill * self.distill_loss(student_logits, teacher_logits)
        return loss
