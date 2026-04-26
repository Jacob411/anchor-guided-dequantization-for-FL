"""
Metrics tracker — records per-round FL statistics.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class RoundMetrics:
    round_num: int
    top1_accuracy: float          # global Top-1 accuracy on test set (%)
    loss: float                   # average test loss
    uplink_bytes: int             # total uplink bytes this round
    correction_loss: float = 0.0  # CorrectionNet training loss
    delta_accuracy: float = 0.0   # round-over-round accuracy change
    client_mean_acc: float = 0.0  # mean per-client local test accuracy (%)
    client_min_acc: float = 0.0   # worst-client local test accuracy (%)
    client_std_acc: float = 0.0   # std dev of per-client local test accuracy


@dataclass
class MetricsTracker:
    """
    Collects and persists per-round FL metrics.
    """
    output_dir: str
    strategy_name: str
    history: List[RoundMetrics] = field(default_factory=list)

    def __post_init__(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        self._csv_path = os.path.join(
            self.output_dir, f"results_{self.strategy_name}.csv"
        )
        # Write CSV header
        with open(self._csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(RoundMetrics(0, 0.0, 0.0, 0)).keys()))
            writer.writeheader()

    def evaluate(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: torch.device,
    ) -> tuple[float, float]:
        """Evaluate model on test_loader. Returns (top1_acc %, avg_loss)."""
        model.eval()
        model.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        correct = total = 0
        total_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                total_loss += criterion(outputs, labels).item() * images.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += images.size(0)
        acc = 100.0 * correct / total
        avg_loss = total_loss / total
        return acc, avg_loss

    def evaluate_clients(
        self,
        model: nn.Module,
        client_test_loaders: List[DataLoader],
        device: torch.device,
    ) -> tuple[float, float, float]:
        """Evaluate global model on each client's local test set.

        Returns (mean_acc, min_acc, std_acc) in %.
        """
        accs = [self.evaluate(model, loader, device)[0] for loader in client_test_loaders]
        accs_arr = torch.tensor(accs)
        return float(accs_arr.mean()), float(accs_arr.min()), float(accs_arr.std())

    def record(self, metrics: RoundMetrics) -> None:
        """Append a round's metrics and write to CSV."""
        if self.history:
            metrics.delta_accuracy = metrics.top1_accuracy - self.history[-1].top1_accuracy
        self.history.append(metrics)
        with open(self._csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(metrics).keys()))
            writer.writerow(asdict(metrics))

    def print_round(self, metrics: RoundMetrics) -> None:
        print(
            f"[Round {metrics.round_num:3d}] "
            f"Acc: {metrics.top1_accuracy:.2f}% "
            f"(Δ{metrics.delta_accuracy:+.2f}) | "
            f"Loss: {metrics.loss:.4f} | "
            f"Client acc — mean: {metrics.client_mean_acc:.2f}% "
            f"min: {metrics.client_min_acc:.2f}% "
            f"std: {metrics.client_std_acc:.2f}% | "
            f"Uplink: {metrics.uplink_bytes / 1e6:.2f} MB | "
            f"CorrLoss: {metrics.correction_loss:.4f}"
        )

    def total_uplink_mb(self) -> float:
        return sum(m.uplink_bytes for m in self.history) / 1e6
