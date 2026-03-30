"""
Simple 3-block CNN for CIFAR-10 classification.
Used as the shared global model across all FL clients.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CIFAR10CNN(nn.Module):
    """
    3-block Conv+BN+ReLU+Pool → FC(256) → FC(10).
    Input:  (B, 3, 32, 32)
    Output: (B, 10)  — raw logits
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3,  64),   # → (B, 64, 16, 16)
            ConvBlock(64, 128),  # → (B, 128, 8, 8)
            ConvBlock(128, 256), # → (B, 256, 4, 4)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(arch: str = "cnn", num_classes: int = 10) -> nn.Module:
    """Factory function — extend here for ResNet-18 etc."""
    if arch == "cnn":
        return CIFAR10CNN(num_classes=num_classes)
    raise ValueError(f"Unknown architecture: {arch}. Supported: ['cnn']")
