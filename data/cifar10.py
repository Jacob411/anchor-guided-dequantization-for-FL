"""
CIFAR-10 data loading with non-IID Dirichlet partitioning.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

def get_transforms(train: bool = True) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


# ---------------------------------------------------------------------------
# Dirichlet partitioning
# ---------------------------------------------------------------------------

def dirichlet_partition(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int = 42,
) -> List[List[int]]:
    """
    Partition dataset indices into `num_clients` non-IID shards using a
    Dirichlet(alpha) distribution over class labels.

    Lower alpha → more skewed (more non-IID).
    """
    rng = np.random.default_rng(seed)
    num_classes = int(labels.max()) + 1

    # Collect per-class indices
    class_indices: dict[int, np.ndarray] = {
        c: np.where(labels == c)[0] for c in range(num_classes)
    }

    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        proportions = rng.dirichlet(alpha=np.repeat(alpha, num_clients))
        proportions = (proportions * len(class_indices[c])).astype(int)
        # Fix rounding: assign remainder to first client
        diff = len(class_indices[c]) - proportions.sum()
        proportions[0] += diff

        idx = class_indices[c]
        rng.shuffle(idx)
        splits = np.split(idx, np.cumsum(proportions)[:-1])
        for i, split in enumerate(splits):
            client_indices[i].extend(split.tolist())

    return client_indices


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_cifar10(
    data_dir: str,
    num_clients: int,
    alpha: float,
    batch_size: int,
    validation_size: int = 500,
    seed: int = 42,
) -> Tuple[List[DataLoader], DataLoader, DataLoader]:
    """
    Returns:
        client_loaders : list of train DataLoaders, one per client
        val_loader     : small server-side validation set (for distillation)
        test_loader    : global test set for evaluation
    """
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True,  download=True, transform=get_transforms(train=True)
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=get_transforms(train=False)
    )

    labels = np.array(train_dataset.targets)

    # Carve out a server validation set *before* client partitioning
    rng = np.random.default_rng(seed)
    val_indices = rng.choice(len(train_dataset), size=validation_size, replace=False)
    train_indices = np.setdiff1d(np.arange(len(train_dataset)), val_indices)

    val_loader = DataLoader(
        Subset(train_dataset, val_indices.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Partition remaining training data among clients
    client_label_arr = labels[train_indices]
    # Re-index so dirichlet_partition sees contiguous indices
    partitions = dirichlet_partition(client_label_arr, num_clients, alpha, seed)
    client_loaders: List[DataLoader] = []
    for part in partitions:
        real_indices = train_indices[part].tolist()
        loader = DataLoader(
            Subset(train_dataset, real_indices),
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        client_loaders.append(loader)

    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return client_loaders, val_loader, test_loader
