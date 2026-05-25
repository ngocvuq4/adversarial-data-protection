"""Dataset loaders for CIFAR-10 benchmarking and LFW cloaking experiments.

All image tensors are float32 in [0, 1]. Normalization is intentionally not
performed here so protection methods can clamp in pixel space.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, random_split
import torchvision.transforms as T
from torchvision.datasets import CIFAR10, ImageFolder, LFWPairs
from torchvision.datasets.utils import download_url, extract_archive


def get_cifar10(root="./data", subset_size=5000, batch_size=128):
    """Return CIFAR-10 train/test loaders with images shaped (3, 32, 32)."""
    transform = T.Compose([T.ToTensor()])
    train_set = CIFAR10(root=root, train=True, download=True, transform=transform)
    test_set = CIFAR10(root=root, train=False, download=True, transform=transform)

    if subset_size:
        train_set = Subset(train_set, range(min(subset_size, len(train_set))))
        test_size = min(max(subset_size // 5, 1), len(test_set))
        test_set = Subset(test_set, range(test_size))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    return train_loader, test_loader


def get_lfw_pairs(root="./data/lfw", img_size=224, max_pairs=200):
    """Return LFW pairs resized to img_size for cloaking experiments."""
    transform = T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ]
    )
    dataset = LFWPairs(root=root, split="test", download=True, transform=transform)
    if max_pairs:
        dataset = Subset(dataset, range(min(max_pairs, len(dataset))))
    return dataset


def _prepare_caltech101(root):
    """Download/extract Caltech-101 into an ImageFolder-compatible directory."""
    root = Path(root)
    dataset_dir = root / "caltech101" / "caltech-101" / "101_ObjectCategories"
    if dataset_dir.exists():
        return dataset_dir

    archive_dir = root / "caltech101"
    zip_path = archive_dir / "caltech-101.zip"
    archive_dir.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists():
        download_url(
            "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip?download=1",
            str(archive_dir),
            filename="caltech-101.zip",
            md5="3138e1922a9193bfa496528edbbc45d0",
        )
    extract_archive(str(zip_path), str(archive_dir))

    inner_tar = archive_dir / "caltech-101" / "101_ObjectCategories.tar.gz"
    if inner_tar.exists() and not dataset_dir.exists():
        extract_archive(str(inner_tar), str(inner_tar.parent))
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Caltech-101 image directory not found: {dataset_dir}")
    return dataset_dir


def get_caltech101(
    root="./data",
    img_size=224,
    subset_size=2000,
    batch_size=32,
    train_ratio=0.8,
    seed=42,
    download=True,
):
    """Return Caltech-101 train/test loaders resized to img_size.

    Returns:
        train_loader, test_loader, num_classes
    """
    dataset_dir = _prepare_caltech101(root) if download else (
        Path(root) / "caltech101" / "caltech-101" / "101_ObjectCategories"
    )
    transform = T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ]
    )
    dataset = ImageFolder(str(dataset_dir), transform=transform)
    if subset_size:
        size = min(subset_size, len(dataset))
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(dataset), generator=generator)[:size].tolist()
        dataset = Subset(dataset, indices)

    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(seed)
    train_set, test_set = random_split(dataset, [train_size, test_size], generator)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    num_classes = len(dataset.dataset.classes) if isinstance(dataset, Subset) else len(dataset.classes)
    return train_loader, test_loader, num_classes
