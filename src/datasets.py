"""Dataset loaders for CIFAR-10 benchmarking and LFW cloaking experiments.

All image tensors are float32 in [0, 1]. Normalization is intentionally not
performed here so protection methods can clamp in pixel space.
"""

from __future__ import annotations

from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
from torchvision.datasets import CIFAR10, LFWPairs


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
