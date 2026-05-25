"""Model builders and training helpers for surrogate and victim networks.

Victim CIFAR models are trained from scratch. The ResNet-18 CIFAR variant uses
a 3x3 first convolution and removes the first max-pool to preserve 32x32 detail.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as M


NORMALIZE = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


class NormalizeLayer(nn.Module):
    def __init__(self, mean=NORMALIZE["mean"], std=NORMALIZE["std"]):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean.to(dtype=x.dtype)) / self.std.to(dtype=x.dtype)


def _freeze_eval(model):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def get_surrogate_resnet50(device):
    """Return ImageNet ResNet-50 feature extractor with 2048-dim output."""
    backbone = M.resnet50(weights=M.ResNet50_Weights.IMAGENET1K_V1)
    backbone.fc = nn.Identity()
    model = nn.Sequential(NormalizeLayer(), backbone)
    return _freeze_eval(model.to(device))


def get_victim_resnet18(num_classes=10, device="cpu"):
    """Return ResNet-18 adapted for CIFAR-10."""
    model = M.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, num_classes)
    return model.to(device)


def get_victim_vgg16(num_classes=10, device="cpu"):
    """Return VGG-16 victim model for transferability tests."""
    model = M.vgg16(weights=None)
    model.classifier[6] = nn.Linear(4096, num_classes)
    return model.to(device)


def get_victim_mobilenet(num_classes=10, device="cpu"):
    """Return MobileNetV2 victim model for transferability tests."""
    model = M.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(1280, num_classes)
    return model.to(device)


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Train one epoch and return rounded average loss and accuracy."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return round(total_loss / max(total, 1), 4), round(correct / max(total, 1), 4)


def evaluate(model, loader, device):
    """Evaluate classification accuracy on a loader."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += x.size(0)
    return round(correct / max(total, 1), 4)
