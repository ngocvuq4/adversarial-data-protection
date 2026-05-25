"""Feature-space cloaking via PGD.

Objective: min_delta ||f(x + delta) - f(t)||_2^2 subject to ||delta||_inf <=
epsilon. Target features are selected from the batch by random permutation or
maximum feature distance. For single-image batches, the target falls back to
the opposite normalized feature direction so the UI path still produces a
feature-disrupting perturbation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _freeze(model):
    model.eval()
    previous = [p.requires_grad for p in model.parameters()]
    for param in model.parameters():
        param.requires_grad = False
    return previous


def _restore(model, previous):
    for param, requires_grad in zip(model.parameters(), previous):
        param.requires_grad = requires_grad


def select_cloak_target(surrogate, x_batch, mode="max_dist", device=None):
    """Select target embeddings for each image in a batch."""
    device = x_batch.device if device is None else device
    surrogate.eval()
    with torch.no_grad():
        feats = F.normalize(surrogate(x_batch.to(device)).float(), dim=1)

    if mode in {"away", "self_negate", "disrupt"}:
        return (-feats).detach()

    if feats.size(0) < 2:
        return (-feats).detach()

    if mode == "random":
        idx = torch.randperm(feats.size(0), device=feats.device)
        return feats[idx].detach()
    if mode == "max_dist":
        dists = torch.cdist(feats, feats, p=2)
        dists.fill_diagonal_(-1.0)
        return feats[dists.argmax(dim=1)].detach()
    raise ValueError(
        f"target_mode must be 'random', 'max_dist', or 'away', got: {mode}"
    )


def cloak_images(
    surrogate,
    x,
    epsilon=0.05,
    pgd_steps=100,
    pgd_alpha=None,
    target_mode="max_dist",
    target_features=None,
    device=None,
):
    """Return cloaked images shaped like x and clamped to [0, 1]."""
    if pgd_alpha is None:
        pgd_alpha = epsilon / 10

    previous = _freeze(surrogate)
    device = x.device if device is None else device
    x_orig = x.detach().to(device)
    if target_features is None:
        target_features = select_cloak_target(
            surrogate, x_orig, mode=target_mode, device=device
        )
    target_features = F.normalize(target_features.detach().to(device).float(), dim=1)

    delta = torch.zeros_like(x_orig, requires_grad=True)
    for _ in range(pgd_steps):
        x_adv = torch.clamp(x_orig + delta, 0.0, 1.0)
        feats = F.normalize(surrogate(x_adv).float(), dim=1)
        loss = F.mse_loss(feats, target_features)
        grad = torch.autograd.grad(loss, delta, only_inputs=True)[0]
        with torch.no_grad():
            delta -= pgd_alpha * grad.sign()
            delta.clamp_(-epsilon, epsilon)
            delta.copy_(torch.clamp(x_orig + delta, 0.0, 1.0) - x_orig)
        delta = delta.detach().requires_grad_(True)

    _restore(surrogate, previous)
    return torch.clamp(x_orig + delta.detach(), 0.0, 1.0)
