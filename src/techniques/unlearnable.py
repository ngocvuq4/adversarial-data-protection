"""Unlearnable Examples via class-wise error-minimizing PGD.

Objective: min_delta min_theta L(f_theta(x + delta), y), approximated by
alternating updates: freeze theta to update class-wise noise, then train theta
on noisy batches. Parameters: epsilon, pgd_steps, pgd_alpha, inner_epochs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tqdm.auto import tqdm


def _freeze(model):
    model.eval()
    previous = [p.requires_grad for p in model.parameters()]
    for param in model.parameters():
        param.requires_grad = False
    return previous


def _restore(model, previous):
    for param, requires_grad in zip(model.parameters(), previous):
        param.requires_grad = requires_grad


def generate_unlearnable_noise(
    model_fn,
    train_loader,
    epsilon=0.03,
    pgd_steps=20,
    pgd_alpha=None,
    inner_epochs=5,
    device="cpu",
    show_progress=True,
):
    """Generate class-wise error-minimizing noise for CIFAR-10."""
    if pgd_alpha is None:
        pgd_alpha = epsilon / 10

    sample_x, _ = next(iter(train_loader))
    img_shape = sample_x.shape[1:]
    num_classes = 10
    noise_dict = {c: torch.zeros(img_shape, device=device) for c in range(num_classes)}

    model = model_fn().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4
    )
    criterion = nn.CrossEntropyLoss()

    pgd_iterator = range(pgd_steps)
    if show_progress:
        pgd_iterator = tqdm(pgd_iterator, desc="PGD noise steps", leave=False)

    for pgd_step in pgd_iterator:
        previous = _freeze(model)
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            for c in range(num_classes):
                mask = y == c
                if mask.sum() == 0:
                    continue
                xc = x[mask]
                yc = y[mask]
                noise = (
                    noise_dict[c]
                    .detach()
                    .unsqueeze(0)
                    .expand_as(xc)
                    .clone()
                    .requires_grad_(True)
                )
                loss = criterion(model(torch.clamp(xc + noise, 0.0, 1.0)), yc)
                grad = torch.autograd.grad(loss, noise, only_inputs=True)[0]
                with torch.no_grad():
                    noise_dict[c] -= pgd_alpha * grad.mean(dim=0).sign()
                    noise_dict[c].clamp_(-epsilon, epsilon)
            del x, y
        _restore(model, previous)

        model.train()
        inner_iterator = range(inner_epochs)
        if show_progress:
            inner_iterator = tqdm(
                inner_iterator,
                desc=f"Inner train {pgd_step + 1}/{pgd_steps}",
                leave=False,
            )
        for _inner in inner_iterator:
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                x_noisy = apply_noise(x, y, noise_dict)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x_noisy), y)
                loss.backward()
                optimizer.step()
                del x, y, x_noisy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {c: noise.detach().cpu() for c, noise in noise_dict.items()}


def apply_noise(x, y, noise_dict):
    """Apply class-wise noise to a batch and clamp to [0, 1]."""
    x_protected = x.clone()
    for c, noise in noise_dict.items():
        mask = y == int(c)
        if mask.sum() == 0:
            continue
        x_protected[mask] = torch.clamp(x[mask] + noise.to(x.device), 0.0, 1.0)
    return x_protected.clamp(0.0, 1.0)


def generate_unlearnable_batch(model, x, y, epsilon=0.03, pgd_steps=20, pgd_alpha=None):
    """Fast sample-wise mini-PGD used by the UI when no precomputed noise exists."""
    if pgd_alpha is None:
        pgd_alpha = epsilon / max(pgd_steps, 1)
    previous = _freeze(model)
    x_orig = x.detach()
    x_protected = x_orig.clone()
    criterion = nn.CrossEntropyLoss()
    for _ in range(pgd_steps):
        x_protected = x_protected.detach().requires_grad_(True)
        loss = criterion(model(x_protected), y)
        grad = torch.autograd.grad(loss, x_protected, only_inputs=True)[0]
        x_protected = x_protected - pgd_alpha * grad.sign()
        delta = torch.clamp(x_protected - x_orig, -epsilon, epsilon)
        x_protected = torch.clamp(x_orig + delta, 0.0, 1.0)
    _restore(model, previous)
    return x_protected.detach()
