"""Evaluation metrics for protected images and victim-model accuracy.

Functions accept float32 tensors in [0, 1] shaped (B, 3, H, W) or (3, H, W).
They do not save files or plot figures.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _round4(value):
    return round(float(value), 4)


def _ensure_batch(x):
    return x.unsqueeze(0) if x.dim() == 3 else x


def compute_psnr(x_orig, x_protected):
    """Peak Signal-to-Noise Ratio in dB. Higher means more similar."""
    x_orig = _ensure_batch(x_orig).float().clamp(0.0, 1.0)
    x_protected = _ensure_batch(x_protected).float().clamp(0.0, 1.0)
    mse = F.mse_loss(x_protected, x_orig).item()
    if mse == 0:
        return float("inf")
    return _round4(10 * np.log10(1.0 / mse))


def compute_ssim(x_orig, x_protected, window_size=11):
    """Structural Similarity Index, averaged over the batch."""
    x_orig = _ensure_batch(x_orig).float().clamp(0.0, 1.0)
    x_protected = _ensure_batch(x_protected).float().clamp(0.0, 1.0)
    try:
        from pytorch_msssim import ssim

        return _round4(ssim(x_orig, x_protected, data_range=1.0).item())
    except ImportError:
        if min(x_orig.shape[-2:]) < window_size:
            window_size = max(3, min(x_orig.shape[-2:]) | 1)
        c1, c2 = 0.01**2, 0.03**2
        pad = window_size // 2
        mu1 = F.avg_pool2d(x_orig, window_size, 1, pad)
        mu2 = F.avg_pool2d(x_protected, window_size, 1, pad)
        mu1_sq, mu2_sq = mu1**2, mu2**2
        sigma1_sq = F.avg_pool2d(x_orig**2, window_size, 1, pad) - mu1_sq
        sigma2_sq = F.avg_pool2d(x_protected**2, window_size, 1, pad) - mu2_sq
        sigma12 = F.avg_pool2d(x_orig * x_protected, window_size, 1, pad) - mu1 * mu2
        ssim_map = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / (
            (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        )
        return _round4(ssim_map.mean().item())


def compute_linf(x_orig, x_protected):
    """Return L-infinity norm of the perturbation in [0, 1] pixel space."""
    return _round4((_ensure_batch(x_protected) - _ensure_batch(x_orig)).abs().max().item())


def compute_lpips(x_orig, x_protected, device="cuda"):
    """Optional LPIPS metric. Requires the lpips package and a GPU/CPU device."""
    import lpips

    metric = lpips.LPIPS(net="alex").to(device)
    metric.eval()
    x_orig = _ensure_batch(x_orig).to(device).float().clamp(0.0, 1.0) * 2 - 1
    x_protected = _ensure_batch(x_protected).to(device).float().clamp(0.0, 1.0) * 2 - 1
    with torch.no_grad():
        score = metric(x_orig, x_protected).mean().item()
    del metric
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return _round4(score)


def compute_attack_success_rate(victim_model, clean_test_loader, device):
    """Evaluate trained victim on clean test data and return accuracy plus ASR."""
    victim_model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in clean_test_loader:
            x, y = x.to(device), y.to(device)
            preds = victim_model(x).argmax(1)
            correct += (preds == y).sum().item()
            total += y.numel()
    accuracy = _round4(correct / max(total, 1))
    return accuracy, _round4(1.0 - accuracy)


def full_evaluation(x_orig, x_protected, victim_model, clean_test_loader, device):
    """Return PSNR, SSIM, L-inf, clean accuracy, and ASR."""
    accuracy, asr = compute_attack_success_rate(victim_model, clean_test_loader, device)
    return {
        "psnr": compute_psnr(x_orig.cpu(), x_protected.cpu()),
        "ssim": compute_ssim(x_orig.cpu(), x_protected.cpu()),
        "linf": compute_linf(x_orig.cpu(), x_protected.cpu()),
        "clean_test_accuracy": accuracy,
        "asr": asr,
    }
