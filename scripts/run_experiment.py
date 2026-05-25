"""Run experiment notebook pipeline and write results/ artifacts.

Usage:
  python scripts/run_experiment.py           # quick local smoke run
  python scripts/run_experiment.py --full    # full ablation grid
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.datasets import get_cifar10
from src.evaluation import compute_attack_success_rate, compute_linf, compute_psnr, compute_ssim
from src.models import (
    evaluate,
    get_surrogate_resnet50,
    get_victim_mobilenet,
    get_victim_resnet18,
    get_victim_vgg16,
    train_one_epoch,
)
from src.techniques.cloaking import cloak_images
from src.techniques.nightshade import load_clip_model, poison_images
from src.techniques.unlearnable import apply_noise, generate_unlearnable_noise
from src.visualization import plot_before_after, plot_epsilon_vs_metrics, plot_technique_comparison

EPSILON_VALUES = [0.01, 0.03, 0.05, 0.08, 0.1]
VICTIM_MODELS = {
    "ResNet-18": get_victim_resnet18,
    "VGG-16": get_victim_vgg16,
    "MobileNet": get_victim_mobilenet,
}
TECHNIQUES = ["unlearnable", "general_cloaking", "concept_poisoning"]


def setup_dirs():
    for path in (
        "results/figures",
        "results/tables",
        "results/protected_samples",
    ):
        os.makedirs(path, exist_ok=True)


def train_classifier(model_fn, loader, epochs, device, lr=0.01):
    model = model_fn().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        loss, acc = train_one_epoch(model, loader, optimizer, criterion, device)
        print(f"  epoch {epoch + 1}/{epochs} loss={loss} acc={acc}")
    return model


def materialize_protected_loader(clean_loader, protected_batches, labels, batch_size, device):
    xs = torch.cat(protected_batches, dim=0).clamp(0.0, 1.0)
    ys = torch.cat(labels, dim=0)
    return DataLoader(TensorDataset(xs, ys), batch_size=batch_size, shuffle=True)


def collect_clean_tensors(loader):
    xs, ys = [], []
    for x, y in loader:
        xs.append(x)
        ys.append(y)
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def protect_cifar_batch_unlearnable(x, y, noise_dict, device):
    return apply_noise(x.to(device), y.to(device), noise_dict).cpu()


def protect_cifar_batch_upscale(technique_fn, x, device, batch_size=32):
    """Upscale 32x32 CIFAR to 224, protect, downscale back."""
    outputs = []
    for start in range(0, x.size(0), batch_size):
        batch = x[start : start + batch_size].to(device)
        up = F.interpolate(batch, size=(224, 224), mode="bilinear", align_corners=False)
        protected = technique_fn(up).detach()
        down = F.interpolate(protected, size=(32, 32), mode="bilinear", align_corners=False)
        outputs.append(down.cpu())
        del batch, up, protected, down
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return torch.cat(outputs, dim=0).clamp(0.0, 1.0)


def build_protected_tensors(
    technique,
    clean_x,
    clean_y,
    device,
    epsilon,
    noise_dict=None,
    surrogate=None,
    clip_model=None,
    pgd_steps=5,
):
    if technique == "unlearnable":
        if noise_dict is None:
            raise ValueError("noise_dict required for unlearnable")
        protected = []
        for start in range(0, clean_x.size(0), 128):
            xb = clean_x[start : start + 128]
            yb = clean_y[start : start + 128]
            protected.append(protect_cifar_batch_unlearnable(xb, yb, noise_dict, device))
        return torch.cat(protected, dim=0)

    if technique in {"general_cloaking", "cloaking"}:
        fn = lambda batch: cloak_images(
            surrogate,
            batch,
            epsilon=epsilon,
            pgd_steps=pgd_steps,
            device=device,
        )
        return protect_cifar_batch_upscale(fn, clean_x, device)

    if technique in {"concept_poisoning", "nightshade"}:
        fn = lambda batch: poison_images(
            clip_model,
            batch,
            target_concept="a photo of a cat",
            epsilon=epsilon,
            pgd_steps=pgd_steps,
            device=device,
        )
        return protect_cifar_batch_upscale(fn, clean_x, device)

    raise ValueError(f"Unknown technique: {technique}")


def run_main_experiment(device, subset_size, batch_size, full):
    print("=== Main experiment: unlearnable noise + baseline ===")
    train_loader, test_loader = get_cifar10(subset_size=subset_size, batch_size=batch_size)
    clean_x, clean_y = collect_clean_tensors(train_loader)

    baseline_epochs = 3 if full else 2
    victim_epochs = 3 if full else 2
    pgd_steps = 20 if full else 5
    inner_epochs = 5 if full else 1

    baseline_model = train_classifier(
        lambda: get_victim_resnet18(device=device),
        train_loader,
        baseline_epochs,
        device,
    )
    baseline_acc = evaluate(baseline_model, test_loader, device)
    print({"baseline_clean_test_accuracy": baseline_acc})

    noise_dict = generate_unlearnable_noise(
        model_fn=lambda: get_victim_resnet18(device=device),
        train_loader=train_loader,
        epsilon=0.03,
        pgd_steps=pgd_steps,
        inner_epochs=inner_epochs,
        device=device,
    )
    torch.save(noise_dict, "results/unlearnable_noise_dict.pt")
    print("Saved results/unlearnable_noise_dict.pt")

    protected_x = build_protected_tensors(
        "unlearnable", clean_x, clean_y, device, epsilon=0.03, noise_dict=noise_dict
    )
    protected_loader = DataLoader(
        TensorDataset(protected_x, clean_y),
        batch_size=batch_size,
        shuffle=True,
    )

    victim_model = train_classifier(
        lambda: get_victim_resnet18(device=device),
        protected_loader,
        victim_epochs,
        device,
    )
    clean_acc, asr = compute_attack_success_rate(victim_model, test_loader, device)
    metrics = {
        "technique": "unlearnable",
        "victim_model": "ResNet-18",
        "epsilon": 0.03,
        "psnr": compute_psnr(clean_x, protected_x),
        "ssim": compute_ssim(clean_x, protected_x),
        "linf": compute_linf(clean_x, protected_x),
        "clean_test_accuracy": clean_acc,
        "asr": asr,
    }
    print(metrics)

    to_pil = torchvision.transforms.ToPILImage()
    sample_idx = 0
    plot_before_after(clean_x[sample_idx], protected_x[sample_idx], "unlearnable")
    to_pil(clean_x[sample_idx]).save("results/protected_samples/original_unlearnable.png")
    to_pil(protected_x[sample_idx]).save("results/protected_samples/protected_unlearnable.png")

    del baseline_model, victim_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return train_loader, test_loader, clean_x, clean_y, noise_dict


def run_ablation(device, train_loader, test_loader, clean_x, clean_y, batch_size, full):
    print("=== Ablation study ===")
    eps_values = EPSILON_VALUES if full else EPSILON_VALUES[:2]
    victims = VICTIM_MODELS if full else {"ResNet-18": get_victim_resnet18}
    techniques = TECHNIQUES if full else ["unlearnable"]
    pgd_unlearnable = 10 if full else 3
    pgd_other = 30 if full else 8
    train_epochs = 3 if full else 1

    surrogate = (
        get_surrogate_resnet50(device)
        if any(t in techniques for t in {"general_cloaking", "cloaking"})
        else None
    )
    clip_model = None
    if any(t in techniques for t in {"concept_poisoning", "nightshade"}):
        clip_model, _ = load_clip_model("ViT-B/32", device)

    records = []
    for technique in techniques:
        for eps in eps_values:
            print(f"[{technique}] epsilon={eps}")
            noise_dict = None
            if technique == "unlearnable":
                noise_dict = generate_unlearnable_noise(
                    model_fn=lambda: get_victim_resnet18(device=device),
                    train_loader=train_loader,
                    epsilon=eps,
                    pgd_steps=pgd_unlearnable,
                    inner_epochs=1 if not full else 3,
                    device=device,
                )

            protected_x = build_protected_tensors(
                technique,
                clean_x,
                clean_y,
                device,
                epsilon=eps,
                noise_dict=noise_dict,
                surrogate=surrogate,
                clip_model=clip_model,
                pgd_steps=pgd_other,
            )
            protected_loader = DataLoader(
                TensorDataset(protected_x, clean_y),
                batch_size=batch_size,
                shuffle=True,
            )

            for model_name, model_fn in victims.items():
                print(f"  victim={model_name}")
                victim = train_classifier(
                    lambda mf=model_fn: mf(device=device),
                    protected_loader,
                    train_epochs,
                    device,
                )
                clean_acc, asr = compute_attack_success_rate(victim, test_loader, device)
                records.append(
                    {
                        "technique": technique,
                        "victim_model": model_name,
                        "epsilon": round(eps, 4),
                        "psnr": compute_psnr(clean_x, protected_x),
                        "ssim": compute_ssim(clean_x, protected_x),
                        "linf": compute_linf(clean_x, protected_x),
                        "clean_test_accuracy": clean_acc,
                        "asr": asr,
                    }
                )
                del victim
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    df = pd.DataFrame(records)
    df.to_csv("results/tables/ablation_results.csv", index=False)
    print(f"Saved results/tables/ablation_results.csv ({len(df)} rows)")

    for technique in df["technique"].unique():
        sub = df[df["technique"] == technique]
        if len(sub):
            plot_epsilon_vs_metrics(sub, technique)

    if 0.05 in set(df["epsilon"]) and len(df["technique"].unique()) > 1:
        plot_technique_comparison(df)

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Run full ablation grid on Colab-scale settings")
    parser.add_argument("--subset-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    setup_dirs()
    subset_size = args.subset_size or (5000 if args.full else 1000)
    batch_size = args.batch_size

    train_loader, test_loader, clean_x, clean_y, _ = run_main_experiment(
        device, subset_size, batch_size, args.full
    )
    run_ablation(device, train_loader, test_loader, clean_x, clean_y, batch_size, args.full)
    print("Experiment complete.")


if __name__ == "__main__":
    main()
