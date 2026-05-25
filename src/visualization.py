"""Visualization helpers for experiment results.

Plot functions accept DataFrames or tensors, save PNG files under
results/figures, return the matplotlib figure, and do not display interactively.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import seaborn as sns

SAVE_DIR = "results/figures"
os.makedirs(SAVE_DIR, exist_ok=True)


def plot_epsilon_vs_metrics(df, technique, save=True):
    """Plot ASR and PSNR against epsilon for one technique."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Ablation: epsilon - {technique}")

    for model_name in df["victim_model"].unique():
        sub = df[(df["technique"] == technique) & (df["victim_model"] == model_name)]
        ax1.plot(sub["epsilon"], sub["asr"], marker="o", label=model_name)
        ax2.plot(sub["epsilon"], sub["psnr"], marker="s", label=model_name)

    ax1.set(xlabel="Epsilon", ylabel="ASR", title="Protection rate vs epsilon")
    ax2.set(xlabel="Epsilon", ylabel="PSNR (dB)", title="Image quality vs epsilon")
    ax1.legend()
    ax2.legend()
    plt.tight_layout()

    if save:
        fig.savefig(f"{SAVE_DIR}/ablation_epsilon_{technique}.png", dpi=150)
    return fig


def plot_technique_comparison(df, save=True):
    """Plot ASR heatmap for techniques by victim model at epsilon=0.05."""
    pivot = df[df["epsilon"] == 0.05].pivot_table(
        index="technique", columns="victim_model", values="asr"
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax, vmin=0, vmax=1)
    ax.set_title("ASR comparison (epsilon=0.05)")
    plt.tight_layout()

    if save:
        fig.savefig(f"{SAVE_DIR}/technique_comparison_heatmap.png", dpi=150)
    return fig


def plot_before_after(x_orig_tensor, x_protected_tensor, technique_name, save=True):
    """Plot original, protected, and amplified perturbation."""
    def t2np(tensor):
        return tensor.detach().cpu().numpy().transpose(1, 2, 0).clip(0, 1)

    orig = t2np(x_orig_tensor)
    prot = t2np(x_protected_tensor)
    delta = ((prot - orig) * 10 + 0.5).clip(0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    axes[0].imshow(orig)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(prot)
    axes[1].set_title("Protected")
    axes[1].axis("off")
    axes[2].imshow(delta)
    axes[2].set_title("Noise x10")
    axes[2].axis("off")
    fig.suptitle(f"Technique: {technique_name}")
    plt.tight_layout()

    if save:
        fig.savefig(f"{SAVE_DIR}/before_after_{technique_name}.png", dpi=150)
    return fig
