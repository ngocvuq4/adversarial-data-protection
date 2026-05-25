"""CLIP-space concept poisoning, implemented as a Nightshade-style proxy.

Objective: move image embeddings toward a false text concept in CLIP space,
using ViT-B/32 to fit Colab T4 VRAM. Stable Diffusion is intentionally not
loaded here.
"""

from __future__ import annotations

import subprocess

import torch
import torch.nn.functional as F

try:
    import clip
except ImportError:
    subprocess.run(
        [
            "pip",
            "install",
            "-q",
            "https://github.com/openai/CLIP/archive/refs/heads/main.zip",
        ],
        check=False,
    )
    import clip


CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def load_clip_model(model_name="ViT-B/32", device="cpu"):
    """Load OpenAI CLIP and freeze it for PGD generation."""
    model, preprocess = clip.load(model_name, device=device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, preprocess


def get_text_embedding(clip_model, text, device="cpu"):
    """Encode text into a normalized CLIP embedding."""
    tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        emb = clip_model.encode_text(tokens)
        emb = F.normalize(emb.float(), dim=-1)
    return emb


def _normalize_clip(x):
    mean = CLIP_MEAN.to(device=x.device, dtype=x.dtype)
    std = CLIP_STD.to(device=x.device, dtype=x.dtype)
    return (x - mean) / std


def poison_images(
    clip_model,
    x,
    target_concept,
    epsilon=0.05,
    pgd_steps=150,
    pgd_alpha=None,
    device=None,
):
    """Return images whose CLIP embeddings move toward target_concept."""
    if pgd_alpha is None:
        pgd_alpha = epsilon * 2 / max(pgd_steps, 1)

    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad = False

    device = x.device if device is None else device
    x_orig = x.detach().to(device)
    target_emb = get_text_embedding(clip_model, target_concept, device).detach()
    delta = torch.zeros_like(x_orig, requires_grad=True)

    for _ in range(pgd_steps):
        x_adv = torch.clamp(x_orig + delta, 0.0, 1.0)
        img_feats = clip_model.encode_image(_normalize_clip(x_adv))
        img_feats = F.normalize(img_feats.float(), dim=-1)
        loss = -F.cosine_similarity(img_feats, target_emb.expand_as(img_feats)).mean()
        grad = torch.autograd.grad(loss, delta, only_inputs=True)[0]
        with torch.no_grad():
            delta -= pgd_alpha * grad.sign()
            delta.clamp_(-epsilon, epsilon)
            delta.copy_(torch.clamp(x_orig + delta, 0.0, 1.0) - x_orig)
        delta = delta.detach().requires_grad_(True)

    return torch.clamp(x_orig + delta.detach(), 0.0, 1.0)
