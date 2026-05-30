"""CLIP-space concept poisoning proxy.

This module implements a lightweight CLIP-space proxy: PGD updates pixels so
the protected image embedding moves toward a target text embedding.
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


def load_clip_model(model_name="ViT-B/32", device="cpu", download_root=None):
    """Load OpenAI CLIP and freeze it for PGD generation."""
    model, preprocess = clip.load(model_name, device=device, download_root=download_root)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, preprocess


def get_text_embedding(clip_model, text, device="cpu"):
    """Encode a target concept string into a normalized CLIP text embedding."""
    tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        emb = clip_model.encode_text(tokens)
        emb = F.normalize(emb.float(), dim=-1)
    return emb


def get_text_embeddings(clip_model, texts, device="cpu"):
    """Encode multiple concept strings into normalized CLIP text embeddings."""
    tokens = clip.tokenize(list(texts)).to(device)
    with torch.no_grad():
        emb = clip_model.encode_text(tokens)
        emb = F.normalize(emb.float(), dim=-1)
    return emb


def normalize_clip_image(x):
    """Apply OpenAI CLIP image normalization to tensors in [0, 1]."""
    mean = CLIP_MEAN.to(device=x.device, dtype=x.dtype)
    std = CLIP_STD.to(device=x.device, dtype=x.dtype)
    return (x - mean) / std


def encode_image_features(clip_model, x):
    """Encode image tensors into normalized CLIP image embeddings."""
    feats = clip_model.encode_image(normalize_clip_image(x))
    return F.normalize(feats.float(), dim=-1)


def poison_images(
    clip_model,
    x,
    target_concept,
    epsilon=0.05,
    pgd_steps=150,
    pgd_alpha=None,
    device=None,
):
    """Return images whose CLIP image embeddings move toward target_concept."""
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
        img_feats = encode_image_features(clip_model, x_adv)
        loss = -F.cosine_similarity(img_feats, target_emb.expand_as(img_feats)).mean()
        grad = torch.autograd.grad(loss, delta, only_inputs=True)[0]
        with torch.no_grad():
            delta -= pgd_alpha * grad.sign()
            delta.clamp_(-epsilon, epsilon)
            delta.copy_(torch.clamp(x_orig + delta, 0.0, 1.0) - x_orig)
        delta = delta.detach().requires_grad_(True)

    return torch.clamp(x_orig + delta.detach(), 0.0, 1.0)


@torch.no_grad()
def clip_target_metrics(clip_model, x_orig, x_protected, target_concept, device=None):
    """Return per-image target cosine before/after/gain for one target concept."""
    device = x_orig.device if device is None else device
    target = get_text_embedding(clip_model, target_concept, device)
    f_orig = encode_image_features(clip_model, x_orig.to(device))
    f_prot = encode_image_features(clip_model, x_protected.to(device))
    before = F.cosine_similarity(f_orig, target.expand_as(f_orig), dim=1)
    after = F.cosine_similarity(f_prot, target.expand_as(f_prot), dim=1)
    return {
        "target_cosine_before": before.detach().cpu(),
        "target_cosine_after": after.detach().cpu(),
        "target_cosine_gain": (after - before).detach().cpu(),
    }


@torch.no_grad()
def clip_text_ranking(clip_model, x, candidate_texts, device=None):
    """Rank candidate text concepts for each image by CLIP similarity."""
    device = x.device if device is None else device
    image_feats = encode_image_features(clip_model, x.to(device))
    text_feats = get_text_embeddings(clip_model, candidate_texts, device)
    sims = image_feats @ text_feats.T
    return sims.detach().cpu()
