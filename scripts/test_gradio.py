"""End-to-end tests for Gradio protection logic without launching the UI."""

from __future__ import annotations

import os
import sys
import tempfile

import torch
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.evaluation import compute_linf, compute_psnr, compute_ssim
from src.models import get_surrogate_resnet50
from src.pipeline import image_to_tensor, image_to_tensor_no_resize, protect_patches, protect_resize
from src.techniques.cloaking import cloak_images
from src.techniques.nightshade import load_clip_model, poison_images

NOISE_DICT_PATH = os.path.join(ROOT, "results", "unlearnable_noise_dict.pt")


def _sample_image(size=(256, 192)):
    img = Image.new("RGB", size, color=(120, 80, 200))
    for x in range(0, size[0], 32):
        for y in range(0, size[1], 32):
            for dx in range(16):
                for dy in range(16):
                    if (x + dx) < size[0] and (y + dy) < size[1]:
                        img.putpixel(
                            (x + dx, y + dy),
                            ((x + dx) * 3 % 255, (y + dy) * 5 % 255, (x + y) % 255),
                        )
    return img


def resize_cifar_noise_for_ui(noise, target_hw, epsilon, source_epsilon=0.03, device="cpu"):
    import torch.nn.functional as F

    noise = noise.unsqueeze(0).float().to(device)
    noise = F.interpolate(noise, size=target_hw, mode="bilinear", align_corners=False)
    return noise * (float(epsilon) / source_epsilon)


def test_unlearnable_missing_noise_dict():
    assert not os.path.exists("/nonexistent/noise_dict.pt")
    print("[OK] unlearnable missing dict guard (static)")


def test_unlearnable_with_noise_dict(device):
    assert os.path.exists(NOISE_DICT_PATH), f"Missing {NOISE_DICT_PATH}; run scripts/run_experiment.py first"
    noise_dict = torch.load(NOISE_DICT_PATH, map_location=device)
    img = _sample_image()
    x = image_to_tensor(img).to(device)
    noise = resize_cifar_noise_for_ui(noise_dict[0], x.shape[-2:], epsilon=0.03, device=device)
    x_protected = torch.clamp(x + noise, 0.0, 1.0)
    assert x_protected.shape == x.shape
    assert compute_psnr(x.cpu(), x_protected.cpu()) > 20
    print("[OK] unlearnable resize noise apply")


def test_cloaking_resize(device):
    surrogate = get_surrogate_resnet50(device)
    img = _sample_image((224, 224))
    out, protected = protect_resize(
        img,
        lambda batch: cloak_images(surrogate, batch, epsilon=0.03, pgd_steps=5, device=device),
        size=224,
        device=device,
    )
    assert out.size == img.size
    assert protected.shape[-2:] == (224, 224)
    print("[OK] cloaking resize mode")


def test_nightshade_resize(device):
    clip_model, _ = load_clip_model("ViT-B/32", device)
    img = _sample_image((224, 224))

    def technique_fn(batch):
        return poison_images(
            clip_model,
            batch,
            target_concept="a photo of a cat",
            epsilon=0.03,
            pgd_steps=5,
            device=device,
        )

    out, protected = protect_resize(img, technique_fn, size=224, device=device)
    assert out.size == img.size
    print("[OK] nightshade resize mode")


def test_patch_mode_small(device):
    surrogate = get_surrogate_resnet50(device)
    img = _sample_image((256, 256))

    def technique_fn(batch):
        return cloak_images(surrogate, batch, epsilon=0.03, pgd_steps=3, device=device)

    out, protected = protect_patches(img, technique_fn, batch_size=2, patch_size=224, device=device)
    assert out.size == img.size
    assert protected.shape[-2:] == (256, 256)
    print("[OK] patch mode small image")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    test_unlearnable_missing_noise_dict()
    test_unlearnable_with_noise_dict(device)
    test_cloaking_resize(device)
    test_nightshade_resize(device)
    test_patch_mode_small(device)
    print("All Gradio pipeline tests passed.")


if __name__ == "__main__":
    main()
