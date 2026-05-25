"""Large-image protection pipeline.

Two modes are supported:
- resize: default for fast Colab T4 demos. Resize image to 224x224, protect,
  then resize the protected output back to the original image size.
- patch: advanced mode. Keep image resolution, split into 224x224 patches,
  protect patch batches, stitch them back, and crop to the original size.

Never pass a full high-resolution image directly into a backward graph.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

TARGET_SIZE = 224


def image_to_tensor(pil_img, size=TARGET_SIZE):
    """Resize PIL Image to tensor (1, 3, size, size) float32 [0, 1]."""
    transform = T.Compose([T.Resize((size, size)), T.ToTensor()])
    return transform(pil_img.convert("RGB")).unsqueeze(0)


def image_to_tensor_no_resize(pil_img):
    """PIL Image to tensor (1, 3, H, W) float32 [0, 1], preserving resolution."""
    return T.ToTensor()(pil_img.convert("RGB")).unsqueeze(0)


def tensor_to_image(tensor, original_size=None):
    """Tensor (1, 3, H, W) or (3, H, W) float32 [0, 1] to PIL Image."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    arr = (
        tensor.detach()
        .cpu()
        .clamp(0.0, 1.0)
        .numpy()
        .transpose(1, 2, 0)
        * 255.0
    )
    img = Image.fromarray(arr.round().clip(0, 255).astype(np.uint8))
    if original_size:
        img = img.resize(original_size, Image.LANCZOS)
    return img


def _pad_to_grid(x, patch_size):
    _, _, height, width = x.shape
    pad_h = (patch_size - height % patch_size) % patch_size
    pad_w = (patch_size - width % patch_size) % patch_size
    return torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="replicate"), (
        height,
        width,
    )


def _extract_patches(x, patch_size):
    _, channels, height, width = x.shape
    return (
        x.unfold(2, patch_size, patch_size)
        .unfold(3, patch_size, patch_size)
        .permute(0, 2, 3, 1, 4, 5)
        .reshape(-1, channels, patch_size, patch_size)
    )


def _stitch_patches(patches, padded_hw, original_hw, patch_size):
    height, width = padded_hw
    rows, cols = height // patch_size, width // patch_size
    channels = patches.shape[1]
    x = (
        patches.reshape(1, rows, cols, channels, patch_size, patch_size)
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(1, channels, height, width)
    )
    original_h, original_w = original_hw
    return x[:, :, :original_h, :original_w].clamp(0.0, 1.0)


def protect_resize(
    pil_img: Image.Image,
    technique_fn: Callable,
    size=TARGET_SIZE,
    device="cpu",
    **technique_kwargs,
):
    """Protect by resizing to size, applying technique_fn, then restoring size."""
    original_size = pil_img.size
    x = image_to_tensor(pil_img, size=size).to(device)
    x_protected = technique_fn(x, **technique_kwargs).detach().cpu().clamp(0.0, 1.0)
    output_img = tensor_to_image(x_protected, original_size=original_size)
    del x
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output_img, x_protected


def protect_patches(
    pil_img: Image.Image,
    technique_fn: Callable,
    batch_size=4,
    patch_size=TARGET_SIZE,
    device="cpu",
    **technique_kwargs,
):
    """Protect by processing non-overlapping patches in small batches."""
    x = image_to_tensor_no_resize(pil_img)
    padded, original_hw = _pad_to_grid(x, patch_size)
    patches = _extract_patches(padded, patch_size)
    outputs = []

    for start in range(0, patches.size(0), batch_size):
        batch = patches[start : start + batch_size].to(device)
        protected = technique_fn(batch, **technique_kwargs)
        outputs.append(protected.detach().cpu().clamp(0.0, 1.0))
        del batch, protected
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    protected_patches = torch.cat(outputs, dim=0)
    stitched = _stitch_patches(
        protected_patches,
        padded_hw=padded.shape[-2:],
        original_hw=original_hw,
        patch_size=patch_size,
    )
    return tensor_to_image(stitched), stitched


def protect_single_image(
    pil_img,
    technique_fn,
    mode="resize",
    patch_size=TARGET_SIZE,
    batch_size=4,
    device="cpu",
    **technique_kwargs,
):
    """Protect one PIL image and return a PIL image with the original size."""
    mode = mode.lower()
    if mode == "resize":
        image, _ = protect_resize(
            pil_img,
            technique_fn,
            size=patch_size,
            device=device,
            **technique_kwargs,
        )
        return image
    if mode == "patch":
        image, _ = protect_patches(
            pil_img,
            technique_fn,
            batch_size=batch_size,
            patch_size=patch_size,
            device=device,
            **technique_kwargs,
        )
        return image
    raise ValueError(f"Unknown processing mode: {mode}")
