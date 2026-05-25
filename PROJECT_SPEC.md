# PROJECT SPEC — Bảo vệ ảnh khỏi AI Training
> Tài liệu này dùng để hướng dẫn AI code (Cursor / Claude / Copilot).  
> Đọc toàn bộ trước khi sinh bất kỳ dòng code nào.

---

## 1. Tổng quan project

**Mục tiêu:** Xây dựng hệ thống thêm perturbation vô hình vào ảnh nhằm ngăn mô hình học sâu trích xuất đặc trưng hữu ích từ ảnh đó.

**Platform:** Google Colab (GPU T4, RAM 12GB, VRAM 16GB)

**Output cuối:**
- `notebook_experiment.ipynb` — thực nghiệm đầy đủ, chạy cell-by-cell
- `app_gradio.ipynb` — demo Gradio UI, người dùng upload ảnh và nhận ảnh đã bảo vệ
- `results/` — thư mục chứa số liệu, biểu đồ, ảnh so sánh

---

## 2. Cấu trúc thư mục (bắt buộc tuân theo)

```
project/
├── notebook_experiment.ipynb     # Notebook thực nghiệm
├── app_gradio.ipynb              # Notebook demo Gradio
├── src/
│   ├── __init__.py
│   ├── datasets.py               # Load CIFAR-10, LFW
│   ├── models.py                 # Định nghĩa surrogate + victim models
│   ├── techniques/
│   │   ├── __init__.py
│   │   ├── unlearnable.py        # Kỹ thuật 1
│   │   ├── cloaking.py           # Kỹ thuật 2
│   │   └── nightshade.py         # Kỹ thuật 3
│   ├── evaluation.py             # Tính ASR, PSNR, SSIM, LPIPS
│   ├── pipeline.py               # Xử lý ảnh lớn (resize default + patch optional)
│   └── visualization.py          # Vẽ biểu đồ, bảng so sánh
├── results/
│   ├── figures/                  # Biểu đồ PNG
│   ├── tables/                   # CSV kết quả
│   ├── protected_samples/        # Ảnh mẫu trước/sau
│   └── unlearnable_noise_dict.pt # Sinh bởi notebook_experiment, load bởi Gradio
└── requirements.txt
```

---

## 3. Môi trường và thư viện

### requirements.txt (phiên bản cố định, không thay đổi)
```
torch==2.1.0
torchvision==0.16.0
Pillow==10.0.0
numpy==1.24.0
scikit-learn==1.3.0
matplotlib==3.7.0
seaborn==0.12.0
lpips==0.1.4
gradio==3.50.0
tqdm==4.66.0
pandas==2.0.0
pytorch-msssim==1.0.0
```

### Cell đầu tiên của MỌI notebook (bắt buộc)
```python
# Cell 1 — Setup (KHÔNG thay đổi thứ tự)
import subprocess
subprocess.run(["pip", "install", "-q", "-r", "requirements.txt"])

import torch
import torchvision
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# Kiểm tra GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

---

## 4. Dataset

### CIFAR-10 (dùng cho Unlearnable + Ablation study)
```python
# src/datasets.py
import torchvision.transforms as T
from torchvision.datasets import CIFAR10, LFWPairs
from torch.utils.data import DataLoader, Subset

def get_cifar10(root="./data", subset_size=5000, batch_size=128):
    """
    Trả về train_loader và test_loader của CIFAR-10.
    subset_size: giới hạn số mẫu để chạy nhanh trên Colab.
    Ảnh được normalize về [0,1], shape (3,32,32).
    """
    transform = T.Compose([T.ToTensor()])  # KHÔNG normalize ở đây, normalize trong model
    train_set = CIFAR10(root=root, train=True, download=True, transform=transform)
    test_set  = CIFAR10(root=root, train=False, download=True, transform=transform)

    if subset_size:
        train_set = Subset(train_set, range(subset_size))
        test_set  = Subset(test_set,  range(subset_size // 5))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader
```

### LFW (dùng cho Cloaking)
```python
def get_lfw_pairs(root="./data/lfw", img_size=224, max_pairs=200):
    """
    Load ảnh khuôn mặt từ LFW.
    Trả về list of (img_tensor, label_str).
    img_tensor shape: (3, img_size, img_size), giá trị [0,1].
    """
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor()
    ])
    # Tải thủ công từ http://vis-www.cs.umass.edu/lfw/lfw.tgz
    # hoặc dùng torchvision.datasets.LFWPairs
    dataset = LFWPairs(root=root, split="test", download=True, transform=transform)
    if max_pairs:
        dataset = Subset(dataset, range(min(max_pairs, len(dataset))))
    return dataset
```

---

## 5. Models

```python
# src/models.py
import torchvision.models as M
import torch.nn as nn

NORMALIZE = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

def get_surrogate_resnet50(device):
    """
    Surrogate model cho Cloaking.
    Dùng ResNet-50 pretrained ImageNet, BỎ lớp fc cuối.
    Output: feature vector 2048-dim.
    KHÔNG train lại.
    """
    model = M.resnet50(weights=M.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    model.eval()
    return model.to(device)

def get_victim_resnet18(num_classes=10, device="cuda"):
    """
    Victim model cho Unlearnable + Ablation.
    Train từ đầu trên CIFAR-10 (KHÔNG dùng pretrained weights).
    """
    model = M.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, num_classes)
    return model.to(device)

def get_victim_vgg16(num_classes=10, device="cuda"):
    """Victim model thứ 2 cho transferability test."""
    model = M.vgg16(weights=None)
    model.classifier[6] = nn.Linear(4096, num_classes)
    return model.to(device)

def get_victim_mobilenet(num_classes=10, device="cuda"):
    """Victim model thứ 3 cho transferability test."""
    model = M.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(1280, num_classes)
    return model.to(device)

def train_one_epoch(model, loader, optimizer, criterion, device):
    """Train 1 epoch. Trả về avg_loss, accuracy."""
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total

def evaluate(model, loader, device):
    """Đánh giá model. Trả về accuracy."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += x.size(0)
    return correct / total
```

---

## 6. Ba kỹ thuật bảo vệ

### Quy tắc chung cho TẤT CẢ kỹ thuật
- Input ảnh: tensor float32, shape `(B, 3, H, W)`, giá trị `[0.0, 1.0]`
- Output ảnh: tensor float32, cùng shape, cùng range `[0.0, 1.0]`
- Perturbation `δ` luôn bị clamp: `x_protected = torch.clamp(x + δ, 0.0, 1.0)`
- Epsilon `ε` luôn tính theo L∞ norm trên range `[0,1]` (ví dụ `ε=0.03` tương đương ~8/255)
- Mọi hàm đều nhận `device` làm tham số, KHÔNG hardcode `"cuda"`

---

### 6.1 Unlearnable Examples

```python
# src/techniques/unlearnable.py
"""
Thuật toán: Error-minimizing noise (Huang et al., ICLR 2021)
Bài toán: min_δ min_θ L(f_θ(x+δ), y)
Giải bằng: alternating update — cố định θ cập nhật δ, rồi ngược lại

Tham số:
    epsilon   : L∞ budget, float, default 0.03 (≈ 8/255)
    pgd_steps : số bước PGD cho outer loop, int, default 20
    pgd_alpha : step size PGD, float, default epsilon/10
    inner_epochs: số epoch train model trong inner loop, int, default 5
"""

import torch
import torch.nn as nn
import copy

def generate_unlearnable_noise(
    model_fn,       # callable: trả về model mới (chưa train)
    train_loader,   # DataLoader của tập cần bảo vệ
    epsilon=0.03,
    pgd_steps=20,
    pgd_alpha=None,
    inner_epochs=5,
    device="cuda"
):
    """
    Sinh class-wise noise cho toàn bộ dataset.
    Trả về: noise_dict — dict {class_idx: noise_tensor shape (3,H,W)}
    """
    if pgd_alpha is None:
        pgd_alpha = epsilon / 10

    # Lấy shape ảnh từ batch đầu tiên
    sample_x, _ = next(iter(train_loader))
    img_shape = sample_x.shape[1:]  # (3, H, W)
    num_classes = 10  # CIFAR-10

    # Khởi tạo class-wise noise = 0
    noise_dict = {
        c: torch.zeros(img_shape, device=device)
        for c in range(num_classes)
    }

    # Alternating update
    model = model_fn().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    for step in range(pgd_steps):
        # --- Outer: cập nhật noise (cố định model) ---
        model.eval()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            for c in range(num_classes):
                mask = (y == c)
                if mask.sum() == 0:
                    continue
                xc = x[mask]
                noise = noise_dict[c].unsqueeze(0).expand_as(xc).clone().requires_grad_(True)
                xc_noisy = torch.clamp(xc + noise, 0, 1)
                loss = criterion(model(xc_noisy), y[mask])
                loss.backward()
                with torch.no_grad():
                    # Gradient descent (minimize loss → error-minimizing)
                    noise_dict[c] -= pgd_alpha * noise.grad.mean(0).sign()
                    noise_dict[c] = noise_dict[c].clamp(-epsilon, epsilon)

        # --- Inner: train model trên noisy data (cố định noise) ---
        model.train()
        for _ in range(inner_epochs):
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                # Thêm noise tương ứng class
                x_noisy = x.clone()
                for c in range(num_classes):
                    mask = (y == c)
                    if mask.sum() == 0:
                        continue
                    x_noisy[mask] = torch.clamp(x[mask] + noise_dict[c], 0, 1)
                optimizer.zero_grad()
                loss = criterion(model(x_noisy), y)
                loss.backward()
                optimizer.step()

    return noise_dict


def apply_noise(x, y, noise_dict):
    """
    Áp dụng noise vào batch ảnh theo class.
    x: (B,3,H,W) float32 [0,1]
    y: (B,) int64
    Trả về: x_protected (B,3,H,W) float32 [0,1]
    """
    x_protected = x.clone()
    for c, noise in noise_dict.items():
        mask = (y == c)
        if mask.sum() == 0:
            continue
        x_protected[mask] = torch.clamp(x[mask] + noise.to(x.device), 0, 1)
    return x_protected
```

### Định dạng bắt buộc của `results/unlearnable_noise_dict.pt`

File `results/unlearnable_noise_dict.pt` bắt buộc lưu đúng class-wise noise dict
do `generate_unlearnable_noise()` trả về, không bọc thêm metadata và không lưu
dưới dạng tensor trần.

```python
noise_dict = {
    0: torch.Tensor,  # shape (3, 32, 32), float32, L∞ <= epsilon
    1: torch.Tensor,  # shape (3, 32, 32), float32, L∞ <= epsilon
    ...
    9: torch.Tensor,  # shape (3, 32, 32), float32, L∞ <= epsilon
}

torch.save(noise_dict, "results/unlearnable_noise_dict.pt")
```

Quy tắc tương thích:
- Keys phải là integer class id `0..9` của CIFAR-10.
- Values phải là tensor noise shape `(3,32,32)`, không phải ảnh đã cộng noise.
- Gradio đọc trực tiếp `unlearnable_noise_dict[0]` cho demo-only UI, nên không đổi
  sang schema `{"type": ..., "noise": ...}` nếu chưa sửa loader tương ứng.
- Nếu sau này dùng sample-wise noise, phải lưu file khác tên và viết loader riêng,
  không ghi đè `results/unlearnable_noise_dict.pt`.

---

### 6.2 General Feature-space Cloaking

```python
# src/techniques/cloaking.py
"""
Thuật toán: General feature-space cloaking, inspired by the feature-space
objective popularized by Fawkes but NOT a full Fawkes reproduction.
Bài toán: min_δ ||f(x+δ) - f(t)||₂²  s.t. ||δ||∞ ≤ ε
Giải bằng: PGD trên feature distance loss

Tham số:
    epsilon   : L∞ budget, float, default 0.05
    pgd_steps : số bước PGD, int, default 100
    pgd_alpha : step size, float, default epsilon/10
    target_mode: cách chọn target embedding — "random", "max_dist", hoặc "away"
                 Nếu batch chỉ có 1 ảnh, "random"/"max_dist" phải fallback sang "away"
                 để UI single-image không bị loss=0.
"""

import torch
import torch.nn.functional as F

def select_cloak_target(surrogate, x_batch, mode="max_dist", device="cuda"):
    """
    Chọn target embedding t cho mỗi ảnh trong batch.
    mode="random": chọn ngẫu nhiên 1 ảnh trong batch làm target
    mode="max_dist": chọn ảnh có embedding xa nhất trong batch
    mode="away": dùng hướng đối nghịch của normalized feature hiện tại
    Trả về: target_features tensor (B, feature_dim)
    """
    with torch.no_grad():
        feats = F.normalize(surrogate(x_batch.to(device)).float(), dim=1)  # (B, 2048)

    if mode in {"away", "self_negate", "disrupt"}:
        return (-feats).detach()
    elif feats.size(0) < 2:
        return (-feats).detach()
    elif mode == "random":
        idx = torch.randperm(feats.size(0))
        return feats[idx]
    elif mode == "max_dist":
        targets = []
        for i in range(feats.size(0)):
            dists = F.pairwise_distance(feats[i].unsqueeze(0), feats)
            targets.append(feats[dists.argmax()])
        return torch.stack(targets)
    else:
        raise ValueError(f"target_mode phải là 'random', 'max_dist', hoặc 'away', nhận được: {mode}")


def cloak_images(
    surrogate,          # General feature extractor pretrained (ResNet-50, fc=Identity)
    x,                  # Ảnh cần bảo vệ (B,3,H,W) float32 [0,1]
    epsilon=0.05,
    pgd_steps=100,
    pgd_alpha=None,
    target_mode="max_dist",
    device="cuda"
):
    """
    Sinh cloaked images cho batch x.
    Trả về: x_cloaked (B,3,H,W) float32 [0,1], cùng device với x
    """
    if pgd_alpha is None:
        pgd_alpha = epsilon / 10

    surrogate.eval()
    x = x.to(device)

    # Lấy target features (không cần gradient)
    target_feats = select_cloak_target(surrogate, x, mode=target_mode, device=device)
    target_feats = target_feats.detach()

    # Khởi tạo delta = 0
    delta = torch.zeros_like(x, requires_grad=True)

    for step in range(pgd_steps):
        x_adv = torch.clamp(x + delta, 0, 1)
        feats = surrogate(x_adv)

        # Feature distance loss: minimize distance đến target
        loss = F.mse_loss(feats, target_feats)
        loss.backward()

        with torch.no_grad():
            # Gradient descent (minimize distance)
            delta -= pgd_alpha * delta.grad.sign()
            # Project về L∞ ball
            delta.clamp_(-epsilon, epsilon)
            # Đảm bảo ảnh kết quả hợp lệ
            delta = torch.clamp(x + delta, 0, 1) - x

        delta = delta.detach().requires_grad_(True)

    return torch.clamp(x + delta.detach(), 0, 1)
```

---

### Optional extension: Face/Fawkes-style Cloaking

This project currently implements `General Feature-space Cloaking` for arbitrary uploaded images.
It is inspired by the feature-space objective used in Fawkes, but it is not a full Fawkes reproduction.

If a dedicated face-protection mode is added later, it must be separated as:

```python
TECH_FACE_CLOAKING = "Face Cloaking (Fawkes-style)"
```

Additional requirements for the face mode:

1. Add face detection/alignment before PGD.
2. Use a face-recognition embedding model instead of ImageNet ResNet-50.
3. Use face datasets such as LFW/CelebA/VGGFace2 for evaluation.
4. Evaluate identity embedding shift, not generic ImageNet feature shift.
5. Do not run face mode automatically on arbitrary landscape/object/animal images.

The default UI mode remains `General Feature-space Cloaking` with ResNet-50 because it is domain-general and does not assume the uploaded image contains a face.

---

### 6.3 CLIP-space Concept Poisoning (Nightshade-style proxy)

```python
# src/techniques/nightshade.py
"""
Thuat toan: CLIP-space concept poisoning (Nightshade-style proxy)
Bài toán: min_δ -cosine(f_CLIP_image(x+δ), f_CLIP_text(target_concept))
          s.t. ||δ||∞ ≤ ε
          tức là: dịch chuyển embedding của ảnh về phía target text concept

Lưu ý Colab:
    - Dùng ViT-B/32 thay ViT-L/14 để tránh OOM trên T4
    - KHÔNG load Stable Diffusion trong module này (quá nặng)
    - Đánh giá bằng cosine similarity shift trong CLIP space

Tham số:
    epsilon   : L∞ budget, float, default 0.05
    pgd_steps : số bước PGD, int, default 150
    pgd_alpha : step size, float, default epsilon/pgd_steps * 2
    target_concept: string mô tả concept đích (ví dụ "a photo of a cat")
"""

import torch
import torch.nn.functional as F

try:
    import clip
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "-q", "git+https://github.com/openai/CLIP.git"])
    import clip


def load_clip_model(model_name="ViT-B/32", device="cuda"):
    """
    Load CLIP model. Dùng ViT-B/32 (không phải ViT-L/14) để fit T4 VRAM.
    Trả về: (model, preprocess)
    """
    model, preprocess = clip.load(model_name, device=device)
    model.eval()
    return model, preprocess


def get_text_embedding(clip_model, text, device="cuda"):
    """
    Encode text thành embedding.
    Trả về: tensor (1, 512) normalized
    """
    tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        emb = clip_model.encode_text(tokens)
        emb = F.normalize(emb, dim=-1)
    return emb


def poison_images(
    clip_model,         # CLIP model đã load
    x,                  # Ảnh cần poison (B,3,H,W) float32 [0,1], đã resize về 224
    target_concept,     # str: concept đích, ví dụ "a photo of a cat"
    epsilon=0.05,
    pgd_steps=150,
    pgd_alpha=None,
    device="cuda"
):
    """
    Sinh poisoned images: ảnh trông như x nhưng CLIP embedding
    dịch chuyển về phía target_concept.

    Trả về: x_poisoned (B,3,H,W) float32 [0,1]
    """
    if pgd_alpha is None:
        pgd_alpha = epsilon * 2 / pgd_steps

    clip_model.eval()
    x = x.to(device)

    # Target embedding từ text
    target_emb = get_text_embedding(clip_model, target_concept, device)  # (1,512)
    target_emb = target_emb.detach()

    # CLIP preprocess: normalize về mean/std của CLIP
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1,3,1,1)
    clip_std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1,3,1,1)

    delta = torch.zeros_like(x, requires_grad=True)

    for step in range(pgd_steps):
        x_adv = torch.clamp(x + delta, 0, 1)
        # Normalize cho CLIP
        x_norm = (x_adv - clip_mean) / clip_std
        img_feats = clip_model.encode_image(x_norm)
        img_feats = F.normalize(img_feats, dim=-1)

        # Maximize cosine similarity với target concept
        loss = -F.cosine_similarity(img_feats, target_emb.expand_as(img_feats)).mean()
        loss.backward()

        with torch.no_grad():
            delta -= pgd_alpha * delta.grad.sign()
            delta.clamp_(-epsilon, epsilon)
            delta = torch.clamp(x + delta, 0, 1) - x

        delta = delta.detach().requires_grad_(True)

    return torch.clamp(x + delta.detach(), 0, 1)
```

---

## 7. Evaluation

```python
# src/evaluation.py
"""
Tất cả hàm đều:
- Nhận tensor float32 [0,1] shape (B,3,H,W) hoặc (3,H,W)
- Trả về float scalar (giá trị trung bình trên batch)
- KHÔNG có side effect (không lưu file, không plot)
"""

import torch
import torch.nn.functional as F
import numpy as np

def compute_psnr(x_orig, x_protected):
    """Peak Signal-to-Noise Ratio. Cao hơn = ảnh giống hơn. Đơn vị: dB."""
    mse = F.mse_loss(x_protected, x_orig).item()
    if mse == 0:
        return float("inf")
    return 10 * np.log10(1.0 / mse)

def compute_ssim(x_orig, x_protected, window_size=11):
    """
    Structural Similarity Index. Range [-1,1], cao hơn = giống hơn.
    Tính xấp xỉ bằng Gaussian window trên từng channel.
    """
    # Dùng thư viện pytorch-msssim nếu có, fallback về tính tay
    try:
        from pytorch_msssim import ssim
        return ssim(x_orig, x_protected, data_range=1.0).item()
    except ImportError:
        # Fallback đơn giản: tính trên 1 patch trung tâm
        C1, C2 = 0.01**2, 0.03**2
        mu1 = F.avg_pool2d(x_orig, window_size, 1, window_size//2)
        mu2 = F.avg_pool2d(x_protected, window_size, 1, window_size//2)
        mu1_sq, mu2_sq = mu1**2, mu2**2
        sigma1_sq = F.avg_pool2d(x_orig**2, window_size, 1, window_size//2) - mu1_sq
        sigma2_sq = F.avg_pool2d(x_protected**2, window_size, 1, window_size//2) - mu2_sq
        sigma12 = F.avg_pool2d(x_orig*x_protected, window_size, 1, window_size//2) - mu1*mu2
        ssim_map = ((2*mu1*mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2))
        return ssim_map.mean().item()

def compute_linf(x_orig, x_protected):
    """L∞ norm của perturbation. Range [0,1]."""
    return (x_protected - x_orig).abs().max().item()

def compute_attack_success_rate(
    victim_model,
    clean_test_loader,
    device
):
    """
    Attack Success Rate (ASR):
    = tỷ lệ mẫu mà victim model dự đoán SAI sau khi train trên protected data.
    Ở đây đo gián tiếp: train victim trên protected data, đo accuracy trên clean test.
    ASR = 1 - clean_test_accuracy (sau khi train trên protected)
    Hàm này chỉ ĐÁNH GIÁ model đã train sẵn, không train lại.
    Trả về: (accuracy, asr), đánh giá trên toàn bộ clean_test_loader.
    """
    victim_model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in clean_test_loader:
            x, y = x.to(device), y.to(device)
            preds = victim_model(x).argmax(1)
            correct += (preds == y).sum().item()
            total += y.numel()
    accuracy = round(correct / max(total, 1), 4)
    return accuracy, round(1.0 - accuracy, 4)

def full_evaluation(x_orig, x_protected, victim_model, clean_test_loader, device):
    """
    Chạy toàn bộ metrics một lần.
    Trả về: dict với keys psnr, ssim, linf, clean_test_accuracy, asr
    """
    psnr = compute_psnr(x_orig.cpu(), x_protected.cpu())
    ssim = compute_ssim(x_orig.cpu(), x_protected.cpu())
    linf = compute_linf(x_orig.cpu(), x_protected.cpu())
    acc, asr = compute_attack_success_rate(victim_model, clean_test_loader, device)
    return {
        "psnr": round(psnr, 4),
        "ssim": round(ssim, 4),
        "linf": round(linf, 4),
        "clean_test_accuracy": round(acc, 4),
        "asr": round(asr, 4)
    }
```

---

## 8. Pipeline xử lý ảnh lớn

### Quyết định cuối cùng cho ảnh lớn trong Gradio

Không chốt cứng một chiến lược duy nhất. Dùng 2 chế độ:

1. **Resize mode — default**
   - Ảnh upload được resize về `224×224`.
   - Chạy PGD trên tensor `1×3×224×224`.
   - Resize kết quả về kích thước gốc.
   - Đây là chế độ mặc định vì chạy nhanh, ít VRAM, phù hợp Colab T4 và demo trực tiếp.

2. **Patch mode — advanced**
   - Ảnh upload được pad về bội số của `224`.
   - Cắt thành patch `224×224`.
   - Chạy PGD theo batch patch nhỏ.
   - Stitch/crop về kích thước gốc.
   - Chế độ này giữ chi tiết tốt hơn nhưng rất chậm. Ví dụ ảnh `2000×2000` có khoảng 81 patch; nếu mỗi patch chạy 50-150 PGD steps thì tổng forward/backward pass rất lớn và có thể timeout trên Colab.

### Public API bắt buộc

```python
# src/pipeline.py
"""
Xử lý ảnh có resolution bất kỳ từ người dùng.

mode="resize" là default cho demo nhanh trên Colab T4.
mode="patch" là option nâng cao khi cần giữ chi tiết ảnh lớn.
"""

TARGET_SIZE = 224

def image_to_tensor(pil_img, size=TARGET_SIZE):
    """Resize PIL Image -> tensor (1,3,size,size) float32 [0,1]."""

def image_to_tensor_no_resize(pil_img):
    """PIL Image -> tensor (1,3,H,W) float32 [0,1], giữ nguyên resolution."""

def tensor_to_image(tensor, original_size=None):
    """tensor (1,3,H,W) hoặc (3,H,W) float32 [0,1] -> PIL Image."""

def protect_resize(pil_img, technique_fn, size=TARGET_SIZE, device="cpu", **technique_kwargs):
    """
    Resize image to size, run technique_fn on tensor (1,3,size,size),
    resize output PIL back to original size.
    Returns: (output_pil, x_protected_tensor).
    """

def protect_patches(pil_img, technique_fn, batch_size=4, patch_size=TARGET_SIZE,
                    device="cpu", **technique_kwargs):
    """
    Pad with replicate padding, unfold into patch_size patches, run technique_fn
    per small batch, stitch/crop back to original resolution.
    Returns: (output_pil, x_protected_tensor).
    """

def protect_single_image(
    pil_img,
    technique_fn,
    mode="resize",
    patch_size=224,
    batch_size=4,
    device="cpu",
    **technique_kwargs,
):
    """
    Bảo vệ 1 ảnh PIL bất kỳ kích thước.

    mode="resize":
        resize -> protect -> resize back.

    mode="patch":
        pad replicate -> unfold patches -> protect patch batches -> stitch -> crop.

    Trả về: PIL Image đã bảo vệ, cùng kích thước ảnh gốc.
    Dùng cho caller chỉ cần ảnh. Gradio nên dùng protect_resize/protect_patches
    để lấy thêm tensor phục vụ PSNR/SSIM/Linf.
    """
```

### Quy tắc triển khai

- `mode="resize"` là default trong Gradio.
- `mode="patch"` phải có cảnh báo trong UI: chậm hơn nhiều, phù hợp ảnh cần giữ chi tiết.
- Không bao giờ đưa ảnh 4K nguyên khối vào backward graph.
- Với patch mode, dùng `replicate padding`, không dùng `reflect padding`, để tránh lỗi khi ảnh nhỏ hơn `224`.
- Sau mỗi batch patch, gọi `del` và `torch.cuda.empty_cache()` nếu có CUDA.

---

## 9. Ablation Study

### Thiết kế thực nghiệm ablation (bắt buộc có trong notebook_experiment.ipynb)

```python
# Ablation 1: Thay đổi epsilon (độ mạnh nhiễu)
EPSILON_VALUES = [0.01, 0.03, 0.05, 0.08, 0.1]

# Ablation 2: Thay đổi victim model (transferability)
VICTIM_MODELS = {
    "ResNet-18": get_victim_resnet18,
    "VGG-16":    get_victim_vgg16,
    "MobileNet": get_victim_mobilenet,
}

# Ablation 3: Thay đổi kỹ thuật (so sánh 3 chiến lược)
TECHNIQUES = ["unlearnable", "general_cloaking", "concept_poisoning"]

# Cấu trúc bảng kết quả (lưu vào results/tables/ablation_results.csv)
# columns: technique, victim_model, epsilon, psnr, ssim, linf, asr
```

### Full train-after-poison benchmark track

Research intent: after each technique generates a protected/poisoned training
dataset, train a fresh victim model on that protected dataset and evaluate how
much downstream performance drops. This is the preferred final benchmark because
it directly measures whether the protected data becomes less useful for model
training.

Benchmark protocol:

1. Build a clean train/test split.
2. Generate `protected_train` from the clean train split with one technique.
3. Train a fresh victim model from scratch on `protected_train`.
4. Evaluate the victim only on the clean test split.
5. Report clean test accuracy, ASR/protection rate, PSNR, SSIM, and L-inf.

Technique-specific notes:

- `unlearnable`: this is the most direct train-after-poison benchmark. Use
  CIFAR-10 labels, train victim on protected CIFAR-10 train set, evaluate on
  clean CIFAR-10 test set.
- `general_cloaking`: this can be benchmarked with train-after-poison, but the
  downstream task must be defined clearly. For the current general ResNet-50
  version, use image classification as a proxy. For a future full Fawkes-style
  mode, use a face-recognition victim and identity evaluation.
- `concept_poisoning`: the current CLIP-space implementation can be evaluated
  by CLIP similarity shift plus image-quality metrics. A full Nightshade-style
  benchmark would require poisoned dataset -> fine-tune/train a generative model
  -> evaluate generated outputs across prompts/concepts. Treat that as a
  heavier optional benchmark, not the default Colab T4 path.

Important: do not claim all three techniques have identical evaluation meaning.
They can share the train-after-poison structure, but the victim task and success
metric must match the protection goal of each technique.

### Evaluation metrics per technique

All techniques must report image-quality metrics:

- `psnr`: higher means the protected image remains closer to the original.
- `ssim`: higher means the protected image preserves visual structure better.
- `linf`: maximum pixel perturbation; must stay within the epsilon budget.
- `visual_samples`: original image, protected image, and amplified noise map.

Technique-specific effectiveness metrics:

| technique | protection goal | victim/evaluation task | primary effectiveness metrics |
|---|---|---|---|
| `unlearnable` | Make protected training data hard to learn from | CIFAR-10 image classification; train victim on protected train set, evaluate on clean test set | `clean_test_accuracy` down, `asr` up, `accuracy_drop` up, train/test generalization gap |
| `general_cloaking` | Distort extracted visual feature/embedding | General feature extractor or classification proxy; optional train-after-poison classifier benchmark | `feature_cosine_original_protected` down, `feature_l2_shift` up, optional `accuracy_drop` if classifier victim is trained |
| `concept_poisoning` | Shift semantic concept signal toward a wrong target | CLIP-space semantic proxy; optional full generative benchmark if diffusion fine-tuning is added | `clip_target_similarity_after` up, `clip_target_similarity_delta` up, `target_margin = sim_target - sim_original` up |

Extended benchmark notes:

- Full Fawkes-style evaluation belongs to a separate face-protection extension:
  train/evaluate a face-recognition victim and report identity verification or
  identity classification drop.
- Full Nightshade-style evaluation belongs to a separate generative-model
  extension: poison dataset, fine-tune/train a diffusion model, then evaluate
  generated outputs across prompts/concepts.
- Do not compare `asr` across all three techniques as if it has the same
  meaning. Use the primary metrics that match each technique's target.

### Hàm chạy ablation tự động
```python
import pandas as pd

def run_ablation(techniques, epsilon_values, victim_model_fns,
                 train_loader, test_loader, device, save_path="results/tables/"):
    """
    Chạy toàn bộ ablation study.
    Lưu kết quả vào CSV.
    Trả về: DataFrame kết quả
    """
    records = []
    for technique in techniques:
        for eps in epsilon_values:
            for model_name, model_fn in victim_model_fns.items():
                print(f"[{technique}] eps={eps} victim={model_name}")
                # Step 1: Generate protected train data.
                #   - "unlearnable": generate_unlearnable_noise() -> apply_noise()
                #   - "general_cloaking": upscale CIFAR batch to 224, cloak_images(), downscale to 32
                #   - "concept_poisoning": upscale CIFAR batch to 224, poison_images(), downscale to 32
                #   - protected_loader = DataLoader(TensorDataset(x_protected, y_train))
                #
                # Step 2: Train victim from scratch on protected_loader.
                #   - smoke mode: 1-3 epochs for pipeline validation
                #   - full mode: 20 epochs only if Colab budget allows
                #   - victim evaluation training MUST stay outside src/techniques/
                #
                # Step 3: Evaluate victim on the full clean test_loader.
                #   metrics = full_evaluation(
                #       x_orig=x_train_sample,
                #       x_protected=x_protected_sample,
                #       victim_model=victim,
                #       clean_test_loader=test_loader,
                #       device=device,
                #   )
                result = {
                    "technique": technique,
                    "victim_model": model_name,
                    "epsilon": eps,
                    # "psnr": metrics["psnr"],
                    # "ssim": metrics["ssim"],
                    # "linf": metrics["linf"],
                    # "clean_test_accuracy": metrics["clean_test_accuracy"],
                    # "asr": metrics["asr"],
                }
                records.append(result)

    df = pd.DataFrame(records)
    df.to_csv(f"{save_path}/ablation_results.csv", index=False)
    return df
```

---

## 10. Visualization

```python
# src/visualization.py
"""
Tất cả hàm plot đều:
- Nhận DataFrame hoặc dict kết quả
- Lưu file PNG vào results/figures/
- Trả về fig object (để hiển thị trong notebook)
- KHÔNG gọi plt.show() bên trong hàm
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

SAVE_DIR = "results/figures"
os.makedirs(SAVE_DIR, exist_ok=True)

def plot_epsilon_vs_metrics(df, technique, save=True):
    """
    Vẽ đường ASR và PSNR theo epsilon cho 1 kỹ thuật.
    df phải có columns: epsilon, asr, psnr, victim_model
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Ablation: epsilon — {technique}")

    for model_name in df["victim_model"].unique():
        sub = df[(df["technique"] == technique) & (df["victim_model"] == model_name)]
        ax1.plot(sub["epsilon"], sub["asr"],   marker="o", label=model_name)
        ax2.plot(sub["epsilon"], sub["psnr"],  marker="s", label=model_name)

    ax1.set(xlabel="Epsilon", ylabel="ASR", title="Protection rate vs epsilon")
    ax2.set(xlabel="Epsilon", ylabel="PSNR (dB)", title="Image quality vs epsilon")
    ax1.legend(); ax2.legend()
    plt.tight_layout()

    if save:
        fig.savefig(f"{SAVE_DIR}/ablation_epsilon_{technique}.png", dpi=150)
    return fig

def plot_technique_comparison(df, save=True):
    """
    Vẽ heatmap so sánh ASR của 3 kỹ thuật × 3 victim model tại epsilon=0.05
    """
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

def plot_before_after(x_orig_tensor, x_protected_tensor,
                      technique_name, save=True):
    """
    Vẽ ảnh gốc vs ảnh bảo vệ vs phóng to perturbation (×10).
    x_orig, x_protected: tensor (3,H,W) float32 [0,1]
    """
    import numpy as np
    def t2np(t):
        return t.detach().cpu().numpy().transpose(1,2,0).clip(0,1)

    orig = t2np(x_orig_tensor)
    prot = t2np(x_protected_tensor)
    delta = np.clip((prot - orig) * 10 + 0.5, 0, 1)  # phóng to ×10

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    axes[0].imshow(orig);  axes[0].set_title("Original");    axes[0].axis("off")
    axes[1].imshow(prot);  axes[1].set_title("Protected");   axes[1].axis("off")
    axes[2].imshow(delta); axes[2].set_title("Noise ×10");   axes[2].axis("off")
    fig.suptitle(f"Technique: {technique_name}")
    plt.tight_layout()

    if save:
        fig.savefig(f"{SAVE_DIR}/before_after_{technique_name}.png", dpi=150)
    return fig
```

---

## 11. Gradio App

### Yêu cầu tiên quyết cho Unlearnable

```
notebook_experiment.ipynb phải được chạy trước và đã lưu:
  results/unlearnable_noise_dict.pt
```

Nếu file này chưa tồn tại, nhánh Unlearnable trong Gradio phải báo lỗi rõ ràng. Không fallback sang random noise.

### Giới hạn khoa học của Unlearnable trong UI

Unlearnable benchmark chính nằm trong `notebook_experiment.ipynb` với CIFAR-10.

Trong Gradio, ảnh user upload là ảnh bất kỳ, thường không thuộc CIFAR-10 và không có label class. Vì vậy:

- Unlearnable trong UI chỉ là **demo minh họa precomputed CIFAR noise**.
- Không được mô tả là bảo vệ đầy đủ cho mọi ảnh upload.
- Noise CIFAR có shape `(3,32,32)`, còn ảnh UI resize default về `(3,224,224)`, nên phải resize noise trước khi cộng.
- Nếu muốn Unlearnable UI nghiêm ngặt hơn, cần precompute noise trên dataset ảnh `224×224` cùng domain với ảnh cần bảo vệ.

### Cell chinh de xuat (UI v2 multi-image)

`app_gradio.ipynb` phai dung UI nhieu anh, khong dung callback single-image cu.

Public callback bat buoc:

```python
def protect_images(
    files,              # list images from gr.Files(file_count="multiple")
    technique,          # "Unlearnable Demo", "General Feature-space Cloaking", or "Concept Poisoning (Nightshade-style)"
    epsilon,
    target_concept,
    pgd_steps,
    processing_mode,    # "resize" or "patch"
    noise_scale,
):
    return original_gallery, protected_gallery, noise_gallery, metrics_text
```

Required flow:

1. Load all uploaded files into a list of `PIL.Image`.
2. If `processing_mode="resize"`:
   - Resize every image to `224x224`, then stack into batch `(B,3,224,224)`.
   - `General Feature-space Cloaking`: call `cloak_images(..., target_mode="max_dist")` on the whole batch.
     With multiple images, each image is pulled toward the farthest image feature in the uploaded batch according to ResNet-50 embedding distance.
     With one image, `cloak_images` must internally fall back to `target_mode="away"` so the loss is not zero.
   - `Concept Poisoning (Nightshade-style)`: run CLIP-space PGD toward `target_concept` on the batch.
   - `Unlearnable Demo`: load `results/unlearnable_noise_dict.pt`, use class `0`, resize noise from `32x32` to `224x224`, then clamp `[0,1]`.
3. If `processing_mode="patch"`:
   - Call `protect_patches()` from `src.pipeline` for each image.
   - Patch mode is advanced and slow. In this mode, cloaking targets are selected inside each patch batch, not as a cross-image target over the whole uploaded album.
4. Output must include:
   - original image gallery,
   - protected image gallery,
   - noise visualization gallery `(x_protected - x_orig) * noise_scale + 0.5`,
   - metrics text with PSNR, SSIM, L-inf; Cloaking also reports feature cosine/feature L2 shift; Concept Poisoning also reports CLIP similarity before/after/delta.

UI labels:

```python
TECH_UNLEARNABLE = "Unlearnable Demo"
TECH_CLOAKING = "General Feature-space Cloaking"
TECH_CONCEPT = "Concept Poisoning (Nightshade-style)"
```

Do not use the short label `Nightshade` in the UI because the current module is a CLIP-space proxy, not full Nightshade/Stable-Diffusion poisoning evaluation.

### Quy tắc UI

- Không dùng `torch.rand`, `uniform_`, hoặc random noise để thay thế Unlearnable.
- `processing_mode="resize"` là default.
- `processing_mode="patch"` là advanced option, phải có cảnh báo chậm.
- Unlearnable UI phải ghi rõ `demo-only`.
- General Feature-space Cloaking va Concept Poisoning la hai mode phu hop hon cho anh upload bat ky.

---

## 12. Quy tắc bắt buộc cho AI code

```
RULES — AI code PHẢI tuân theo, không được phép vi phạm:

[R01] Mọi tensor đều là float32, range [0.0, 1.0], KHÔNG phải [0, 255].
[R02] Device agnostic: KHÔNG dùng `.cuda()` hoặc `.to("cuda")` hardcode trong `src/`.
      Mọi hàm phải nhận `device` từ caller hoặc suy ra từ tensor đầu vào bằng `x.device`
      khi hợp lý. Notebook/script là nơi tạo `device = torch.device("cuda" if
      torch.cuda.is_available() else "cpu")` và truyền xuống các hàm.
[R03] Mọi hàm perturbation đều clamp output: torch.clamp(x + delta, 0.0, 1.0).
[R04] KHÔNG import trong thân hàm (trừ fallback đã ghi rõ). Import ở đầu file.
[R05] KHÔNG gọi plt.show() trong src/. Chỉ return fig.
[R06] Mỗi file src/ có docstring đầu file mô tả: thuật toán, bài toán, tham số.
[R07] Tên biến nhất quán: x_orig, x_protected, delta, epsilon, pgd_steps, pgd_alpha.
[R08] KHÔNG train victim evaluation model bên trong hàm technique. Train victim
      để đánh giá phải nằm trong notebook/pipeline thực nghiệm riêng. Riêng
      Unlearnable được phép train surrogate/internal model trong inner loop để
      sinh noise, vì đây là một phần của thuật toán.
[R09] Kết quả số luôn round(value, 4) trước khi lưu vào dict/DataFrame.
[R10] Gradio app KHÔNG chứa logic tính toán nặng — gọi hàm từ src/.
[R11] Mọi file lưu kết quả đều dùng đường dẫn tương đối từ root project.
[R12] Colab: đặt !pip install ở setup cell đầu tiên, KHÔNG lẫn với algorithm/experiment logic.
[R13] KHÔNG dùng random/uniform noise để thay thế Unlearnable noise trong demo
      hoặc test nhanh. Nếu chưa có `results/unlearnable_noise_dict.pt` thì báo
      lỗi rõ ràng, không fallback random.
[R14] Nếu dùng CIFAR noise `(3,32,32)` trong UI, phải resize noise về đúng shape
      ảnh đang xử lý trước khi cộng. Không được cộng trực tiếp vào ảnh `224×224`.
[R15] `mode="resize"` là default cho Gradio vì phù hợp Colab T4. `mode="patch"`
      là option nâng cao, phải cảnh báo chậm và không đưa ảnh lớn nguyên khối
      vào backward graph.
[R16] Concept Poisoning objective phai duoc mo ta nhat quan la toi da cosine similarity
      với target text concept, hoặc tương đương minimize negative cosine similarity.
```

---

## 13. Thứ tự implement (AI code follow theo thứ tự này)

```
Bước 1: Tạo requirements.txt và kiểm tra cài đặt thành công
Bước 2: Implement src/datasets.py — test get_cifar10() ra đúng shape
Bước 3: Implement src/models.py — test forward pass không lỗi
Bước 4: Implement src/techniques/unlearnable.py — test với 100 ảnh CIFAR
        → cuối bước này lưu noise_dict bằng torch.save(noise_dict, "results/unlearnable_noise_dict.pt")
Bước 5: Implement src/evaluation.py — test compute_psnr, compute_ssim với tensor synthetic
Bước 6: Implement src/techniques/cloaking.py — test với 10 ảnh LFW
Bước 7: Implement src/techniques/nightshade.py — test với 5 ảnh
Bước 8: Implement src/pipeline.py
        → test mode="resize" với ảnh PIL 1200×800, output cùng kích thước
        → test mode="patch" với ảnh PIL 1200×800, output cùng kích thước
Bước 9: Implement src/visualization.py — test plot_before_after
Bước 10: Xây dựng notebook_experiment.ipynb — chạy ablation đầy đủ
Bước 11: Xây dựng app_gradio.ipynb — test demo end-to-end
        → test Unlearnable khi thiếu noise_dict: báo lỗi rõ ràng
        → test Unlearnable khi có noise_dict: resize noise 32×32 về 224×224, không crash
        → test General Feature-space Cloaking/Concept Poisoning với mode resize
        → test patch mode trên ảnh nhỏ trước khi dùng ảnh lớn
```

---

*Tài liệu này là nguồn sự thật duy nhất cho project.  
Nếu có mâu thuẫn giữa tài liệu này và bất kỳ nguồn khác, ưu tiên tài liệu này.*
