# Notebook Guide

Tai lieu nay ghi ro notebook nao la chinh, notebook nao la thu nghiem phu.

## Notebook chinh cho Unlearnable

### `colab_unlearnable_full_final_experiment.ipynb`

Dung de tao ket qua final:

- Train 45000 anh
- Validation 5000 anh
- Test 10000 anh
- Epsilon chinh: `0.03`
- Luu protected dataset, model, tables, figures vao Drive

Day la notebook nen dung khi can tai tao ket qua chinh trong bao cao.

### `colab_unlearnable_transfer_victims.ipynb`

Dung de kiem tra transferability tren cac victim model khac:

- MobileNetV2
- VGG-16

Notebook nay khong sinh lai noise. No doc file `protected_dataset.pt` tu ket qua full-final.

## Notebook dung cho validation / ablation

### `colab_unlearnable_validation_experiment.ipynb`

Dung de chon epsilon bang validation:

- Epsilon: `0.01`, `0.03`, `0.05`
- Train 10000 anh
- Validation 2000 anh
- Test 10000 anh

Ket qua tu notebook nay giai thich vi sao chon `epsilon=0.03`.

## Notebook cu / exploratory

### `colab_unlearnable_experiment.ipynb`

Notebook thuc nghiem ban dau, dung subset 5000. Ket qua chi nen xem la preliminary/debug, khong dung lam benchmark chinh.

## Notebook cho cac ky thuat khac

### Cloaking

- `colab_cloaking_caltech101_experiment.ipynb`
  - Notebook ablation chinh cho Cloaking.
  - Chay subset 500 voi epsilon `0.03` va `0.05` de so sanh trade-off feature shift / chat luong anh.
  - Benchmark dung muc tieu feature-space: cosine(original, protected), target cosine gain, feature L2 shift, PSNR, SSIM, L-infinity.
  - Khong train classifier victim vi classification accuracy khong phai metric chinh cua Cloaking.
- `colab_cloaking_caltech101_full_final_experiment.ipynb`
  - Notebook final cho Cloaking.
  - Mac dinh chay full Caltech-101 voi epsilon `0.03`, PGD steps `20`.
  - Neu Colab khong du thoi gian, doi `SUBSET_SIZE = 2000` trong cell config.
  - Benchmark dung muc tieu feature-space: cosine(original, protected), target cosine gain, feature L2 shift, PSNR, SSIM, L-infinity.
  - Khong train classifier victim vi classification accuracy khong phai metric chinh cua Cloaking.
- `colab_cloaking_cifar100_experiment.ipynb`
- `colab_cloaking_experiment.ipynb`
  - Notebook cu / exploratory, khong dung lam ket qua chinh.

### Concept Poisoning

- `colab_concept_poisoning_caltech101_experiment.ipynb`
- `colab_concept_poisoning_experiment.ipynb`

Hai nhom nay chua nen dua vao bang ket qua chinh neu chua chay va validate day du nhu Unlearnable.

## Thu muc ket qua tren Drive

Cau truc khuyen nghi:

```text
adversarial-data-protection/
├── data/
└── results/
    ├── unlearnable_validation/
    ├── unlearnable_full_final/
    └── unlearnable_transfer_victims/
```

## Thu muc ket qua local

`results/` tren local chi la output tam khi chay notebook/script tren may ca nhan. Ket qua benchmark chinh duoc luu tren Google Drive, con anh/bang can cho bao cao da duoc xuat vao:

```text
reports/assets/benchmark_tables/
```

Vi vay co the xoa `results/` local sau khi da xac nhan cac artifact bao cao nam trong `reports/`.
