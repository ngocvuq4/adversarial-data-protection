# Final Results - Adversarial Data Protection Framework

This file summarizes the final benchmark results selected for the report.
Use this as the main reference when writing the thesis/report, preparing slides,
or asking another GPT model to inspect the repository and generate a written report.

## Project Scope

Project title:

**Phan tich va phat trien ky thuat lam nhieu du lieu anh nham han che viec trich xuat dac trung phuc vu huan luyen mo hinh hoc sau**

English description:

**Adversarial Data Protection Framework for Image Data**

The project implements three PGD-based image protection directions:

1. **Unlearnable Examples**
   - Dataset-level protection.
   - Main goal: make a victim classifier fail to learn useful features from a protected training set.

2. **General Feature-space Cloaking**
   - Feature-space perturbation.
   - Main goal: shift image embeddings extracted by a pretrained feature model.
   - This is inspired by feature-space cloaking ideas, but it is not a full Fawkes face-recognition implementation.

3. **CLIP-space Concept Poisoning Proxy**
   - Text-image embedding manipulation.
   - Main goal: move image embeddings closer to a target text concept in CLIP space.
   - This is a proxy experiment, not a full Nightshade reproduction with Stable Diffusion fine-tuning.

Common perturbation constraint:

```text
x' = x + delta
||delta||_inf <= epsilon
x' in [0, 1]
```

PGD is used as the optimization method to update `delta` under the above constraints.

---

## 1. Unlearnable Examples - Final Benchmark

### Final Run Configuration

| Item | Value |
|---|---:|
| Dataset | CIFAR-10 |
| Train split | 45,000 images |
| Validation split | 5,000 images |
| Test split | 10,000 clean CIFAR-10 test images |
| Seed | 42 |
| Surrogate model | CIFAR ResNet-18 |
| Main victim model | CIFAR ResNet-18 |
| Baseline epochs | 20 |
| Victim epochs | 20 |
| PGD steps | 10 |
| Inner epochs | 2 |
| Epsilon | 0.03 |

Important evaluation rule:

The victim model is trained on the protected training set, but evaluated on a strictly clean test set.

### Main Result

| Metric | Value | Interpretation |
|---|---:|---|
| Baseline clean test accuracy | 0.8528 | Victim trained on clean data performs well. |
| Protected clean test accuracy | 0.1037 | Victim trained on protected data falls near random guessing for CIFAR-10. |
| Accuracy drop | 0.7491 | Strong degradation caused by protected dataset. |
| ASR proxy | 0.8963 | High attack/protection success proxy. |
| PSNR | 31.5549 dB | Protected images remain visually close to original images. |
| SSIM | 0.9419 | Structural similarity remains high. |
| L-infinity | 0.0300 | Perturbation respects epsilon bound. |

Short interpretation:

The victim model trained on clean data achieves `85.28%` test accuracy, while the victim trained on the protected dataset drops to `10.37%`, which is close to random guessing on CIFAR-10. This supports the main claim that Unlearnable Examples can make the dataset difficult to learn while maintaining acceptable visual quality.

### Transferability Across Victim Architectures

Protected dataset:

```text
results/unlearnable_full_final/train45000_val5000_seed42_base20_victim20_pgd10_inner2/tensors/eps0p03/protected_dataset.pt
```

| Victim model | Clean test accuracy | Protected test accuracy | Accuracy drop | ASR proxy |
|---|---:|---:|---:|---:|
| ResNet-18 | 0.8528 | 0.1037 | 0.7491 | 0.8963 |
| MobileNetV2 | 0.6570 | 0.1008 | 0.5562 | 0.8992 |
| VGG-16 | 0.8291 | 0.1000 | 0.7291 | 0.9000 |

Short interpretation:

The protected dataset degrades multiple victim architectures, not only the architecture used in the main experiment. This supports a transferability argument: the perturbation is not limited to one specific classifier architecture.

---

## 2. General Feature-space Cloaking - Final Benchmark

### Final Run Configuration

| Item | Value |
|---|---:|
| Dataset | Caltech-101 |
| Feature extractor / surrogate | Pretrained ResNet-50 |
| Target mode | Max-distance target in feature space |
| Epsilon | 0.03 |
| PGD steps | 20 |
| Runtime | 1733.47 seconds |

### Final Full Result

| Metric | Value | Direction | Interpretation |
|---|---:|---|---|
| Cosine(original, protected) | 0.5510 | Lower is better | Protected features become less similar to original features. |
| Feature L2 shift | 0.9458 | Higher is better | Feature vector is shifted significantly. |
| Target cosine before | 0.4580 | Reference | Original image is not very close to the selected target feature. |
| Target cosine after | 0.9236 | Higher is better | Protected image becomes much closer to the target feature. |
| Target cosine gain | 0.4656 | Higher is better | Strong movement toward target feature. |
| PSNR | 35.8575 dB | Higher is better | Visual quality remains high. |
| SSIM | 0.9142 | Higher is better | Structural similarity remains good. |
| L-infinity | 0.0300 | Bounded | Perturbation respects epsilon bound. |

Short interpretation:

Cloaking does not mainly measure classification accuracy. Instead, it measures whether the protected image's feature representation has shifted. The final result shows a strong feature-space shift while maintaining high visual quality.

### Prototype vs Final Cloaking Result

| Run | Epsilon | PGD steps | Cosine(original, protected) | Feature L2 shift | Target cosine gain | PSNR | SSIM |
|---|---:|---:|---:|---:|---:|---:|---:|
| Prototype subset | 0.03 | 10 | 0.6103 | 0.8797 | 0.4126 | 37.8700 | 0.9422 |
| Prototype subset | 0.05 | 10 | 0.5953 | 0.8969 | 0.4171 | 35.8811 | 0.9135 |
| Final full run | 0.03 | 20 | 0.5510 | 0.9458 | 0.4656 | 35.8575 | 0.9142 |

Report image:

```text
reports/assets/benchmark_tables/cloaking_prototype_vs_final.png
```

---

## 3. CLIP-space Concept Poisoning Proxy - Final Benchmark

### Scope Note

This project implements **CLIP-space Concept Poisoning Proxy**, not full Nightshade.

Full Nightshade would require:

```text
poisoned dataset
-> fine-tune/train Stable Diffusion
-> generate images from prompts
-> evaluate generated output
```

This project instead evaluates whether PGD can move an image embedding closer to a target text embedding in CLIP space.

### Single-target Dog Benchmark

Dataset: Caltech-101 subset.

Target text:

```text
a photo of a dog
```

| Run | Epsilon | PGD steps | Target similarity before | Target similarity after | Gain | Target rank before | Target rank after | Top-1 target rate after | PSNR | SSIM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Smoke run | 0.03 | 5 | 0.1998 | 0.3033 | 0.1035 | 3.77 | 1.20 | 0.8300 | 37.1977 | 0.9398 |
| Benchmark | 0.03 | 10 | 0.2024 | 0.3340 | 0.1316 | 3.66 | 1.05 | 0.9540 | 37.4607 | 0.9415 |
| Benchmark | 0.05 | 10 | 0.2024 | 0.3374 | 0.1350 | 3.66 | 1.03 | 0.9700 | 35.4821 | 0.9137 |

Short interpretation:

For target `"a photo of a dog"`, the CLIP target similarity increases and the target rank moves close to rank 1. This shows that the image embeddings are being pulled toward the selected target concept in CLIP space.

Report image:

```text
reports/assets/benchmark_tables/concept_poisoning_dog_epsilon_ablation.png
```

### Multi-target Benchmark

Dataset: Caltech-101 subset of 500 images.

Configuration:

| Item | Value |
|---|---:|
| Epsilon | 0.03 |
| PGD steps | 10 |
| Dataset subset | 500 images |

| Target concept | Similarity before | Similarity after | Gain | Target rank before | Target rank after | Top-1 target rate after | PSNR | SSIM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| a photo of a dog | 0.2024 | 0.3340 | 0.1316 | 3.6600 | 1.0500 | 0.9540 | 37.4607 | 0.9415 |
| a photo of a car | 0.1963 | 0.3318 | 0.1356 | 4.8720 | 1.0760 | 0.9440 | 37.4720 | 0.9415 |
| a photo of a flower | 0.1865 | 0.3436 | 0.1570 | 6.8440 | 1.0340 | 0.9680 | 37.3474 | 0.9407 |
| a photo of an airplane | 0.1871 | 0.3491 | 0.1619 | 7.1520 | 1.0220 | 0.9780 | 37.5613 | 0.9421 |

Short interpretation:

The target concepts consistently move close to rank 1 after PGD optimization. This supports the claim that the method can steer image embeddings toward multiple target text concepts in CLIP space.

Report image:

```text
reports/assets/benchmark_tables/concept_poisoning_multitarget_subset500.png
```

---

## 4. Metric Interpretation Summary

### Image Quality Metrics

| Metric | Meaning | Better direction |
|---|---|---|
| PSNR | Pixel-level similarity between original and protected images | Higher |
| SSIM | Structural similarity between original and protected images | Higher |
| L-infinity | Maximum allowed per-pixel perturbation | Must stay within epsilon |

### Unlearnable Metrics

| Metric | Meaning | Better direction |
|---|---|---|
| Baseline clean test accuracy | Accuracy of model trained on clean data | Higher |
| Protected clean test accuracy | Accuracy of model trained on protected data and tested on clean data | Lower |
| Accuracy drop | Baseline accuracy minus protected accuracy | Higher |
| ASR proxy | Proxy attack success rate, usually `1 - protected accuracy` | Higher |

### Cloaking Metrics

| Metric | Meaning | Better direction |
|---|---|---|
| Cosine(original, protected) | Similarity between original and protected feature embeddings | Lower |
| Feature L2 shift | L2 distance between original and protected feature embeddings | Higher |
| Target cosine gain | Increase in similarity between protected image and target feature | Higher |

### Concept Poisoning Proxy Metrics

| Metric | Meaning | Better direction |
|---|---|---|
| Target similarity gain | Increase in CLIP similarity to target text | Higher |
| Target rank after | Rank of target text after poisoning | Closer to 1 |
| Top-1 target rate after | Fraction of images where target text becomes top-1 | Higher |

---

## 5. Recommended Tables/Figures for the Report

Use these as the main report artifacts:

1. **Unlearnable main benchmark table**
   - Baseline vs protected accuracy.
   - Include PSNR, SSIM, L-infinity.

2. **Unlearnable transferability table**
   - ResNet-18, MobileNetV2, VGG-16.

3. **Cloaking prototype vs final table**
   - Use `cloaking_prototype_vs_final.png`.

4. **Concept Poisoning dog epsilon ablation**
   - Use `concept_poisoning_dog_epsilon_ablation.png`.

5. **Concept Poisoning multi-target table**
   - Use `concept_poisoning_multitarget_subset500.png`.

6. **Demo UI screenshots**
   - Original images.
   - Protected images.
   - Noise maps.
   - Metrics panel.

---

## 6. Important Limitations to State Clearly

1. **Concept Poisoning is a proxy**
   - The project measures CLIP-space target shift.
   - It does not fine-tune Stable Diffusion on the poisoned dataset.
   - Therefore, it should not be claimed as a full Nightshade reproduction.

2. **Cloaking is general feature-space cloaking**
   - The project uses ResNet-50 feature embeddings.
   - It is inspired by feature-space cloaking.
   - It is not a full Fawkes implementation with a face recognition system.

3. **Unlearnable is dataset-level**
   - UI image demo only shows visual perturbation.
   - The real evidence is victim training on protected data and evaluation on clean test data.

4. **Surrogate dependence**
   - All techniques depend on the surrogate model used to compute gradients.
   - Transferability is tested for Unlearnable but not fully exhausted for every possible victim model.

5. **Compute limitation**
   - Full Stable Diffusion poisoning and large-scale patch-based high-resolution experiments are outside the current scope.

---

## 7. Suggested Final Claim

The project demonstrates that small PGD-based image perturbations can reduce or alter the usefulness of image data for deep learning models in three different settings:

1. **Dataset learning disruption** through Unlearnable Examples.
2. **Feature extraction disruption** through General Feature-space Cloaking.
3. **Image-text concept steering** through CLIP-space Concept Poisoning Proxy.

The strongest completed benchmark is Unlearnable Examples, where the victim model's clean test accuracy drops from `85.28%` to `10.37%` while maintaining `PSNR = 31.5549 dB` and `SSIM = 0.9419`.
