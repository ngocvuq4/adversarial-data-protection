# Demo Assets

This folder contains sample images for running the Gradio demo quickly.

## Folder layout

```text
demo_assets/
└── input_images/
    ├── general/
    ├── face/
    └── high_res/
```

## Recommended usage

- `input_images/general/`: use for General Feature-space Cloaking and CLIP-space Concept Poisoning.
- `input_images/face/`: use when explaining a face-protection scenario. The current project is not a full Fawkes implementation, but these images are useful for demonstrating the face-image use case.
- `input_images/high_res/`: use to compare resize mode and patch mode in the UI.

## Files

| File | Suggested demo | Source / license note |
|---|---|---|
| `input_images/general/dog_millie_public_domain.jpg` | Concept Poisoning target text such as `a photo of a dog` | Wikimedia Commons, public domain, U.S. Government work |
| `input_images/general/car_fetherstonhaugh_public_domain.jpg` | Concept Poisoning target text such as `a photo of a car` | Wikimedia Commons, public domain |
| `input_images/general/flower_daisy_public_domain.jpg` | Concept Poisoning target text such as `a photo of a flower`; high-res quality comparison | Wikimedia Commons, public domain |
| `input_images/face/astronaut_eileen_collins_public_domain.png` | Face-image demo | scikit-image sample data, NASA image, public domain |
| `input_images/face/astronaut_face_crop_public_domain.jpg` | Face-focused demo | Derived crop from the astronaut sample image |
| `input_images/high_res/flower_daisy_public_domain_high_res.jpg` | High-resolution resize/patch test | Copy of public-domain daisy image |
| `input_images/high_res/flower_daisy_4k_stress_public_domain.jpg` | 4K-style stress test for patch mode | Resized derivative of public-domain daisy image |
| `input_images/high_res/dog_millie_large_public_domain.jpg` | Larger animal-image test | Resized derivative of public-domain dog image |

Avoid placing private personal photos here if the repository is public.
