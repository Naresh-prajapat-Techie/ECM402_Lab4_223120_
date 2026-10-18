# ECM-402 Lab 4: CNNs for Multi-Dimensional Signals

Fashion-MNIST classification in PyTorch for **ECM-402: AI for Signal Processing** (NIT Andhra Pradesh). The lab treats an image as a 2-D signal and a convolutional layer as a *learned* 2-D filter. The project covers a custom CNN, a controlled ablation study, ResNet-18 transfer learning, and Grad-CAM explainability.

- **GitHub:** https://github.com/rajgigyashu-2412/ecm402-lab4-123110/tree/main
- **Roll no:** <123110>

## Highlights
- Custom CNN (421,642 parameters) reaches **92.4% validation accuracy** and **91.9% test accuracy**, ahead of an MLP with a similar parameter budget (88.7% val).
- One-factor-at-a-time ablations on pooling, depth and augmentation.
- Three ResNet-18 fine-tuning strategies at 64x64 input on a stratified 10k/2k subset.
- Grad-CAM from scratch with forward and backward hooks on `layer4[-1]`.
- Strict protocol: model selection on validation only, test set used once.

## Dataset
[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist): 28x28 grayscale, 10 classes. It downloads automatically through `torchvision.datasets` into `data/` (git-ignored).
Split: 60,000 training images -> 48,000 train / 12,000 validation (`random_split`, seed 42). The official 10,000-image test set is evaluated once at the end.

## Project structure
```
ecm402-lab4-<rollno>/
|-- README.md
|-- requirements.txt
|-- report.pdf
|-- src/
|   |-- data.py        # transforms, splits, loaders, device utility   (Task 1)
|   |-- models.py      # custom CNN, MLP baseline, ResNet-18 setup     (Tasks 2, 4)
|   |-- train.py       # training / validation loops
|   |-- ablations.py   # controlled experiments                        (Task 3)
|   |-- transfer.py    # fine-tuning strategies                        (Task 4)
|   |-- gradcam.py     # Grad-CAM implementation                       (Task 6)
|   `-- main.py        # reproduces all results and figures
|-- results/
|   |-- figures/
|   `-- tables/
`-- notebooks/         # optional
```

## How to run
```bash
pip install -r requirements.txt
python src/main.py
```
Optional environment variables: `EPOCHS=12` (custom CNN and ablations), `TL_EPOCHS=6` (transfer learning).
All figures go to `results/figures/` and all tables to `results/tables/`.

## Method
| Task | What was done |
|---|---|
| 1. Data | Sobel and Gaussian filters via `conv2d`; augmentation (horizontal flip, small affine); train-only augmentation; device-agnostic code |
| 2. Custom CNN | Conv32-ReLU-MaxPool-Conv64-ReLU-MaxPool-FC128-Dropout-FC10; hand-computed layer table verified against code; receptive field of 8 px |
| 3. Ablations | MLP baseline, max / avg / strided-conv downsampling, 1-4 conv blocks, augmentation on/off |
| 4. Transfer learning | ResNet-18 (ImageNet weights), 64x64 grayscale->RGB; feature extraction, unfreeze `layer4`, differential LRs; cosine LR schedule |
| 5. Evaluation | Single test evaluation, macro-F1, confusion matrix, efficiency table |
| 6. Grad-CAM | Heatmaps for correct Coat and Shirt, plus one misclassified image |

## Results

### Ablations (validation accuracy)
| Variant | Params | Val acc | Sec / epoch |
|---|---|---|---|
| MLP baseline | 435,402 | 88.67% | 11.9 |
| CNN, 2 blocks, max-pool | 421,642 | **92.39%** | 13.3 |
| CNN, 2 blocks, avg-pool | 421,642 | 91.84% | 12.9 |
| CNN, 2 blocks, strided conv | 421,642 | 91.51% | 11.6 |
| CNN, 1 block | 804,554 | 91.83% | 12.4 |
| CNN, 3 blocks | 241,546 | 92.33% | 13.3 |
| CNN, 4 blocks | 258,058 | 92.25% | 13.6 |
| CNN, 2 blocks + augmentation | 421,642 | 89.72% | 19.5 |

Accuracy stops improving after 2 conv blocks. Augmentation lowered validation accuracy within 12 epochs because the augmented model was still improving when training stopped.

### Transfer learning (ResNet-18, 64x64, 10k train subset)
| Strategy | Trainable params | Best val acc | Sec / epoch |
|---|---|---|---|
| Feature extraction | 5,130 | 85.55% | 7.3 |
| Unfreeze `layer4` | 8,398,858 | 89.65% | 8.0 |
| Differential LR (all layers) | 11,181,642 | **90.65%** | 9.4 |
| Differential LR + cosine schedule | 11,181,642 | 90.50% | 9.3 |

### Final test results (evaluated once)
Fill these from `results/tables/efficiency_table.csv`.

| Model | Trainable params | Test acc | Macro-F1 | Sec / epoch |
|---|---|---|---|---|
| MLP | 435,402 | <fill> | <fill> | 11.9 |
| Custom CNN | 421,642 | 91.94% | <fill> | 13.3 |
| ResNet-18 (differential LR) | 11,181,642 | <fill> | <fill> | 9.4 |

The most confused classes are **Shirt and T-shirt**, since sleeve length and collar are only a few pixels at 28x28.

## Reproducibility
- **Seed:** 42 (splits, stratified subset, weight initialisation, data loaders)
- **Hardware:** Google Colab, <GPU model, e.g. NVIDIA T4>
- **Per-epoch times:** listed in the tables above and in `results/tables/*.csv`
- Every number in `report.pdf` comes from `python src/main.py`.

## Figures
All figures are in `results/figures/`: `task1a_filters.png`, `task2d_curves.png`, `task2e_kernels.png`, `task2e_featuremaps.png`, `task3c_depth.png`, `task3d_augmentation.png`, `task4c_strategies.png`, `task5_confusion.png`, `task6_gradcam.png`.

## Notes
- Grad-CAM heatmaps are coarse because ResNet-18's last feature map is only 2x2 at 64x64 input.
- The optional denoising bonus task is not included.

## Author
<Gigyashu Raj>, B.Tech Biotechnology, NIT Andhra Pradesh
