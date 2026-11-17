# Penn-Fudan Pedestrian Binary Semantic Segmentation

A production-grade computer vision pipeline and deep learning framework for binary person segmentation using the Penn-Fudan Pedestrian Dataset, implemented in Python and PyTorch.

---

## 1. Project Objective

Prepare the Penn-Fudan Pedestrian Dataset and build a high-fidelity semantic segmentation pipeline to identify pedestrian pixels versus background pixels. While the original dataset provides instance-level bounding boxes and individual pedestrian IDs, this project transforms the task into **binary semantic segmentation** ($0 = \text{background}, 1 = \text{person}$) for downstream U-Net model training.

---

## 2. Dataset Overview & Source

- **Dataset Source**: [Penn-Fudan Pedestrian Database](https://www.cis.upenn.edu/~jshi/ped_html/)
- **Download Archive**: `https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip`
- **Total Photographs**: 170 photographs taken from campus and urban pedestrian scenes.
- **Annotated Instances**: 345 original pedestrian annotations plus 78 additional objects documented in `added-object-list.txt`, totaling 423 ground-truth instances.
- **License / Credit**: University of Pennsylvania & Fudan University (Liming Wang, Jianbo Shi, Gang Song, I-fan Shen).

---

## 3. Directory Structure

```
person-segmentation/
├── README.md                          # Project documentation
├── requirements.txt                   # Pinned dependency environment
├── configs/
│   ├── config.yaml                    # Dataset configuration
│   └── training.yaml                  # Model and training hyperparameters
├── data/
│   ├── raw/
│   │   └── PennFudanPed/              # Extracted official dataset
│   │       ├── Annotation/            # Pascal VOC-style text annotations
│   │       ├── PedMasks/              # Raw instance masks (mode L)
│   │       └── PNGImages/             # Raw photographs (RGB PNG)
│   └── processed/
│       ├── images/                    # Standardized 256x256 RGB images (170 PNGs)
│       └── masks/                     # Standardized 256x256 Binary masks {0,1} (170 PNGs)
├── splits/
│   ├── train.txt                      # 119 samples (70.0%)
│   ├── val.txt                        # 25 samples (14.71%)
│   ├── test.txt                       # 26 samples (15.29%)
│   └── splits_metadata.json           # Split audit metadata
├── models/
│   ├── baseline_unet.pth              # Checkpoint: Unaugmented U-Net (Exp A)
│   └── augmented_unet.pth             # Checkpoint: Augmented U-Net (Exp B)
├── src/
│   ├── download_dataset.py            # Dataset acquisition and extraction
│   ├── inspect_dataset.py             # Raw data dimension and property audit
│   ├── prepare_masks.py               # Instance-to-binary mask merging logic
│   ├── preprocess.py                  # Bilinear/nearest 256x256 standardization
│   ├── create_splits.py               # Seed-42 train/val/test partitioning
│   ├── visualize.py                   # Generation of publication-quality plots
│   ├── models/
│   │   └── unet.py                    # Reusable PyTorch U-Net architecture
│   ├── datasets/
│   │   └── pedestrian_dataset.py      # PyTorch Dataset and DataLoader
│   ├── augmentation/
│   │   └── transforms.py              # Synchronized geometric transformations
│   ├── evaluation/
│   │   └── metrics.py                 # Precision, Recall, F1, IoU metrics
│   ├── training/
│   │   ├── trainer.py                 # Controlled training and validation loop
│   │   └── train.py                   # Experiment runner CLI
│   └── utils/
│       ├── utils.py                   # Config loading and path helpers
│       └── reproducibility.py         # Seed locking and hardware info
├── results/
│   ├── dataset_samples/               # Visual grids, overlays, comparisons
│   ├── dataset_statistics/            # Dataset distribution plots
│   ├── training_curves/               # Loss, IoU, and F1 training curves
│   └── experiment_logs/               # Detailed JSON training metrics
└── tests/
    ├── test_dataset.py                # Dataset pipeline validation test suite
    ├── test_unet.py                   # U-Net forward/backward and shape tests
    └── test_transforms.py             # Augmentation alignment and integrity tests
```

---

## 4. Engineering Methodology

### A. Mask Merging Logic
The raw ground-truth masks in `PedMasks/*.png` contain separate integer IDs for each pedestrian instance (e.g., $1, 2, \dots, N$).
Our semantic segmentation target collapses all person instances into class `1`:
$$\text{binary\_mask} = (\text{original\_mask} > 0).\text{astype}(\text{uint8})$$
- `0`: Background
- `1`: Pedestrian foreground
Every mask is strictly checked to ensure only $\{0, 1\}$ values are present.

### B. Preprocessing & Resizing
All images and masks are standardized to $256 \times 256$ pixels:
1. **RGB Images**: Converted to 3-channel RGB; resized using **bilinear interpolation** (`PIL.Image.Resampling.BILINEAR`) to preserve edge smoothness and natural transitions.
2. **Binary Masks**: Resized using **nearest-neighbor interpolation** (`PIL.Image.Resampling.NEAREST`) to prevent interpolation artifacts and ensure output pixels remain strictly $\{0, 1\}$.
3. Dimensions and filenames are strictly paired and verified across all 170 samples.

### C. Split Methodology & Leakage Prevention
An image-level random split is computed using a deterministic random seed of `42`:
- **Training Set**: 119 images ($70.0\%$)
- **Validation Set**: 25 images ($14.71\%$)
- **Testing Set**: 26 images ($15.29\%$)

**Data Leakage Guards**:
- Strict set disjointness: $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$.
- Exact union: $\text{Train} \cup \text{Val} \cup \text{Test} = 170$.
- No test set statistics or augmentations are ever used during preprocessing or training.

---

## 5. Dataset Statistics & Key Findings

| Metric | Value |
|---|---|
| Total Photographs | 170 |
| Total Ground-Truth Pedestrian Instances | 423 (345 original + 78 added) |
| Raw Image Width Range | 253 px to 1,017 px (Mean: 491.8 px) |
| Raw Image Height Range | 311 px to 581 px (Mean: 396.5 px) |
| Processed Spatial Resolution | 256 × 256 px |
| Total Processed Pixels Across Dataset | 11,141,120 px |
| Foreground (Person) Coverage | 17.28% (Mean per image: 17.28% ± 7.66%) |
| Background Coverage | 82.72% |
| Minimum Person Coverage in an Image | 4.85% |
| Maximum Person Coverage in an Image | 42.42% |

### Dataset Limitations
1. **Sample Size**: 170 images is a compact dataset; regularization and data augmentation are essential to prevent severe overfitting.
2. **Class Imbalance**: Pedestrians cover approximately 17.3% of pixels. While binary cross-entropy works well, IoU and F1-score are critical for assessing true boundary quality.
3. **Occlusion & Scene Diversity**: Images depict daytime campus/outdoor scenes; performance may vary under night or heavy weather conditions.

---

## 6. How to Reproduce Dataset Engineering

Execute commands from within the `person-segmentation` directory:

```powershell
# 1. Download / Extract raw dataset
python -m src.download_dataset

# 2. Inspect raw data properties and dimensions
python -m src.inspect_dataset

# 3. Validate binary mask merging
python -m src.prepare_masks

# 4. Preprocess images and masks to 256x256
python -m src.preprocess

# 5. Generate deterministic 70/15/15 splits (Seed 42)
python -m src.create_splits

# 6. Generate all publication figures and distribution plots
python -m src.visualize

# 7. Run automated test suite
python -m unittest tests/test_dataset.py -v
```
