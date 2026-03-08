# Water Body Segmentation

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![License](https://img.shields.io/badge/License-MIT-green)

Semantic segmentation of water bodies from 12-band multispectral satellite
imagery using U-Net and transfer learning.

---

## Problem

Accurate water body detection from satellite imagery is critical for flood
monitoring, water resource management, and environmental conservation.
This project trains a deep learning model to produce pixel-level binary masks
separating water from non-water regions across diverse satellite scenes.

---

## Dataset

- 306 multispectral GeoTIFF images
- 12 spectral bands per image (visible + infrared)
- Resolution: 128 × 128 pixels
- Labels: binary masks (1 = water, 0 = not water)
- Split: 70% train / 15% validation / 15% test

---

## Results

| Model | IoU | F1 | Precision | Recall |
|---|---|---|---|---|
| U-Net from Scratch | 0.817 | 0.896 | 0.905 | 0.888 |
| Transfer — Pre-Layer (ResNet50) | 0.828 | 0.905 | 0.958 | 0.859 |
| Transfer — Replace Layer (ResNet50) | 0.842 | 0.913 | 0.960 | 0.873 |
| Transfer — MiT-B2 Transformer (**best**) | **0.854** | **0.920** | 0.939 | **0.903** |

---

## Key Findings

- MiT-B2 transformer encoder outperformed all CNN-based approaches
- Pre-layer approach (Exp1) outperformed replace-layer (Exp2) at 50 epochs
  but underperformed at 100 epochs — replace layer needs more time to adapt
- Transfer learning gave +0.037 IoU improvement over training from scratch
  with only 306 images
- Band 9 contained -9999 no-data values in 7 images that caused NaN
  poisoning during training — fixed by detecting and replacing before
  computing global normalization statistics

---

## Project Structure

```
water-segmentation/
├── config.py                  ← all settings in one place
├── main.py                    ← runs full pipeline end to end
├── requirements.txt
│
├── data/
│   ├── raw/                   ← original .tif files (not tracked by git)
│   └── processed/
│       └── stats.json         ← global normalization stats (tracked)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_transfer_learning.ipynb
│
├── src/
│   ├── preprocessing.py       ← load, clean, normalize
│   ├── dataset.py             ← PyTorch Dataset + DataLoader
│   ├── train.py               ← training loop + metrics
│   ├── evaluate.py            ← test set evaluation + visualizations
│   ├── visualize.py           ← plotting functions
│   └── models/
│       ├── unet_scratch.py    ← baseline U-Net
│       ├── unet_prelayer.py   ← Experiment 1
│       ├── unet_replace.py    ← Experiment 2
│       └── unet_satellite.py  ← Experiment 3
│
├── scripts/
│   └── run_all_experiments.py ← runs all 3 transfer learning experiments
│
└── outputs/
    ├── plots/                 ← all visualizations
    └── results/               ← JSON results per experiment
```

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/water-segmentation.git
cd water-segmentation
pip install -r requirements.txt
```

---

## How to Run

### Full pipeline from scratch
```bash
python main.py
```

### Run all transfer learning experiments
```bash
python scripts/run_all_experiments.py

# Skip already finished experiments
python scripts/run_all_experiments.py --skip-done

# Run one specific experiment
python scripts/run_all_experiments.py --only exp3
```

### Explore interactively
Open notebooks in order:
```
notebooks/01_data_exploration.ipynb
notebooks/02_preprocessing.ipynb
notebooks/03_transfer_learning.ipynb
```

---

## Preprocessing Pipeline

```
Raw .tif → replace_nodata() → compute_global_stats() → normalize_with_stats()
```

Key challenge: Band 9 contained -9999 sentinel values (satellite no-data flag)
in 3 images where the entire band was missing. This caused NaN propagation
through the network. Fix: detect fully corrupted bands and replace with 0
before computing dataset-wide normalization statistics.

---

## Visualizations

The pipeline auto-generates all plots in `outputs/plots/`:

| Plot | Description |
|------|-------------|
| `all_bands.png` | All 12 spectral bands from one sample |
| `image_vs_mask.png` | Input image next to its water mask |
| `before_after_norm.png` | 4 bands before/after normalization |
| `band_distributions.png` | Pixel value histograms for key bands |
| `first_batch.png` | 4 samples from first training batch |
| `predictions_epoch_*.png` | Model predictions at key epochs |
| `loss_curves.png` | Train vs validation loss |
| `metrics.png` | IoU, F1, Precision, Recall over epochs |
| `final_evaluation.png` | Test samples with TP/FP/FN overlay |

### Sample Output

<p align="center">
  <img src="outputs/plots/final_evaluation.png" width="700" alt="Final evaluation overlay showing TP, FP, FN">
</p>

---

## Stack

Python · PyTorch · Rasterio · NumPy · Matplotlib · segmentation-models-pytorch
