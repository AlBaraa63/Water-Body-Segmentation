# 🛰️ Water Body Segmentation

Binary segmentation of water bodies from 12-band multispectral satellite imagery using a U-Net architecture.

**Input:** 12-band GeoTIFF (128×128) → **Output:** Binary water mask (128×128)

---

## 📁 Project Structure

```
water-segmentation/
│
├── data/
│   ├── raw/                    # original files
│   │   ├── images/             # .tif satellite images (12 bands)
│   │   └── masks/              # .png binary water masks
│   └── processed/              # cached stats.json
│
├── notebooks/                  # Jupyter exploration
│   ├── 01_data_exploration.ipynb
│   └── 02.ipynb
│
├── src/                        # all Python code
│   ├── dataset.py              # PyTorch dataset + data splits
│   ├── preprocessing.py        # normalization, no-data handling
│   ├── model.py                # U-Net + BCEDiceLoss
│   ├── train.py                # full training loop
│   ├── evaluate.py             # test set evaluation
│   └── visualize.py            # all 12 plotting functions
│
├── outputs/
│   ├── checkpoints/            # best_model.pth
│   ├── plots/                  # all visualizations
│   ├── predictions/            # predicted masks
│   ├── results/                # results.json, test_results.json
│   └── logs/                   # training logs
│
├── docs/
│   ├── ARCHITECTURE.md         # model design details
│   └── TRAINING_GUIDE.md       # step-by-step guide
│
├── config.py                   # all settings in one place
├── main.py                     # runs the whole pipeline
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place your data
Put satellite `.tif` images in `data/raw/images/` and matching `.png` masks in `data/raw/masks/`.

### 3. Run the full pipeline
```bash
python main.py
```

Or run specific phases:
```bash
python main.py --mode train    # training only
python main.py --mode eval     # evaluation only
```

---

## 🧠 Architecture

**U-Net** with 4-level encoder/decoder:

```
Input (12, 128, 128)
  ↓
Encoder:  64 → 128 → 256 → 512  (MaxPool between each)
  ↓
Bottleneck: 1024
  ↓
Decoder:  512 → 256 → 128 → 64  (Skip connections from encoder)
  ↓
Output (1, 128, 128) — raw logits
```

**Loss:** BCE + Dice (handles class imbalance — water is often small)

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for full details.

---

## 📊 Visualizations

The pipeline auto-generates all of these in `outputs/plots/`:

| Plot | Description |
|------|-------------|
| `all_bands.png` | All 12 spectral bands from one sample |
| `image_vs_mask.png` | Input image next to its water mask |
| `before_after_norm.png` | 4 bands before/after normalization |
| `band_distributions.png` | Pixel value histograms for bands 9, 10, 12 |
| `first_batch.png` | 4 samples from first training batch |
| `predictions_epoch_000.png` | Model predictions BEFORE training |
| `predictions_epoch_010.png` | Predictions at epoch 10 |
| `predictions_epoch_020.png` | Predictions at epoch 20 |
| `predictions_epoch_050.png` | Predictions at final epoch |
| `loss_curves.png` | Train vs validation loss |
| `metrics.png` | IoU, F1, Precision, Recall over epochs |
| `final_evaluation.png` | Test samples with TP/FP/FN overlay |

---

## ⚙️ Configuration

All settings in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `BATCH_SIZE` | 16 | Samples per batch |
| `EPOCHS` | 50 | Maximum training epochs |
| `LR` | 0.001 | Learning rate |
| `EARLY_STOP_PATIENCE` | 10 | Stop if no improvement for N epochs |
| `TRAIN_SPLIT` | 0.70 | Training data fraction |
| `VAL_SPLIT` | 0.15 | Validation data fraction |
| `TEST_SPLIT` | 0.15 | Test data fraction |
| `SEED` | 42 | Random seed for reproducibility |

---

## 📈 Training Features

- **BCEDiceLoss** — handles water/non-water class imbalance
- **Early Stopping** — stops training when IoU plateaus
- **Gradient Clipping** — prevents exploding gradients (max_norm=1.0)
- **ReduceLROnPlateau** — halves LR when IoU stops improving
- **tqdm Progress Bars** — per-batch training progress
- **Auto JSON Saving** — results saved to `outputs/results/`
- **Console Banners** — structured 3-level output protocol

---

## 📦 Dataset

- **Source:** 12-band Sentinel-2 multispectral satellite imagery
- **Image size:** 128 × 128 pixels
- **Bands:** Coastal, Blue, Green, Red, 3× Vegetation Red Edge, NIR, Narrow NIR, Water Vapour, SWIR 1, SWIR 2
- **Labels:** Binary masks (1 = water, 0 = not water)
- **Samples:** 306 valid image-mask pairs

---

## 📄 Documentation

- [Architecture Details](docs/ARCHITECTURE.md)
- [Training Guide](docs/TRAINING_GUIDE.md)
