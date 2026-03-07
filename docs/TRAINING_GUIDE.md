# Training Guide

Step-by-step guide from raw data to trained model.

---

## Step 1: Prepare Data

Place your files:
```
data/raw/images/   ← .tif satellite images (12 bands, 128×128)
data/raw/masks/    ← .png binary masks (matching filenames)
```

The dataset auto-pairs images and masks by filename (e.g., `image_001.tif` → `image_001.png`).

## Step 2: Run Training

```bash
python main.py --mode train
```

**What happens automatically:**
1. Global band statistics (mean/std) are computed from training images and cached to `data/processed/stats.json`
2. Dataset is split: 70% train / 15% val / 15% test
3. Pre-training visualizations are saved (all bands, image vs mask, batch preview)
4. Model predictions at epoch 0 (before learning) are saved
5. Training begins with progress bars and per-epoch metrics
6. Model checkpoints are saved when IoU improves
7. Training stops early if IoU doesn't improve for 10 epochs
8. Post-training plots (loss curves, metrics) are saved
9. Results are saved to `outputs/results/results.json`

## Step 3: Evaluate

```bash
python main.py --mode eval
```

Loads the best checkpoint, runs inference on the test set, and generates `final_evaluation.png` with TP/FP/FN overlays.

---

## Config Reference

All settings live in `config.py`. Key parameters:

| Parameter | Default | What it controls |
|-----------|---------|------------------|
| `EPOCHS` | 50 | Maximum training epochs |
| `BATCH_SIZE` | 16 | Samples per gradient step |
| `LR` | 0.001 | Adam learning rate |
| `EARLY_STOP_PATIENCE` | 10 | Epochs without improvement before stopping |
| `SEED` | 42 | Random seed |

---

## Training Features Explained

### Early Stopping
If the validation IoU doesn't improve for `EARLY_STOP_PATIENCE` consecutive epochs, training stops automatically. This prevents overfitting and saves time.

### Gradient Clipping
`clip_grad_norm_(max_norm=1.0)` — caps the magnitude of gradients during backpropagation. Prevents the "exploding gradient" problem which can destabilize training.

### ReduceLROnPlateau
The learning rate is halved when the IoU stops improving for 5 epochs. This allows the model to make finer adjustments as it converges.

### BCEDiceLoss
Combined loss that handles class imbalance:
- BCE handles per-pixel accuracy
- Dice handles region overlap (critical when water covers a small portion of the image)

---

## Console Output Protocol

The training script uses a 3-level banner system:

```
##############################################################     ← Level 1: Script header
  Water Body Segmentation — Training
  Device : cuda  |  Seed : 42  |  Epochs : 50
##############################################################

==============================================================     ← Level 2: Phase header
  PHASE 2 / 3 — Training
==============================================================

  Epoch 010/050 | Train: 0.41 | Val: 0.38 | IoU: 0.62 | ...       ← Level 3: Per-epoch
  ✅ New best model saved! IoU=0.6201

--------------------------------------------------------------     ← Summary block
  >> Best Epoch   : 38
  >> Best IoU     : 0.7231
  >> Total Time   : 0h 12m 44s
--------------------------------------------------------------
```

---

## Output Files

After a training run, you'll find:

```
outputs/
├── checkpoints/best_model.pth        # best model weights
├── plots/
│   ├── all_bands.png                 # 12-band grid
│   ├── image_vs_mask.png             # input vs ground truth
│   ├── band_distributions.png        # pixel histograms
│   ├── first_batch.png               # batch preview
│   ├── predictions_epoch_000.png     # before training
│   ├── predictions_epoch_010.png     # mid-training
│   ├── predictions_epoch_020.png     # mid-training
│   ├── predictions_epoch_050.png     # end of training
│   ├── loss_curves.png               # train vs val loss
│   ├── metrics.png                   # IoU/F1/Prec/Recall
│   └── final_evaluation.png          # test set overlay
└── results/
    ├── results.json                  # training results
    └── test_results.json             # test set metrics
```
