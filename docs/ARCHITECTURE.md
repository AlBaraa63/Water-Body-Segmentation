# Architecture Details

## U-Net for Water Body Segmentation

### Why U-Net?
U-Net is the standard architecture for binary segmentation tasks. Its encoder-decoder structure with skip connections preserves both high-level semantic understanding AND low-level spatial detail — critical for accurately drawing water body boundaries.

### Input/Output
- **Input:** (batch, 12, 128, 128) — 12 spectral bands from Sentinel-2
- **Output:** (batch, 1, 128, 128) — raw logits (no sigmoid applied)

### Architecture Details

```
ENCODER (contracting path)
──────────────────────────
Level 1:  DoubleConv(12 → 64)     128×128
          MaxPool2d                 → 64×64

Level 2:  DoubleConv(64 → 128)     64×64
          MaxPool2d                 → 32×32

Level 3:  DoubleConv(128 → 256)    32×32
          MaxPool2d                 → 16×16

Level 4:  DoubleConv(256 → 512)    16×16
          MaxPool2d                 → 8×8

BOTTLENECK
──────────
          DoubleConv(512 → 1024)    8×8

DECODER (expanding path)
────────────────────────
Level 4:  ConvTranspose(1024 → 512) → 16×16
          Concat(skip from enc4)    → 1024 channels
          DoubleConv(1024 → 512)

Level 3:  ConvTranspose(512 → 256)  → 32×32
          Concat(skip from enc3)    → 512 channels
          DoubleConv(512 → 256)

Level 2:  ConvTranspose(256 → 128)  → 64×64
          Concat(skip)              → 256 channels
          DoubleConv(256 → 128)

Level 1:  ConvTranspose(128 → 64)   → 128×128
          Concat(skip)              → 128 channels
          DoubleConv(128 → 64)

OUTPUT
──────
          Conv2d(64 → 1, kernel=1)  → 128×128
```

### DoubleConv Block
Every encoder/decoder level uses this repeating block:
```
Conv2d(3×3, padding=1) → BatchNorm2d → ReLU
Conv2d(3×3, padding=1) → BatchNorm2d → ReLU
```
- `bias=False` because BatchNorm already has its own bias term
- `padding=1` preserves spatial dimensions

### Skip Connections
The encoder features at each level are concatenated (not added) to the corresponding decoder features. This doubles the channel count at concat time, which is why each decoder DoubleConv takes 2× the expected channels.

---

## Loss Function: BCEDiceLoss

### Why not just BCE?
Water bodies are often a small fraction of the image (class imbalance). Pure BCE gets dominated by the majority class (land) — the model can predict "all land" and still get low loss.

### Why BCE + Dice?
- **BCE (per-pixel):** Penalizes individual pixel errors
- **Dice (region-based):** Directly optimizes overlap between prediction and ground truth
- **Combined:** `loss = BCE(logits, targets) + DiceLoss(logits, targets)`

The Dice component uses a smooth term (1e-6) to prevent division by zero on empty masks.

---

## Preprocessing

### No-Data Handling
Satellite imagery contains no-data pixels (value -9999) from sensor gaps or cloud masking. These are replaced with the median of valid pixels in the same band.

### Normalization
Z-score normalization per band using **global statistics** computed across the entire training set:
```
normalized = (pixel - global_mean) / global_std
```
Stats are cached to `data/processed/stats.json` so they only need to be computed once.
