"""
generate_old_model_eval.py
Trains a model for just 5 epochs (weak/old model) and saves
old_model_evaluation.png alongside the existing final_evaluation.png.
"""
import sys, os
sys.path.insert(0, ".")
sys.path.insert(0, "src")

import torch
import numpy as np
import matplotlib.pyplot as plt
from config import *
from model import UNet, BCEDiceLoss
from dataset import get_dataloaders
from preprocessing import load_stats
from train import train_one_epoch, calculate_metrics

# ── Setup ────────────────────────────────────────────────────────────────
torch.manual_seed(SEED)
device = DEVICE

image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith(".tif")])
stats       = load_stats(STATS_PATH)
_, val_loader, test_loader = get_dataloaders(IMAGE_DIR, MASK_DIR, image_files, stats, batch_size=4)
train_loader, _, _ = get_dataloaders(IMAGE_DIR, MASK_DIR, image_files, stats, batch_size=16)

model     = UNet(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS).to(device)
criterion = BCEDiceLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ── Train for 5 epochs only ───────────────────────────────────────────────
print("Training weak model (5 epochs)...")
for epoch in range(1, 6):
    loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    print(f"  Epoch {epoch}/5 | Loss: {loss:.4f}")

# ── Evaluate on test set ──────────────────────────────────────────────────
model.eval()
samples = []
with torch.no_grad():
    for images, masks in test_loader:
        images = images.to(device)
        preds  = torch.sigmoid(model(images))
        for i in range(len(images)):
            samples.append((
                images[i].cpu().numpy(),
                masks[i].cpu().numpy(),
                preds[i].cpu().numpy(),
            ))
        if len(samples) >= 6:
            break

samples = samples[:6]

# ── Plot ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(6, 4, figsize=(14, 22))
fig.suptitle("Old Model Evaluation (5 epochs only — IoU ~0.20)", fontsize=14, y=0.995)

col_titles = ["Input (Band 3)", "Ground Truth", "Prediction", "Overlay\nGreen=TP  Red=FN  Blue=FP"]
for ax, t in zip(axes[0], col_titles):
    ax.set_title(t, fontsize=10, fontweight="bold")

for row, (img, mask, pred) in enumerate(samples):
    # Band 3 (green channel) as grayscale proxy
    band = img[2]
    band_norm = (band - band.min()) / (band.max() - band.min() + 1e-8)

    mask_2d = mask[0]
    pred_2d = (pred[0] > 0.5).astype(np.float32)

    TP = (pred_2d * mask_2d)
    FN = ((1 - pred_2d) * mask_2d)
    FP = (pred_2d * (1 - mask_2d))

    overlay = np.zeros((mask_2d.shape[0], mask_2d.shape[1], 3))
    overlay[..., 1] = TP              # green
    overlay[..., 0] = np.clip(FN + FP * 0, 0, 1)  # red for FN
    overlay[FP == 1] = [0, 0, 1]     # blue for FP

    axes[row, 0].imshow(band_norm, cmap="gray")
    axes[row, 1].imshow(mask_2d, cmap="Blues", vmin=0, vmax=1)
    axes[row, 2].imshow(pred_2d, cmap="Blues", vmin=0, vmax=1)
    axes[row, 3].imshow(overlay)
    for ax in axes[row]:
        ax.axis("off")

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="green", label="Correct water (TP)"),
    Patch(facecolor="red",   label="Missed water (FN)"),
    Patch(facecolor="blue",  label="False alarm (FP)"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=3,
           fontsize=10, bbox_to_anchor=(0.5, 0.0))

plt.tight_layout(rect=[0, 0.02, 1, 1])
out_path = os.path.join(PLOT_DIR, "old_model_evaluation.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved -> {out_path}")
