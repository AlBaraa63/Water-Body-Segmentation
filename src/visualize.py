"""
src/visualize.py — All plotting functions for the project.

Covers every plot in the visualization checklist:
  1.  all_bands.png
  2.  image_vs_mask.png
  3.  before_after_norm.png
  4.  band_distributions.png
  5.  first_batch.png
  6.  predictions_epoch_NNN.png
  7.  loss_curves.png
  8.  metrics.png
  9.  final_evaluation.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import torch

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import NUM_BANDS, BAND_NAMES, PLOT_DIR, DEVICE, RGB_BANDS


# ======================== 1. all_bands.png ========================
def plot_all_bands(dataset, sample_idx=0):
    """Grid of all 12 bands from one sample + mask."""
    image, mask = dataset[sample_idx]
    image = image.numpy()
    mask  = mask.numpy()[0]

    fig, axes = plt.subplots(3, 5, figsize=(18, 10))
    axes = axes.flatten()

    for b in range(NUM_BANDS):
        axes[b].imshow(image[b], cmap="gray")
        axes[b].set_title(BAND_NAMES[b], fontsize=8)
        axes[b].axis("off")

    axes[12].imshow(mask, cmap="Blues")
    axes[12].set_title("Ground Truth Mask", fontsize=8)
    axes[12].axis("off")

    for i in range(13, len(axes)):
        axes[i].axis("off")

    plt.suptitle(f"Sample #{sample_idx} — All {NUM_BANDS} Bands + Mask", fontsize=13)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "all_bands.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  >> Saved -> {path}")


# ======================== 2. image_vs_mask.png ========================
def plot_image_vs_mask(dataset, sample_idx=0):
    """One image (band 3) next to its water mask."""
    image, mask = dataset[sample_idx]
    img_np = image[2].numpy()  # band 3 (green)
    msk_np = mask[0].numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.imshow(img_np, cmap="gray")
    ax1.set_title("Input Image (Band 3 — Green)", fontsize=11)
    ax1.axis("off")

    ax2.imshow(msk_np, cmap="Blues")
    ax2.set_title("Water Mask (Ground Truth)", fontsize=11)
    ax2.axis("off")

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "image_vs_mask.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  >> Saved -> {path}")


# ======================== 3. before_after_norm.png ========================
def plot_before_after_norm(raw_image, norm_image, bands=[1, 3, 8, 11]):
    """
    4 bands before and after normalization.
    raw_image  : (H, W, 12) — before normalization
    norm_image : (H, W, 12) — after normalization
    bands      : which band indices to show (0-indexed)
    """
    fig, axes = plt.subplots(2, len(bands), figsize=(4 * len(bands), 7))

    for i, b in enumerate(bands):
        axes[0, i].imshow(raw_image[:, :, b], cmap="gray")
        axes[0, i].set_title(f"{BAND_NAMES[b]}\n(Raw)", fontsize=8)
        axes[0, i].axis("off")

        axes[1, i].imshow(norm_image[:, :, b], cmap="gray")
        axes[1, i].set_title(f"{BAND_NAMES[b]}\n(Normalized)", fontsize=8)
        axes[1, i].axis("off")

    plt.suptitle("Before vs After Normalization", fontsize=13)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "before_after_norm.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  >> Saved -> {path}")


# ======================== 4. band_distributions.png ========================
def plot_band_distributions(dataset, bands=[8, 9, 11], n_samples=30):
    """
    Histogram of pixel distributions for specific bands.
    Bands 9, 10, 12 in the checklist -> indices 8, 9, 11 (0-indexed).
    """
    bins = np.linspace(-3, 3, 60)
    accum = {b: np.zeros(len(bins) - 1) for b in bands}

    n = min(n_samples, len(dataset))
    for i in range(n):
        img, _ = dataset[i]
        for b in bands:
            h, _ = np.histogram(img[b].numpy().flatten(), bins=bins)
            accum[b] += h

    centers = 0.5 * (bins[:-1] + bins[1:])
    fig, axes = plt.subplots(1, len(bands), figsize=(5 * len(bands), 4))
    if len(bands) == 1:
        axes = [axes]

    for ax, b in zip(axes, bands):
        ax.bar(centers, accum[b], width=centers[1] - centers[0],
               color="steelblue", alpha=0.85, edgecolor="none")
        ax.set_title(BAND_NAMES[b], fontsize=10)
        ax.set_xlabel("Normalized Value")
        ax.set_ylabel("Count")

    plt.suptitle(f"Pixel Distributions (n={n} samples)", fontsize=13)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "band_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  >> Saved -> {path}")


# ======================== 5. first_batch.png ========================
def plot_first_batch(loader, num_samples=4):
    """4 samples from the first training batch with masks."""
    images, masks = next(iter(loader))

    fig, axes = plt.subplots(num_samples, 2, figsize=(8, num_samples * 3.5))
    cols = ["Input (Band 3)", "Ground Truth Mask"]
    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontsize=12, fontweight="bold")

    for i in range(min(num_samples, images.shape[0])):
        axes[i, 0].imshow(images[i, 2].numpy(), cmap="gray")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(masks[i, 0].numpy(), cmap="Blues")
        axes[i, 1].axis("off")

    plt.suptitle("First Training Batch", fontsize=14)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "first_batch.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  >> Saved -> {path}")


# ======================== 6. predictions_epoch_NNN.png ========================
def show_predictions(model, loader, device, epoch, num_samples=4):
    """
    Side by side: Input | Ground Truth | Prediction.
    Called at specific epochs to track visual progress.
    """
    model.eval()
    images, masks = next(iter(loader))
    images_gpu = images.to(device)

    with torch.no_grad():
        outputs = torch.sigmoid(model(images_gpu))
        preds   = (outputs > 0.5).float()

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, num_samples * 3.5))
    cols = ["Input (Band 3)", "Ground Truth", "Prediction"]
    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontsize=12, fontweight="bold")

    for i in range(min(num_samples, images.shape[0])):
        axes[i, 0].imshow(images[i, 2].cpu().numpy(), cmap="gray")
        axes[i, 1].imshow(masks[i, 0].numpy(), cmap="Blues")
        axes[i, 2].imshow(preds[i, 0].cpu().numpy(), cmap="Blues")
        for ax in axes[i]:
            ax.axis("off")

    plt.suptitle(f"Epoch {epoch} — Predictions", fontsize=14)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"predictions_epoch_{epoch:03d}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  >> Saved -> {path}")


# ======================== 7. loss_curves.png ========================
def plot_loss_curves(train_losses, val_losses):
    """Train vs validation loss over all epochs."""
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(train_losses) + 1)

    ax.plot(epochs, train_losses, "b-o", label="Train Loss", linewidth=2, markersize=3)
    ax.plot(epochs, val_losses,   "r-o", label="Val Loss",   linewidth=2, markersize=3)

    ax.set_title("Training vs Validation Loss", fontsize=14)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "loss_curves.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  >> Saved -> {path}")


# ======================== 8. metrics.png ========================
def plot_metrics(metrics_history):
    """IoU, F1, Precision, Recall — 4-panel subplot."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    names  = ["IoU", "F1", "Precision", "Recall"]
    colors = ["steelblue", "green", "orange", "purple"]

    for i, (name, color) in enumerate(zip(names, colors)):
        ax = axes[i // 2, i % 2]
        values = metrics_history[name]
        epochs = range(1, len(values) + 1)

        ax.plot(epochs, values, color=color, linewidth=2, marker="o", markersize=3)
        ax.set_title(name, fontsize=13)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(name)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0.8, color="red", linestyle="--", alpha=0.5, label="0.8 target")
        ax.legend()

    plt.suptitle("Metrics Over Training", fontsize=15)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "metrics.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  >> Saved -> {path}")


# ======================== 9. final_evaluation.png ========================
def plot_final_evaluation(model, test_loader, device, num_samples=6):
    """
    6 test samples with overlay: green=correct, red=missed, blue=false alarm.
    """
    model.eval()
    images, masks = next(iter(test_loader))
    images_gpu = images.to(device)

    with torch.no_grad():
        raw_out = model(images_gpu)
        probs   = torch.sigmoid(raw_out)
        preds   = (probs > 0.5).float()

    n = min(num_samples, images.shape[0])
    fig, axes = plt.subplots(n, 4, figsize=(16, n * 3.5))

    cols = ["Input (Band 3)", "Ground Truth", "Prediction", "Overlay"]
    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontsize=12, fontweight="bold")

    for i in range(n):
        img  = images[i, 2].numpy()
        gt   = masks[i, 0].numpy()
        pred = preds[i, 0].cpu().numpy()

        # TP/FP/FN overlay
        overlay = np.zeros((*gt.shape, 3))
        overlay[..., 1] = ((pred == 1) & (gt == 1))  # green = correct
        overlay[..., 0] = ((pred == 0) & (gt == 1))  # red   = missed
        overlay[..., 2] = ((pred == 1) & (gt == 0))  # blue  = false alarm

        axes[i, 0].imshow(img,     cmap="gray")
        axes[i, 1].imshow(gt,      cmap="Blues")
        axes[i, 2].imshow(pred,    cmap="Blues")
        axes[i, 3].imshow(overlay)

        # Per-sample IoU label
        tp  = ((pred == 1) & (gt == 1)).sum()
        fp  = ((pred == 1) & (gt == 0)).sum()
        fn  = ((pred == 0) & (gt == 1)).sum()
        iou = tp / (tp + fp + fn + 1e-8)
        axes[i, 0].set_ylabel(f"IoU: {iou:.3f}", fontsize=11)

        for ax in axes[i]:
            ax.axis("off")

    legend = [
        Patch(color="green", label="Correct water (TP)"),
        Patch(color="red",   label="Missed water (FN)"),
        Patch(color="blue",  label="False alarm (FP)"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=11, frameon=True)

    plt.suptitle("Final Test Evaluation", fontsize=15)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "final_evaluation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  >> Saved -> {path}")
