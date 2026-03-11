"""
src/visualize.py — Publication-quality plotting functions.

All plots use a unified dark style with a consistent color palette.
Every function saves to PLOT_DIR (or a custom save_dir) at 150 dpi.

Plots produced:
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
import sys
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import NUM_BANDS, BAND_NAMES, PLOT_DIR, RGB_BANDS


# ═══════════════════════════════════════════════════════════════════════════
# Global Style
# ═══════════════════════════════════════════════════════════════════════════

# Dark background palette
BG       = "#0d1117"   # page / figure background
PANEL    = "#161b22"   # axes background
BORDER   = "#30363d"   # spine / tick color
TEXT     = "#e6edf3"   # primary text
SUBTEXT  = "#8b949e"   # secondary text / labels
ACCENT1  = "#58a6ff"   # blue  (train / IoU)
ACCENT2  = "#f78166"   # coral (val / Recall)
ACCENT3  = "#3fb950"   # green (F1)
ACCENT4  = "#d2a8ff"   # lavender (Precision)
GRID     = "#21262d"   # grid lines

# TP / FP / FN colours for overlay
TP_COLOR = "#3fb950"   # green
FN_COLOR = "#f85149"   # red
FP_COLOR = "#58a6ff"   # blue

# Custom water colormap: dark→rich-blue
_WATER_CMAP = LinearSegmentedColormap.from_list(
    "water", ["#0d1117", "#1c4f8a", "#2b7be8", "#58c4f8"], N=256
)

def _apply_style(fig, axes_list):
    """Apply the global dark style to a figure and list of axes."""
    fig.patch.set_facecolor(BG)
    for ax in axes_list:
        ax.set_facecolor(PANEL)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        ax.tick_params(colors=SUBTEXT, labelsize=8)
        ax.xaxis.label.set_color(SUBTEXT)
        ax.yaxis.label.set_color(SUBTEXT)
        ax.title.set_color(TEXT)

def _save(fig, path):
    """Save figure and close cleanly."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  >> Saved -> {path}")


# ═══════════════════════════════════════════════════════════════════════════
# 1. all_bands.png
# ═══════════════════════════════════════════════════════════════════════════

def plot_all_bands(dataset, sample_idx=0):
    """Grid of all 12 spectral bands from one sample + ground truth mask."""
    image, mask = dataset[sample_idx]
    image = image.numpy()
    mask  = mask.numpy()[0]

    fig = plt.figure(figsize=(20, 11), facecolor=BG)
    fig.suptitle(
        f"All {NUM_BANDS} Spectral Bands  ·  Sample #{sample_idx}",
        fontsize=15, color=TEXT, fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(3, 5, figure=fig, hspace=0.35, wspace=0.08)

    for b in range(NUM_BANDS):
        row, col = divmod(b, 5) if b < 10 else (2, b - 10)
        row, col = b // 5, b % 5
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(image[b], cmap="gray", interpolation="nearest")
        ax.set_title(BAND_NAMES[b], fontsize=8.5, color=SUBTEXT, pad=4)
        ax.axis("off")
        ax.set_facecolor(PANEL)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
            spine.set_linewidth(0.5)

    # Mask in the 13th slot (row 2, col 2)
    ax_mask = fig.add_subplot(gs[2, 2])
    ax_mask.imshow(mask, cmap=_WATER_CMAP, interpolation="nearest")
    ax_mask.set_title("Ground Truth Mask", fontsize=8.5, color=ACCENT1, pad=4, fontweight="bold")
    ax_mask.axis("off")
    ax_mask.set_facecolor(PANEL)

    # Hide unused slots
    for col in [3, 4]:
        ax = fig.add_subplot(gs[2, col])
        ax.axis("off")
        ax.set_facecolor(BG)

    _save(fig, os.path.join(PLOT_DIR, "all_bands.png"))


# ═══════════════════════════════════════════════════════════════════════════
# 2. image_vs_mask.png
# ═══════════════════════════════════════════════════════════════════════════

def plot_image_vs_mask(dataset, sample_idx=0):
    """Input satellite image (RGB composite) beside its binary water mask."""
    image, mask = dataset[sample_idx]
    img_np = image.numpy()
    msk_np = mask.numpy()[0]

    # Build a rough RGB composite from bands 3, 2, 1
    def norm(arr):
        lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
        return np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1)

    r = norm(img_np[RGB_BANDS[0]])
    g = norm(img_np[RGB_BANDS[1]])
    b = norm(img_np[RGB_BANDS[2]])
    rgb = np.stack([r, g, b], axis=-1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), facecolor=BG)
    fig.suptitle("Satellite Image vs Ground Truth Water Mask",
                 fontsize=13, color=TEXT, fontweight="bold", y=1.01)

    axes[0].imshow(rgb, interpolation="nearest")
    axes[0].set_title("Input Image  (R=B04, G=B03, B=B02)", fontsize=10, color=SUBTEXT)
    axes[0].axis("off")

    axes[1].imshow(msk_np, cmap=_WATER_CMAP, interpolation="nearest", vmin=0, vmax=1)
    axes[1].set_title("Water Mask  (Ground Truth)", fontsize=10, color=ACCENT1)
    axes[1].axis("off")

    _apply_style(fig, axes)
    for ax in axes:
        ax.set_facecolor(PANEL)

    plt.tight_layout()
    _save(fig, os.path.join(PLOT_DIR, "image_vs_mask.png"))


# ═══════════════════════════════════════════════════════════════════════════
# 3. before_after_norm.png
# ═══════════════════════════════════════════════════════════════════════════

def plot_before_after_norm(raw_image, norm_image, bands=None):
    """
    4 bands before and after Z-score normalization.

    raw_image  : (H, W, 12) float32 — raw pixel values
    norm_image : (H, W, 12) float32 — after normalize_with_stats()
    bands      : list of 0-indexed band indices to display
    """
    if bands is None:
        bands = [1, 3, 8, 11]

    ncols = len(bands)
    fig, axes = plt.subplots(2, ncols, figsize=(4 * ncols, 8), facecolor=BG)
    fig.suptitle("Normalization  ·  Before vs After",
                 fontsize=14, color=TEXT, fontweight="bold", y=1.02)

    row_labels = ["Raw", "Normalized"]
    row_colors = [SUBTEXT, ACCENT1]

    for i, b in enumerate(bands):
        for row in range(2):
            src = raw_image if row == 0 else norm_image
            ax  = axes[row, i]
            ax.imshow(src[:, :, b], cmap="gray", interpolation="nearest")
            ax.axis("off")
            ax.set_facecolor(PANEL)
            for spine in ax.spines.values():
                spine.set_edgecolor(BORDER)
                spine.set_linewidth(0.5)

            if row == 0:
                ax.set_title(BAND_NAMES[b], fontsize=9, color=TEXT, pad=5)
            if i == 0:
                ax.set_ylabel(row_labels[row], fontsize=10,
                              color=row_colors[row], rotation=90, labelpad=6)
                ax.yaxis.label.set_visible(True)

    plt.tight_layout()
    _save(fig, os.path.join(PLOT_DIR, "before_after_norm.png"))


# ═══════════════════════════════════════════════════════════════════════════
# 4. band_distributions.png
# ═══════════════════════════════════════════════════════════════════════════

def plot_band_distributions(dataset, bands=None, n_samples=30):
    """
    Overlapping histogram of pixel value distributions for key bands.
    Bands 9, 10, 12 in Sentinel-2 notation → indices 8, 9, 11 (0-indexed).
    """
    if bands is None:
        bands = [8, 9, 11]

    bins   = np.linspace(-3.5, 3.5, 70)
    accum  = {b: np.zeros(len(bins) - 1) for b in bands}
    n      = min(n_samples, len(dataset))
    colors = [ACCENT1, ACCENT3, ACCENT4]

    for i in range(n):
        img, _ = dataset[i]
        for b in bands:
            h, _ = np.histogram(img[b].numpy().flatten(), bins=bins)
            accum[b] += h

    centers = 0.5 * (bins[:-1] + bins[1:])
    bar_w   = centers[1] - centers[0]

    fig, axes = plt.subplots(1, len(bands), figsize=(5.5 * len(bands), 4.5), facecolor=BG)
    if len(bands) == 1:
        axes = [axes]

    for ax, b, color in zip(axes, bands, colors):
        ax.set_facecolor(PANEL)
        ax.bar(centers, accum[b], width=bar_w, color=color,
               alpha=0.85, edgecolor="none", linewidth=0)
        # Overlay smoothed KDE-like line
        from scipy.ndimage import gaussian_filter1d
        smooth = gaussian_filter1d(accum[b].astype(float), sigma=1.5)
        ax.plot(centers, smooth, color="white", linewidth=1.2, alpha=0.6)

        ax.set_title(BAND_NAMES[b], fontsize=11, color=TEXT, fontweight="bold", pad=8)
        ax.set_xlabel("Normalised value (σ)", fontsize=9, color=SUBTEXT)
        ax.set_ylabel("Pixel count", fontsize=9, color=SUBTEXT)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(colors=SUBTEXT, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)

    fig.suptitle(f"Pixel Distributions after Z-score Normalisation  (n={n} images)",
                 fontsize=13, color=TEXT, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, os.path.join(PLOT_DIR, "band_distributions.png"))


# ═══════════════════════════════════════════════════════════════════════════
# 5. first_batch.png
# ═══════════════════════════════════════════════════════════════════════════

def plot_first_batch(loader, num_samples=4):
    """Grid of the first N training samples: RGB composite + water mask."""
    images, masks = next(iter(loader))

    def norm(arr):
        lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
        return np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1)

    fig, axes = plt.subplots(num_samples, 2,
                              figsize=(8, num_samples * 3.8), facecolor=BG)
    fig.suptitle("First Training Batch", fontsize=14,
                 color=TEXT, fontweight="bold", y=1.01)

    col_titles  = ["Input Image", "Water Mask"]
    col_colors  = [SUBTEXT, ACCENT1]
    for col, (title, color) in enumerate(zip(col_titles, col_colors)):
        axes[0, col].set_title(title, fontsize=11, color=color, fontweight="bold", pad=8)

    n = min(num_samples, images.shape[0])
    for i in range(n):
        img_np = images[i].numpy()
        r = norm(img_np[RGB_BANDS[0]])
        g = norm(img_np[RGB_BANDS[1]])
        b = norm(img_np[RGB_BANDS[2]])
        rgb = np.stack([r, g, b], axis=-1)

        axes[i, 0].imshow(rgb, interpolation="nearest")
        axes[i, 0].axis("off")
        axes[i, 0].set_facecolor(PANEL)

        axes[i, 1].imshow(masks[i, 0].numpy(), cmap=_WATER_CMAP,
                          interpolation="nearest", vmin=0, vmax=1)
        axes[i, 1].axis("off")
        axes[i, 1].set_facecolor(PANEL)

        # Sample number label
        axes[i, 0].set_ylabel(f"#{i+1}", fontsize=9, color=SUBTEXT,
                              rotation=0, labelpad=12, va="center")
        axes[i, 0].yaxis.label.set_visible(True)

    plt.tight_layout()
    _save(fig, os.path.join(PLOT_DIR, "first_batch.png"))


# ═══════════════════════════════════════════════════════════════════════════
# 6. predictions_epoch_NNN.png
# ═══════════════════════════════════════════════════════════════════════════

def show_predictions(model, loader, device, epoch, num_samples=4):
    """
    Side-by-side: Input | Ground Truth | Predicted Mask.
    Called at key epochs to track visual progress across training.
    """
    model.eval()
    images, masks = next(iter(loader))
    images_gpu    = images.to(device)

    with torch.no_grad():
        probs = torch.sigmoid(model(images_gpu))
        preds = (probs > 0.5).float()

    def norm(arr):
        lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
        return np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1)

    n   = min(num_samples, images.shape[0])
    fig, axes = plt.subplots(n, 3, figsize=(13, n * 3.8), facecolor=BG)
    if n == 1:
        axes = axes[np.newaxis, :]

    stage = "Before Training" if epoch == 0 else f"Epoch {epoch}"
    fig.suptitle(f"Predictions  ·  {stage}",
                 fontsize=14, color=TEXT, fontweight="bold", y=1.01)

    col_titles  = ["Input Image", "Ground Truth", "Prediction"]
    col_colors  = [SUBTEXT, ACCENT1, ACCENT3]
    for col, (title, color) in enumerate(zip(col_titles, col_colors)):
        axes[0, col].set_title(title, fontsize=11, color=color, fontweight="bold", pad=8)

    for i in range(n):
        img_np = images[i].numpy()
        r = norm(img_np[RGB_BANDS[0]])
        g = norm(img_np[RGB_BANDS[1]])
        b = norm(img_np[RGB_BANDS[2]])
        rgb = np.stack([r, g, b], axis=-1)

        axes[i, 0].imshow(rgb, interpolation="nearest")
        axes[i, 1].imshow(masks[i, 0].numpy(), cmap=_WATER_CMAP,
                          interpolation="nearest", vmin=0, vmax=1)
        axes[i, 2].imshow(preds[i, 0].cpu().numpy(), cmap=_WATER_CMAP,
                          interpolation="nearest", vmin=0, vmax=1)

        for ax in axes[i]:
            ax.axis("off")
            ax.set_facecolor(PANEL)

    plt.tight_layout()
    _save(fig, os.path.join(PLOT_DIR, f"predictions_epoch_{epoch:03d}.png"))


# ═══════════════════════════════════════════════════════════════════════════
# 7. loss_curves.png
# ═══════════════════════════════════════════════════════════════════════════

def plot_loss_curves(train_losses, val_losses):
    """Training vs validation loss with shaded region and best-epoch marker."""
    epochs = list(range(1, len(train_losses) + 1))
    best_e = int(np.argmin(val_losses)) + 1
    best_v = min(val_losses)

    fig, ax = plt.subplots(figsize=(11, 5), facecolor=BG)
    ax.set_facecolor(PANEL)

    ax.plot(epochs, train_losses, color=ACCENT1, linewidth=2.2,
            label="Train Loss", zorder=3)
    ax.plot(epochs, val_losses,   color=ACCENT2, linewidth=2.2,
            label="Val Loss",   zorder=3)

    # Shaded area between curves
    ax.fill_between(epochs, train_losses, val_losses,
                    alpha=0.08, color=ACCENT4, zorder=2)

    # Best val epoch marker
    ax.axvline(best_e, color=ACCENT3, linestyle="--", linewidth=1.2,
               alpha=0.7, zorder=2)
    ax.annotate(f" Best val\n epoch {best_e}\n loss {best_v:.4f}",
                xy=(best_e, best_v), xytext=(best_e + max(1, len(epochs) * 0.04), best_v),
                color=ACCENT3, fontsize=8, va="center",
                arrowprops=dict(arrowstyle="->", color=ACCENT3, lw=1))

    ax.set_title("Training vs Validation Loss", fontsize=14, color=TEXT,
                 fontweight="bold", pad=12)
    ax.set_xlabel("Epoch", fontsize=10, color=SUBTEXT)
    ax.set_ylabel("Loss (BCE + Dice)", fontsize=10, color=SUBTEXT)
    ax.tick_params(colors=SUBTEXT, labelsize=9)
    ax.grid(color=GRID, linewidth=0.6, zorder=1)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)

    legend = ax.legend(fontsize=10, facecolor=PANEL, edgecolor=BORDER,
                       labelcolor=TEXT, framealpha=0.9)

    plt.tight_layout()
    _save(fig, os.path.join(PLOT_DIR, "loss_curves.png"))


# ═══════════════════════════════════════════════════════════════════════════
# 8. metrics.png
# ═══════════════════════════════════════════════════════════════════════════

def plot_metrics(metrics_history):
    """IoU, F1, Precision, Recall — 4-panel subplot with target line."""
    names  = ["IoU", "F1", "Precision", "Recall"]
    colors = [ACCENT1, ACCENT3, ACCENT4, ACCENT2]
    final  = {k: metrics_history[k][-1] for k in names}

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor=BG)
    fig.suptitle("Validation Metrics over Training", fontsize=15,
                 color=TEXT, fontweight="bold", y=1.02)

    for idx, (name, color) in enumerate(zip(names, colors)):
        ax     = axes[idx // 2, idx % 2]
        values = metrics_history[name]
        epochs = list(range(1, len(values) + 1))

        ax.set_facecolor(PANEL)
        ax.fill_between(epochs, values, alpha=0.15, color=color)
        ax.plot(epochs, values, color=color, linewidth=2.2, zorder=3)

        # 0.8 target line
        ax.axhline(0.8, color=BORDER, linestyle="--", linewidth=1,
                   alpha=0.6, zorder=2, label="Target 0.80")

        # Final value badge
        ax.annotate(f"{final[name]:.3f}",
                    xy=(epochs[-1], final[name]),
                    xytext=(-5, 8), textcoords="offset points",
                    color=color, fontsize=9, fontweight="bold", ha="right")

        ax.set_title(name, fontsize=12, color=color, fontweight="bold", pad=8)
        ax.set_xlabel("Epoch", fontsize=9, color=SUBTEXT)
        ax.set_ylabel(name, fontsize=9, color=SUBTEXT)
        ax.set_ylim(max(0, min(values) - 0.05), min(1.02, max(values) + 0.05))
        ax.tick_params(colors=SUBTEXT, labelsize=8)
        ax.grid(color=GRID, linewidth=0.6, zorder=1)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        ax.legend(fontsize=8, facecolor=PANEL, edgecolor=BORDER,
                  labelcolor=SUBTEXT, framealpha=0.9)

    plt.tight_layout()
    _save(fig, os.path.join(PLOT_DIR, "metrics.png"))


# ═══════════════════════════════════════════════════════════════════════════
# 9. final_evaluation.png
# ═══════════════════════════════════════════════════════════════════════════

def plot_final_evaluation(model, test_loader, device,
                          num_samples=6, save_dir=PLOT_DIR):
    """
    6 test samples — 4 columns: Input | Ground Truth | Prediction | Overlay.
    Overlay: green = TP, red = FN, blue = FP, dark = TN.
    """
    model.eval()
    images, masks = next(iter(test_loader))
    images_gpu    = images.to(device)

    with torch.no_grad():
        probs = torch.sigmoid(model(images_gpu))
        preds = (probs > 0.5).float()

    def norm(arr):
        lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
        return np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1)

    def hex2rgb(h):
        h = h.lstrip("#")
        return tuple(int(h[j:j+2], 16) / 255 for j in (0, 2, 4))

    n = min(num_samples, images.shape[0])

    # Header row is thin; image rows share the rest
    height_ratios = [0.18] + [1.0] * n
    fig = plt.figure(figsize=(17, n * 3.4 + 0.8), facecolor=BG)
    gs  = gridspec.GridSpec(
        n + 1, 4, figure=fig,
        height_ratios=height_ratios,
        hspace=0.04, wspace=0.04,
        top=0.95, bottom=0.07,
        left=0.02, right=0.98,
    )

    fig.suptitle("Final Test Set Evaluation  ·  Best Model (MiT-B2 Transformer)",
                 fontsize=14, color=TEXT, fontweight="bold", y=0.99)

    col_titles  = ["Input Image", "Ground Truth", "Prediction", "Error Overlay"]
    col_colors  = [SUBTEXT, ACCENT1, ACCENT3, TEXT]

    for col, (title, color) in enumerate(zip(col_titles, col_colors)):
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor(BG)
        ax.axis("off")
        ax.text(0.5, 0.5, title, transform=ax.transAxes,
                ha="center", va="center", fontsize=11,
                color=color, fontweight="bold")

    for i in range(n):
        img_np = images[i].numpy()
        gt     = masks[i, 0].numpy()
        pred   = preds[i, 0].cpu().numpy()

        r = norm(img_np[RGB_BANDS[0]])
        g = norm(img_np[RGB_BANDS[1]])
        b = norm(img_np[RGB_BANDS[2]])
        rgb = np.stack([r, g, b], axis=-1)

        # Error overlay
        overlay = np.zeros((*gt.shape, 3), dtype=np.float32)
        tp_mask = (pred == 1) & (gt == 1)
        fn_mask = (pred == 0) & (gt == 1)
        fp_mask = (pred == 1) & (gt == 0)

        overlay[tp_mask] = hex2rgb(TP_COLOR)
        overlay[fn_mask] = hex2rgb(FN_COLOR)
        overlay[fp_mask] = hex2rgb(FP_COLOR)

        # Per-sample IoU
        tp_n  = tp_mask.sum()
        fp_n  = fp_mask.sum()
        fn_n  = fn_mask.sum()
        iou   = tp_n / (tp_n + fp_n + fn_n + 1e-8)

        panels = [rgb, gt, pred, overlay]
        cmaps  = [None, _WATER_CMAP, _WATER_CMAP, None]
        first_ax = None

        for col, (data, cmap) in enumerate(zip(panels, cmaps)):
            ax = fig.add_subplot(gs[i + 1, col])
            ax.set_facecolor(PANEL)
            ax.axis("off")
            if cmap:
                ax.imshow(data, cmap=cmap, interpolation="nearest", vmin=0, vmax=1)
            else:
                ax.imshow(data, interpolation="nearest")

            # Thin border
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(BORDER)
                spine.set_linewidth(0.8)
            if col == 0:
                first_ax = ax

        # IoU badge drawn directly on the input image axes
        first_ax.text(
            0.04, 0.05, f"IoU  {iou:.3f}",
            transform=first_ax.transAxes,
            ha="left", va="bottom", fontsize=8.5,
            color=TEXT, fontweight="bold",
            bbox=dict(facecolor="#0d111799", edgecolor=BORDER,
                      boxstyle="round,pad=0.3", linewidth=0.8)
        )

    # Legend
    legend_patches = [
        Patch(facecolor=TP_COLOR, label="Correct water  (TP)", edgecolor="none"),
        Patch(facecolor=FN_COLOR, label="Missed water  (FN)", edgecolor="none"),
        Patch(facecolor=FP_COLOR, label="False alarm  (FP)",  edgecolor="none"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3,
               fontsize=10, facecolor=PANEL, edgecolor=BORDER,
               labelcolor=TEXT, framealpha=0.95,
               bbox_to_anchor=(0.5, 0.005))

    os.makedirs(save_dir, exist_ok=True)
    _save(fig, os.path.join(save_dir, "final_evaluation.png"))
