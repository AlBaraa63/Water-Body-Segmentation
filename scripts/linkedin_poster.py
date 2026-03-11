import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "outputs", "plots") + os.sep

imgs = {
    "mask"    : mpimg.imread(BASE + "image_vs_mask.png"),
    "bands"   : mpimg.imread(BASE + "all_bands.png"),
    "norm"    : mpimg.imread(BASE + "before_after_norm.png"),
    "dist"    : mpimg.imread(BASE + "band_distributions.png"),
    "loss"    : mpimg.imread(BASE + "loss_curves.png"),
    "metrics" : mpimg.imread(BASE + "metrics.png"),
    "base"    : mpimg.imread(BASE + "final_evaluation.png"),
    "exp1"    : mpimg.imread(BASE + "exp1" + os.sep + "final_evaluation.png"),
    "exp2"    : mpimg.imread(BASE + "exp2" + os.sep + "final_evaluation.png"),
    "exp3"    : mpimg.imread(BASE + "exp3" + os.sep + "final_evaluation.png"),
}

BG    = "#0d1117"
WHITE = "#e6edf3"
GRAY  = "#8b949e"
BLUE  = "#58a6ff"
ACCENT= "#3fb950"
DIM   = "#30363d"

# 4:5 Aspect ratio for standard portrait LinkedIn post
fig = plt.figure(figsize=(16, 20), facecolor=BG)

gs = GridSpec(
    9, 4,
    figure=fig,
    height_ratios=[1.2, 0.2, 3, 0.2, 3, 0.2, 4.5, 0.2, 2],
    hspace=0.2,
    wspace=0.15,
    top=0.95,
    bottom=0.05,
    left=0.04,
    right=0.96
)

def section_title(row, text):
    ax = fig.add_subplot(gs[row, :])
    ax.axis("off")
    ax.text(0, 0, text, transform=ax.transAxes, va="bottom", 
            color=BLUE, fontsize=14, fontweight="bold", 
            bbox=dict(facecolor=BG, edgecolor='none', pad=0))
    # Line below title
    ax.plot([0, 1], [-0.2, -0.2], transform=ax.transAxes, color=DIM, linewidth=1)

def img_ax(subplot, img, title):
    ax = fig.add_subplot(subplot)
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(title, fontsize=12, color=WHITE, pad=10, fontweight="medium")
    return ax

# ══════════════════════════════════════════════════════
# 0. HEADER
# ══════════════════════════════════════════════════════
header_ax = fig.add_subplot(gs[0, :])
header_ax.axis("off")
header_ax.text(
    0.5, 0.7, "Water Body Segmentation",
    transform=header_ax.transAxes, ha="center", va="center",
    fontsize=32, fontweight="bold", color=WHITE
)
header_ax.text(
    0.5, 0.15, "Deep Learning • Multispectral Imagery • Transfer Learning\nDataset: 306 12-band GeoTIFFs (128x128)",
    transform=header_ax.transAxes, ha="center", va="center",
    fontsize=14, color=GRAY, linespacing=1.6
)

# ══════════════════════════════════════════════════════
# 1. DATA
# ══════════════════════════════════════════════════════
section_title(1, "1. DATA UNDERSTANDING")
img_ax(gs[2, 0], imgs["mask"],  "Satellite vs Mask")
img_ax(gs[2, 1], imgs["bands"], "12 Spectral Bands")
img_ax(gs[2, 2], imgs["norm"],  "Normalization Effect")
img_ax(gs[2, 3], imgs["dist"],  "Pixel Distributions")

# ══════════════════════════════════════════════════════
# 2. TRAINING
# ══════════════════════════════════════════════════════
section_title(3, "2. MODEL TRAINING")
img_ax(gs[4, :2], imgs["loss"],    "Training & Validation Loss")
img_ax(gs[4, 2:], imgs["metrics"], "Evaluation Metrics over Time")


# ══════════════════════════════════════════════════════
# 3. COMPARISON EVALUATION
# ══════════════════════════════════════════════════════
section_title(5, "3. TEST SET PERFORMANCE")
img_ax(gs[6, 0], imgs["base"], "Baseline U-Net\n(IoU: 0.817)")
img_ax(gs[6, 1], imgs["exp1"], "Pre-Trained Encoder\n(IoU: 0.828)")
img_ax(gs[6, 2], imgs["exp2"], "Replace Layer Strategy\n(IoU: 0.842)")
img_ax(gs[6, 3], imgs["exp3"], "MiT-B2 Transformer [BEST]\n(IoU: 0.854)")


# ══════════════════════════════════════════════════════
# 4. RESULTS TABLE
# ══════════════════════════════════════════════════════
section_title(7, "4. FINAL RESULTS")

tab = fig.add_subplot(gs[8, :])
tab.axis("off")

headers = ["Architecture", "IoU", "F1 Score", "Precision", "Recall", "Parameters"]
cx = [0.15, 0.35, 0.48, 0.61, 0.74, 0.88]

# Header row
for h, x in zip(headers, cx):
    tab.text(x, 0.85, h, ha="center", va="center", fontsize=11, color=GRAY, fontweight="bold")

tab.axhline(0.65, color=DIM, linewidth=1)

rows = [
    ("U-Net (Baseline)",        "0.817", "0.896", "0.905", "0.888", "31M", WHITE),
    ("ResNet-50 Encoder",       "0.828", "0.905", "0.958", "0.859", "47M", WHITE),
    ("ResNet-50 w/ Repl. Layer","0.842", "0.913", "0.960", "0.873", "47M", WHITE),
    ("MiT-B2 Transformer",      "0.854", "0.920", "0.939", "0.903", "27M", ACCENT),
]

ys = [0.45, 0.25, 0.05, -0.15]
for (name, iou, f1, prec, rec, params, color), y in zip(rows, ys):
    for val, x in zip([name, iou, f1, prec, rec, params], cx):
        fw = "bold" if color == ACCENT else "normal"
        tab.text(x, y, val, ha="center", va="center", fontsize=11, color=color, fontweight=fw)

# Footer
fig.text(0.5, 0.02, "Stack: Python, PyTorch, Rasterio, segmentation-models-pytorch", 
         ha="center", fontsize=10, color=GRAY)

# plt.savefig(os.path.join(ROOT, "outputs", "plots", "linkedin_poster.png"), dpi=150, facecolor=BG)
plt.show()
