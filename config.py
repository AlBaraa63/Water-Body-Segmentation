# config.py
# ─────────────────────────────────────────
# Central control panel for the whole project
# Change settings HERE only
# ─────────────────────────────────────────
import os
import torch

# ── Paths ─────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR   = os.path.join(BASE_DIR, "data", "raw", "images")
MASK_DIR    = os.path.join(BASE_DIR, "data", "raw", "masks")
STATS_PATH  = os.path.join(BASE_DIR, "data", "processed", "stats.json")
CKPT_DIR    = os.path.join(BASE_DIR, "outputs", "checkpoints")
PLOT_DIR    = os.path.join(BASE_DIR, "outputs", "plots")
PRED_DIR    = os.path.join(BASE_DIR, "outputs", "predictions")
RESULTS_DIR = os.path.join(BASE_DIR, "outputs", "results")
LOG_DIR     = os.path.join(BASE_DIR, "outputs", "logs")

# ── Data ──────────────────────────────────
NUM_BANDS   = 12
IMG_SIZE    = (128, 128)
NODATA_VAL  = -9999

BAND_NAMES = [
    "B01 (Coastal)", "B02 (Blue)", "B03 (Green)", "B04 (Red)",
    "B05 (Veg RE)", "B06 (Veg RE)", "B07 (Veg RE)", "B08 (NIR)",
    "B8A (Narrow NIR)", "B09 (Water Vap)", "B11 (SWIR 1)", "B12 (SWIR 2)"
]

# RGB composite → Red (B04=idx3), Green (B03=idx2), Blue (B02=idx1)
RGB_BANDS = [3, 2, 1]

# ── Training ──────────────────────────────
BATCH_SIZE  = 16
EPOCHS      = 150
LR          = 0.001   # best LR for this dataset and Adam optimizer
TRAIN_SPLIT = 0.70
VAL_SPLIT   = 0.15
TEST_SPLIT  = 0.15
SEED        = 42

EARLY_STOP_PATIENCE = 15   # stop if no IoU improvement for N epochs


# ── Model ─────────────────────────────────
IN_CHANNELS  = 12
OUT_CHANNELS = 1

# ── Device ────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Ensure dirs exist ─────────────────────
for d in [CKPT_DIR, PLOT_DIR, PRED_DIR, RESULTS_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)
