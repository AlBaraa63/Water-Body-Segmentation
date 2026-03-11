"""
app/inference.py — The kitchen.

One job: take a .tif file, return a water mask as PNG bytes.
Flask does not exist here. Pure Python.
"""

import os
import sys
import io
import numpy as np
import torch
from PIL import Image

# ── Make src/ and project root importable ────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from config import DEVICE, STATS_PATH, CKPT_DIR, IN_CHANNELS, OUT_CHANNELS
from preprocessing import load_tif, replace_nodata, normalize_with_stats, load_stats


# ── Cache variables — loaded once, reused forever ────────────────────
_stats = None
_model = None


def get_stats():
    """Load stats.json once. Return cached version on every call after."""
    global _stats
    if _stats is None:
        _stats = load_stats(STATS_PATH)
    return _stats


def get_model():
    """
    Load the best checkpoint once at startup.
    Tries MiT-B2 first, falls back to base U-Net if SMP unavailable.
    """
    global _model
    if _model is not None:
        return _model

    # ── Load checkpoint first to inspect keys ────
    ckpt_path = os.path.join(CKPT_DIR, "best_model.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"No checkpoint found at {ckpt_path}. "
            "Run training first: python main.py"
        )

    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    keys = list(checkpoint["model_state_dict"].keys())

    # ── Detect architecture from checkpoint keys ──
    # Base U-Net keys start with "enc1", "enc2", "bottleneck" etc.
    # MiT-B2 keys start with "encoder.patch_embed1" etc.
    if keys[0].startswith("enc1"):
        print("  Detected architecture: Base U-Net (from scratch)")
        from models.unet_scratch import UNet
        model = UNet(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS)
    else:
        print("  Detected architecture: MiT-B2 (Transformer U-Net)")
        import segmentation_models_pytorch as smp
        model = smp.Unet(
            encoder_name    = "mit_b2",
            encoder_weights = None,
            in_channels     = IN_CHANNELS,
            classes         = OUT_CHANNELS,
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()       # switch to inference mode — never forget this

    epoch = checkpoint.get("epoch", "?")
    iou   = checkpoint.get("best_iou", "?")
    print(f"  Checkpoint loaded — epoch {epoch}, best IoU: {iou}")

    _model = model
    return _model


def predict_from_bytes(file_bytes: bytes) -> bytes:
    """
    Full pipeline: raw .tif bytes → PNG mask bytes.

    Steps:
        1. Save bytes to temp file (rasterio needs a path)
        2. load_tif → replace_nodata → normalize  (same as training)
        3. numpy → tensor (1, 12, H, W)
        4. model forward pass → sigmoid → threshold
        5. mask → PIL image → PNG bytes
        6. clean up temp file (always, even if crash)
    """
    import tempfile

    # Step 1 — write to temp file
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        # Step 2 — exact same pipeline as training
        image = load_tif(tmp_path)                        # (H, W, 12)
        image = replace_nodata(image, verbose=False)      # fix -9999
        image = normalize_with_stats(image, get_stats())  # z-score

        # Step 3 — numpy to tensor
        tensor = torch.tensor(image, dtype=torch.float32)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)    # (1, 12, H, W)
        tensor = tensor.to(DEVICE)

        # Step 4 — predict
        model = get_model()
        with torch.no_grad():                             # no memory waste
            logits = model(tensor)                        # (1, 1, H, W)
            probs  = torch.sigmoid(logits)
            mask   = (probs > 0.5).float()               # binary

        # Step 5 — tensor to PNG bytes
        mask_np    = mask[0, 0].cpu().numpy()             # (H, W)
        mask_uint8 = (mask_np * 255).astype(np.uint8)     # 0 or 255
        pil_img    = Image.fromarray(mask_uint8, mode="L")

        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)
        return buf.getvalue()

    finally:
        os.remove(tmp_path)    # always clean up — even if step 2-5 crash