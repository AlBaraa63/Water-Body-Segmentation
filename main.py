"""
main.py — Entry point for the whole pipeline.

Usage:
    python main.py                    # train + evaluate
    python main.py --mode train       # train only
    python main.py --mode eval        # evaluate only
"""

import os
import sys
import argparse
import random
import numpy as np
import torch

# Project root is always the working directory when you run `python main.py`
# Add src/ so internal imports (train.py, evaluate.py etc.) can reach each other
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from config import *
from models import UNet, count_parameters
from dataset import WaterSegmentationDataset, split_dataset, get_dataloaders
from preprocessing import compute_global_stats, save_stats, load_stats
from train import train
from evaluate import evaluate


def set_seed(seed=SEED):
    """Lock all random sources for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="Water Segmentation Pipeline")
    parser.add_argument("--mode", choices=["train", "eval", "all"], default="all")
    args = parser.parse_args()

    set_seed()

    # ── Compute or load stats ─────────────────────────
    image_files = sorted([
        f for f in os.listdir(IMAGE_DIR)
        if f.endswith(".tif")
    ])
    print(f"  Found {len(image_files)} images in {IMAGE_DIR}")

    os.makedirs(os.path.dirname(STATS_PATH), exist_ok=True)
    if os.path.exists(STATS_PATH):
        stats = load_stats(STATS_PATH)
    else:
        train_files, _, _ = split_dataset(image_files)
        stats = compute_global_stats(IMAGE_DIR, train_files)
        save_stats(stats, STATS_PATH)

    # ── Build DataLoaders ─────────────────────────────
    train_loader, val_loader, test_loader = get_dataloaders(
        IMAGE_DIR, MASK_DIR, image_files, stats, batch_size=BATCH_SIZE
    )

    # ── Build Model ───────────────────────────────────
    model = UNet(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS).to(DEVICE)
    print(f"  Model parameters: {count_parameters(model):,}")

    # ── Train ─────────────────────────────────────────
    if args.mode in ["train", "all"]:
        train(model, train_loader, val_loader, DEVICE)

    # ── Evaluate ──────────────────────────────────────
    if args.mode in ["eval", "all"]:
        ckpt_path = os.path.join(CKPT_DIR, "best_model.pth")
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=DEVICE)
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"\n  Loaded best model from epoch {checkpoint['epoch']}")
        else:
            print(f"\n  ⚠️  No checkpoint found at {ckpt_path} — using untrained model")

        evaluate(model, test_loader, DEVICE)

    print("\n=== Pipeline Complete ===")


if __name__ == "__main__":
    main()