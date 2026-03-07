"""
src/train.py — Full training pipeline with all optimization features.

Features:
  - 3-level banner console output
  - tqdm per batch
  - EarlyStopping with patience
  - Gradient clipping
  - ReduceLROnPlateau
  - Per-epoch timing
  - Auto-save results to JSON
  - Prediction snapshots at key epochs
"""

# ======================== Imports ========================
import os
import sys
import json
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from config import *
from visualize import (
    show_predictions, plot_loss_curves, plot_metrics,
    plot_all_bands, plot_image_vs_mask, plot_band_distributions,
    plot_first_batch
)


# ======================== Helpers ========================
class EarlyStopping:
    """
    Stops training when validation metric stops improving.
    patience = how many epochs to wait before stopping.
    """
    def __init__(self, patience=10, min_delta=0.0001):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_score = None
        self.counter    = 0
        self.should_stop = False

    def __call__(self, val_metric):
        if self.best_score is None:
            self.best_score = val_metric
        elif val_metric < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = val_metric
            self.counter = 0


def calculate_metrics(preds, masks, threshold=0.5):
    """
    Calculate IoU, F1, Precision, Recall for water class.
    preds -> raw model output (before sigmoid)
    masks -> ground truth binary masks
    """
    preds = (torch.sigmoid(preds) > threshold).float()
    masks = masks.float()

    preds = preds.view(-1)
    masks = masks.view(-1)

    TP = (preds * masks).sum()
    FP = (preds * (1 - masks)).sum()
    FN = ((1 - preds) * masks).sum()

    precision = TP / (TP + FP + 1e-8)
    recall    = TP / (TP + FN + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    iou       = TP / (TP + FP + FN + 1e-8)

    return {
        "IoU":       iou.item(),
        "F1":        f1.item(),
        "Precision": precision.item(),
        "Recall":    recall.item()
    }


# ======================== Training Functions ========================
def train_one_epoch(model, loader, optimizer, criterion, device):
    """One full pass through training data with tqdm progress."""
    model.train()
    total_loss = 0

    pbar = tqdm(loader, desc="  Training", leave=False)
    for images, masks in pbar:
        images = images.to(device)
        masks  = masks.to(device)

        optimizer.zero_grad()
        predictions = model(images)
        loss = criterion(predictions, masks)
        loss.backward()

        # Gradient clipping — prevents exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)


def validate_one_epoch(model, loader, criterion, device):
    """One full pass through validation data."""
    model.eval()
    total_loss = 0
    all_metrics = {"IoU": [], "F1": [], "Precision": [], "Recall": []}

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks  = masks.to(device)

            predictions = model(images)
            loss = criterion(predictions, masks)
            total_loss += loss.item()

            batch_metrics = calculate_metrics(predictions, masks)
            for key in all_metrics:
                all_metrics[key].append(batch_metrics[key])

    avg_metrics = {k: sum(v) / len(v) for k, v in all_metrics.items()}
    return total_loss / len(loader), avg_metrics


# ======================== Main Training Loop ========================
def train(model, train_loader, val_loader, device,
          epochs=EPOCHS, lr=LR, save_dir=CKPT_DIR):
    """
    Full training loop with all features:
    - EarlyStopping, gradient clipping, LR scheduler
    - tqdm, banners, timing, JSON saving
    - Prediction snapshots at key epochs
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Setup ─────────────────────────────────────────
    from model import BCEDiceLoss, count_parameters
    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # ReduceLROnPlateau: halves LR only when IoU stagnates (patience=5 epochs).
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5
    )
    early_stop = EarlyStopping(patience=EARLY_STOP_PATIENCE)

    # Track history
    train_losses = []
    val_losses   = []
    metrics_history = {"IoU": [], "F1": [], "Precision": [], "Recall": []}
    best_iou   = 0.0
    best_epoch = 0
    n_params   = count_parameters(model)

    # Epochs to save prediction snapshots
    snapshot_epochs = {0, 10, 20, epochs}

    # ── Level 1 Banner ────────────────────────────────
    print("##############################################################")
    print(f"  Water Body Segmentation — Training")
    print(f"  Device : {device}  |  Seed : {SEED}  |  Epochs : {epochs}")
    print(f"  Params : {n_params:,}  |  LR : {lr}  |  Batch : {BATCH_SIZE}")
    print("##############################################################")

    # ── Pre-training visualization ────────────────────
    print("\n==============================================================")
    print("  PHASE 1 / 3 — Data Exploration")
    print("==============================================================")
    plot_all_bands(train_loader.dataset, sample_idx=0)
    plot_image_vs_mask(train_loader.dataset, sample_idx=0)
    plot_band_distributions(train_loader.dataset)
    plot_first_batch(train_loader)

    # Predictions BEFORE training (epoch 0)
    show_predictions(model, val_loader, device, epoch=0)

    # ── Training ──────────────────────────────────────
    print("\n==============================================================")
    print("  PHASE 2 / 3 — Training")
    print("==============================================================")

    total_start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # Validate
        val_loss, metrics = validate_one_epoch(
            model, val_loader, criterion, device
        )

        # LR scheduler steps on IoU
        scheduler.step(metrics["IoU"])
        current_lr = optimizer.param_groups[0]["lr"]

        # Track history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        for key in metrics_history:
            metrics_history[key].append(metrics[key])

        epoch_time = time.time() - epoch_start

        # ── Level 3 — Per-epoch summary ───────────────
        print(f"  Epoch {epoch:03d}/{epochs} | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"IoU: {metrics['IoU']:.4f} | F1: {metrics['F1']:.4f} | "
              f"Time: {epoch_time:.0f}s | LR: {current_lr:.2e}")

        # Save best model
        if metrics["IoU"] > best_iou:
            best_iou   = metrics["IoU"]
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_iou": best_iou,
            }, os.path.join(save_dir, "best_model.pth"))
            print(f"  [BEST] New best model saved! IoU={best_iou:.4f}")

        # Prediction snapshots
        if epoch in snapshot_epochs:
            show_predictions(model, val_loader, device, epoch)

        # Early stopping check
        early_stop(metrics["IoU"])
        if early_stop.should_stop:
            print(f"\n  [STOP] Early stopping triggered at epoch {epoch} "
                  f"(no improvement for {EARLY_STOP_PATIENCE} epochs)")
            break

    total_time = time.time() - total_start

    # ── Post-training ─────────────────────────────────
    print("\n==============================================================")
    print("  PHASE 3 / 3 — Post-Training Summary")
    print("==============================================================")

    # Final summary banner
    print("--------------------------------------------------------------")
    print(f"  >> Best Epoch   : {best_epoch}")
    print(f"  >> Best IoU     : {best_iou:.4f}")
    print(f"  >> Final F1     : {metrics_history['F1'][-1]:.4f}")
    print(f"  >> Params       : {n_params:,}")
    hours   = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    print(f"  >> Total Time   : {hours}h {minutes}m {seconds}s")
    print(f"  >> Time/Epoch   : {total_time / len(train_losses):.1f}s")
    print(f"  >> Early Stopped: {early_stop.should_stop}")
    print("--------------------------------------------------------------")

    # Save plots
    plot_loss_curves(train_losses, val_losses)
    plot_metrics(metrics_history)

    # Save results to JSON
    results = {
        "best_iou":       best_iou,
        "best_epoch":     best_epoch,
        "early_stopped":  early_stop.should_stop,
        "final_lr":       current_lr,
        "total_time":     f"{hours}h {minutes}m {seconds}s",
        "avg_epoch_time": round(total_time / len(train_losses), 1),
        "params":         n_params,
        "epochs_trained": len(train_losses),
        "final_metrics": {
            "IoU":       metrics_history["IoU"][-1],
            "F1":        metrics_history["F1"][-1],
            "Precision": metrics_history["Precision"][-1],
            "Recall":    metrics_history["Recall"][-1],
        }
    }
    results_path = os.path.join(RESULTS_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  >> Results saved to: {results_path}")

    return train_losses, val_losses, metrics_history
