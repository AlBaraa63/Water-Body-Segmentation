"""
src/evaluate.py — Final evaluation on the held-out test set.

Features:
  - Loads best checkpoint
  - Computes all metrics
  - Generates final_evaluation.png overlay
  - Saves results to JSON
"""

import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from config import *
from train import calculate_metrics
from visualize import plot_final_evaluation


def evaluate(model, test_loader, device, save_dir=RESULTS_DIR):
    """
    Final evaluation on test data.
    Model weights are FROZEN — no learning happens here.
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    all_metrics = {"IoU": [], "F1": [], "Precision": [], "Recall": []}

    print("==============================================================")
    print("  EVALUATION — Test Set")
    print("==============================================================")

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks  = masks.to(device)

            preds   = model(images)
            metrics = calculate_metrics(preds, masks)

            for key in all_metrics:
                all_metrics[key].append(metrics[key])

    final = {k: float(np.mean(v)) for k, v in all_metrics.items()}

    # Banner output
    print("--------------------------------------------------------------")
    print("  FINAL TEST RESULTS")
    print("--------------------------------------------------------------")
    print(f"  >> IoU       : {final['IoU']:.4f}")
    print(f"  >> F1        : {final['F1']:.4f}")
    print(f"  >> Precision : {final['Precision']:.4f}")
    print(f"  >> Recall    : {final['Recall']:.4f}")
    print("--------------------------------------------------------------")

    # Save to JSON
    results_path = os.path.join(save_dir, "test_results.json")
    with open(results_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"  >> Results saved to: {results_path}")

    # Generate final evaluation visualization
    plot_final_evaluation(model, test_loader, device)

    return final

def evaluate_and_visualize(model, test_loader, device, save_dir=RESULTS_DIR):
    """Generate final evaluation visualizations without running full metric computation."""
    os.makedirs(save_dir, exist_ok=True)
    plot_final_evaluation(model, test_loader, device, save_dir=save_dir)
