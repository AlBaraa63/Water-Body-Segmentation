"""
scripts/run_all_experiments.py
──────────────────────────────────────────────────────────────────────────────
Runs ALL three transfer-learning experiments back-to-back.

Features:
  - Crash resistant  → one failure won't stop the others
  - Resume support   → continues from checkpoint if interrupted
  - GPU OOM fallback → drops to CPU if CUDA runs out of memory
  - Auto evaluation  → runs test set evaluation after each experiment
  - Full logging     → everything saved to outputs/logs/
  - Final summary    → comparison table of all experiments at the end

Usage:
    python scripts/run_all_experiments.py              # run all
    python scripts/run_all_experiments.py --skip-done  # skip finished ones
    python scripts/run_all_experiments.py --only exp1  # run one only
"""

# ═══════════════════════════════════════════════════════════════════════════
# Imports
# ═══════════════════════════════════════════════════════════════════════════
import os
import sys
import json
import time
import argparse
import traceback
import logging
from datetime import datetime
from pathlib import Path

# Suppress albumentations update ping (hangs when offline)
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

# ── Project root on sys.path ───────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
SRC  = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

import torch
import segmentation_models_pytorch as smp

from config import (
    IMAGE_DIR, MASK_DIR, STATS_PATH,
    CKPT_DIR, RESULTS_DIR, LOG_DIR,
    BATCH_SIZE, IN_CHANNELS, OUT_CHANNELS,
)
from preprocessing import load_stats
from dataset       import get_dataloaders
from train         import train as run_training
from evaluate      import evaluate, evaluate_and_visualize
from models        import UNetPreLayer, UNetReplace


# ═══════════════════════════════════════════════════════════════════════════
# Experiment Configurations
# ═══════════════════════════════════════════════════════════════════════════
EPOCHS = 100

EXPERIMENTS = {
    "exp1": {
        "label"   : "Experiment 1 — UNet + Pre-Layer (ResNet50)",
        "lr"      : 0.001,
        "save_dir": os.path.join(CKPT_DIR, "exp1"),
        "plot_dir": os.path.join(str(ROOT), "outputs", "plots", "exp1"),
    },
    "exp2": {
        "label"   : "Experiment 2 — UNet + Replace First Layer (ResNet50)",
        "lr"      : 0.001,
        "save_dir": os.path.join(CKPT_DIR, "exp2"),
        "plot_dir": os.path.join(str(ROOT), "outputs", "plots", "exp2"),
    },
    "exp3": {
        "label"   : "Experiment 3 — SMP UNet + MiT-B2 (Transformer)",
        "lr"      : 0.0001,
        "save_dir": os.path.join(CKPT_DIR, "exp3"),
        "plot_dir": os.path.join(str(ROOT), "outputs", "plots", "exp3"),
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Logger Setup
# ═══════════════════════════════════════════════════════════════════════════
def setup_logger(name: str, log_file: str) -> logging.Logger:
    """Logger that writes to both console and a file."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ═══════════════════════════════════════════════════════════════════════════
# Model Factory
# ═══════════════════════════════════════════════════════════════════════════
def build_model(exp_key: str, device: torch.device,
                logger: logging.Logger) -> torch.nn.Module:
    """Build the correct model for each experiment."""

    if exp_key == "exp1":
        model = UNetPreLayer(
            in_channels  = IN_CHANNELS,
            out_channels = OUT_CHANNELS
        )

    elif exp_key == "exp2":
        model = UNetReplace(
            in_channels  = IN_CHANNELS,
            out_channels = OUT_CHANNELS
        )

    elif exp_key == "exp3":
        model = smp.Unet(
            encoder_name    = "mit_b2",
            encoder_weights = "imagenet",
            in_channels     = IN_CHANNELS,
            classes         = OUT_CHANNELS,
            activation      = None,
        )

    else:
        raise ValueError(f"Unknown experiment: {exp_key!r}")

    model    = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model ready — {n_params:,} parameters on {device}")
    return model


# ═══════════════════════════════════════════════════════════════════════════
# Checkpoint Resume
# ═══════════════════════════════════════════════════════════════════════════
def maybe_resume(model: torch.nn.Module, save_dir: str,
                 device: torch.device, logger: logging.Logger) -> int:
    """
    Load checkpoint if it exists.
    Returns the epoch number it was saved at, or 0 if no checkpoint.
    """
    ckpt_path = os.path.join(save_dir, "best_model.pth")

    if not os.path.exists(ckpt_path):
        return 0

    try:
        ckpt  = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        epoch = ckpt.get("epoch", 0)
        iou   = ckpt.get("best_iou", float("nan"))
        logger.info(f"  ↺ Resumed from checkpoint — epoch {epoch}, best IoU={iou:.4f}")
        return epoch

    except Exception as e:
        logger.warning(f"  Could not load checkpoint ({e}) — starting fresh.")
        return 0


# ═══════════════════════════════════════════════════════════════════════════
# Single Experiment Runner
# ═══════════════════════════════════════════════════════════════════════════
def run_experiment(
    exp_key     : str,
    cfg         : dict,
    train_loader,
    val_loader,
    test_loader,
    device      : torch.device,
    logger      : logging.Logger,
    skip_done   : bool = False,
) -> dict | None:
    """
    Runs one full experiment:
      1. Build model
      2. Resume from checkpoint if available
      3. Train
      4. Evaluate on test set
      5. Save results to JSON
      6. Save visual predictions

    Returns test metrics dict on success, None if skipped/failed.
    """
    save_dir  = cfg["save_dir"]
    plot_dir  = cfg["plot_dir"]
    best_ckpt = os.path.join(save_dir, "best_model.pth")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # ── Skip if already finished ───────────────────────────────────────────
    if skip_done and os.path.exists(best_ckpt):
        logger.info(f"[{exp_key.upper()}] Checkpoint exists — skipping.")
        return None

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"  STARTING  {cfg['label']}")
    logger.info(f"  Epochs={EPOCHS}  LR={cfg['lr']}  Device={device}")
    logger.info("=" * 70)

    # ── Build model ────────────────────────────────────────────────────────
    model        = build_model(exp_key, device, logger)
    resume_epoch = maybe_resume(model, save_dir, device, logger)

    remaining = EPOCHS - resume_epoch
    if remaining <= 0:
        logger.info(f"  Already trained for {resume_epoch} epochs. Skipping training.")
    else:
        if resume_epoch > 0:
            logger.info(
                f"  Continuing from epoch {resume_epoch} → {EPOCHS} "
                f"({remaining} epochs remaining)."
            )

        # ── Train (with GPU OOM fallback) ──────────────────────────────────
        active_device = device
        t0 = time.time()

        for attempt in range(2):
            try:
                if attempt == 1:
                    logger.warning("  ⚠  CUDA OOM — retrying on CPU.")
                    active_device = torch.device("cpu")
                    model = build_model(exp_key, active_device, logger)
                    maybe_resume(model, save_dir, active_device, logger)

                run_training(
                    model        = model,
                    train_loader = train_loader,
                    val_loader   = val_loader,
                    device       = active_device,
                    epochs       = remaining,
                    lr           = cfg["lr"],
                    save_dir     = save_dir,
                )
                break  # success

            except RuntimeError as e:
                if "out of memory" in str(e).lower() and attempt == 0:
                    torch.cuda.empty_cache()
                    continue
                raise

        elapsed = time.time() - t0
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = int(elapsed % 60)
        logger.info(f"  Training complete in {h}h {m}m {s}s")

    # ── Load best checkpoint for evaluation ───────────────────────────────
    logger.info("  Loading best checkpoint for evaluation...")
    best_model = build_model(exp_key, device, logger)

    ckpt = torch.load(best_ckpt, map_location=device)
    best_model.load_state_dict(ckpt["model_state_dict"])
    best_iou = ckpt.get("best_iou", float("nan"))
    logger.info(f"  Best checkpoint IoU (val) : {best_iou:.4f}")

    # ── Test set evaluation ────────────────────────────────────────────────
    logger.info("  Running final test set evaluation...")
    test_metrics = evaluate(best_model, test_loader, device)

    logger.info("  ── Test Results ──────────────────────────────")
    logger.info(f"  IoU       : {test_metrics['IoU']:.4f}")
    logger.info(f"  F1        : {test_metrics['F1']:.4f}")
    logger.info(f"  Precision : {test_metrics['Precision']:.4f}")
    logger.info(f"  Recall    : {test_metrics['Recall']:.4f}")
    logger.info("  ──────────────────────────────────────────────")

    # ── Save visual predictions ────────────────────────────────────────────
    logger.info("  Saving visual predictions...")
    evaluate_and_visualize(
        best_model, test_loader, device,
        save_dir    = plot_dir,
    )

    # ── Save results to JSON ───────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(RESULTS_DIR, f"{exp_key}_results.json")

    results_data = {
        "experiment"  : exp_key,
        "label"       : cfg["label"],
        "best_val_iou": round(float(best_iou), 4),
        "test_metrics": {k: round(float(v), 4) for k, v in test_metrics.items()},
        "timestamp"   : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=4)

    logger.info(f"  Results saved → {results_path}")
    logger.info(f"  ✅  {exp_key.upper()} complete.")

    return test_metrics


# ═══════════════════════════════════════════════════════════════════════════
# Final Comparison Table
# ═══════════════════════════════════════════════════════════════════════════
def print_comparison(all_results: dict, logger: logging.Logger):
    """Print a clean comparison table of all experiments."""

    # Add baseline from scratch
    baseline = {
        "label"    : "From Scratch (U-Net)",
        "IoU"      : 0.8170,
        "F1"       : 0.8959,
        "Precision": 0.9053,
        "Recall"   : 0.8879,
    }

    logger.info("")
    logger.info("═" * 72)
    logger.info("        FINAL COMPARISON — ALL EXPERIMENTS")
    logger.info("═" * 72)
    logger.info(f"  {'Model':<30} | {'IoU':>6} | {'F1':>6} | {'Precision':>9} | {'Recall':>6}")
    logger.info("  " + "-" * 68)

    # Print baseline first
    logger.info(
        f"  {'From Scratch (U-Net)':<30} | "
        f"{baseline['IoU']:>6.4f} | "
        f"{baseline['F1']:>6.4f} | "
        f"{baseline['Precision']:>9.4f} | "
        f"{baseline['Recall']:>6.4f}"
    )

    # Print each experiment
    best_iou  = baseline["IoU"]
    best_name = "From Scratch (U-Net)"

    for exp_key, metrics in all_results.items():
        if metrics is None:
            continue

        label = EXPERIMENTS[exp_key]["label"].split("—")[-1].strip()
        iou   = metrics["IoU"]
        f1    = metrics["F1"]
        prec  = metrics["Precision"]
        rec   = metrics["Recall"]

        logger.info(
            f"  {label:<30} | "
            f"{iou:>6.4f} | "
            f"{f1:>6.4f} | "
            f"{prec:>9.4f} | "
            f"{rec:>6.4f}"
        )

        if iou > best_iou:
            best_iou  = iou
            best_name = label

    logger.info("═" * 72)
    logger.info(f"  🏆 Best Model : {best_name}")
    logger.info(f"     IoU        : {best_iou:.4f}")
    logger.info("═" * 72)

    # Save comparison to JSON
    os.makedirs(RESULTS_DIR, exist_ok=True)
    comparison_path = os.path.join(RESULTS_DIR, "full_comparison.json")

    comparison_data = {
        "baseline"   : baseline,
        "experiments": {
            k: {
                "label"  : EXPERIMENTS[k]["label"],
                "metrics": {m: round(float(v), 4) for m, v in metrics.items()}
            }
            for k, metrics in all_results.items()
            if metrics is not None
        },
        "winner"     : best_name,
        "best_iou"   : round(best_iou, 4),
        "timestamp"  : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(comparison_path, "w") as f:
        json.dump(comparison_data, f, indent=4)

    logger.info(f"  Full comparison saved → {comparison_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Run all water-body experiments.")
    parser.add_argument(
        "--skip-done", action="store_true",
        help="Skip experiments that already have a checkpoint."
    )
    parser.add_argument(
        "--only", choices=["exp1", "exp2", "exp3"], default=None,
        help="Run only one specific experiment."
    )
    args = parser.parse_args()

    # ── Master logger ──────────────────────────────────────────────────────
    os.makedirs(LOG_DIR, exist_ok=True)
    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    master_log = os.path.join(LOG_DIR, f"run_{ts}.log")
    logger     = setup_logger("master", master_log)

    logger.info("╔══════════════════════════════════════════════════════════════════╗")
    logger.info("║         Water Body Segmentation — All Experiments                ║")
    logger.info(f"║         Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                         ║")
    logger.info("╚══════════════════════════════════════════════════════════════════╝")

    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu     = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"Device : {device}  ({gpu}, {vram_gb:.1f} GB VRAM)")
    else:
        logger.info(f"Device : {device}  (no CUDA — training will be slow)")

    # ── Load data once — shared across all experiments ────────────────────
    logger.info("\nLoading dataset...")
    image_files = sorted(os.listdir(IMAGE_DIR))
    stats       = load_stats(STATS_PATH)

    train_loader, val_loader, test_loader = get_dataloaders(
        IMAGE_DIR, MASK_DIR, image_files, stats, batch_size=BATCH_SIZE
    )

    logger.info(f"  Train batches : {len(train_loader)}")
    logger.info(f"  Val   batches : {len(val_loader)}")
    logger.info(f"  Test  batches : {len(test_loader)}")

    # ── Decide which experiments to run ───────────────────────────────────
    to_run = [args.only] if args.only else list(EXPERIMENTS.keys())

    # ── Run each experiment ────────────────────────────────────────────────
    all_results = {}
    summary     = {}

    for exp_key in to_run:
        cfg        = EXPERIMENTS[exp_key]
        exp_logger = setup_logger(
            exp_key,
            os.path.join(LOG_DIR, f"{exp_key}_{ts}.log")
        )

        try:
            metrics = run_experiment(
                exp_key      = exp_key,
                cfg          = cfg,
                train_loader = train_loader,
                val_loader   = val_loader,
                test_loader  = test_loader,
                device       = device,
                logger       = exp_logger,
                skip_done    = args.skip_done,
            )

            all_results[exp_key] = metrics
            summary[exp_key]     = "skipped" if metrics is None else "success ✅"

        except KeyboardInterrupt:
            logger.warning(f"\n  ⚡  Interrupted during {exp_key.upper()}.")
            logger.warning("     Progress saved — re-run to continue.")
            summary[exp_key] = "interrupted (checkpoint saved)"
            break

        except Exception:
            tb         = traceback.format_exc()
            error_path = os.path.join(LOG_DIR, f"{exp_key}_error_{ts}.txt")

            with open(error_path, "w", encoding="utf-8") as f:
                f.write(tb)

            exp_logger.error(f"  ❌  {exp_key.upper()} FAILED.")
            exp_logger.error(f"  Traceback saved → {error_path}")
            exp_logger.error(tb)

            summary[exp_key]     = f"FAILED ❌ (see {os.path.basename(error_path)})"
            all_results[exp_key] = None
            continue

        finally:
            torch.cuda.empty_cache()

    # ── Final comparison table ─────────────────────────────────────────────
    finished = {k: v for k, v in all_results.items() if v is not None}
    if finished:
        print_comparison(finished, logger)

    # ── Run status summary ─────────────────────────────────────────────────
    logger.info("")
    logger.info("═" * 72)
    logger.info("  RUN STATUS SUMMARY")
    logger.info("═" * 72)
    for k, status in summary.items():
        logger.info(f"  {k.upper()}  →  {status}")
    logger.info("═" * 72)
    logger.info(f"  Logs saved to : {LOG_DIR}")
    logger.info("  Done.")


if __name__ == "__main__":
    main()
