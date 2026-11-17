"""Verify and compare saved checkpoints on validation split."""

import json
from pathlib import Path
from typing import Dict
import torch
import torch.nn as nn

from src.datasets.pedestrian_dataset import get_dataloader
from src.evaluation.metrics import GlobalMetricTracker
from src.models.unet import UNet
from src.utils import get_project_root


def evaluate_checkpoint(ckpt_path: Path, split: str = "val") -> Dict[str, float]:
    """Evaluate a saved model checkpoint on a specific split without any data augmentation."""
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    model = UNet(in_channels=3, out_channels=1)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    criterion = nn.BCEWithLogitsLoss()
    loader = get_dataloader(split=split, batch_size=8, shuffle=False)
    tracker = GlobalMetricTracker()

    with torch.no_grad():
        for imgs, masks, _ in loader:
            logits = model(imgs)
            loss = criterion(logits, masks)
            tracker.update(logits, masks, loss=loss.item(), batch_size=imgs.size(0))

    metrics = tracker.compute()
    metrics["best_epoch"] = ckpt.get("epoch", -1)
    return metrics


def run_comparison():
    root = get_project_root()
    baseline_path = root / "models" / "baseline_unet.pth"
    augmented_path = root / "models" / "augmented_unet.pth"

    results = {}
    if baseline_path.exists():
        results["Baseline U-Net (Exp A)"] = evaluate_checkpoint(baseline_path, split="val")
    if augmented_path.exists():
        results["Augmented U-Net (Exp B)"] = evaluate_checkpoint(augmented_path, split="val")

    print("\n" + "=" * 80)
    print("      CONTROLLED EXPERIMENT COMPARISON (VALIDATION SET — 25 IMAGES)")
    print("=" * 80)
    print(f"{'Metric':<20s} | {'Baseline (Unaugmented)':<25s} | {'Augmented (HFlip + Zoom)':<25s} | {'Delta':<10s}")
    print("-" * 80)

    base = results.get("Baseline U-Net (Exp A)", {})
    aug = results.get("Augmented U-Net (Exp B)", {})

    metrics_to_show = [
        ("Best Epoch", "best_epoch", "{:d}"),
        ("Val Loss", "loss", "{:.4f}"),
        ("Val IoU", "iou", "{:.4f}"),
        ("Val F1-Score", "f1", "{:.4f}"),
        ("Val Precision", "precision", "{:.4f}"),
        ("Val Recall", "recall", "{:.4f}"),
        ("Val Pixel Accuracy", "accuracy", "{:.4f}"),
    ]

    for label, key, fmt in metrics_to_show:
        b_val = base.get(key, 0.0)
        a_val = aug.get(key, 0.0)
        delta_str = ""
        if isinstance(b_val, (int, float)) and isinstance(a_val, (int, float)) and key != "best_epoch":
            diff = a_val - b_val
            delta_str = f"{diff:+.4f}"

        b_str = fmt.format(b_val) if isinstance(b_val, (int, float)) else str(b_val)
        a_str = fmt.format(a_val) if isinstance(a_val, (int, float)) else str(a_val)
        print(f"{label:<20s} | {b_str:<25s} | {a_str:<25s} | {delta_str:<10s}")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_comparison()
