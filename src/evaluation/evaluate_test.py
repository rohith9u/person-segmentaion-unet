"""Unbiased final test evaluation script for Baseline and Augmented U-Net models on the 26 test images."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.datasets.pedestrian_dataset import get_dataloader
from src.evaluation.metrics import calculate_metrics_from_counts, compute_contingency
from src.models.unet import UNet
from src.utils import get_project_root, load_config, resolve_path


def evaluate_model_on_test(
    model_path: Path,
    device: torch.device,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Evaluate a saved model checkpoint on the untouched 26 test images."""
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {model_path}")

    # Load checkpoint
    checkpoint = torch.load(model_path, weights_only=False, map_location=device)
    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    criterion = nn.BCEWithLogitsLoss()
    # Test split is strictly unaugmented, batch_size=1 to enable per-image tracking
    test_loader = get_dataloader(split="test", batch_size=1, shuffle=False)

    global_tp = 0
    global_fp = 0
    global_fn = 0
    global_tn = 0
    total_loss = 0.0

    per_image_results: List[Dict[str, Any]] = []

    with torch.no_grad():
        for images, masks, fnames in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            fname = fnames[0]

            logits = model(images)
            loss = criterion(logits, masks).item()
            total_loss += loss

            tp, fp, fn, tn = compute_contingency(logits, masks, threshold=threshold)
            global_tp += tp
            global_fp += fp
            global_fn += fn
            global_tn += tn

            img_metrics = calculate_metrics_from_counts(tp, fp, fn, tn, eps=eps)
            per_image_results.append({
                "filename": fname,
                "loss": round(loss, 5),
                "iou": round(img_metrics["iou"], 4),
                "f1": round(img_metrics["f1"], 4),
                "precision": round(img_metrics["precision"], 4),
                "recall": round(img_metrics["recall"], 4),
                "accuracy": round(img_metrics["accuracy"], 4),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            })

    num_samples = len(per_image_results)
    avg_loss = total_loss / num_samples

    # Compute Primary Global Aggregated Metrics
    global_metrics = calculate_metrics_from_counts(
        global_tp, global_fp, global_fn, global_tn, eps=eps
    )
    global_metrics["loss"] = float(avg_loss)
    global_metrics["sample_count"] = num_samples
    global_metrics["best_epoch"] = int(checkpoint.get("epoch", -1))
    global_metrics["best_val_iou"] = float(checkpoint.get("best_val_iou", -1.0))

    # Compute Per-Image Distribution Statistics (Supplementary)
    ious = [r["iou"] for r in per_image_results]
    f1s = [r["f1"] for r in per_image_results]

    per_image_summary = {
        "mean_iou": float(np.mean(ious)),
        "std_iou": float(np.std(ious)),
        "median_iou": float(np.median(ious)),
        "min_iou": float(np.min(ious)),
        "max_iou": float(np.max(ious)),
        "mean_f1": float(np.mean(f1s)),
        "std_f1": float(np.std(f1s)),
        "median_f1": float(np.median(f1s)),
        "min_f1": float(np.min(f1s)),
        "max_f1": float(np.max(f1s)),
    }

    results = {
        "global_metrics": global_metrics,
        "per_image_summary": per_image_summary,
    }

    return results, per_image_results


def run_test_evaluation() -> None:
    """Execute test evaluation on both checkpoints and produce JSON and CSV deliverables."""
    root = get_project_root()
    eval_dir = root / "results" / "test_evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = root / "models" / "baseline_unet.pth"
    augmented_path = root / "models" / "augmented_unet.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 80)
    print("      FINAL UNBIASED TEST EVALUATION (PENN-FUDAN TEST SPLIT — 26 IMAGES)")
    print("=" * 80)
    print(f"Device               : {device}")
    print(f"Inference Threshold  : 0.5")
    print(f"Test Split Size      : 26 photographs (completely untouched)")
    print(f"Baseline Checkpoint  : {baseline_path}")
    print(f"Augmented Checkpoint : {augmented_path}")
    print("-" * 80)

    # Evaluate Baseline
    print("Evaluating Baseline U-Net (Experiment A)...")
    base_res, base_per_img = evaluate_model_on_test(baseline_path, device=device)

    # Evaluate Augmented
    print("Evaluating Augmented U-Net (Experiment B)...")
    aug_res, aug_per_img = evaluate_model_on_test(augmented_path, device=device)

    # Print Side-by-Side Summary Table
    print("\n" + "=" * 80)
    print("               FINAL TEST METRICS COMPARATIVE BENCHMARK")
    print("=" * 80)
    print(f"{'Metric':<25s} | {'Baseline (Unaug)':<18s} | {'Augmented (HFlip+Zoom)':<22s} | {'Absolute Delta':<15s} | {'Rel Change':<10s}")
    print("-" * 85)

    metrics_def = [
        ("Test BCE Loss", "loss", "{:.4f}", False),
        ("Global Test IoU", "iou", "{:.4f}", True),
        ("Global Test F1-Score", "f1", "{:.4f}", True),
        ("Global Test Precision", "precision", "{:.4f}", True),
        ("Global Test Recall", "recall", "{:.4f}", True),
        ("Global Pixel Accuracy", "accuracy", "{:.4f}", True),
    ]

    table_data = []

    for label, key, fmt, higher_is_better in metrics_def:
        b_val = base_res["global_metrics"][key]
        a_val = aug_res["global_metrics"][key]
        diff = a_val - b_val
        rel_change = (diff / b_val) * 100.0 if b_val != 0 else 0.0

        diff_str = f"{diff:+.4f}"
        rel_str = f"{rel_change:+.2f}%"

        b_str = fmt.format(b_val)
        a_str = fmt.format(a_val)

        print(f"{label:<25s} | {b_str:<18s} | {a_str:<22s} | {diff_str:<15s} | {rel_str:<10s}")
        table_data.append({
            "Metric": label,
            "Baseline": b_str,
            "Augmented": a_str,
            "Absolute_Delta": diff_str,
            "Relative_Change": rel_str,
        })

    print("-" * 85)
    print("--- Supplementary Per-Image Distributions (Mean ± Std) ---")
    p_b = base_res["per_image_summary"]
    p_a = aug_res["per_image_summary"]
    print(f"{'Per-Image Mean IoU':<25s} | {p_b['mean_iou']:.4f} ± {p_b['std_iou']:.4f}  | {p_a['mean_iou']:.4f} ± {p_a['std_iou']:.4f}   | {p_a['mean_iou']-p_b['mean_iou']:+.4f}")
    print(f"{'Per-Image Mean F1':<25s} | {p_b['mean_f1']:.4f} ± {p_b['std_f1']:.4f}  | {p_a['mean_f1']:.4f} ± {p_a['std_f1']:.4f}   | {p_a['mean_f1']-p_b['mean_f1']:+.4f}")
    print("=" * 80 + "\n")

    # Save test_metrics.json
    metrics_payload = {
        "evaluation_protocol": {
            "test_sample_count": 26,
            "threshold": 0.5,
            "device": str(device),
            "primary_aggregation": "Global pixel contingency (summed TP, FP, FN, TN over all 26 test images)",
            "supplementary_aggregation": "Per-image metric distribution (mean, std, median, min, max)",
            "zero_division_epsilon": 1e-7,
            "test_augmentation": "None (strictly original images)",
        },
        "baseline_model": {
            "checkpoint": str(baseline_path),
            "best_epoch": base_res["global_metrics"]["best_epoch"],
            "best_val_iou": base_res["global_metrics"]["best_val_iou"],
            "global_test_metrics": base_res["global_metrics"],
            "per_image_summary": base_res["per_image_summary"],
            "per_image_samples": base_per_img,
        },
        "augmented_model": {
            "checkpoint": str(augmented_path),
            "best_epoch": aug_res["global_metrics"]["best_epoch"],
            "best_val_iou": aug_res["global_metrics"]["best_val_iou"],
            "global_test_metrics": aug_res["global_metrics"],
            "per_image_summary": aug_res["per_image_summary"],
            "per_image_samples": aug_per_img,
        },
        "comparison_deltas": {
            "iou_delta": round(aug_res["global_metrics"]["iou"] - base_res["global_metrics"]["iou"], 5),
            "f1_delta": round(aug_res["global_metrics"]["f1"] - base_res["global_metrics"]["f1"], 5),
            "precision_delta": round(aug_res["global_metrics"]["precision"] - base_res["global_metrics"]["precision"], 5),
            "recall_delta": round(aug_res["global_metrics"]["recall"] - base_res["global_metrics"]["recall"], 5),
            "loss_delta": round(aug_res["global_metrics"]["loss"] - base_res["global_metrics"]["loss"], 5),
            "accuracy_delta": round(aug_res["global_metrics"]["accuracy"] - base_res["global_metrics"]["accuracy"], 5),
        },
    }

    json_path = eval_dir / "test_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"[SAVED] Test metrics JSON saved to: {json_path}")

    # Save test_comparison.csv
    csv_path = eval_dir / "test_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Metric", "Baseline", "Augmented", "Absolute_Delta", "Relative_Change"])
        writer.writeheader()
        for row in table_data:
            writer.writerow(row)
        # Add per-image summary rows
        writer.writerow({
            "Metric": "Per-Image Mean IoU",
            "Baseline": f"{p_b['mean_iou']:.4f} ± {p_b['std_iou']:.4f}",
            "Augmented": f"{p_a['mean_iou']:.4f} ± {p_a['std_iou']:.4f}",
            "Absolute_Delta": f"{p_a['mean_iou']-p_b['mean_iou']:+.4f}",
            "Relative_Change": f"{((p_a['mean_iou']-p_b['mean_iou'])/p_b['mean_iou'])*100:+.2f}%",
        })
        writer.writerow({
            "Metric": "Per-Image Mean F1",
            "Baseline": f"{p_b['mean_f1']:.4f} ± {p_b['std_f1']:.4f}",
            "Augmented": f"{p_a['mean_f1']:.4f} ± {p_a['std_f1']:.4f}",
            "Absolute_Delta": f"{p_a['mean_f1']-p_b['mean_f1']:+.4f}",
            "Relative_Change": f"{((p_a['mean_f1']-p_b['mean_f1'])/p_b['mean_f1'])*100:+.2f}%",
        })
    print(f"[SAVED] Test comparison CSV saved to: {csv_path}")


if __name__ == "__main__":
    run_test_evaluation()
