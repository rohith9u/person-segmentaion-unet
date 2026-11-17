"""Segmentation metrics (Precision, Recall, F1, IoU) using global pixel contingency aggregation."""

from typing import Dict, Tuple
import torch


def compute_contingency(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> Tuple[int, int, int, int]:
    """Compute true positives, false positives, false negatives, and true negatives.
    
    Parameters:
        logits: Raw model predictions [B, 1, H, W]
        targets: Binary ground-truth masks [B, 1, H, W] in {0, 1}
        threshold: Classification probability threshold (default: 0.5)
        
    Returns:
        (tp, fp, fn, tn): Integer counts of pixel-level contingency table entries.
    """
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).to(torch.uint8)
        targets_bin = (targets >= 0.5).to(torch.uint8)

        tp = int(torch.sum((preds == 1) & (targets_bin == 1)).item())
        fp = int(torch.sum((preds == 1) & (targets_bin == 0)).item())
        fn = int(torch.sum((preds == 0) & (targets_bin == 1)).item())
        tn = int(torch.sum((preds == 0) & (targets_bin == 0)).item())

    return tp, fp, fn, tn


def calculate_metrics_from_counts(
    tp: int,
    fp: int,
    fn: int,
    tn: int,
    eps: float = 1e-7,
) -> Dict[str, float]:
    """Calculate Precision, Recall, F1-Score, and IoU from contingency counts.
    
    Formulae:
        Precision = TP / (TP + FP + eps)
        Recall    = TP / (TP + FN + eps)
        F1        = (2 * TP) / (2 * TP + FP + FN + eps)
        IoU       = TP / (TP + FP + FN + eps)
        Accuracy  = (TP + TN) / (TP + TN + FP + FN + eps)
    """
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    total_pixels = tp + fp + fn + tn
    accuracy = (tp + tn) / (total_pixels + eps) if total_pixels > 0 else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "accuracy": float(accuracy),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


class GlobalMetricTracker:
    """Accumulates pixel-level contingency across all batches and calculates global metrics."""

    def __init__(self, eps: float = 1e-7):
        self.eps = eps
        self.reset()

    def reset(self) -> None:
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        self.total_tn = 0
        self.loss_sum = 0.0
        self.sample_count = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor, loss: float = 0.0, batch_size: int = 1) -> None:
        tp, fp, fn, tn = compute_contingency(logits, targets)
        self.total_tp += tp
        self.total_fp += fp
        self.total_fn += fn
        self.total_tn += tn
        self.loss_sum += loss * batch_size
        self.sample_count += batch_size

    def compute(self) -> Dict[str, float]:
        metrics = calculate_metrics_from_counts(
            self.total_tp,
            self.total_fp,
            self.total_fn,
            self.total_tn,
            eps=self.eps,
        )
        avg_loss = (self.loss_sum / self.sample_count) if self.sample_count > 0 else 0.0
        metrics["loss"] = float(avg_loss)
        return metrics
