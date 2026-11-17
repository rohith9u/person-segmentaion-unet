"""Generate publication-ready comparative visualization of test evaluation results."""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils import get_project_root


def generate_test_comparison_plot():
    root = get_project_root()
    eval_dir = root / "results" / "test_evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    json_path = eval_dir / "test_metrics.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    b_glob = data["baseline_model"]["global_test_metrics"]
    a_glob = data["augmented_model"]["global_test_metrics"]

    b_samples = [s["iou"] for s in data["baseline_model"]["per_image_samples"]]
    a_samples = [s["iou"] for s in data["augmented_model"]["per_image_samples"]]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), gridspec_kw={"width_ratios": [2.2, 1.2, 1]})
    fig.suptitle("Final Unbiased Test Set Benchmark: Baseline vs. Augmented U-Net (26 Test Images)", fontsize=16, fontweight="bold", y=1.02)

    # -------------------------------------------------------------
    # Panel 1: Global Metrics Comparison Bar Chart
    # -------------------------------------------------------------
    metrics = ["Global IoU", "Global F1", "Precision", "Recall", "Pixel Accuracy"]
    b_vals = [b_glob["iou"], b_glob["f1"], b_glob["precision"], b_glob["recall"], b_glob["accuracy"]]
    a_vals = [a_glob["iou"], a_glob["f1"], a_glob["precision"], a_glob["recall"], a_glob["accuracy"]]

    x = np.arange(len(metrics))
    width = 0.35

    rects1 = axes[0].bar(x - width/2, b_vals, width, label="Baseline (Unaugmented)", color="#2b5c8f", edgecolor="black", linewidth=1.1)
    rects2 = axes[0].bar(x + width/2, a_vals, width, label="Augmented (HFlip + Zoom)", color="#16a34a", edgecolor="black", linewidth=1.1)

    axes[0].set_ylabel("Score", fontsize=12, fontweight="bold")
    axes[0].set_title("Global Pixel-Level Contingency Metrics", fontsize=13, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics, fontsize=11, fontweight="bold")
    axes[0].set_ylim(0.5, 1.05)
    axes[0].legend(loc="upper left", fontsize=11)
    axes[0].grid(axis="y", linestyle="--", alpha=0.6)

    for idx, (r1, r2) in enumerate(zip(rects1, rects2)):
        h1 = r1.get_height()
        h2 = r2.get_height()
        diff = h2 - h1
        axes[0].annotate(f"{h1:.4f}", xy=(r1.get_x() + r1.get_width() / 2, h1),
                         xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
        axes[0].annotate(f"{h2:.4f}\n({diff:+.2f})", xy=(r2.get_x() + r2.get_width() / 2, h2),
                         xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#15803d")

    # -------------------------------------------------------------
    # Panel 2: Per-Image IoU Distribution Boxplot & Scatter
    # -------------------------------------------------------------
    box_data = [b_samples, a_samples]
    box = axes[1].boxplot(box_data, tick_labels=["Baseline", "Augmented"], patch_artist=True, widths=0.45)
    colors = ["#93c5fd", "#86efac"]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.2)
    for median in box["medians"]:
        median.set(color="black", linewidth=2)

    # Overlay jitter points
    for i, pts in enumerate(box_data, start=1):
        jitter = np.random.normal(0, 0.04, size=len(pts))
        axes[1].scatter([i] * len(pts) + jitter, pts, alpha=0.65, color="#1e293b", edgecolors="none", s=30)

    axes[1].set_ylabel("Per-Image IoU", fontsize=12, fontweight="bold")
    axes[1].set_title(f"Per-Image IoU Distribution (N=26)\nMean: {np.mean(b_samples):.3f} vs. {np.mean(a_samples):.3f}", fontsize=13, fontweight="bold")
    axes[1].grid(axis="y", linestyle="--", alpha=0.6)

    # -------------------------------------------------------------
    # Panel 3: Test BCE Loss Comparison
    # -------------------------------------------------------------
    losses = [b_glob["loss"], a_glob["loss"]]
    loss_bars = axes[2].bar(["Baseline", "Augmented"], losses, color=["#2b5c8f", "#16a34a"], width=0.48, edgecolor="black", linewidth=1.1)
    axes[2].set_ylabel("BCE Loss (Raw Logits)", fontsize=12, fontweight="bold")
    axes[2].set_title("Test BCE Loss (Lower is Better)", fontsize=13, fontweight="bold")
    axes[2].set_ylim(0, 0.28)
    axes[2].grid(axis="y", linestyle="--", alpha=0.6)

    for bar, val in zip(loss_bars, losses):
        h = bar.get_height()
        delta_str = f"\n(-{(1 - val/losses[0])*100:.1f}%)" if val < losses[0] else ""
        axes[2].annotate(f"{val:.4f}{delta_str}", xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    out_path = eval_dir / "test_comparison.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Test comparison graphic saved to: {out_path}")


if __name__ == "__main__":
    generate_test_comparison_plot()
