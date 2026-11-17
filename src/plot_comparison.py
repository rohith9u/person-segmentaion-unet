"""Generate publication-ready comparative visualization of Baseline vs Augmented U-Net."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils import get_project_root


def plot_side_by_side_performance():
    root = get_project_root()
    curves_dir = root / "results" / "training_curves"
    curves_dir.mkdir(parents=True, exist_ok=True)

    metrics = ["Validation IoU", "Validation F1", "Precision", "Recall", "Pixel Accuracy"]
    baseline_vals = [0.6342, 0.7762, 0.7207, 0.8409, 0.9284]
    augmented_vals = [0.7409, 0.8512, 0.8459, 0.8566, 0.9558]

    x = np.arange(len(metrics))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [2.2, 1]})
    fig.suptitle("Controlled Experiment Benchmark: Baseline vs. Augmented U-Net (Validation Set)", fontsize=15, fontweight="bold")

    # Bar chart
    rects1 = ax1.bar(x - width/2, baseline_vals, width, label="Baseline (Unaugmented)", color="#2b5c8f", edgecolor="black")
    rects2 = ax1.bar(x + width/2, augmented_vals, width, label="Augmented (HFlip + Zoom)", color="#2ca02c", edgecolor="black")

    ax1.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax1.set_title("Validation Metric Comparison (Higher is Better)", fontsize=13, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=11, fontweight="bold")
    ax1.set_ylim(0.5, 1.05)
    ax1.legend(loc="upper left", fontsize=11)
    ax1.grid(axis="y", linestyle="--", alpha=0.6)

    def autolabel(rects, is_aug=False):
        for idx, rect in enumerate(rects):
            h = rect.get_height()
            delta = f"(+{augmented_vals[idx]-baseline_vals[idx]:.2f})" if is_aug else ""
            txt = f"{h:.4f}\n{delta}".strip()
            ax1.annotate(txt,
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

    autolabel(rects1)
    autolabel(rects2, is_aug=True)

    # Loss comparison
    losses = [0.1865, 0.1165]
    colors = ["#2b5c8f", "#2ca02c"]
    bars_loss = ax2.bar(["Baseline", "Augmented"], losses, color=colors, width=0.5, edgecolor="black")
    ax2.set_ylabel("BCE Loss", fontsize=12, fontweight="bold")
    ax2.set_title("Validation BCE Loss (Lower is Better)", fontsize=13, fontweight="bold")
    ax2.set_ylim(0, 0.25)
    ax2.grid(axis="y", linestyle="--", alpha=0.6)

    for bar, val in zip(bars_loss, losses):
        ax2.annotate(f"{val:.4f}\n(-{(1 - val/0.1865)*100:.1f}%)" if val < 0.1865 else f"{val:.4f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    out_path = curves_dir / "comparative_experiment_summary.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Comparative graph saved to: {out_path}")


if __name__ == "__main__":
    plot_side_by_side_performance()
