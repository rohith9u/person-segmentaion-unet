"""CLI training entrypoint for Experiment A (Baseline) and Experiment B (Augmented) U-Net."""

import argparse
from pathlib import Path
from typing import Dict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam

from src.augmentation.transforms import get_transforms
from src.datasets.pedestrian_dataset import get_dataloader
from src.models.unet import UNet
from src.training.trainer import UNetTrainer
from src.utils import get_project_root, load_config
from src.utils.reproducibility import set_seed


def run_experiment(
    experiment_type: str = "baseline",
    batch_size: int = 8,
    max_epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 10,
    seed: int = 42,
) -> Dict:
    """Execute a single controlled U-Net training experiment.
    
    Parameters:
        experiment_type: 'baseline' (no augmentation) or 'augmented' (hflip + zoom)
        batch_size: Mini-batch size (identical across experiments)
        max_epochs: Upper bound on training epochs
        lr: Learning rate for Adam optimizer
        patience: Early stopping patience epochs on validation IoU
        seed: Fixed random seed for complete reproducibility
    """
    if experiment_type not in ["baseline", "augmented"]:
        raise ValueError(f"Unknown experiment: {experiment_type}. Choose 'baseline' or 'augmented'.")

    # 1. Enforce strict seed locking
    set_seed(seed)

    # 2. Setup DataLoaders
    # Augmentation applied ONLY to training set if experiment is 'augmented'
    augment_flag = (experiment_type == "augmented")
    train_transform = get_transforms(split="train", augment=augment_flag)
    # Validation is strictly unaugmented across both experiments
    val_transform = None

    train_loader = get_dataloader(split="train", batch_size=batch_size, shuffle=True, transform=train_transform)
    val_loader = get_dataloader(split="val", batch_size=batch_size, shuffle=False, transform=val_transform)

    # 3. Initialize identical architecture
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, out_channels=1)

    # 4. Initialize identical loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    # 5. Initialize Trainer
    trainer = UNetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        max_epochs=max_epochs,
        early_stopping_patience=patience,
        experiment_name=experiment_type,
        seed=seed,
    )

    # 6. Fit and return results
    results = trainer.fit()
    return results


def plot_comparative_summary(baseline_history: list, augmented_history: list, out_path: Path) -> None:
    """Generate a side-by-side comparative graph contrasting Baseline vs. Augmented dynamics."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("Controlled Experiment Comparison: Baseline vs. Augmented U-Net", fontsize=15, fontweight="bold")

    b_epochs = [r["epoch"] for r in baseline_history]
    b_val_iou = [r["val_iou"] for r in baseline_history]
    b_val_loss = [r["val_loss"] for r in baseline_history]

    a_epochs = [r["epoch"] for r in augmented_history]
    a_val_iou = [r["val_iou"] for r in augmented_history]
    a_val_loss = [r["val_loss"] for r in augmented_history]

    # Panel 1: Validation IoU
    axes[0].plot(b_epochs, b_val_iou, label="Baseline Val IoU", color="#1f77b4", linewidth=2.2)
    axes[0].plot(a_epochs, a_val_iou, label="Augmented Val IoU", color="#2ca02c", linewidth=2.2)
    axes[0].set_title("Validation IoU Comparison", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("IoU", fontsize=11)
    axes[0].legend(loc="lower right")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Panel 2: Validation Loss
    axes[1].plot(b_epochs, b_val_loss, label="Baseline Val Loss", color="#1f77b4", linewidth=2.2, linestyle="--")
    axes[1].plot(a_epochs, a_val_loss, label="Augmented Val Loss", color="#2ca02c", linewidth=2.2, linestyle="--")
    axes[1].set_title("Validation BCE Loss Comparison", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("BCE Loss", fontsize=11)
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Comparative training curves saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Train U-Net on Penn-Fudan Pedestrian Dataset.")
    parser.add_argument(
        "--experiment",
        type=str,
        default="both",
        choices=["baseline", "augmented", "both"],
        help="Experiment to run: baseline, augmented, or both",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Mini-batch size (default: 8)")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum epochs (default: 50)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 0.001)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Fixed random seed (default: 42)")
    args = parser.parse_args()

    root = get_project_root()
    curves_dir = root / "results" / "training_curves"
    curves_dir.mkdir(parents=True, exist_ok=True)

    if args.experiment == "both":
        print("\n" + "#" * 75)
        print("   RUNNING FAIR CONTROLLED BENCHMARK: EXPERIMENT A & EXPERIMENT B")
        print("#" * 75)
        res_a = run_experiment("baseline", batch_size=args.batch_size, max_epochs=args.epochs, lr=args.lr, patience=args.patience, seed=args.seed)
        res_b = run_experiment("augmented", batch_size=args.batch_size, max_epochs=args.epochs, lr=args.lr, patience=args.patience, seed=args.seed)

        # Comparative summary
        comp_path = curves_dir / "comparative_experiment_summary.png"
        plot_comparative_summary(res_a["history"], res_b["history"], comp_path)

        print("\n" + "=" * 75)
        print("               FINAL EXPERIMENTAL PROTOCOL SUMMARY")
        print("=" * 75)
        print(f"Baseline U-Net  : Best Val IoU = {res_a['best_val_iou']:.4f} (Epoch {res_a['best_epoch']}) | Duration: {res_a['total_duration_sec']:.1f}s")
        print(f"Augmented U-Net : Best Val IoU = {res_b['best_val_iou']:.4f} (Epoch {res_b['best_epoch']}) | Duration: {res_b['total_duration_sec']:.1f}s")
        print("=" * 75 + "\n")
    else:
        run_experiment(args.experiment, batch_size=args.batch_size, max_epochs=args.epochs, lr=args.lr, patience=args.patience, seed=args.seed)


if __name__ == "__main__":
    main()
