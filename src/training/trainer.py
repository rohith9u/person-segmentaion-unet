"""Controlled training loop, evaluation engine, and checkpoint manager for U-Net."""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.evaluation.metrics import GlobalMetricTracker
from src.utils import get_project_root, resolve_path
from src.utils.reproducibility import get_environment_info


class UNetTrainer:
    """Manages training, validation, early stopping, checkpointing, and metric logging."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optional[torch.optim.Optimizer] = None,
        criterion: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        max_epochs: int = 50,
        early_stopping_patience: int = 10,
        experiment_name: str = "baseline",
        checkpoint_dir: Optional[Path] = None,
        curves_dir: Optional[Path] = None,
        logs_dir: Optional[Path] = None,
        seed: int = 42,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = optimizer or Adam(self.model.parameters(), lr=1e-3)
        self.criterion = criterion or nn.BCEWithLogitsLoss()

        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.experiment_name = experiment_name
        self.seed = seed

        root = get_project_root()
        self.checkpoint_dir = checkpoint_dir or (root / "models")
        self.curves_dir = curves_dir or (root / "results" / "training_curves")
        self.logs_dir = logs_dir or (root / "results" / "experiment_logs")

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.curves_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.history: List[Dict[str, Any]] = []
        self.best_val_iou = -1.0
        self.best_epoch = -1
        self.patience_counter = 0

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train the model for a single epoch."""
        self.model.train()
        tracker = GlobalMetricTracker()

        for batch_idx, (images, masks, _) in enumerate(self.train_loader):
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, masks)

            loss.backward()
            self.optimizer.step()

            tracker.update(logits, masks, loss=loss.item(), batch_size=images.size(0))

        return tracker.compute()

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate the model on validation or test split."""
        self.model.eval()
        tracker = GlobalMetricTracker()

        for images, masks, _ in dataloader:
            images = images.to(self.device)
            masks = masks.to(self.device)

            logits = self.model(images)
            loss = self.criterion(logits, masks)

            tracker.update(logits, masks, loss=loss.item(), batch_size=images.size(0))

        return tracker.compute()

    def fit(self) -> Dict[str, Any]:
        """Execute the complete training run with early stopping and checkpointing."""
        print("\n" + "=" * 75)
        print(f"  STARTING TRAINING: {self.experiment_name.upper()} EXPERIMENT")
        print("=" * 75)
        print(f"Device        : {self.device}")
        print(f"Max Epochs    : {self.max_epochs}")
        print(f"Early Stopping: Patience = {self.early_stopping_patience} on val_iou")
        print(f"Model Params  : {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Train batches : {len(self.train_loader)} (Batch size: {self.train_loader.batch_size})")
        print(f"Val batches   : {len(self.val_loader)}")

        start_time = time.time()
        checkpoint_path = self.checkpoint_dir / f"{self.experiment_name}_unet.pth"

        for epoch in range(1, self.max_epochs + 1):
            epoch_start = time.time()
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.evaluate(self.val_loader)
            epoch_time = time.time() - epoch_start

            current_val_iou = val_metrics["iou"]

            record = {
                "epoch": epoch,
                "epoch_time_sec": round(epoch_time, 2),
                "train_loss": round(train_metrics["loss"], 5),
                "train_iou": round(train_metrics["iou"], 4),
                "train_f1": round(train_metrics["f1"], 4),
                "val_loss": round(val_metrics["loss"], 5),
                "val_iou": round(val_metrics["iou"], 4),
                "val_f1": round(val_metrics["f1"], 4),
                "val_precision": round(val_metrics["precision"], 4),
                "val_recall": round(val_metrics["recall"], 4),
                "is_best": bool(current_val_iou > self.best_val_iou),
            }
            self.history.append(record)

            status_mark = ""
            if record["is_best"]:
                self.best_val_iou = current_val_iou
                self.best_epoch = epoch
                self.patience_counter = 0
                status_mark = " [* BEST CHECKPOINT]"
                # Save checkpoint with extensive metadata
                self._save_checkpoint(checkpoint_path, epoch, val_metrics)
            else:
                self.patience_counter += 1

            print(f"Epoch [{epoch:02d}/{self.max_epochs:02d}] ({epoch_time:.1f}s) | "
                  f"Train Loss: {record['train_loss']:.4f}, IoU: {record['train_iou']:.4f} | "
                  f"Val Loss: {record['val_loss']:.4f}, IoU: {record['val_iou']:.4f}, F1: {record['val_f1']:.4f}"
                  f"{status_mark}", flush=True)

            if self.patience_counter >= self.early_stopping_patience:
                print(f"\n[EARLY STOPPING TRIGGERED] Validation IoU did not improve for {self.early_stopping_patience} consecutive epochs.", flush=True)
                break

        total_duration = time.time() - start_time
        print("\n" + "=" * 75)
        print(f"  TRAINING COMPLETED: {self.experiment_name.upper()}")
        print(f"Total Duration : {total_duration:.1f} seconds ({total_duration/60:.2f} mins)")
        print(f"Best Val IoU   : {self.best_val_iou:.4f} (at epoch {self.best_epoch})")
        print(f"Checkpoint     : {checkpoint_path}")
        print("=" * 75 + "\n")

        # Save curves and log json
        self._plot_curves()
        self._save_experiment_log(total_duration)

        return {
            "experiment_name": self.experiment_name,
            "best_epoch": self.best_epoch,
            "best_val_iou": self.best_val_iou,
            "total_duration_sec": total_duration,
            "checkpoint_path": str(checkpoint_path),
            "history": self.history,
        }

    def _save_checkpoint(self, path: Path, epoch: int, val_metrics: Dict[str, float]) -> None:
        """Save the model checkpoint with complete training context and environment details."""
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_iou": self.best_val_iou,
            "val_metrics": val_metrics,
            "experiment_name": self.experiment_name,
            "seed": self.seed,
            "device": str(self.device),
            "environment": get_environment_info(),
        }
        torch.save(state, path)

    def _plot_curves(self) -> None:
        """Plot and save training and validation loss, IoU, and F1 curves."""
        epochs = [r["epoch"] for r in self.history]
        train_loss = [r["train_loss"] for r in self.history]
        val_loss = [r["val_loss"] for r in self.history]
        train_iou = [r["train_iou"] for r in self.history]
        val_iou = [r["val_iou"] for r in self.history]
        train_f1 = [r["train_f1"] for r in self.history]
        val_f1 = [r["val_f1"] for r in self.history]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Training Dynamics: {self.experiment_name.upper()} U-Net Experiment", fontsize=15, fontweight="bold")

        # Subplot 1: Loss
        axes[0].plot(epochs, train_loss, label="Train Loss", color="#1f77b4", linewidth=2)
        axes[0].plot(epochs, val_loss, label="Val Loss", color="#ff7f0e", linewidth=2, linestyle="--")
        axes[0].set_title("BCEWithLogits Loss", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Epoch", fontsize=11)
        axes[0].set_ylabel("Loss", fontsize=11)
        axes[0].legend(loc="upper right")
        axes[0].grid(True, linestyle="--", alpha=0.5)

        # Subplot 2: IoU
        axes[1].plot(epochs, train_iou, label="Train IoU", color="#2ca02c", linewidth=2)
        axes[1].plot(epochs, val_iou, label="Val IoU", color="#d62728", linewidth=2, linestyle="--")
        axes[1].axvline(self.best_epoch, color="gray", linestyle=":", label=f"Best Ep {self.best_epoch} ({self.best_val_iou:.4f})")
        axes[1].set_title("Intersection over Union (IoU)", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Epoch", fontsize=11)
        axes[1].set_ylabel("IoU", fontsize=11)
        axes[1].legend(loc="lower right")
        axes[1].grid(True, linestyle="--", alpha=0.5)

        # Subplot 3: F1-Score
        axes[2].plot(epochs, train_f1, label="Train F1", color="#9467bd", linewidth=2)
        axes[2].plot(epochs, val_f1, label="Val F1", color="#8c564b", linewidth=2, linestyle="--")
        axes[2].set_title("F1-Score (Dice Coefficient)", fontsize=12, fontweight="bold")
        axes[2].set_xlabel("Epoch", fontsize=11)
        axes[2].set_ylabel("F1", fontsize=11)
        axes[2].legend(loc="lower right")
        axes[2].grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        out_path = self.curves_dir / f"{self.experiment_name}_curves.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[SAVED] Training curves saved to: {out_path}")

    def _save_experiment_log(self, total_duration: float) -> None:
        """Save structured training logs to JSON."""
        log_data = {
            "experiment_name": self.experiment_name,
            "seed": self.seed,
            "best_epoch": self.best_epoch,
            "best_val_iou": self.best_val_iou,
            "total_duration_sec": round(total_duration, 2),
            "environment": get_environment_info(),
            "epochs": self.history,
        }
        out_path = self.logs_dir / f"{self.experiment_name}_log.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2)
        print(f"[SAVED] Experiment log saved to: {out_path}")
