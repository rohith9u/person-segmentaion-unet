"""Generate comprehensive test prediction visualizations and color-coded error maps."""

from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.datasets.pedestrian_dataset import get_dataloader
from src.models.unet import UNet
from src.utils import get_project_root, resolve_path


def create_error_map(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Create RGB error map: Green=TP, Red=FP, Blue=FN, Dark Gray=TN."""
    h, w = pred.shape
    error_map = np.zeros((h, w, 3), dtype=np.uint8)

    # True Negative: background
    error_map[(pred == 0) & (gt == 0)] = [25, 25, 30]
    # True Positive: correctly detected person (Green)
    error_map[(pred == 1) & (gt == 1)] = [34, 197, 94]
    # False Positive: predicted person, actually background (Red)
    error_map[(pred == 1) & (gt == 0)] = [239, 68, 68]
    # False Negative: missed person (Blue/Cyan)
    error_map[(pred == 0) & (gt == 1)] = [59, 130, 246]

    return error_map


def generate_test_visualizations(
    selected_filenames: List[str] = None,
    threshold: float = 0.5,
) -> List[Path]:
    """Generate 5-panel visualization for representative test images."""
    root = get_project_root()
    out_dir = root / "results" / "test_visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Baseline and Augmented models
    base_ckpt = torch.load(root / "models" / "baseline_unet.pth", weights_only=False, map_location=device)
    aug_ckpt = torch.load(root / "models" / "augmented_unet.pth", weights_only=False, map_location=device)

    base_model = UNet(in_channels=3, out_channels=1).to(device)
    base_model.load_state_dict(base_ckpt["model_state_dict"])
    base_model.eval()

    aug_model = UNet(in_channels=3, out_channels=1).to(device)
    aug_model.load_state_dict(aug_ckpt["model_state_dict"])
    aug_model.eval()

    # Representative diverse selection:
    # 1. FudanPed00066.png -> Highest augmentation gain (+0.4640 IoU)
    # 2. FudanPed00025.png -> Complex crowded scene (6 pedestrians, multi-person)
    # 3. FudanPed00023.png -> Large false-positive suppression (+0.2564 IoU)
    # 4. PennPed00017.png  -> High confidence clear pedestrian on both (>0.86 IoU)
    # 5. PennPed00012.png  -> Difficult failure / regression case (0.2179 IoU)
    # 6. PennPed00009.png  -> Challenging background clutter (large FP)
    # 7. PennPed00038.png  -> Significant boundary recovery (+0.3525 IoU)
    # 8. PennPed00077.png  -> Low IoU on both (<0.43 IoU, difficult lighting/contrast)
    if selected_filenames is None:
        selected_filenames = [
            "FudanPed00066.png",
            "FudanPed00025.png",
            "FudanPed00023.png",
            "PennPed00017.png",
            "PennPed00012.png",
            "PennPed00009.png",
            "PennPed00038.png",
            "PennPed00077.png",
        ]

    images_dir = resolve_path("data/processed/images")
    masks_dir = resolve_path("data/processed/masks")

    saved_paths = []

    print("\n" + "=" * 80)
    print("      GENERATING 5-PANEL TEST VISUALIZATIONS & ERROR MAPS")
    print("=" * 80)

    for idx, fname in enumerate(selected_filenames):
        img_path = images_dir / fname
        mask_path = masks_dir / fname

        with Image.open(img_path) as im:
            rgb_img = im.convert("RGB")
        with Image.open(mask_path) as mk:
            gt_mask_pil = mk.convert("L")

        img_np = np.array(rgb_img)
        gt_np = (np.array(gt_mask_pil) > 0.5).astype(np.uint8)

        # Tensor input
        input_tensor = torch.from_numpy(img_np.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            base_logits = base_model(input_tensor)
            aug_logits = aug_model(input_tensor)

            base_pred = (torch.sigmoid(base_logits)[0, 0] >= threshold).cpu().numpy().astype(np.uint8)
            aug_pred = (torch.sigmoid(aug_logits)[0, 0] >= threshold).cpu().numpy().astype(np.uint8)

        # Metrics for this specific sample
        base_iou = np.sum((base_pred == 1) & (gt_np == 1)) / (np.sum((base_pred == 1) | (gt_np == 1)) + 1e-7)
        aug_iou = np.sum((aug_pred == 1) & (gt_np == 1)) / (np.sum((aug_pred == 1) | (gt_np == 1)) + 1e-7)

        # Error map for augmented model
        error_map_aug = create_error_map(aug_pred, gt_np)
        error_map_base = create_error_map(base_pred, gt_np)

        # 6-Panel Layout:
        # Col 1: Original Image
        # Col 2: Ground Truth
        # Col 3: Baseline Pred
        # Col 4: Augmented Pred
        # Col 5: Baseline Error Map
        # Col 6: Augmented Error Map
        fig, axes = plt.subplots(1, 6, figsize=(22, 4.2))
        fig.suptitle(f"Test Sample: {fname} | Baseline IoU: {base_iou:.4f} vs. Augmented IoU: {aug_iou:.4f} (Delta: {aug_iou - base_iou:+.4f})",
                     fontsize=14, fontweight="bold", y=1.02)

        axes[0].imshow(img_np)
        axes[0].set_title("1. Original RGB", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(gt_np, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("2. Ground Truth", fontsize=11, fontweight="bold")
        axes[1].axis("off")

        axes[2].imshow(base_pred, cmap="gray", vmin=0, vmax=1)
        axes[2].set_title(f"3. Baseline Pred\nIoU: {base_iou:.4f}", fontsize=11, fontweight="bold")
        axes[2].axis("off")

        axes[3].imshow(aug_pred, cmap="gray", vmin=0, vmax=1)
        axes[3].set_title(f"4. Augmented Pred\nIoU: {aug_iou:.4f}", fontsize=11, fontweight="bold")
        axes[3].axis("off")

        axes[4].imshow(error_map_base)
        axes[4].set_title("5. Baseline Error Map\n[G:TP, R:FP, B:FN]", fontsize=10, fontweight="bold")
        axes[4].axis("off")

        axes[5].imshow(error_map_aug)
        axes[5].set_title("6. Augmented Error Map\n[G:TP, R:FP, B:FN]", fontsize=10, fontweight="bold")
        axes[5].axis("off")

        plt.tight_layout()
        stem = Path(fname).stem
        out_path = out_dir / f"pred_{stem}.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        saved_paths.append(out_path)
        print(f"  [SAVED] {out_path.name:<25s} | Base IoU: {base_iou:.4f} -> Aug IoU: {aug_iou:.4f} (Delta {aug_iou-base_iou:+.4f})")

    # Generate an all-in-one 8-sample showcase collage
    _generate_all_in_one_collage(selected_filenames, base_model, aug_model, images_dir, masks_dir, out_dir / "00_test_predictions_overview_grid.png", device, threshold)

    print("=" * 80 + "\n")
    return saved_paths


def _generate_all_in_one_collage(
    filenames: List[str],
    base_model: nn.Module,
    aug_model: nn.Module,
    images_dir: Path,
    masks_dir: Path,
    out_path: Path,
    device: torch.device,
    threshold: float = 0.5,
):
    """Generate a multi-sample grid comparing Original, Ground Truth, Baseline, Augmented, and Error Map."""
    num_samples = len(filenames)
    fig, axes = plt.subplots(num_samples, 5, figsize=(18, 3.2 * num_samples))
    fig.suptitle("Final Test Evaluation: Prediction & Error Map Showcase Across Diverse Test Cases", fontsize=16, fontweight="bold", y=0.995)

    col_headers = [
        "1. Original RGB",
        "2. Ground Truth Mask",
        "3. Baseline Prediction",
        "4. Augmented Prediction",
        "5. Augmented Error Map\n(Green:TP, Red:FP, Blue:FN)",
    ]

    for row_idx, fname in enumerate(filenames):
        img_path = images_dir / fname
        mask_path = masks_dir / fname

        with Image.open(img_path) as im:
            rgb_img = im.convert("RGB")
        with Image.open(mask_path) as mk:
            gt_mask_pil = mk.convert("L")

        img_np = np.array(rgb_img)
        gt_np = (np.array(gt_mask_pil) > 0.5).astype(np.uint8)
        input_tensor = torch.from_numpy(img_np.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            base_logits = base_model(input_tensor)
            aug_logits = aug_model(input_tensor)

            base_pred = (torch.sigmoid(base_logits)[0, 0] >= threshold).cpu().numpy().astype(np.uint8)
            aug_pred = (torch.sigmoid(aug_logits)[0, 0] >= threshold).cpu().numpy().astype(np.uint8)

        base_iou = np.sum((base_pred == 1) & (gt_np == 1)) / (np.sum((base_pred == 1) | (gt_np == 1)) + 1e-7)
        aug_iou = np.sum((aug_pred == 1) & (gt_np == 1)) / (np.sum((aug_pred == 1) | (gt_np == 1)) + 1e-7)
        err_map = create_error_map(aug_pred, gt_np)

        axes[row_idx, 0].imshow(img_np)
        axes[row_idx, 0].set_ylabel(f"{fname}\nBase: {base_iou:.2f}\nAug: {aug_iou:.2f}", fontsize=10, fontweight="bold")
        axes[row_idx, 0].set_xticks([])
        axes[row_idx, 0].set_yticks([])

        axes[row_idx, 1].imshow(gt_np, cmap="gray", vmin=0, vmax=1)
        axes[row_idx, 1].axis("off")

        axes[row_idx, 2].imshow(base_pred, cmap="gray", vmin=0, vmax=1)
        axes[row_idx, 2].axis("off")

        axes[row_idx, 3].imshow(aug_pred, cmap="gray", vmin=0, vmax=1)
        axes[row_idx, 3].axis("off")

        axes[row_idx, 4].imshow(err_map)
        axes[row_idx, 4].axis("off")

        if row_idx == 0:
            for col_idx, text in enumerate(col_headers):
                axes[row_idx, col_idx].set_title(text, fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Complete test predictions overview grid saved to: {out_path}")


if __name__ == "__main__":
    generate_test_visualizations()
