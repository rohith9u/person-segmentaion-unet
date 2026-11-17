"""Visualize synchronized data augmentations: horizontal flips, zoom in, and zoom out."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.augmentation.transforms import JointSegmentationTransform
from src.datasets.pedestrian_dataset import PedestrianDataset
from src.utils import get_project_root, resolve_path


def generate_augmentation_visualizations(out_dir: Path = None) -> Path:
    """Generate visual demonstration of synchronized geometric transformations."""
    root = get_project_root()
    out_dir = out_dir or (root / "results" / "dataset_samples")
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = PedestrianDataset(split="train")
    # Choose a sample with multiple pedestrians or distinctive bounding box
    sample_idx = 4
    img_tensor, mask_tensor, fname = dataset[sample_idx]

    img_path = resolve_path(f"data/processed/images/{fname}")
    mask_path = resolve_path(f"data/processed/masks/{fname}")

    with Image.open(img_path) as im, Image.open(mask_path) as mk:
        orig_img = im.convert("RGB")
        orig_mask = mk.convert("L")

    # Define deterministic transformations for visualization
    tf_flip = JointSegmentationTransform(hflip_prob=1.0, zoom_range=(1.0, 1.0))
    tf_zoom_in = JointSegmentationTransform(hflip_prob=0.0, zoom_range=(1.1, 1.1))
    tf_zoom_out = JointSegmentationTransform(hflip_prob=0.0, zoom_range=(0.9, 0.9))

    flip_img, flip_mask = tf_flip(orig_img, orig_mask)
    zin_img, zin_mask = tf_zoom_in(orig_img, orig_mask)
    zout_img, zout_mask = tf_zoom_out(orig_img, orig_mask)

    stages = [
        ("Original (Unaugmented)", orig_img, orig_mask),
        ("Horizontal Flip (p=1.0)", flip_img, flip_mask),
        ("Zoom In (Scale 1.1x)", zin_img, zin_mask),
        ("Zoom Out (Scale 0.9x + 0-pad)", zout_img, zout_mask),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    fig.suptitle(f"Synchronized Data Augmentation Showcase (Sample: {fname})", fontsize=16, fontweight="bold", y=0.98)

    for col, (title, img, mask) in enumerate(stages):
        img_np = np.array(img)
        mask_np = np.array(mask)

        # Row 0: Image
        axes[0, col].imshow(img_np)
        axes[0, col].set_title(f"{title}\nImage (RGB)", fontsize=11, fontweight="bold")
        axes[0, col].axis("off")

        # Row 1: Mask
        axes[1, col].imshow(mask_np, cmap="gray", vmin=0, vmax=1)
        axes[1, col].set_title(f"Binary Mask {{0, 1}}\nNonzero Pixels: {np.sum(mask_np == 1):,}", fontsize=11)
        axes[1, col].axis("off")

        # Row 2: Overlay
        overlay = img_np.copy().astype(np.float32)
        m_bool = (mask_np == 1)
        overlay[m_bool, 0] = 0.5 * overlay[m_bool, 0] + 0.5 * 0
        overlay[m_bool, 1] = 0.5 * overlay[m_bool, 1] + 0.5 * 255
        overlay[m_bool, 2] = 0.5 * overlay[m_bool, 2] + 0.5 * 80
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        axes[2, col].imshow(overlay)
        axes[2, col].set_title("Overlay (Alignment Verification)", fontsize=11)
        axes[2, col].axis("off")

    plt.tight_layout()
    out_file = out_dir / "augmentation_showcase.png"
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Augmentation visualization saved to: {out_file}")
    return out_file


if __name__ == "__main__":
    generate_augmentation_visualizations()
