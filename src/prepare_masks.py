"""Convert multi-instance pedestrian masks into binary semantic masks (0=background, 1=person)."""

import argparse
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
from PIL import Image

from src.utils import load_config, resolve_path


def convert_instance_mask_to_binary(original_mask: np.ndarray) -> Tuple[np.ndarray, Dict[str, any]]:
    """Convert an instance segmentation mask to a binary semantic segmentation mask.
    
    Formula:
        binary_mask = (original_mask > 0).astype(np.uint8)
        
    Parameters:
        original_mask: 2D numpy array where 0 is background and positive integers are instance IDs.
        
    Returns:
        binary_mask: 2D numpy array with values in {0, 1}
        stats: dictionary containing instance and pixel count information
    """
    if original_mask.ndim != 2:
        raise ValueError(f"Expected 2D mask array, got shape {original_mask.shape}")

    unique_before = np.unique(original_mask)
    nonzero_ids = unique_before[unique_before > 0]

    # Explicit transformation
    binary_mask = (original_mask > 0).astype(np.uint8)

    unique_after = np.unique(binary_mask)
    if not np.all(np.isin(unique_after, [0, 1])):
        raise ValueError(f"Binary mask contains invalid values: {unique_after}")

    total_pixels = original_mask.size
    foreground_pixels = int(np.sum(binary_mask == 1))
    background_pixels = int(np.sum(binary_mask == 0))

    stats = {
        "unique_before": unique_before.tolist(),
        "instance_count": len(nonzero_ids),
        "instance_ids": nonzero_ids.tolist(),
        "unique_after": unique_after.tolist(),
        "total_pixels": total_pixels,
        "foreground_pixels": foreground_pixels,
        "background_pixels": background_pixels,
        "foreground_ratio": round(foreground_pixels / total_pixels, 6) if total_pixels > 0 else 0.0,
    }
    return binary_mask, stats


def prepare_all_masks(verbose: bool = True) -> int:
    """Validate and demonstrate mask conversion across all raw masks in the dataset."""
    config = load_config()
    raw_masks_dir = resolve_path(config["paths"]["raw_dir"]) / "PennFudanPed" / "PedMasks"

    if not raw_masks_dir.exists():
        raise FileNotFoundError(f"Raw masks directory not found at: {raw_masks_dir}")

    mask_paths = sorted(list(raw_masks_dir.glob("*.png")))
    if verbose:
        print(f"\n[INFO] Validating and testing binary mask conversion for {len(mask_paths)} masks...")

    total_instances_counted = 0
    anomalies = []

    for idx, mp in enumerate(mask_paths):
        with Image.open(mp) as img:
            arr = np.array(img)

        binary_mask, stats = convert_instance_mask_to_binary(arr)
        total_instances_counted += stats["instance_count"]

        # Anomaly check: Empty mask (zero foreground)
        if stats["foreground_pixels"] == 0:
            anomalies.append(f"Empty mask warning: {mp.name} has 0 foreground pixels.")

        if verbose and idx < 5:
            print(f"  {mp.name}: Instances={stats['instance_count']} {stats['instance_ids']} | "
                  f"Raw values={stats['unique_before']} -> Binary values={stats['unique_after']} | "
                  f"Person pixel ratio={stats['foreground_ratio'] * 100:.2f}%")

    if verbose:
        print(f"\n[SUMMARY] Total pedestrian instances verified across all masks: {total_instances_counted}")
        if anomalies:
            for warn in anomalies:
                print(f"[WARNING] {warn}")
        else:
            print("[SUCCESS] Zero empty or corrupt masks found. All nonzero instance regions successfully merged.")

    return len(mask_paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare and verify binary semantic masks.")
    parser.add_argument("--verbose", action="store_true", default=True, help="Print detailed conversion stats")
    args = parser.parse_args()

    prepare_all_masks(verbose=args.verbose)
