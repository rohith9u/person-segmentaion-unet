"""Preprocess Penn-Fudan images and masks: resizing to 256x256, RGB conversion, and binary mask generation."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
from PIL import Image

from src.inspect_dataset import find_mask_for_image
from src.prepare_masks import convert_instance_mask_to_binary
from src.utils import load_config, resolve_path


def preprocess_dataset(target_size: tuple = (256, 256)) -> Dict[str, Any]:
    """Resize all images to 256x256 (RGB, bilinear) and masks to 256x256 (binary {0,1}, nearest-neighbor)."""
    config = load_config()
    raw_dir = resolve_path(config["paths"]["raw_dir"]) / "PennFudanPed"
    images_dir = raw_dir / "PNGImages"
    masks_dir = raw_dir / "PedMasks"

    out_images_dir = resolve_path(config["paths"]["processed_images_dir"])
    out_masks_dir = resolve_path(config["paths"]["processed_masks_dir"])
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_masks_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(list(images_dir.glob("*.png")))
    if not image_files:
        raise FileNotFoundError(f"No PNG images found in {images_dir}. Run download_dataset.py first.")

    print("\n" + "=" * 70)
    print(f"  PREPROCESSING DATASET TO {target_size[0]}x{target_size[1]}")
    print("=" * 70)
    print(f"Target Image Specs: {target_size[0]}x{target_size[1]} | Mode: RGB | Interpolation: Bilinear")
    print(f"Target Mask Specs : {target_size[0]}x{target_size[1]} | Mode: L (uint8) | Interpolation: Nearest | Values: {{0, 1}}")

    records: List[Dict[str, Any]] = []
    total_fg_pixels = 0
    total_pixels_all = 0
    fg_percentages = []

    for idx, img_path in enumerate(image_files):
        mask_path = find_mask_for_image(img_path, masks_dir)
        stem = img_path.stem

        # --- 1. Load and process Image ---
        with Image.open(img_path) as img:
            orig_w, orig_h = img.size
            # Convert to RGB (standard 3 channels)
            img_rgb = img.convert("RGB")
            # Resize using bilinear interpolation
            img_resized = img_rgb.resize(target_size, resample=Image.Resampling.BILINEAR)

        # Save processed image
        save_img_path = out_images_dir / f"{stem}.png"
        img_resized.save(save_img_path, format="PNG")

        # --- 2. Load and process Mask ---
        with Image.open(mask_path) as mask_img:
            m_orig_w, m_orig_h = mask_img.size
            if (orig_w, orig_h) != (m_orig_w, m_orig_h):
                raise ValueError(f"Size mismatch in raw files for {stem}: Image {img.size} vs Mask {mask_img.size}")

            mask_raw_arr = np.array(mask_img)

        # Convert instance mask to binary mask (0 = background, 1 = person)
        binary_mask_orig, stats = convert_instance_mask_to_binary(mask_raw_arr)

        # Resize binary mask using Nearest Neighbor interpolation to maintain strictly {0, 1}
        binary_pil = Image.fromarray(binary_mask_orig, mode="L")
        mask_resized_pil = binary_pil.resize(target_size, resample=Image.Resampling.NEAREST)
        mask_resized_arr = np.array(mask_resized_pil)

        # Assert strict binary values
        unique_mask_values = np.unique(mask_resized_arr).tolist()
        if not set(unique_mask_values).issubset({0, 1}):
            raise ValueError(f"CRITICAL ERROR: Resized mask for {stem} has non-binary values: {unique_mask_values}")

        # Assert dimensions
        assert img_resized.size == target_size, f"Image {stem} has wrong size: {img_resized.size}"
        assert mask_resized_pil.size == target_size, f"Mask {stem} has wrong size: {mask_resized_pil.size}"

        # Save processed mask
        save_mask_path = out_masks_dir / f"{stem}.png"
        mask_resized_pil.save(save_mask_path, format="PNG")

        # Calculate foreground statistics
        img_fg_pixels = int(np.sum(mask_resized_arr == 1))
        img_total_pixels = mask_resized_arr.size
        img_fg_pct = (img_fg_pixels / img_total_pixels) * 100.0

        total_fg_pixels += img_fg_pixels
        total_pixels_all += img_total_pixels
        fg_percentages.append(img_fg_pct)

        records.append({
            "sample_id": stem,
            "orig_size": [orig_w, orig_h],
            "processed_size": list(target_size),
            "instance_count": stats["instance_count"],
            "raw_unique_values": stats["unique_before"],
            "binary_unique_values": unique_mask_values,
            "foreground_pixels": img_fg_pixels,
            "foreground_pct": round(img_fg_pct, 4),
        })

        if (idx + 1) % 34 == 0 or idx == len(image_files) - 1:
            print(f"  Processed {idx + 1}/{len(image_files)} pairs ({((idx + 1)/len(image_files))*100:.1f}%)")

    overall_fg_ratio = total_fg_pixels / total_pixels_all
    summary = {
        "total_processed_samples": len(records),
        "target_size": list(target_size),
        "total_pixels_across_dataset": total_pixels_all,
        "total_foreground_pixels": total_fg_pixels,
        "total_background_pixels": total_pixels_all - total_fg_pixels,
        "overall_foreground_percentage": round(overall_fg_ratio * 100.0, 4),
        "mean_foreground_pct_per_image": round(float(np.mean(fg_percentages)), 4),
        "std_foreground_pct_per_image": round(float(np.std(fg_percentages)), 4),
        "min_foreground_pct": round(float(np.min(fg_percentages)), 4),
        "max_foreground_pct": round(float(np.max(fg_percentages)), 4),
        "samples": records,
    }

    print("\n--- Preprocessing Metrics ---")
    print(f"Successfully processed images: {len(records)} in {out_images_dir}")
    print(f"Successfully processed masks : {len(records)} in {out_masks_dir}")
    print(f"Overall Foreground (Person) Coverage: {summary['overall_foreground_percentage']:.2f}%")
    print(f"Mean Coverage per Image             : {summary['mean_foreground_pct_per_image']:.2f}% ± {summary['std_foreground_pct_per_image']:.2f}%")
    print(f"Min / Max Person Coverage           : {summary['min_foreground_pct']:.2f}% / {summary['max_foreground_pct']:.2f}%")

    # Save summary json
    stats_dir = resolve_path(config["paths"]["statistics_results_dir"])
    stats_dir.mkdir(parents=True, exist_ok=True)
    summary_path = stats_dir / "processed_dataset_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Saved processed dataset summary to: {summary_path}")
    print("=" * 70 + "\n")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Penn-Fudan images and masks.")
    parser.add_argument("--size", type=int, nargs=2, default=[256, 256], help="Target width and height")
    args = parser.parse_args()

    preprocess_dataset(target_size=tuple(args.size))
