"""Inspect raw Penn-Fudan dataset structure, dimensions, formats, and annotations."""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
from PIL import Image

from src.utils import load_config, resolve_path


def find_mask_for_image(img_path: Path, masks_dir: Path) -> Path:
    """Find the corresponding mask file for an image, supporting Penn-Fudan naming conventions."""
    stem = img_path.stem
    # Standard naming: <stem>_mask.png
    candidate1 = masks_dir / f"{stem}_mask.png"
    if candidate1.exists():
        return candidate1
    # Alternative naming: <stem>.png
    candidate2 = masks_dir / f"{stem}.png"
    if candidate2.exists():
        return candidate2
    # Search by stem prefix
    matches = list(masks_dir.glob(f"{stem}*.png"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No mask found for image: {img_path.name} in {masks_dir}")


def inspect_dataset() -> Dict[str, Any]:
    """Inspect the extracted raw dataset and report all properties and statistics."""
    config = load_config()
    raw_dir = resolve_path(config["paths"]["raw_dir"]) / "PennFudanPed"
    images_dir = raw_dir / "PNGImages"
    masks_dir = raw_dir / "PedMasks"
    annotation_dir = raw_dir / "Annotation"

    if not images_dir.exists() or not masks_dir.exists():
        raise FileNotFoundError(f"Raw dataset directories missing in {raw_dir}. Run download_dataset.py first.")

    image_files = sorted(list(images_dir.glob("*.png")))
    mask_files = sorted(list(masks_dir.glob("*.png")))

    print("\n" + "=" * 70)
    print("        PENN-FUDAN PEDESTRIAN DATASET: RAW DATA INSPECTION")
    print("=" * 70)
    print(f"Total Image Files found: {len(image_files)}")
    print(f"Total Mask Files found : {len(mask_files)}")
    print(f"Annotation files found : {len(list(annotation_dir.glob('*.txt')))}")

    if len(image_files) != config["dataset"]["expected_images"]:
        print(f"[WARNING] Expected {config['dataset']['expected_images']} images, but found {len(image_files)}.")

    image_details: List[Dict[str, Any]] = []
    widths = []
    heights = []
    aspect_ratios = []
    formats = set()
    modes = set()
    mask_modes = set()
    total_pedestrian_instances = 0
    instances_per_image: List[int] = []
    unmatched_images: List[str] = []
    unmatched_masks: List[str] = []

    matched_mask_paths = set()

    for idx, img_path in enumerate(image_files):
        try:
            mask_path = find_mask_for_image(img_path, masks_dir)
            matched_mask_paths.add(mask_path.resolve())
        except FileNotFoundError as e:
            unmatched_images.append(str(e))
            continue

        # Inspect image
        with Image.open(img_path) as img:
            w, h = img.size
            fmt = img.format
            mode = img.mode
            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h)
            formats.add(fmt)
            modes.add(mode)

        # Inspect mask
        with Image.open(mask_path) as mask_img:
            mw, mh = mask_img.size
            mask_fmt = mask_img.format
            mask_mode = mask_img.mode
            mask_modes.add(mask_mode)
            mask_arr = np.array(mask_img)

        # Check dimension match
        if (w, h) != (mw, mh):
            raise ValueError(f"Dimension mismatch for {img_path.name}: Image is {(w, h)}, mask is {(mw, mh)}")

        unique_vals = np.unique(mask_arr).tolist()
        # Instance IDs are non-zero values
        instance_ids = [val for val in unique_vals if val > 0]
        num_pedestrians = len(instance_ids)
        total_pedestrian_instances += num_pedestrians
        instances_per_image.append(num_pedestrians)

        image_details.append({
            "image_name": img_path.name,
            "mask_name": mask_path.name,
            "width": w,
            "height": h,
            "aspect_ratio": round(w / h, 4),
            "image_mode": mode,
            "mask_mode": mask_mode,
            "unique_mask_values": unique_vals,
            "num_pedestrians": num_pedestrians,
        })

    # Check for orphan masks
    for mp in mask_files:
        if mp.resolve() not in matched_mask_paths:
            unmatched_masks.append(mp.name)

    # Summary statistics
    summary = {
        "num_images": len(image_files),
        "num_masks": len(mask_files),
        "all_pairs_matched": (len(unmatched_images) == 0 and len(unmatched_masks) == 0),
        "unmatched_images": unmatched_images,
        "unmatched_masks": unmatched_masks,
        "image_formats": list(formats),
        "image_modes": list(modes),
        "mask_modes": list(mask_modes),
        "total_pedestrian_instances": total_pedestrian_instances,
        "min_width": int(np.min(widths)),
        "max_width": int(np.max(widths)),
        "mean_width": float(np.mean(widths)),
        "min_height": int(np.min(heights)),
        "max_height": int(np.max(heights)),
        "mean_height": float(np.mean(heights)),
        "pedestrians_per_image_distribution": {
            str(k): int(v) for k, v in zip(*np.unique(instances_per_image, return_counts=True))
        },
    }

    # Print report
    print("\n--- Image & Mask Formats ---")
    print(f"Image Formats : {summary['image_formats']}")
    print(f"Image Modes   : {summary['image_modes']}")
    print(f"Mask Modes    : {summary['mask_modes']}")

    print("\n--- Dimensions (Width x Height) ---")
    print(f"Widths  : Min={summary['min_width']}, Max={summary['max_width']}, Mean={summary['mean_width']:.1f} px")
    print(f"Heights : Min={summary['min_height']}, Max={summary['max_height']}, Mean={summary['mean_height']:.1f} px")

    print("\n--- Ground Truth Annotations & Pedestrian Instances ---")
    print(f"Total pedestrian instances across all photographs: {total_pedestrian_instances}")
    print(f"Expected instances according to dataset spec      : {config['dataset']['expected_instances']}")
    print(f"Pedestrians per photograph breakdown (Count: Occurrences):")
    for count, occ in sorted(summary["pedestrians_per_image_distribution"].items(), key=lambda x: int(x[0])):
        print(f"  {count} pedestrian(s): in {occ} photographs")

    print("\n--- Pairwise Integrity Check ---")
    if summary["all_pairs_matched"]:
        print("[PASSED] 100% of images and masks matched with identical spatial dimensions.")
    else:
        print(f"[FAILED] Unmatched images: {len(unmatched_images)}, Unmatched masks: {len(unmatched_masks)}")

    # Print sample 5 entries
    print("\nSample 5 Image-Mask Pairs:")
    for item in image_details[:5]:
        print(f"  {item['image_name']} <-> {item['mask_name']} | Size: {item['width']}x{item['height']} | Mask Values: {item['unique_mask_values']} | People: {item['num_pedestrians']}")

    # Save summary json in results/dataset_statistics/
    stats_dir = resolve_path(config["paths"]["statistics_results_dir"])
    stats_dir.mkdir(parents=True, exist_ok=True)
    summary_out = stats_dir / "raw_inspection_summary.json"
    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[INFO] Saved raw inspection summary to: {summary_out}")
    print("=" * 70 + "\n")

    return summary


if __name__ == "__main__":
    inspect_dataset()
