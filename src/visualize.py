"""Generate dataset visualizations: samples, side-by-side comparisons, overlays, and statistics."""

import argparse
from pathlib import Path
from typing import List
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.inspect_dataset import find_mask_for_image
from src.prepare_masks import convert_instance_mask_to_binary
from src.utils import load_config, resolve_path


def generate_visualizations(num_samples: int = 8, seed: int = 42) -> None:
    """Generate all required sample visualisations and statistical distribution charts."""
    config = load_config()
    raw_dir = resolve_path(config["paths"]["raw_dir"]) / "PennFudanPed"
    raw_images_dir = raw_dir / "PNGImages"
    raw_masks_dir = raw_dir / "PedMasks"

    proc_images_dir = resolve_path(config["paths"]["processed_images_dir"])
    proc_masks_dir = resolve_path(config["paths"]["processed_masks_dir"])

    samples_dir = resolve_path(config["paths"]["sample_results_dir"])
    stats_dir = resolve_path(config["paths"]["statistics_results_dir"])
    samples_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("           GENERATING DATASET VISUALIZATIONS")
    print("=" * 70)

    # -------------------------------------------------------------
    # 1. Select representative samples (diverse aspect ratios & pedestrian counts)
    # -------------------------------------------------------------
    all_raw_images = sorted(list(raw_images_dir.glob("*.png")))
    rng = np.random.RandomState(seed)
    # Pick a deterministic set of 8 images spanning both single and multi-person scenes
    sample_indices = [0, 4, 15, 42, 70, 95, 120, 155]
    if len(all_raw_images) < len(sample_indices):
        sample_indices = list(range(min(num_samples, len(all_raw_images))))
    selected_raw_paths = [all_raw_images[i] for i in sample_indices]

    # --- Plot A: 8 Original Photographs ---
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Original Pedestrian Photographs (Penn-Fudan Pedestrian Dataset)", fontsize=16, fontweight="bold")
    for idx, (ax, img_path) in enumerate(zip(axes.flat, selected_raw_paths)):
        with Image.open(img_path) as img:
            w, h = img.size
            ax.imshow(img.convert("RGB"))
            ax.set_title(f"{img_path.stem}\nRaw: {w}x{h} px", fontsize=11)
            ax.axis("off")
    plt.tight_layout()
    p_orig = samples_dir / "01_original_photographs_8.png"
    plt.savefig(p_orig, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] 8 Original Photographs: {p_orig}")

    # --- Plot B: Corresponding Ground-Truth Binary Masks ---
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Ground-Truth Binary Segmentation Masks (0: Background, 1: Person)", fontsize=16, fontweight="bold")
    for idx, (ax, img_path) in enumerate(zip(axes.flat, selected_raw_paths)):
        mask_path = find_mask_for_image(img_path, raw_masks_dir)
        with Image.open(mask_path) as mask_img:
            arr = np.array(mask_img)
            bin_mask, stats = convert_instance_mask_to_binary(arr)
            ax.imshow(bin_mask, cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"{mask_path.stem}\nInstances: {stats['instance_count']} merged", fontsize=11)
            ax.axis("off")
    plt.tight_layout()
    p_masks = samples_dir / "02_ground_truth_masks_8.png"
    plt.savefig(p_masks, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] 8 Ground-Truth Binary Masks: {p_masks}")

    # --- Plot C: Side-by-Side Comparisons (8 pairs) ---
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle("Side-by-Side Comparison: Original Photograph vs. Binary Semantic Mask", fontsize=16, fontweight="bold", y=0.99)
    for pair_idx, img_path in enumerate(selected_raw_paths):
        row = pair_idx // 2
        col_img = (pair_idx % 2) * 2
        col_mask = col_img + 1

        mask_path = find_mask_for_image(img_path, raw_masks_dir)
        with Image.open(img_path) as img:
            rgb_arr = np.array(img.convert("RGB"))
        with Image.open(mask_path) as mask_img:
            bin_mask, stats = convert_instance_mask_to_binary(np.array(mask_img))

        ax_img = axes[row, col_img]
        ax_mask = axes[row, col_mask]

        ax_img.imshow(rgb_arr)
        ax_img.set_title(f"{img_path.stem} (Image)", fontsize=10, fontweight="bold")
        ax_img.axis("off")

        ax_mask.imshow(bin_mask, cmap="Blues_r", vmin=0, vmax=1)
        ax_mask.set_title(f"Binary Mask ({stats['instance_count']} people)", fontsize=10, fontweight="bold")
        ax_mask.axis("off")

    plt.tight_layout()
    p_side = samples_dir / "03_side_by_side_comparison_8.png"
    plt.savefig(p_side, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Side-by-Side Comparisons: {p_side}")

    # --- Plot D: Image with Segmentation Mask Overlay ---
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Segmentation Overlays (Semi-Transparent Green Mask on Photograph)", fontsize=16, fontweight="bold")
    for idx, (ax, img_path) in enumerate(zip(axes.flat, selected_raw_paths)):
        mask_path = find_mask_for_image(img_path, raw_masks_dir)
        with Image.open(img_path) as img:
            img_rgb = np.array(img.convert("RGB"))
        with Image.open(mask_path) as mask_img:
            bin_mask, stats = convert_instance_mask_to_binary(np.array(mask_img))

        # Create RGBA colored overlay (vibrant lime green: [0, 230, 80])
        overlay = img_rgb.copy().astype(np.float32)
        alpha = config["visualization"]["overlay_alpha"]
        mask_bool = (bin_mask == 1)

        overlay[mask_bool, 0] = (1 - alpha) * overlay[mask_bool, 0] + alpha * 0
        overlay[mask_bool, 1] = (1 - alpha) * overlay[mask_bool, 1] + alpha * 235
        overlay[mask_bool, 2] = (1 - alpha) * overlay[mask_bool, 2] + alpha * 80
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        ax.imshow(overlay)
        ax.set_title(f"{img_path.stem}\n({stats['instance_count']} Pedestrians)", fontsize=11)
        ax.axis("off")

    plt.tight_layout()
    p_overlay = samples_dir / "04_segmentation_overlays_8.png"
    plt.savefig(p_overlay, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Segmentation Overlays: {p_overlay}")

    # -------------------------------------------------------------
    # 2. Dataset Statistical Visualizations across all 170 samples
    # -------------------------------------------------------------
    widths = []
    heights = []
    aspect_ratios = []
    pedestrian_counts = []
    fg_pixel_counts = []
    bg_pixel_counts = []
    fg_percentages = []

    for img_path in all_raw_images:
        mask_path = find_mask_for_image(img_path, raw_masks_dir)
        with Image.open(img_path) as img:
            w, h = img.size
            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h)

        with Image.open(mask_path) as mask_img:
            bin_mask, stats = convert_instance_mask_to_binary(np.array(mask_img))
            pedestrian_counts.append(stats["instance_count"])
            fg_pixel_counts.append(stats["foreground_pixels"])
            bg_pixel_counts.append(stats["background_pixels"])
            fg_percentages.append(stats["foreground_ratio"] * 100.0)

    # --- Plot E: Foreground vs Background Pixel Distribution ---
    fig, (ax_bar, ax_pie) = plt.subplots(1, 2, figsize=(14, 6))
    total_fg = sum(fg_pixel_counts)
    total_bg = sum(bg_pixel_counts)
    total_px = total_fg + total_bg
    fg_pct = (total_fg / total_px) * 100.0
    bg_pct = (total_bg / total_px) * 100.0

    colors = ["#2b5c8f", "#ff7f0e"]
    categories = [f"Background\n(Class 0)", f"Pedestrians\n(Class 1)"]
    pixel_counts_m = [total_bg / 1e6, total_fg / 1e6]

    bars = ax_bar.bar(categories, pixel_counts_m, color=colors, width=0.5, edgecolor="black", linewidth=1.2)
    ax_bar.set_ylabel("Total Pixels (Millions)", fontsize=12, fontweight="bold")
    ax_bar.set_title("Total Pixel Count per Semantic Class", fontsize=13, fontweight="bold")
    ax_bar.grid(axis="y", linestyle="--", alpha=0.6)
    for bar in bars:
        h_val = bar.get_height()
        ax_bar.annotate(f"{h_val:.2f} M px\n({h_val/sum(pixel_counts_m)*100:.1f}%)",
                        xy=(bar.get_x() + bar.get_width() / 2, h_val),
                        xytext=(0, 5), textcoords="offset points",
                        ha="center", va="bottom", fontsize=11, fontweight="bold")

    wedges, texts, autotexts = ax_pie.pie(
        [total_bg, total_fg],
        labels=[f"Background\n({bg_pct:.1f}%)", f"Foreground Person\n({fg_pct:.1f}%)"],
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        explode=(0, 0.08),
        textprops={"fontsize": 11, "fontweight": "bold"},
        wedgeprops={"edgecolor": "black", "linewidth": 1.2}
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontweight("bold")
    ax_pie.set_title("Foreground vs. Background Class Balance", fontsize=13, fontweight="bold")

    plt.suptitle("Semantic Segmentation Pixel Distribution Across All 170 Images", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    p_pixels = stats_dir / "01_class_pixel_distribution.png"
    plt.savefig(p_pixels, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Class Pixel Distribution: {p_pixels}")

    # --- Plot F: Number of Pedestrians per Photograph ---
    fig, ax = plt.subplots(figsize=(10, 6))
    unique_counts, occs = np.unique(pedestrian_counts, return_counts=True)
    total_people = sum(c * o for c, o in zip(unique_counts, occs))

    bars = ax.bar(unique_counts, occs, color="#2ca02c", edgecolor="black", linewidth=1.2, width=0.55)
    ax.set_xlabel("Number of Pedestrians in Photograph", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Photographs", fontsize=12, fontweight="bold")
    ax.set_title(f"Pedestrian Density Distribution (Total Pedestrians: {total_people} across 170 Images)",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(unique_counts)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    for bar, occ in zip(bars, occs):
        ax.annotate(f"{occ} images\n({(occ/len(all_raw_images))*100:.1f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    p_ped_counts = stats_dir / "02_pedestrians_per_photograph.png"
    plt.savefig(p_ped_counts, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Pedestrians Per Photograph: {p_ped_counts}")

    # --- Plot G: Raw Image Dimension Distributions Before Preprocessing ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Raw Image Spatial Dimensions & Aspect Ratios (Before 256x256 Standardization)",
                 fontsize=15, fontweight="bold", y=1.03)

    # Subplot 1: Width Distribution
    ax_w = axes[0]
    ax_w.hist(widths, bins=15, color="#1f77b4", edgecolor="black", alpha=0.75)
    ax_w.axvline(np.mean(widths), color="red", linestyle="--", linewidth=1.5, label=f"Mean: {np.mean(widths):.1f} px")
    ax_w.set_title("Image Width Distribution", fontsize=12, fontweight="bold")
    ax_w.set_xlabel("Width (pixels)", fontsize=11)
    ax_w.set_ylabel("Frequency", fontsize=11)
    ax_w.legend(loc="upper right")
    ax_w.grid(axis="y", linestyle="--", alpha=0.5)

    # Subplot 2: Height Distribution
    ax_h = axes[1]
    ax_h.hist(heights, bins=15, color="#ff7f0e", edgecolor="black", alpha=0.75)
    ax_h.axvline(np.mean(heights), color="red", linestyle="--", linewidth=1.5, label=f"Mean: {np.mean(heights):.1f} px")
    ax_h.set_title("Image Height Distribution", fontsize=12, fontweight="bold")
    ax_h.set_xlabel("Height (pixels)", fontsize=11)
    ax_h.set_ylabel("Frequency", fontsize=11)
    ax_h.legend(loc="upper right")
    ax_h.grid(axis="y", linestyle="--", alpha=0.5)

    # Subplot 3: Width vs Height Scatter & Aspect Ratio
    ax_s = axes[2]
    scatter = ax_s.scatter(widths, heights, c=pedestrian_counts, cmap="viridis", s=45, edgecolors="black", linewidth=0.6)
    cbar = plt.colorbar(scatter, ax=ax_s)
    cbar.set_label("Pedestrians in Image", fontsize=10)
    ax_s.set_title("Width vs. Height Scatter", fontsize=12, fontweight="bold")
    ax_s.set_xlabel("Width (pixels)", fontsize=11)
    ax_s.set_ylabel("Height (pixels)", fontsize=11)
    ax_s.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    p_dims = stats_dir / "03_raw_dimension_distribution.png"
    plt.savefig(p_dims, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Raw Dimension Distribution: {p_dims}")

    # --- Plot H: Foreground Coverage Percentage Distribution ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(fg_percentages, bins=20, color="#9467bd", edgecolor="black", alpha=0.8)
    ax.axvline(np.mean(fg_percentages), color="red", linestyle="--", linewidth=1.8,
               label=f"Mean: {np.mean(fg_percentages):.2f}%")
    ax.axvline(np.median(fg_percentages), color="green", linestyle=":", linewidth=1.8,
               label=f"Median: {np.median(fg_percentages):.2f}%")
    ax.set_title("Distribution of Person Foreground Coverage Percentage per Image", fontsize=13, fontweight="bold")
    ax.set_xlabel("Foreground Coverage (% of total image area)", fontsize=11)
    ax.set_ylabel("Number of Images", fontsize=11)
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    p_fg = stats_dir / "04_foreground_coverage_distribution.png"
    plt.savefig(p_fg, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Foreground Coverage Distribution: {p_fg}")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset visualizations and statistical charts.")
    parser.add_argument("--samples", type=int, default=8, help="Number of sample images to plot")
    args = parser.parse_args()

    generate_visualizations(num_samples=args.samples)
