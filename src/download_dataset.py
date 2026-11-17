"""Download and extract the Penn-Fudan Pedestrian Dataset."""

import argparse
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

from src.utils import get_project_root, load_config, resolve_path


def download_dataset(force: bool = False) -> Path:
    """Download the dataset zip file if not present, and extract it to data/raw."""
    config = load_config()
    raw_dir = resolve_path(config["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    extracted_dir = raw_dir / "PennFudanPed"
    images_dir = extracted_dir / "PNGImages"
    masks_dir = extracted_dir / "PedMasks"

    if not force and images_dir.exists() and masks_dir.exists():
        num_imgs = len(list(images_dir.glob("*.png")))
        num_masks = len(list(masks_dir.glob("*.png")))
        if num_imgs == config["dataset"]["expected_images"] and num_masks == config["dataset"]["expected_masks"]:
            print(f"[INFO] Raw dataset already extracted at {extracted_dir} with {num_imgs} images and {num_masks} masks.")
            return extracted_dir

    # Check for existing zip archive in raw_dir, project root, or parent workspace
    zip_name = config["dataset"]["zip_filename"]
    candidate_zips = [
        raw_dir / zip_name,
        get_project_root() / zip_name,
        get_project_root().parent / zip_name,
    ]

    target_zip = None
    for cand in candidate_zips:
        if cand.exists() and cand.stat().st_size > 10_000_000:
            target_zip = cand
            print(f"[INFO] Found existing archive at: {target_zip} ({target_zip.stat().st_size} bytes)")
            break

    if target_zip is None or force:
        target_zip = raw_dir / zip_name
        url = config["dataset"]["url"]
        print(f"[INFO] Downloading Penn-Fudan Pedestrian Dataset from {url}...")
        
        # Download with progress report
        def reporthook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100.0, downloaded * 100.0 / total_size)
                if block_num % 100 == 0:
                    print(f"  Downloading: {percent:.1f}% ({downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB)", end="\r")

        urllib.request.urlretrieve(url, str(target_zip), reporthook=reporthook)
        print(f"\n[INFO] Download completed: {target_zip} ({target_zip.stat().st_size} bytes)")

    # Extract safely
    print(f"[INFO] Extracting {target_zip} into {raw_dir}...")
    with zipfile.ZipFile(target_zip, "r") as z:
        # Sanity check for path traversal vulnerability (Zip Slip)
        for member in z.namelist():
            member_path = (raw_dir / member).resolve()
            if not str(member_path).startswith(str(raw_dir.resolve())):
                raise RuntimeError(f"Unsafe zip entry detected: {member}")
        z.extractall(str(raw_dir))

    # Verify extracted contents
    if not (images_dir.exists() and masks_dir.exists()):
        raise RuntimeError(f"Extraction failed: Expected {images_dir} and {masks_dir} to exist.")

    num_imgs = len(list(images_dir.glob("*.png")))
    num_masks = len(list(masks_dir.glob("*.png")))
    print(f"[SUCCESS] Dataset extracted to {extracted_dir}: {num_imgs} images, {num_masks} masks.")
    return extracted_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and extract Penn-Fudan Pedestrian Dataset.")
    parser.add_argument("--force", action="store_true", help="Force re-download and re-extraction")
    args = parser.parse_args()

    download_dataset(force=args.force)
