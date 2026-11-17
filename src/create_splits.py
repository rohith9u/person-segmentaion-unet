"""Create reproducible train, validation, and test splits (70/15/15) with fixed random seed 42."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split

from src.utils import load_config, resolve_path


def create_dataset_splits(seed: int = 42) -> Dict[str, List[str]]:
    """Partition the 170 samples into 70% train (119), 15% val (25), and 15% test (26)."""
    config = load_config()
    processed_images_dir = resolve_path(config["paths"]["processed_images_dir"])
    splits_dir = resolve_path(config["paths"]["splits_dir"])
    splits_dir.mkdir(parents=True, exist_ok=True)

    if not processed_images_dir.exists():
        raise FileNotFoundError(f"Processed images directory not found at: {processed_images_dir}. Run preprocess.py first.")

    # Collect and sort all filenames deterministically
    all_files = sorted([f.name for f in processed_images_dir.glob("*.png")])
    total_samples = len(all_files)

    if total_samples != config["dataset"]["expected_images"]:
        print(f"[WARNING] Expected {config['dataset']['expected_images']} images, but found {total_samples}.")

    # Deterministic 2-stage split: 70% train, 15% val, 15% test
    # 170 * 0.70 = 119 train, 51 remaining
    # 51 -> 25 val, 26 test
    train_files, temp_files = train_test_split(
        all_files,
        train_size=119,
        test_size=51,
        random_state=seed,
        shuffle=True
    )

    val_files, test_files = train_test_split(
        temp_files,
        train_size=25,
        test_size=26,
        random_state=seed,
        shuffle=True
    )

    # Sort each split for clean, reproducible file outputs
    train_files = sorted(train_files)
    val_files = sorted(val_files)
    test_files = sorted(test_files)

    # Rigorous mutual exclusivity and completeness checks
    set_train = set(train_files)
    set_val = set(val_files)
    set_test = set(test_files)

    assert len(train_files) == 119, f"Expected 119 train samples, got {len(train_files)}"
    assert len(val_files) == 25, f"Expected 25 val samples, got {len(val_files)}"
    assert len(test_files) == 26, f"Expected 26 test samples, got {len(test_files)}"
    assert len(set_train & set_val) == 0, "Data leakage detected: train and val overlap!"
    assert len(set_train & set_test) == 0, "Data leakage detected: train and test overlap!"
    assert len(set_val & set_test) == 0, "Data leakage detected: val and test overlap!"
    assert len(set_train | set_val | set_test) == total_samples, "Not all samples assigned to splits!"

    # Save to text files
    split_paths = {
        "train": splits_dir / "train.txt",
        "val": splits_dir / "val.txt",
        "test": splits_dir / "test.txt",
    }

    for split_name, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
        out_file = split_paths[split_name]
        with open(out_file, "w", encoding="utf-8") as f:
            for fname in files:
                f.write(f"{fname}\n")
        print(f"[INFO] Saved {len(files)} {split_name} filenames to: {out_file}")

    # Save summary metadata
    meta = {
        "random_seed": seed,
        "total_samples": total_samples,
        "train_count": len(train_files),
        "val_count": len(val_files),
        "test_count": len(test_files),
        "train_percentage": round((len(train_files) / total_samples) * 100.0, 2),
        "val_percentage": round((len(val_files) / total_samples) * 100.0, 2),
        "test_percentage": round((len(test_files) / total_samples) * 100.0, 2),
        "mutually_exclusive": True,
        "collectively_exhaustive": True,
    }

    meta_path = splits_dir / "splits_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 70)
    print("                 DATASET SPLITS SUMMARY")
    print("=" * 70)
    print(f"Total Images    : {total_samples}")
    print(f"Training Set    : {len(train_files)} images ({meta['train_percentage']}%)")
    print(f"Validation Set  : {len(val_files)} images ({meta['val_percentage']}%)")
    print(f"Testing Set     : {len(test_files)} images ({meta['test_percentage']}%)")
    print(f"Random Seed     : {seed}")
    print(f"Data Leakage    : None (Disjoint sets verified)")
    print("=" * 70 + "\n")

    return {"train": train_files, "val": val_files, "test": test_files}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create deterministic train/val/test splits.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    create_dataset_splits(seed=args.seed)
