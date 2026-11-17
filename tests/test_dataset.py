"""Automated test suite validating Penn-Fudan Person Segmentation Dataset Engineering."""

import unittest
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from src.inspect_dataset import find_mask_for_image
from src.utils import load_config, resolve_path


class TestPennFudanDataset(unittest.TestCase):
    """Automated validation test suite for dataset pipeline."""

    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        cls.raw_dir = resolve_path(cls.config["paths"]["raw_dir"]) / "PennFudanPed"
        cls.raw_images_dir = cls.raw_dir / "PNGImages"
        cls.raw_masks_dir = cls.raw_dir / "PedMasks"
        cls.processed_images_dir = resolve_path(cls.config["paths"]["processed_images_dir"])
        cls.processed_masks_dir = resolve_path(cls.config["paths"]["processed_masks_dir"])
        cls.splits_dir = resolve_path(cls.config["paths"]["splits_dir"])

    def test_01_raw_image_mask_correspondence(self):
        """Test that all 170 raw images have a valid corresponding mask file."""
        self.assertTrue(self.raw_images_dir.exists(), f"Raw images dir missing: {self.raw_images_dir}")
        self.assertTrue(self.raw_masks_dir.exists(), f"Raw masks dir missing: {self.raw_masks_dir}")

        image_files = sorted(list(self.raw_images_dir.glob("*.png")))
        mask_files = sorted(list(self.raw_masks_dir.glob("*.png")))

        self.assertEqual(len(image_files), 170, f"Expected 170 images, found {len(image_files)}")
        self.assertEqual(len(mask_files), 170, f"Expected 170 masks, found {len(mask_files)}")

        matched_masks = set()
        for img_path in image_files:
            mask_path = find_mask_for_image(img_path, self.raw_masks_dir)
            self.assertTrue(mask_path.exists(), f"Mask {mask_path} does not exist for {img_path.name}")
            matched_masks.add(mask_path.resolve())

            # Verify identical raw dimensions
            with Image.open(img_path) as im, Image.open(mask_path) as mk:
                self.assertEqual(im.size, mk.size, f"Raw size mismatch for {img_path.name}: {im.size} vs {mk.size}")

        self.assertEqual(len(matched_masks), 170, "Not all mask files were matched 1-to-1 with images.")

    def test_02_raw_pedestrian_instance_count(self):
        """Test that the raw masks represent the verified pedestrian instances (345 original + 78 added = 423 total)."""
        mask_files = sorted(list(self.raw_masks_dir.glob("*.png")))
        total_instances = 0
        for mp in mask_files:
            with Image.open(mp) as img:
                arr = np.array(img)
                unique_vals = np.unique(arr)
                instances = unique_vals[unique_vals > 0]
                total_instances += len(instances)

        # The PennFudanPed release includes 345 original instances + 78 added objects = 423 total
        self.assertIn(total_instances, [345, 423], f"Expected 345 or 423 pedestrian instances, got {total_instances}")

    def test_03_processed_images_integrity(self):
        """Test that all 170 processed images are 256x256x3, RGB format, uint8, and loadable."""
        self.assertTrue(self.processed_images_dir.exists(), "Processed images dir does not exist.")
        proc_images = sorted(list(self.processed_images_dir.glob("*.png")))
        self.assertEqual(len(proc_images), 170, f"Expected 170 processed images, got {len(proc_images)}")

        for img_path in proc_images:
            with Image.open(img_path) as img:
                self.assertEqual(img.size, (256, 256), f"Wrong size for {img_path.name}: {img.size}")
                self.assertEqual(img.mode, "RGB", f"Wrong mode for {img_path.name}: {img.mode}")
                arr = np.array(img)
                self.assertEqual(arr.shape, (256, 256, 3), f"Wrong shape for {img_path.name}: {arr.shape}")
                self.assertEqual(arr.dtype, np.uint8, f"Wrong dtype for {img_path.name}: {arr.dtype}")

    def test_04_processed_masks_integrity(self):
        """Test that all 170 processed masks are 256x256, uint8, and contain strictly {0, 1}."""
        self.assertTrue(self.processed_masks_dir.exists(), "Processed masks dir does not exist.")
        proc_masks = sorted(list(self.processed_masks_dir.glob("*.png")))
        self.assertEqual(len(proc_masks), 170, f"Expected 170 processed masks, got {len(proc_masks)}")

        empty_mask_count = 0
        for mask_path in proc_masks:
            with Image.open(mask_path) as img:
                self.assertEqual(img.size, (256, 256), f"Wrong size for {mask_path.name}: {img.size}")
                arr = np.array(img)
                self.assertEqual(arr.shape, (256, 256), f"Wrong shape for {mask_path.name}: {arr.shape}")
                self.assertEqual(arr.dtype, np.uint8, f"Wrong dtype for {mask_path.name}: {arr.dtype}")
                
                unique_vals = np.unique(arr).tolist()
                # Must be strictly a subset of {0, 1}
                self.assertTrue(set(unique_vals).issubset({0, 1}),
                                f"Non-binary values found in {mask_path.name}: {unique_vals}")
                
                fg_pixels = np.sum(arr == 1)
                if fg_pixels == 0:
                    empty_mask_count += 1

        self.assertEqual(empty_mask_count, 0, f"Found {empty_mask_count} empty masks without any pedestrian pixels!")

    def test_05_image_mask_matching_filenames(self):
        """Test that processed images and masks share matching filenames and paired dimensions."""
        proc_images = sorted([f.name for f in self.processed_images_dir.glob("*.png")])
        proc_masks = sorted([f.name for f in self.processed_masks_dir.glob("*.png")])
        self.assertEqual(proc_images, proc_masks, "Processed images and masks do not have identical filenames.")

    def test_06_splits_exclusivity_and_completeness(self):
        """Test that train (119), val (25), and test (26) splits are mutually exclusive and collectively exhaustive."""
        train_file = self.splits_dir / "train.txt"
        val_file = self.splits_dir / "val.txt"
        test_file = self.splits_dir / "test.txt"

        self.assertTrue(train_file.exists(), f"Missing split file: {train_file}")
        self.assertTrue(val_file.exists(), f"Missing split file: {val_file}")
        self.assertTrue(test_file.exists(), f"Missing split file: {test_file}")

        with open(train_file, "r", encoding="utf-8") as f:
            train_set = set(line.strip() for line in f if line.strip())
        with open(val_file, "r", encoding="utf-8") as f:
            val_set = set(line.strip() for line in f if line.strip())
        with open(test_file, "r", encoding="utf-8") as f:
            test_set = set(line.strip() for line in f if line.strip())

        # Exact split counts
        self.assertEqual(len(train_set), 119, f"Expected 119 train samples, got {len(train_set)}")
        self.assertEqual(len(val_set), 25, f"Expected 25 val samples, got {len(val_set)}")
        self.assertEqual(len(test_set), 26, f"Expected 26 test samples, got {len(test_set)}")

        # Pairwise mutual exclusivity (zero data leakage)
        self.assertEqual(len(train_set & val_set), 0, "Train and Val splits share overlapping samples!")
        self.assertEqual(len(train_set & test_set), 0, "Train and Test splits share overlapping samples!")
        self.assertEqual(len(val_set & test_set), 0, "Val and Test splits share overlapping samples!")

        # Completeness
        union_set = train_set | val_set | test_set
        self.assertEqual(len(union_set), 170, f"Union of splits does not equal 170 total images ({len(union_set)})")

        all_processed = set(f.name for f in self.processed_images_dir.glob("*.png"))
        self.assertEqual(union_set, all_processed, "Split files do not match processed image set.")

    def test_07_pytorch_tensor_compatibility(self):
        """Test that processed images and masks convert directly into PyTorch tensors for U-Net."""
        sample_img_path = next(self.processed_images_dir.glob("*.png"))
        sample_mask_path = self.processed_masks_dir / sample_img_path.name

        with Image.open(sample_img_path) as im, Image.open(sample_mask_path) as mk:
            im_arr = np.array(im, dtype=np.float32) / 255.0  # Normalize to [0, 1]
            mk_arr = np.array(mk, dtype=np.int64)           # Binary class labels

        # Convert to PyTorch tensors (C, H, W)
        tensor_img = torch.from_numpy(im_arr).permute(2, 0, 1) # (3, 256, 256)
        tensor_mask = torch.from_numpy(mk_arr).unsqueeze(0)   # (1, 256, 256)

        self.assertEqual(tensor_img.shape, (3, 256, 256))
        self.assertEqual(tensor_mask.shape, (1, 256, 256))
        self.assertEqual(tensor_img.dtype, torch.float32)
        self.assertEqual(tensor_mask.dtype, torch.int64)
        self.assertTrue(torch.all((tensor_mask == 0) | (tensor_mask == 1)))


if __name__ == "__main__":
    unittest.main()
