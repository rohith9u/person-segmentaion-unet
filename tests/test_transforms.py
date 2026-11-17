"""Automated tests for synchronized augmentations, dataset loader, and determinism."""

import unittest
import numpy as np
from PIL import Image
import torch

from src.augmentation.transforms import JointSegmentationTransform, get_transforms
from src.datasets.pedestrian_dataset import PedestrianDataset, get_dataloader


class TestAugmentationAndDataset(unittest.TestCase):
    """Test suite validating data augmentation synchronization and dataset loading."""

    def setUp(self):
        # Create a synthetic image and mask with a known asymmetric pattern
        self.img = Image.new("RGB", (256, 256), (50, 100, 150))
        # Place a bright square on the top-left quadrant of image and mask
        for y in range(40, 100):
            for x in range(30, 90):
                self.img.putpixel((x, y), (255, 255, 255))

        self.mask = Image.new("L", (256, 256), 0)
        for y in range(40, 100):
            for x in range(30, 90):
                self.mask.putpixel((x, y), 1)

    def test_01_mask_remains_strictly_binary_after_augmentation(self):
        """Verify that repeated random augmentations strictly preserve binary {0, 1} mask values."""
        transform = JointSegmentationTransform(hflip_prob=0.5, zoom_range=(0.9, 1.1))
        for trial in range(30):
            aug_img, aug_mask = transform(self.img, self.mask)
            mask_arr = np.array(aug_mask)
            unique_vals = np.unique(mask_arr)
            self.assertTrue(
                set(unique_vals).issubset({0, 1}),
                f"Trial {trial}: Non-binary mask values encountered: {unique_vals}",
            )
            self.assertEqual(aug_img.size, (256, 256))
            self.assertEqual(aug_mask.size, (256, 256))

    def test_02_transform_spatial_synchronization(self):
        """Verify that horizontal flip and zoom transformations are applied identically to image and mask."""
        transform_flip = JointSegmentationTransform(hflip_prob=1.0, zoom_range=(1.0, 1.0))
        aug_img, aug_mask = transform_flip(self.img, self.mask)

        # In original, box was x in [30, 90] (left side).
        # After horizontal flip, box must be at x in [256 - 90, 256 - 30] = [166, 226].
        mask_arr = np.array(aug_mask)
        # Check that right quadrant has 1s
        self.assertEqual(mask_arr[70, 196], 1, "Mask was not flipped to the right side!")
        # Check that left quadrant is 0
        self.assertEqual(mask_arr[70, 60], 0, "Left side still has pedestrian pixels after flip!")

    def test_03_validation_and_test_are_strictly_unaugmented(self):
        """Verify that get_transforms returns None for validation and test splits."""
        self.assertIsNone(get_transforms(split="val", augment=False))
        self.assertIsNone(get_transforms(split="val", augment=True))
        self.assertIsNone(get_transforms(split="test", augment=False))
        self.assertIsNone(get_transforms(split="test", augment=True))
        self.assertIsNone(get_transforms(split="train", augment=False))
        self.assertIsNotNone(get_transforms(split="train", augment=True))

    def test_04_dataset_loading_and_tensor_normalization(self):
        """Verify PedestrianDataset produces tensors with expected shapes, ranges, and types."""
        dataset = PedestrianDataset(split="train")
        self.assertEqual(len(dataset), 119)

        img_tensor, mask_tensor, fname = dataset[0]
        self.assertEqual(img_tensor.shape, (3, 256, 256))
        self.assertEqual(mask_tensor.shape, (1, 256, 256))
        self.assertEqual(img_tensor.dtype, torch.float32)
        self.assertEqual(mask_tensor.dtype, torch.float32)

        # Image normalized to [0, 1]
        self.assertGreaterEqual(img_tensor.min().item(), 0.0)
        self.assertLessEqual(img_tensor.max().item(), 1.0)

        # Mask strictly 0.0 or 1.0
        unique_mask = torch.unique(mask_tensor).tolist()
        self.assertTrue(set(unique_mask).issubset({0.0, 1.0}), f"Mask values not in {{0,1}}: {unique_mask}")

    def test_05_dataloader_batching(self):
        """Verify DataLoader batches images into [B, 3, 256, 256] and masks into [B, 1, 256, 256]."""
        loader = get_dataloader(split="val", batch_size=4, shuffle=False)
        self.assertEqual(len(loader.dataset), 25)

        for batch_idx, (imgs, masks, fnames) in enumerate(loader):
            self.assertEqual(imgs.shape[1:], (3, 256, 256))
            self.assertEqual(masks.shape[1:], (1, 256, 256))
            self.assertEqual(len(fnames), imgs.size(0))
            if batch_idx == 0:
                self.assertEqual(imgs.size(0), 4)
            break


if __name__ == "__main__":
    unittest.main()
