"""Synchronized geometric transformations for semantic segmentation pairs."""

import random
from typing import Optional, Tuple
import numpy as np
from PIL import Image


class JointSegmentationTransform:
    """Synchronized geometric transformations applied identically to both image and mask.
    
    Transforms:
      1. Random horizontal flip with probability p=0.5.
      2. Small random zoom with scale in [zoom_min, zoom_max] (e.g., 0.9 to 1.1).
    
    Interpolation:
      - Image: Bilinear interpolation
      - Mask: Nearest-neighbor interpolation (preserves strict {0, 1} binary values)
    """

    def __init__(
        self,
        hflip_prob: float = 0.5,
        zoom_range: Tuple[float, float] = (0.9, 1.1),
        target_size: Tuple[int, int] = (256, 256),
    ):
        self.hflip_prob = hflip_prob
        self.zoom_min, self.zoom_max = zoom_range
        self.target_size = target_size

    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        orig_w, orig_h = image.size
        m_w, m_h = mask.size
        if (orig_w, orig_h) != (m_w, m_h):
            raise ValueError(f"Input size mismatch: Image is {(orig_w, orig_h)}, mask is {(m_w, m_h)}")

        # 1. Random Horizontal Flip (p = 0.5)
        if random.random() < self.hflip_prob:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        # 2. Random Zoom (0.9 to 1.1)
        scale = random.uniform(self.zoom_min, self.zoom_max)

        if scale > 1.0:
            # Zoom In: Crop an inner sub-region and resize up to target_size
            crop_w = max(1, int(round(orig_w / scale)))
            crop_h = max(1, int(round(orig_h / scale)))
            crop_w = min(orig_w, crop_w)
            crop_h = min(orig_h, crop_h)

            max_dx = orig_w - crop_w
            max_dy = orig_h - crop_h
            left = random.randint(0, max_dx) if max_dx > 0 else 0
            top = random.randint(0, max_dy) if max_dy > 0 else 0
            box = (left, top, left + crop_w, top + crop_h)

            image = image.crop(box).resize(self.target_size, resample=Image.Resampling.BILINEAR)
            mask = mask.crop(box).resize(self.target_size, resample=Image.Resampling.NEAREST)

        elif scale < 1.0:
            # Zoom Out: Scale down and pad onto a target_size canvas
            scaled_w = max(1, int(round(orig_w * scale)))
            scaled_h = max(1, int(round(orig_h * scale)))

            scaled_img = image.resize((scaled_w, scaled_h), resample=Image.Resampling.BILINEAR)
            scaled_mask = mask.resize((scaled_w, scaled_h), resample=Image.Resampling.NEAREST)

            max_px = orig_w - scaled_w
            max_py = orig_h - scaled_h
            offset_x = random.randint(0, max_px) if max_px > 0 else 0
            offset_y = random.randint(0, max_py) if max_py > 0 else 0

            # Blank canvases (0 for background)
            canvas_img = Image.new("RGB", self.target_size, (0, 0, 0))
            canvas_mask = Image.new("L", self.target_size, 0)

            canvas_img.paste(scaled_img, (offset_x, offset_y))
            canvas_mask.paste(scaled_mask, (offset_x, offset_y))

            image = canvas_img
            mask = canvas_mask

        # Sanity check: Ensure mask values remain strictly binary {0, 1}
        mask_arr = np.array(mask)
        unique_vals = np.unique(mask_arr)
        if not set(unique_vals).issubset({0, 1}):
            raise RuntimeError(f"Data corruption in augmentation: mask values {unique_vals} are not in {{0, 1}}.")

        return image, mask


def get_transforms(split: str = "train", augment: bool = False) -> Optional[JointSegmentationTransform]:
    """Factory function returning transforms for training or evaluation."""
    if split == "train" and augment:
        return JointSegmentationTransform(hflip_prob=0.5, zoom_range=(0.9, 1.1), target_size=(256, 256))
    return None
