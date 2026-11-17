"""PyTorch Dataset and DataLoader for Penn-Fudan Person Segmentation."""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from src.utils import get_project_root, load_config, resolve_path


class PedestrianDataset(Dataset):
    """Penn-Fudan Pedestrian Segmentation Dataset.
    
    Loads preprocessed 256x256 RGB images and binary {0, 1} masks based on split files.
    Image tensor: [3, 256, 256], normalized float32 in [0.0, 1.0].
    Mask tensor:  [1, 256, 256], binary float32 in {0.0, 1.0}.
    """

    def __init__(
        self,
        split: str = "train",
        transform: Optional[Callable[[Image.Image, Image.Image], Tuple[Image.Image, Image.Image]]] = None,
        config: Optional[Dict] = None,
    ):
        super().__init__()
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Invalid split '{split}'. Must be one of ['train', 'val', 'test']")

        self.split = split
        self.transform = transform
        self.config = config or load_config()

        self.images_dir = resolve_path(self.config["paths"]["processed_images_dir"])
        self.masks_dir = resolve_path(self.config["paths"]["processed_masks_dir"])
        splits_dir = resolve_path(self.config["paths"]["splits_dir"])
        split_file = splits_dir / f"{split}.txt"

        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}. Run create_splits.py first.")

        with open(split_file, "r", encoding="utf-8") as f:
            self.filenames = [line.strip() for line in f if line.strip()]

        if len(self.filenames) == 0:
            raise RuntimeError(f"Split file {split_file} is empty!")

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        filename = self.filenames[idx]
        img_path = self.images_dir / filename
        mask_path = self.masks_dir / filename

        if not img_path.exists():
            raise FileNotFoundError(f"Missing processed image: {img_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing processed mask: {mask_path}")

        # Open image and mask with PIL
        with Image.open(img_path) as im:
            image = im.convert("RGB")
        with Image.open(mask_path) as mk:
            mask = mk.convert("L")

        # Apply synchronized data augmentations if specified
        if self.transform is not None:
            image, mask = self.transform(image, mask)

        # Convert image to float32 normalized to [0, 1]
        img_arr = np.array(image, dtype=np.float32) / 255.0  # [256, 256, 3]
        # Convert mask to float32 with values strictly 0.0 or 1.0
        mask_arr = np.array(mask, dtype=np.float32)         # [256, 256]

        # Ensure mask is strictly binary {0.0, 1.0}
        mask_arr = (mask_arr > 0.5).astype(np.float32)

        # Transpose image to [C, H, W] -> [3, 256, 256]
        img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).contiguous()
        # Add channel dimension to mask -> [1, 256, 256]
        mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0).contiguous()

        return img_tensor, mask_tensor, filename


def get_dataloader(
    split: str = "train",
    batch_size: int = 8,
    shuffle: Optional[bool] = None,
    num_workers: int = 0,
    transform: Optional[Callable] = None,
    config: Optional[Dict] = None,
) -> DataLoader:
    """Build a DataLoader for a given dataset split."""
    if shuffle is None:
        shuffle = (split == "train")

    dataset = PedestrianDataset(split=split, transform=transform, config=config)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )
    return dataloader
