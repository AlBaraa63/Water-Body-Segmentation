import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing import load_tif, replace_nodata, normalize_with_stats, normalize_mask


class WaterSegmentationDataset(Dataset):
    """
    Loads multispectral .tif images and binary water masks.
    Handles preprocessing on the fly for each sample.
    """

    def __init__(self, image_dir, mask_dir, image_files, stats, transform=None):
        """
        Setup — store paths, filenames, and global stats.

        image_dir    -> folder containing .tif images
        mask_dir     -> folder containing .tif masks
        image_files  -> list of filenames to use
        stats        -> global mean/std from training data
        transform    -> optional albumentations transform (train only)
        """
        self.image_dir   = image_dir
        self.mask_dir    = mask_dir
        self.image_files = image_files
        self.stats       = stats
        self.transform   = transform

        print(f"Dataset created with {len(image_files)} samples")

    def __len__(self):
        """
        Returns how many samples are in this dataset.
        PyTorch calls this automatically.
        """
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Load and return ONE sample by index.
        PyTorch calls this automatically when batching.

        Returns:
            image → tensor shape (12, 128, 128)  channels first for PyTorch
            mask  → tensor shape (1,  128, 128)
        """
        fname = self.image_files[idx]

        # Build full paths
        img_path  = os.path.join(self.image_dir, fname)
        base_name = os.path.splitext(fname)[0]
        mask_path = os.path.join(self.mask_dir,  f"{base_name}.png")

        # Load
        image = load_tif(img_path)   # (128, 128, 12)
        mask  = load_tif(mask_path)  # (128, 128, 1)

        # Clean and normalize
        image = replace_nodata(image, verbose=False)
        image = normalize_with_stats(image, self.stats)
        mask  = normalize_mask(mask)

        # Apply augmentation (image and mask in sync) — HWC format for albumentations
        if self.transform is not None:
            mask_hwc = mask  # (H, W, 1)
            augmented = self.transform(image=image, mask=mask_hwc[:, :, 0])
            image = augmented["image"]
            mask  = augmented["mask"][:, :, np.newaxis]  # restore (H, W, 1)

        # Convert to PyTorch tensors: (H, W, C) -> (C, H, W)
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        mask  = torch.tensor(mask,  dtype=torch.float32).permute(2, 0, 1)

        return image, mask
def split_dataset(image_files, train=0.70, val=0.15, test=0.15, seed=42):
    """
    Splits file list into train, validation, test sets.
    
    seed=42 means the split is REPRODUCIBLE
    → same split every time you run the code
    """
    assert train + val + test == 1.0, "Splits must add up to 1.0"

    # Shuffle files randomly but consistently
    np.random.seed(seed)
    files = np.array(image_files.copy())
    np.random.shuffle(files)

    n       = len(files)
    n_train = int(n * train)
    n_val   = int(n * val)

    train_files = list(files[:n_train])
    val_files   = list(files[n_train:n_train + n_val])
    test_files  = list(files[n_train + n_val:])

    print(f"Total   : {n}")
    print(f"Train   : {len(train_files)} ({len(train_files)/n*100:.0f}%)")
    print(f"Val     : {len(val_files)}   ({len(val_files)/n*100:.0f}%)")
    print(f"Test    : {len(test_files)}  ({len(test_files)/n*100:.0f}%)")

    return train_files, val_files, test_files


def get_dataloaders(image_dir, mask_dir, image_files, stats, batch_size=16):
    """
    Creates all three DataLoaders in one call.
    No augmentation — produces best results on this 214-image training set.
    """
    train_files, val_files, test_files = split_dataset(image_files)

    train_dataset = WaterSegmentationDataset(image_dir, mask_dir, train_files, stats,
                                             transform=None)   # augmentation off
    val_dataset   = WaterSegmentationDataset(image_dir, mask_dir, val_files,   stats)
    test_dataset  = WaterSegmentationDataset(image_dir, mask_dir, test_files,  stats)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                              num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,
                              num_workers=0)

    return train_loader, val_loader, test_loader
