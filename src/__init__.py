# src/__init__.py
# Keep this minimal — just expose key classes.

from .dataset import WaterSegmentationDataset, split_dataset, get_dataloaders
from .models import UNet, BCEDiceLoss, count_parameters

__all__ = [
    "WaterSegmentationDataset",
    "split_dataset",
    "get_dataloaders",
    "UNet",
    "BCEDiceLoss",
    "count_parameters",
]
