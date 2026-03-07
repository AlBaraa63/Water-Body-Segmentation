# src/__init__.py
# Keep this minimal — just expose key classes.

from .dataset import WaterSegmentationDataset, split_dataset, get_dataloaders
from .model import UNet, BCEDiceLoss, count_parameters

__all__ = [
    "WaterSegmentationDataset",
    "split_dataset",
    "get_dataloaders",
    "UNet",
    "BCEDiceLoss",
    "count_parameters",
]
