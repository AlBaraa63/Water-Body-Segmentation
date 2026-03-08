import torch
import torch.nn as nn
from .unet_scratch import UNet

class SatelliteUNet(nn.Module):
    """
    Experiment 3: Satellite-Specific Pre-training Route
    Instead of trying to adapt a natural-image (RGB) pretrained model
    like ResNet from ImageNet, this architecture is meant to use weights 
    specifically trained on 12-channel multispectral satellite imagery 
    (e.g., Sentinel-2 via BigEarthNet).
    
    Here we implement a standard U-Net built natively for 12 channels.
    """
    def __init__(self, in_channels=12, out_channels=1):
        super().__init__()
        
        # When true satellite-pretrained weights are available, one would load
        # a specialized backbone like EuroSAT/BigEarthNet ResNet here.
        # For our architecture, this simply uses the full 12-channel UNet natively.
        self.model = UNet(in_channels=in_channels, out_channels=out_channels)
        
    def forward(self, x):
        # x is (batch, 12, H, W)
        return self.model(x)
