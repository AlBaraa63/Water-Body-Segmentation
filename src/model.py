import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    The basic building block of U-Net.
    Used at every level of encoder and decoder.
    
    Two rounds of: Conv → BatchNorm → ReLU
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv = nn.Sequential(
            # First convolution
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            # Second convolution
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net for water segmentation.
    
    Input  → (batch, 12, 128, 128)   12 bands
    Output → (batch,  1, 128, 128)   binary water mask
    """
    def __init__(self, in_channels=12, out_channels=1):
        super().__init__()
        
        # ── ENCODER (going down) ──────────────────────────
        self.enc1 = DoubleConv(in_channels, 64)   # 128x128
        self.enc2 = DoubleConv(64, 128)            # 64x64
        self.enc3 = DoubleConv(128, 256)           # 32x32
        self.enc4 = DoubleConv(256, 512)           # 16x16
        
        # ── BOTTLENECK ───────────────────────────────────
        self.bottleneck = DoubleConv(512, 1024)    # 8x8
        
        # ── DECODER (going up) ───────────────────────────
        # ConvTranspose2d = learnable upsampling (opposite of pooling)
        self.up4    = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4   = DoubleConv(1024, 512)  # 1024 because skip connection doubles channels
        
        self.up3    = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3   = DoubleConv(512, 256)
        
        self.up2    = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2   = DoubleConv(256, 128)
        
        self.up1    = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1   = DoubleConv(128, 64)
        
        # ── OUTPUT ───────────────────────────────────────
        # 1x1 conv to collapse 64 channels → 1 channel
        self.output = nn.Conv2d(64, out_channels, kernel_size=1)
        
        # Pooling to shrink image at each encoder step
        self.pool   = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        """
        Forward pass — data flows through the U shape.
        x shape: (batch, 12, 128, 128)
        """
        
        # ── ENCODER ──────────────────────────────────────
        e1 = self.enc1(x)           # (batch, 64,  128, 128)
        e2 = self.enc2(self.pool(e1)) # (batch, 128,  64,  64)
        e3 = self.enc3(self.pool(e2)) # (batch, 256,  32,  32)
        e4 = self.enc4(self.pool(e3)) # (batch, 512,  16,  16)
        
        # ── BOTTLENECK ───────────────────────────────────
        b  = self.bottleneck(self.pool(e4))  # (batch, 1024, 8, 8)
        
        # ── DECODER ──────────────────────────────────────
        # up → then concatenate skip connection → then DoubleConv
        
        d4 = self.up4(b)                        # (batch, 512, 16, 16)
        d4 = torch.cat([d4, e4], dim=1)         # (batch, 1024, 16, 16) ← skip!
        d4 = self.dec4(d4)                      # (batch, 512, 16, 16)
        
        d3 = self.up3(d4)                       # (batch, 256, 32, 32)
        d3 = torch.cat([d3, e3], dim=1)         # (batch, 512, 32, 32) ← skip!
        d3 = self.dec3(d3)                      # (batch, 256, 32, 32)
        
        d2 = self.up2(d3)                       # (batch, 128, 64, 64)
        d2 = torch.cat([d2, e2], dim=1)         # (batch, 256, 64, 64) ← skip!
        d2 = self.dec2(d2)                      # (batch, 128, 64, 64)
        
        d1 = self.up1(d2)                       # (batch, 64, 128, 128)
        d1 = torch.cat([d1, e1], dim=1)         # (batch, 128, 128, 128) ← skip!
        d1 = self.dec1(d1)                      # (batch, 64, 128, 128)
        
        # ── OUTPUT ───────────────────────────────────────
        return self.output(d1)                  # (batch, 1, 128, 128)


# ── Loss Function ────────────────────────────────────────────
class BCEDiceLoss(nn.Module):
    """
    Combined BCE + Dice loss.
    BCE handles per-pixel accuracy.
    Dice handles region overlap (critical for imbalanced masks).
    """
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.bce    = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def dice_loss(self, logits, targets):
        probs = torch.sigmoid(logits)
        num   = 2 * (probs * targets).sum(dim=(2, 3)) + self.smooth
        den   = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + self.smooth
        return 1 - (num / den).mean()

    def forward(self, logits, targets):
        return self.bce(logits, targets) + self.dice_loss(logits, targets)


# ── Utility ──────────────────────────────────────────────────
def count_parameters(model):
    """Returns total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
