import torch
import torch.nn as nn
import torchvision.models as models


class DecoderBlock(nn.Module):
    """Same decoder block as Experiment 1."""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.up   = nn.ConvTranspose2d(
            in_channels, in_channels // 2,
            kernel_size=2, stride=2
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels // 2 + skip_channels,
                      out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetReplace(nn.Module):
    """
    Experiment 2 — Replace First Layer

    Surgically replaces ONLY the first conv layer
    of ResNet50 to accept 12 channels instead of 3.
    All other pretrained weights kept intact.
    Uses smart weight initialization — copies RGB
    weights to first 3 channels of new layer.
    """
    def __init__(self, in_channels=12, out_channels=1):
        super().__init__()

        # ── Load pretrained ResNet50 ──────────────────
        resnet = models.resnet50(weights='IMAGENET1K_V1')

        # ── Replace first layer ───────────────────────
        old_conv    = resnet.conv1
        # New layer: 12 channels in, same everything else
        new_conv    = nn.Conv2d(
            in_channels, 64,
            kernel_size = old_conv.kernel_size,
            stride      = old_conv.stride,
            padding     = old_conv.padding,
            bias        = False
        )

        # ── Smart initialization ──────────────────────
        with torch.no_grad():
            # Copy pretrained weights for first 3 channels
            new_conv.weight[:, :3, :, :] = old_conv.weight.clone()

            # Initialize remaining 9 channels with small values
            # Use std of pretrained weights as reference
            std = old_conv.weight.std().item()
            nn.init.normal_(new_conv.weight[:, 3:, :, :], mean=0, std=std)

        # Replace in ResNet
        resnet.conv1 = new_conv

        print(f"First layer replaced: 3→{in_channels} channels ✅")
        print(f"Pretrained RGB weights copied to channels 0-2 ✅")
        print(f"Channels 3-{in_channels-1} initialized with std={std:.6f} ✅")

        # ── Break ResNet into encoder pieces ──────────
        self.encoder0 = nn.Sequential(
            resnet.conv1,   # now accepts 12 channels
            resnet.bn1,
            resnet.relu,
        )                                    # → (64,  64, 64)
        self.pool     = resnet.maxpool       # → (64,  32, 32)
        self.encoder1 = resnet.layer1        # → (256, 32, 32)
        self.encoder2 = resnet.layer2        # → (512, 16, 16)
        self.encoder3 = resnet.layer3        # → (1024, 8,  8)
        self.encoder4 = resnet.layer4        # → (2048, 4,  4)

        # ── Decoder ───────────────────────────────────
        self.dec1 = DecoderBlock(2048, 1024, 512)
        self.dec2 = DecoderBlock(512,  512,  256)
        self.dec3 = DecoderBlock(256,  256,  128)
        self.dec4 = DecoderBlock(128,  64,   64)
        self.dec5 = DecoderBlock(64,   0,    32)

        # ── Output ────────────────────────────────────
        self.output = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # ── Encoder ───────────────────────────────────
        # No pre-layer needed — first layer handles 12 channels directly
        e0  = self.encoder0(x)         # (64,   64,  64)
        e0p = self.pool(e0)            # (64,   32,  32)
        e1  = self.encoder1(e0p)       # (256,  32,  32)
        e2  = self.encoder2(e1)        # (512,  16,  16)
        e3  = self.encoder3(e2)        # (1024,  8,   8)
        e4  = self.encoder4(e3)        # (2048,  4,   4)

        # ── Decoder ───────────────────────────────────
        d1  = self.dec1(e4, e3)        # 4→8
        d2  = self.dec2(d1, e2)        # 8→16
        d3  = self.dec3(d2, e1)        # 16→32
        d4  = self.dec4(d3, e0)        # 32→64
        d5  = self.dec5(d4)            # 64→128

        return self.output(d5)         # (1, 128, 128)