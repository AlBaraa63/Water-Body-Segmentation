import torch
import torch.nn as nn
import torchvision.models as models


class DecoderBlock(nn.Module):
    """
    One step of the decoder.
    Upsample → concatenate skip → DoubleConv
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        # Upsample — double the spatial size
        self.up   = nn.ConvTranspose2d(
            in_channels, in_channels // 2,
            kernel_size=2, stride=2
        )

        # After concatenating skip connection
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


class UNetPreLayer(nn.Module):
    """
    Experiment 1 — Pre-Layer Approach

    Adds a small conv layer before ResNet50
    to compress 12 bands → 3 channels.
    ResNet50 encoder then processes normally.
    """
    def __init__(self, in_channels=12, out_channels=1):
        super().__init__()

        # ── Pre-layer: 12 → 3 ────────────────────────
        self.pre_layer = nn.Sequential(
            nn.Conv2d(in_channels, 3,
                      kernel_size=1, bias=False),  # 1x1 conv
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )

        # ── ResNet50 Encoder ──────────────────────────
        resnet = models.resnet50(weights='IMAGENET1K_V1')

        # Break ResNet into pieces we can use separately
        self.encoder0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
        )                                    # → (64,  32, 32)
        self.pool     = resnet.maxpool       # → (64,  32, 32) same
        self.encoder1 = resnet.layer1        # → (256, 32, 32)
        self.encoder2 = resnet.layer2        # → (512, 16, 16)
        self.encoder3 = resnet.layer3        # → (1024, 8,  8)
        self.encoder4 = resnet.layer4        # → (2048, 4,  4)

        # ── Decoder ───────────────────────────────────
        # in_channels, skip_channels, out_channels
        self.dec1 = DecoderBlock(2048, 1024, 512)   # 4→8
        self.dec2 = DecoderBlock(512,  512,  256)   # 8→16
        self.dec3 = DecoderBlock(256,  256,  128)   # 16→32
        self.dec4 = DecoderBlock(128,  64,   64)    # 32→64
        self.dec5 = DecoderBlock(64,   0,    32)    # 64→128 no skip

        # ── Output ────────────────────────────────────
        self.output = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # ── Pre-layer ─────────────────────────────────
        x = self.pre_layer(x)          # (12, 128, 128) → (3, 128, 128)

        # ── Encoder ───────────────────────────────────
        e0 = self.encoder0(x)          # (64,   32,  32)
        e0p= self.pool(e0)             # (64,   32,  32) — note: same size
        e1 = self.encoder1(e0p)        # (256,  32,  32)
        e2 = self.encoder2(e1)         # (512,  16,  16)
        e3 = self.encoder3(e2)         # (1024,  8,   8)
        e4 = self.encoder4(e3)         # (2048,  4,   4)

        # ── Decoder with skip connections ─────────────
        d1 = self.dec1(e4, e3)         # 4→8,   + skip e3
        d2 = self.dec2(d1, e2)         # 8→16,  + skip e2
        d3 = self.dec3(d2, e1)         # 16→32, + skip e1
        d4 = self.dec4(d3, e0)         # 32→64, + skip e0
        d5 = self.dec5(d4)             # 64→128, no skip

        return self.output(d5)         # → (1, 128, 128)