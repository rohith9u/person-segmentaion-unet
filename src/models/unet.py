"""U-Net Semantic Segmentation Architecture implemented in PyTorch."""

import torch
import torch.nn as nn
from typing import Tuple


class DoubleConv(nn.Module):
    """Two 3x3 convolutions with Batch Normalization and ReLU activations."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """U-Net architecture for binary semantic person segmentation.
    
    Specification:
        Input: [B, 3, 256, 256]
        Encoder:
            Block 1: 32 channels
            Block 2: 64 channels
            Block 3: 128 channels
            Block 4: 256 channels
        Bottleneck:
            512 channels
        Decoder:
            256 channels
            128 channels
            64 channels
            32 channels
        Output:
            Raw logits [B, 1, 256, 256] (no sigmoid applied)
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1, use_bilinear_up: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bilinear_up = use_bilinear_up

        # --- Encoder ---
        self.enc1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Bottleneck ---
        self.bottleneck = DoubleConv(256, 512)

        # --- Decoder ---
        if use_bilinear_up:
            self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.dec1 = DoubleConv(512 + 256, 256)

            self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.dec2 = DoubleConv(256 + 128, 128)

            self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.dec3 = DoubleConv(128 + 64, 64)

            self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.dec4 = DoubleConv(64 + 32, 32)
        else:
            self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
            self.dec1 = DoubleConv(256 + 256, 256)

            self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
            self.dec2 = DoubleConv(128 + 128, 128)

            self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
            self.dec3 = DoubleConv(64 + 64, 64)

            self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
            self.dec4 = DoubleConv(32 + 32, 32)

        # --- Final 1x1 Convolution ---
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw unscaled logits."""
        # Encoder forward
        x1 = self.enc1(x)         # [B, 32, 256, 256]
        p1 = self.pool1(x1)       # [B, 32, 128, 128]

        x2 = self.enc2(p1)        # [B, 64, 128, 128]
        p2 = self.pool2(x2)       # [B, 64, 64, 64]

        x3 = self.enc3(p2)        # [B, 128, 64, 64]
        p3 = self.pool3(x3)       # [B, 128, 32, 32]

        x4 = self.enc4(p3)        # [B, 256, 32, 32]
        p4 = self.pool4(x4)       # [B, 256, 16, 16]

        # Bottleneck forward
        b = self.bottleneck(p4)   # [B, 512, 16, 16]

        # Decoder forward with channel-wise skip connections
        u1 = self.up1(b)          # [B, 256, 32, 32]
        d1 = torch.cat([u1, x4], dim=1)  # [B, 512, 32, 32]
        d1 = self.dec1(d1)        # [B, 256, 32, 32]

        u2 = self.up2(d1)         # [B, 128, 64, 64]
        d2 = torch.cat([u2, x3], dim=1)  # [B, 256, 64, 64]
        d2 = self.dec2(d2)        # [B, 128, 64, 64]

        u3 = self.up3(d2)         # [B, 64, 128, 128]
        d3 = torch.cat([u3, x2], dim=1)  # [B, 128, 128, 128]
        d3 = self.dec3(d3)        # [B, 64, 128, 128]

        u4 = self.up4(d3)         # [B, 32, 256, 256]
        d4 = torch.cat([u4, x1], dim=1)  # [B, 64, 256, 256]
        d4 = self.dec4(d4)        # [B, 32, 256, 256]

        logits = self.final_conv(d4)  # [B, 1, 256, 256]
        return logits

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = UNet(in_channels=3, out_channels=1)
    dummy_input = torch.randn(2, 3, 256, 256)
    out = model(dummy_input)
    print("U-Net Model Created Successfully!")
    print(f"Input shape : {dummy_input.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Trainable Parameters: {model.count_parameters():,}")
