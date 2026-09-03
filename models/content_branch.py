"""
Content Branch — learns higher-level structural/content artifacts
(unnatural smoothness, GAN upsampling patterns, frequency anomalies).

Expects a content-preprocessed image as input (no wavelet denoising
applied — computed upstream, not part of this file's scope).

Architecture (AI Handoff spec, Section 8):
    Block1 -> Block2 -> Block3 -> GAP -> FC -> f_C
Intentionally deeper than ELA/PRNU since it handles higher-level features.

The internal structure of each block (how many conv layers, filters,
kernel size, stride, padding, whether/how downsampling happens) is
explicitly NOT specified in the research doc — the doc even warns not
to assume a VGG/AlexNet-style block just because those are cited as
inspiration. The block below (Conv-BN-ReLU-MaxPool) is a plain,
commonly-used default, not a documented requirement. Confirm with the
team before treating it as final. See Handoff Section 19, "DO NOT INVENT".
"""

import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    """One implementation-default block: Conv -> BN -> ReLU -> MaxPool (downsample)."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(self.relu(self.bn(self.conv(x))))


class ContentBranch(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        # NOT specified in research doc — implementation default (progressive widening)
        block_channels: tuple = (32, 64, 128),
        kernel_size: int = 3,       # NOT specified in research doc
        stride: int = 1,            # NOT specified in research doc
        padding: int = 1,           # NOT specified in research doc
        feature_dim: int = 128,     # "n" in the doc — NOT specified, must match other branches
    ):
        super().__init__()
        assert len(block_channels) == 3, "Handoff doc specifies exactly 3 CNN blocks"

        channels = [in_channels] + list(block_channels)
        self.blocks = nn.Sequential(*[
            _ConvBlock(channels[i], channels[i + 1], kernel_size, stride, padding)
            for i in range(3)
        ])

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(block_channels[-1], feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W) content-preprocessed image tensor.
        returns: f_c, shape (B, feature_dim)
        """
        x = self.blocks(x)
        x = self.gap(x).flatten(1)
        f_c = self.fc(x)
        return f_c
