"""
ELA Branch — learns JPEG recompression / compression-artifact patterns.

Expects a precomputed ELA image as input:
    ELA = image - JPEG^-1[JPEG(image, quality=95)]
(computed upstream by data/ela.py — not part of this file's scope).

Architecture (AI Handoff spec, Section 6):
    Conv1 -> BatchNorm -> ReLU -> Conv2 -> BatchNorm -> ReLU -> GAP -> FC -> f_ELA

Everything the research doc leaves unspecified (filter counts, kernel
size, stride, padding, output dimension n) is exposed as a constructor
argument with a clearly-marked default. These are implementation
choices, not requirements from the methodology — confirm with the team
before treating them as final. See Handoff Section 19, "DO NOT INVENT".
"""

import torch
import torch.nn as nn


class ELABranch(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        conv1_channels: int = 32,   # NOT specified in research doc — implementation default
        conv2_channels: int = 64,   # NOT specified in research doc — implementation default
        kernel_size: int = 3,       # NOT specified in research doc — implementation default
        stride: int = 1,            # NOT specified in research doc — implementation default
        padding: int = 1,           # NOT specified in research doc — implementation default
        feature_dim: int = 128,     # "n" in the doc — NOT specified, must match other branches
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, conv1_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(conv1_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(conv1_channels, conv2_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm2d(conv2_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(conv2_channels, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W) precomputed ELA image tensor.
        returns: f_ela, shape (B, feature_dim)
        """
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.gap(x).flatten(1)
        f_ela = self.fc(x)
        return f_ela
