"""Content branch for higher-level image features."""
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
        block_channels: tuple = (32, 64, 128),
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        feature_dim: int = 128,
        baseline_mode: bool = False,
    ):
        super().__init__()
        self.baseline_mode = baseline_mode

        if baseline_mode:
            self.blocks = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=5, stride=1, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(128, feature_dim)
            return

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
