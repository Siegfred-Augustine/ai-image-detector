"""
PRNU Branch — learns sensor-noise-residual patterns.

Expects a precomputed wavelet residual as input:
    W = X - IDWT(SoftThreshold(DWT(X)))
(computed upstream by data/wavelet.py — not part of this file's scope).

Architecture (AI Handoff spec, Section 7):
    Conv1 -> (NO BatchNorm) -> Conv2 -> BatchNorm -> ReLU -> GAP -> FC -> f_PRNU

DESIGN CONSTRAINT, explicit in the handoff doc:
    Do NOT put BatchNorm after Conv1. The residual's raw amplitude
    carries forensic information; BatchNorm would normalize it away.
    This is enforced structurally below, not just by convention.

The doc's diagram does not specify an activation after Conv1 either.
An activation is included here by default (`conv1_activation=True`)
because stacking two linear convolutions with no nonlinearity between
them collapses to a single linear layer — but this is an
implementation judgment call, not a documented requirement, so it's
exposed as a toggle. Filter counts, kernel size, stride, padding, and
the output dimension n are likewise NOT specified in the research doc.
See Handoff Section 19, "DO NOT INVENT".
"""

import torch
import torch.nn as nn


class PRNUBranch(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        conv1_channels: int = 32,     # NOT specified in research doc — implementation default
        conv2_channels: int = 64,     # NOT specified in research doc — implementation default
        kernel_size: int = 3,         # NOT specified in research doc — implementation default
        stride: int = 1,              # NOT specified in research doc — implementation default
        padding: int = 1,             # NOT specified in research doc — implementation default
        feature_dim: int = 128,       # "n" in the doc — NOT specified, must match other branches
        conv1_activation: bool = True,  # judgment call — doc's diagram doesn't mention one
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, conv1_channels, kernel_size, stride, padding)
        # Intentionally NO BatchNorm here — do not add one.
        self.relu1 = nn.ReLU(inplace=True) if conv1_activation else nn.Identity()

        self.conv2 = nn.Conv2d(conv1_channels, conv2_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm2d(conv2_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(conv2_channels, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W) precomputed wavelet residual tensor W.
        returns: f_prnu, shape (B, feature_dim)
        """
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.gap(x).flatten(1)
        f_prnu = self.fc(x)
        return f_prnu
