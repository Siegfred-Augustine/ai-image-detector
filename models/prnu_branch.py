"""
PRNU Branch — learns sensor-noise-residual patterns.

Input: the RAW (minimally-preprocessed) content image tensor, shape
(B, C, H, W). This branch internally runs the Hybrid Wavelet Layer
(models/wavelet_layer.py — 1-level D4 DWT, learnable per-subband soft
threshold, IDWT) to compute the wavelet residual W = X - D, per Handoff
Section 4, and only THEN feeds W through the two-conv PRNU CNN
described in Handoff Section 7:

    Raw image X
       |
    Hybrid Wavelet Layer  ->  W = X - D          (Section 4)
       |
    Conv1 -> (NO BatchNorm) -> Conv2 -> BatchNorm -> ReLU -> GAP -> FC -> f_PRNU

WHY THE WAVELET STEP LIVES INSIDE THIS BRANCH (not precomputed upstream)
--------------------------------------------------------------------
Handoff Section 4 requires a *learnable* per-subband threshold, and
Section 13 says PyTorch is required specifically "because the Hybrid
Wavelet Layer needs end-to-end gradient flow." A one-off NumPy residual
computed in the data pipeline cannot receive gradients from the
classification loss, so W must be produced by an nn.Module that sits
inside the forward pass — hence `HybridWaveletLayer` is instantiated
here rather than in data/wavelet.py. See that file's docstring for the
NumPy utilities that remain available for *offline* analysis only.

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

from .wavelet_layer import HybridWaveletLayer


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
        wavelet_init_threshold: float = 0.05,  # NOT specified — see wavelet_layer.py
    ):
        super().__init__()

        # Section 4: D4 DWT -> learnable soft threshold -> IDWT -> residual.
        self.wavelet = HybridWaveletLayer(in_channels=in_channels, init_threshold=wavelet_init_threshold)

        self.conv1 = nn.Conv2d(in_channels, conv1_channels, kernel_size, stride, padding)
        # Intentionally NO BatchNorm here — do not add one.
        self.relu1 = nn.ReLU(inplace=True) if conv1_activation else nn.Identity()

        self.conv2 = nn.Conv2d(conv1_channels, conv2_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm2d(conv2_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(conv2_channels, feature_dim)

    def forward(self, x: torch.Tensor, return_residual: bool = False):
        """
        x: (B, C, H, W) raw/minimally-preprocessed image tensor (NOT a
           precomputed residual — the wavelet residual is computed here).
        return_residual: if True, also return the wavelet residual W and
            the learned tau values (useful for visualization/debugging).
        returns:
            f_prnu, shape (B, feature_dim)
            (optionally) (residual, tau) if return_residual=True
        """
        residual, tau = self.wavelet(x)

        h = self.relu1(self.conv1(residual))
        h = self.relu2(self.bn2(self.conv2(h)))
        h = self.gap(h).flatten(1)
        f_prnu = self.fc(h)

        if return_residual:
            return f_prnu, (residual, tau)
        return f_prnu
