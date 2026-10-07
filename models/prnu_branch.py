"""PRNU branch with optional classical or learnable wavelet residuals."""
import torch
import torch.nn as nn

from .wavelet_layer import HybridWaveletLayer


class PRNUBranch(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        conv1_channels: int = 32,
        conv2_channels: int = 64,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        feature_dim: int = 128,
        conv1_activation: bool = True,
        wavelet_init_threshold: float = 0.05,
        use_hybrid_wavelet: bool = True,
    ):
        super().__init__()

        self.use_hybrid_wavelet = use_hybrid_wavelet
        if use_hybrid_wavelet:
            self.wavelet = HybridWaveletLayer(
                in_channels=in_channels, init_threshold=wavelet_init_threshold
            )
        else:
            self.wavelet = None

        self.conv1 = nn.Conv2d(
            in_channels, conv1_channels, kernel_size, stride, padding
        )
        self.relu1 = nn.ReLU(inplace=True) if conv1_activation else nn.Identity()

        self.conv2 = nn.Conv2d(
            conv1_channels, conv2_channels, kernel_size, stride, padding
        )
        self.bn2 = nn.BatchNorm2d(conv2_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(conv2_channels, feature_dim)

    def forward(self, x: torch.Tensor, return_residual: bool = False):
        """
        x: (B, C, H, W). If use_hybrid_wavelet=True (default), this is
           the raw/minimally-preprocessed image tensor, and the wavelet
           residual is computed here. If use_hybrid_wavelet=False, x is
           assumed to already BE the residual (e.g. from
           data/classical_prnu.py), and is used as-is.
        return_residual: if True, also return the residual W and the
            learned tau values (tau is None in classical mode, since
            there's no learnable threshold in that path).
        returns:
            f_prnu, shape (B, feature_dim)
            (optionally) (residual, tau) if return_residual=True
        """
        if self.use_hybrid_wavelet:
            residual, tau = self.wavelet(x)
        else:
            residual, tau = x, None  # x is already a precomputed residual

        h = self.relu1(self.conv1(residual))
        h = self.relu2(self.bn2(self.conv2(h)))
        h = self.gap(h).flatten(1)
        f_prnu = self.fc(h)

        if return_residual:
            return f_prnu, (residual, tau)
        return f_prnu
