"""Differentiable 1-level D4 wavelet residual layer used by the PRNU branch."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

D4_LOWPASS = [0.4830, 0.8365, 0.2241, -0.1294]
SUBBAND_ORDER = ("LL", "LH", "HL", "HH")


def _qmf_highpass(lowpass: torch.Tensor) -> torch.Tensor:
    """Derive the orthogonal-wavelet high-pass filter from a low-pass filter via QMF."""
    n = lowpass.shape[0]
    idx = torch.arange(n)
    signs = torch.where(idx % 2 == 0, 1.0, -1.0)
    return signs * lowpass.flip(0)


def soft_threshold(x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """
    ST(x, tau) = sign(x) * max(|x| - tau, 0)     -- Handoff Section 4.
    `tau` is broadcastable against `x` (here: one scalar per subband).
    """
    return torch.sign(x) * torch.clamp(x.abs() - tau, min=0.0)


class HybridWaveletLayer(nn.Module):
    """
    Differentiable 1-level D4 DWT -> learnable per-subband soft
    thresholding -> IDWT -> residual, feeding the PRNU branch.

    Args:
        in_channels: number of channels of the input image (3 for RGB).
        init_threshold: initial value for every subband's tau, passed
            through softplus so tau stays >= 0 throughout training
            regardless of the underlying (unconstrained) parameter's
            sign. NOT specified in the research doc -- implementation
            default; small and positive so the layer starts close to
            an identity (near-zero shrinkage) and learns how aggressive
            to be. See Handoff Section 19, "DO NOT INVENT".

    Shapes:
        forward(x): x is (B, C, H, W) with H, W even (224 satisfies this).
        Returns residual W, same shape (B, C, H, W), plus the learned
        (post-softplus) tau values for logging/inspection.
    """

    def __init__(self, in_channels: int = 3, init_threshold: float = 0.05):
        super().__init__()

        low = torch.tensor(D4_LOWPASS, dtype=torch.float32)
        high = _qmf_highpass(low)
        self.filter_len = low.shape[0]  # 4
        self.in_channels = in_channels

        self.register_buffer("dec_lo", low.view(1, 1, -1))
        self.register_buffer("dec_hi", high.view(1, 1, -1))
        self.register_buffer("rec_lo", low.view(1, 1, -1))
        self.register_buffer("rec_hi", high.view(1, 1, -1))

        self.register_buffer("k_LL", torch.outer(low, low).view(1, 1, self.filter_len, self.filter_len))
        self.register_buffer("k_LH", torch.outer(high, low).view(1, 1, self.filter_len, self.filter_len))
        self.register_buffer("k_HL", torch.outer(low, high).view(1, 1, self.filter_len, self.filter_len))
        self.register_buffer("k_HH", torch.outer(high, high).view(1, 1, self.filter_len, self.filter_len))

        raw_init = torch.log(torch.expm1(torch.tensor(init_threshold)))
        self._raw_tau = nn.Parameter(raw_init.expand(4).clone())

        self._fwd_pad = (self.filter_len - 2) // 2 + (self.filter_len % 2)
        self._inv_pad = self._fwd_pad

    @property
    def tau(self) -> torch.Tensor:
        """Current (non-negative) per-subband thresholds, shape (4,) in SUBBAND_ORDER."""
        return F.softplus(self._raw_tau)

    def _depthwise(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """Apply a (1,1,k,k) kernel depthwise (same kernel per channel) via conv2d."""
        c = x.shape[1]
        w = kernel.expand(c, 1, -1, -1)
        return F.conv2d(x, w, stride=2, padding=0, groups=c)

    def _depthwise_transpose(self, x: torch.Tensor, kernel: torch.Tensor, out_hw) -> torch.Tensor:
        """Adjoint of `_depthwise`: upsample by 2 and apply the kernel via conv_transpose2d."""
        c = x.shape[1]
        w = kernel.expand(c, 1, -1, -1)
        out = F.conv_transpose2d(x, w, stride=2, padding=self._inv_pad, groups=c)
        h, w_ = out_hw
        out = out[..., :h, :w_]
        if out.shape[-2] < h or out.shape[-1] < w_:
            out = F.pad(out, (0, max(0, w_ - out.shape[-1]), 0, max(0, h - out.shape[-2])))
        return out

    def dwt(self, x: torch.Tensor) -> dict:
        """1-level D4 DWT. Returns a dict of 4 subbands, each (B, C, H/2, W/2)."""
        pad = self._fwd_pad
        x_padded = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        return {
            "LL": self._depthwise(x_padded, self.k_LL),
            "LH": self._depthwise(x_padded, self.k_LH),
            "HL": self._depthwise(x_padded, self.k_HL),
            "HH": self._depthwise(x_padded, self.k_HH),
        }

    def idwt(self, subbands: dict, out_hw) -> torch.Tensor:
        """Inverse of `dwt`: reconstruct a (B, C, H, W) image from the 4 subbands."""
        recon = 0.0
        for name, kernel in (("LL", self.k_LL), ("LH", self.k_LH), ("HL", self.k_HL), ("HH", self.k_HH)):
            recon = recon + self._depthwise_transpose(subbands[name], kernel, out_hw)
        return recon

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, C, H, W) input image tensor (H, W should be even; 224
               satisfies this).
        Returns:
            residual: (B, C, H, W) = X - D, the input to the PRNU branch.
            tau: (4,) the current learned thresholds, in SUBBAND_ORDER
                 (useful for logging/inspection, e.g. TensorBoard).
        """
        b, c, h, w = x.shape
        subbands = self.dwt(x)

        tau = self.tau  # (4,), softplus-mapped, >= 0
        thresholded = {
            name: soft_threshold(subbands[name], tau[i])
            for i, name in enumerate(SUBBAND_ORDER)
        }

        denoised = self.idwt(thresholded, out_hw=(h, w))
        residual = x - denoised
        return residual, tau.detach()
