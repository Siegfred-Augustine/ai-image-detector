"""
models/wavelet_layer.py

Hybrid Wavelet Layer — AI Handoff spec, Section 4.

    Input X
      |
    D4 DWT                      (1 level -> LL, LH, HL, HH)
      |
    Learnable Soft Thresholding  (one tau per subband)
      |
    IDWT                        (reconstruct denoised image D)
      |
    Residual = X - D            (-> PRNU CNN Branch, models/prnu_branch.py)

WHY THIS IS A MODEL FILE, NOT A data/ PREPROCESSING STEP
----------------------------------------------------------
Handoff Section 13 is explicit: "PyTorch is the intended primary
framework because the Hybrid Wavelet Layer needs end-to-end gradient
flow." The per-subband threshold tau is *learnable*, so the DWT ->
threshold -> IDWT chain has to be a differentiable nn.Module that sits
inside the network and gets gradients from the classification loss —
it cannot be precomputed once, offline, with plain NumPy (that's what
data/wavelet.py's helper functions do, and they remain useful for
*offline visualization/analysis*, but they are NOT what feeds the PRNU
branch during training; see the note at the top of that file).

D4 FILTER
----------------------------------------------------------
Handoff Section 4 gives the D4 (Daubechies-4, i.e. pywt's "db2") low-pass
decomposition coefficients explicitly:

    L = [0.4830, 0.8365, 0.2241, -0.1294]

The document explicitly states the corresponding **high-pass filter is
NOT specified**. Rather than inventing arbitrary coefficients, this
implementation derives the high-pass filter from the given low-pass
filter using the standard orthogonal-wavelet quadrature-mirror-filter
(QMF) relation:

    g[n] = (-1)^n * h[N - 1 - n],   N = filter length

This is the standard mathematical construction that makes {h, g} a
valid perfect-reconstruction analysis pair for any given orthogonal
low-pass filter h (Daubechies, 1992, "Ten Lectures on Wavelets", ch. 5).
It is not a free implementation choice — it is the unique, well known
way to complete a QMF pair from an orthogonal scaling filter — but it
is flagged here in case the research team has a different convention
in mind (e.g. Daubechies' own sign convention flips g's sign).

DECOMPOSITION LEVEL
----------------------------------------------------------
The doc's diagram shows a single LL/LH/HL/HH split (no further
decomposition of LL), so this implements exactly **1 level**, per
Handoff Section 4 ("Wavelet decomposition level: 1 level is implied").

BOUNDARY HANDLING
----------------------------------------------------------
The classic DWT/IDWT filter-bank achieves *exact* perfect reconstruction
only under periodic (or carefully matched symmetric) boundary handling.
This implementation uses reflect padding for the forward transform and
crops the inverse transform back to the input's exact H, W. This is a
standard engineering approximation (matches common learned-wavelet
literature) — reconstruction is not bit-exact at the image border, but
the whole point of this layer is to produce a *learned* forensic
residual, not a mathematically perfect wavelet reconstruction, so this
tradeoff is reasonable. Flag to the team if bit-exact periodic
reconstruction is required instead.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# D4 / Daubechies-4 / pywt "db2" low-pass decomposition filter,
# exactly as given in Handoff Section 4.
D4_LOWPASS = [0.4830, 0.8365, 0.2241, -0.1294]

# Subband order used consistently throughout this module and by callers.
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

        # Analysis (DWT) filters, as (1, 1, k) buffers -- fixed constants
        # from the documented D4 filter, not learned.
        self.register_buffer("dec_lo", low.view(1, 1, -1))
        self.register_buffer("dec_hi", high.view(1, 1, -1))
        # Synthesis (IDWT) filters. For an orthogonal wavelet the
        # synthesis filters equal the analysis filters (conv_transpose2d
        # performs the adjoint operation internally), so we reuse the
        # same coefficients rather than re-deriving a separate pair.
        self.register_buffer("rec_lo", low.view(1, 1, -1))
        self.register_buffer("rec_hi", high.view(1, 1, -1))

        # 4 separable 2D analysis kernels: LL, LH, HL, HH = outer(row, col)
        # filters, matching Handoff Section 4's LL=L(x)L, LH=H(x)L,
        # HL=L(x)H, HH=H(x)H (row filter listed first).
        self.register_buffer("k_LL", torch.outer(low, low).view(1, 1, self.filter_len, self.filter_len))
        self.register_buffer("k_LH", torch.outer(high, low).view(1, 1, self.filter_len, self.filter_len))
        self.register_buffer("k_HL", torch.outer(low, high).view(1, 1, self.filter_len, self.filter_len))
        self.register_buffer("k_HH", torch.outer(high, high).view(1, 1, self.filter_len, self.filter_len))

        # One learnable threshold per subband (LL, LH, HL, HH), per
        # Handoff Section 4: "One learnable threshold tau per subband."
        # Stored as an unconstrained parameter and mapped through
        # softplus in forward() so tau >= 0 is guaranteed by
        # construction rather than by clamping (keeps gradients smooth).
        raw_init = torch.log(torch.expm1(torch.tensor(init_threshold)))  # inverse-softplus
        self._raw_tau = nn.Parameter(raw_init.expand(4).clone())

        # Padding so that a stride-2, kernel=filter_len conv maps
        # H -> H // 2 exactly (see derivation in the module docstring).
        self._fwd_pad = (self.filter_len - 2) // 2 + (self.filter_len % 2)  # = 1 for filter_len=4
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
        # Crop/pad to the exact original spatial size (boundary handling
        # is approximate -- see module docstring).
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
