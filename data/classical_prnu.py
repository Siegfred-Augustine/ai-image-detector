"""
data/classical_prnu.py

Classical (non-learnable) PRNU-style residual extraction, matching the
baseline paper's approach (Martin-Rodriguez et al., 2023, "Detection of
AI-Created Images Using Pixel-Wise Feature Extraction and Convolutional
Neural Networks", Sensors 23(22):9037):

    1. Convert to grayscale.
    2. Denoise with a wavelet-domain adaptive filter.
    3. Residual W = grayscale - denoise(grayscale).

USED ONLY by config/prnu_only.yaml (via PRNUBranch(use_hybrid_wavelet=False)),
as an alternative to this project's own learnable Hybrid Wavelet Layer
(models/wavelet_layer.py), which every other config (full, no_ela,
no_content, etc.) continues to use completely unchanged.

HONEST CAVEAT: the paper's residual specifically comes from a Matlab
implementation (Goljan's "Camera Fingerprint" toolbox) using a
Wavelet-Transform-based Wiener filter. That exact tool isn't available
here. This implementation uses BayesShrink adaptive wavelet denoising
(scikit-image's `denoise_wavelet`), the standard, well-known open-source
technique in the same family (wavelet-domain adaptive denoising ->
residual) -- but it is NOT a byte-for-byte reproduction of their
specific toolbox. State this plainly in your methodology section rather
than implying exact replication.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.restoration import denoise_wavelet


def extract_classical_prnu(pil_image: Image.Image) -> np.ndarray:
    """
    Args:
        pil_image: RGB PIL image (ideally already center-cropped to the
            paper's 512x512 via data.preprocessing.center_crop_or_resize).
    Returns:
        (H, W) float32 residual array, NOT normalized to [0, 1] --
        pass through data.preprocessing.to_unit_tensor for that.
    """
    gray = rgb2gray(np.asarray(pil_image)).astype(np.float32)  # [0, 1]
    denoised = denoise_wavelet(
        gray,
        method="BayesShrink",
        mode="soft",
        rescale_sigma=True,
    ).astype(np.float32)
    residual = gray - denoised
    return residual
