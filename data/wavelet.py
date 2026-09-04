"""
data/wavelet.py

Wavelet-domain utilities, covering both uses implied by the thesis title
("Wavelet-Enhanced Pixel-Wise Feature Fusion"):

1. Wavelet denoising -> noise residual extraction, used as the PRNU-style
   input to models/prnu_branch.py. True camera-PRNU reference patterns
   aren't available for arbitrary web-sourced images, so we follow the
   standard wavelet-based-denoising-filter approach (Lyu & Farid /
   Lukas et al.) to estimate the sensor/generator noise residual instead.
2. Multi-level DWT detail maps, upsampled back to pixel resolution so
   they can be concatenated with the content stream before fusion.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pywt


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    return image.mean(axis=-1) if image.ndim == 3 else image


def wavelet_denoise(image: np.ndarray, wavelet: str = "db8", level: int = 4) -> np.ndarray:
    """
    Soft-threshold wavelet denoising (per-channel if image is HxWxC).
    Threshold follows the VisuShrink universal-threshold rule, with sigma
    estimated from the finest detail sub-band's median absolute deviation.
    """

    def _denoise_channel(channel: np.ndarray) -> np.ndarray:
        coeffs = pywt.wavedec2(channel, wavelet=wavelet, level=level)
        cA, detail_coeffs = coeffs[0], coeffs[1:]

        finest_cH = detail_coeffs[-1][0]
        sigma = np.median(np.abs(finest_cH)) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(channel.size))

        denoised_details = [
            tuple(pywt.threshold(band, threshold, mode="soft") for band in level_bands)
            for level_bands in detail_coeffs
        ]
        return pywt.waverec2([cA] + denoised_details, wavelet=wavelet)

    if image.ndim == 2:
        out = _denoise_channel(image)
        return out[: image.shape[0], : image.shape[1]]

    h, w = image.shape[:2]
    channels = [_denoise_channel(image[..., c])[:h, :w] for c in range(image.shape[-1])]
    return np.stack(channels, axis=-1)


def extract_noise_residual(image: np.ndarray, wavelet: str = "db8", level: int = 4) -> np.ndarray:
    """
    PRNU-style noise residual: original image minus its wavelet-denoised
    version. Consumed by models/prnu_branch.py.
    """
    denoised = wavelet_denoise(image, wavelet=wavelet, level=level)
    return image.astype(np.float32) - denoised.astype(np.float32)


def dwt_detail_maps(
    image: np.ndarray,
    wavelet: str = "haar",
    level: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Single-level 2D DWT on a grayscale version of `image`.
    Returns (LL, LH, HL, HH); each is roughly half the input's H and W
    (approximation, and horizontal/vertical/diagonal detail).
    """
    gray = _to_grayscale(image)
    LL, (LH, HL, HH) = pywt.dwt2(gray, wavelet)
    return LL, LH, HL, HH


def wavelet_pixel_features(image: np.ndarray, wavelet: str = "haar") -> np.ndarray:
    """
    Build a pixel-resolution feature stack for the fusion step: each
    detail sub-band is nearest-neighbor-upsampled back to the original
    HxW so it can be concatenated channel-wise with the content stream.

    Returns: float32 HxWx3 array (LH, HL, HH upsampled to input size).
    """
    h, w = image.shape[:2]
    _, LH, HL, HH = dwt_detail_maps(image, wavelet=wavelet, level=1)

    def _upsample(band: np.ndarray) -> np.ndarray:
        return np.kron(band, np.ones((2, 2)))[:h, :w]

    stacked = np.stack([_upsample(LH), _upsample(HL), _upsample(HH)], axis=-1)
    return stacked.astype(np.float32)
