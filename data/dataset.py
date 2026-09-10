"""
data/wavelet.py

OFFLINE / EXPLORATORY wavelet-domain utilities -- built on NumPy + PyWavelets,
NOT used by the main training pipeline.

IMPORTANT: these functions do NOT power models/prnu_branch.py.
The Handoff doc (Section 4) requires the PRNU residual to be computed by
a *learnable* per-subband soft threshold with end-to-end gradient flow
(Section 13), which means it has to live inside the network as a real
nn.Module -- see models/wavelet_layer.py:HybridWaveletLayer, which
implements the documented D4 / 1-level / learnable-threshold pipeline
and is called directly by models/prnu_branch.py on every forward pass.
data/dataset.py accordingly feeds PRNUBranch a raw (minimally
preprocessed) image, not a residual from this file.

What THIS file is still useful for:
    - Offline visualization / EDA notebooks (e.g. "what does a fixed,
      non-learned wavelet residual look like for this image?").
    - A fixed-filter (db8, multi-level, universal-threshold) baseline
      if the team ever wants to compare a classical, non-learned
      residual against the learnable Hybrid Wavelet Layer's output.
    - `wavelet_pixel_features`, which produces pixel-resolution detail
      maps for exploratory pixel-wise fusion outside the main model.

1. Wavelet denoising -> noise residual extraction: a classical (non-
   learned) estimate of the sensor/generator noise residual, following
   the standard wavelet-based-denoising-filter approach (Lyu & Farid /
   Lukas et al.). True camera-PRNU reference patterns aren't available
   for arbitrary web-sourced images, hence this residual-based proxy.
2. Multi-level DWT detail maps, upsampled back to pixel resolution so
   they can be concatenated with the content stream before fusion.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pywt

def build_datasets(config: dict, data_root: Optional[str] = None):
    """Index the raw data folder, split it, and return (train_ds, val_ds, test_ds) built from the same config."""
    data_cfg = config.get("data", {})
    root = data_root or data_cfg.get("root", "./data/raw")
    val_ratio = data_cfg.get("val_ratio", 0.15)
    test_ratio = data_cfg.get("test_ratio", 0.15)
    seed = data_cfg.get("seed", 42)
    # NOT specified in research doc -- optional dataset-size control.
    # None (default) uses every image found under root/real, root/fake.
    max_samples_per_class = data_cfg.get("max_samples_per_class", None)

    all_samples = index_dataset(root, max_per_class=max_samples_per_class, seed=seed)
    train_samples, val_samples, test_samples = stratified_split(
        all_samples, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
    )

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
