"""
data/ela.py

Error Level Analysis (ELA): re-compress an image at a known JPEG quality
and measure the pixel-wise difference from the original. Regions that
were edited or AI-generated tend to show a different recompression
error signature than untouched regions -- that signal feeds
models/ela_branch.py.
"""

from __future__ import annotations

import io
from typing import Sequence

import numpy as np
from PIL import Image, ImageChops


def compute_ela_image(
    image: Image.Image,
    quality: int = 90,
    scale: float = 15.0,
) -> np.ndarray:
    """
    Args:
        image: PIL RGB image.
        quality: JPEG quality used for recompression (lower = stronger,
            noisier signal).
        scale: linear amplification applied to the raw difference so the
            signal is visible/learnable.

    Returns:
        float32 HxWx3 array in [0, 255]: the amplified per-pixel
        difference between `image` and its once-recompressed JPEG version.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert("RGB")

    diff = ImageChops.difference(image, recompressed)
    diff_arr = np.asarray(diff, dtype=np.float32)

    return np.clip(diff_arr * scale, 0, 255)


def compute_ela_multi_quality(
    image: Image.Image,
    qualities: Sequence[int] = (70, 80, 90, 95),
    scale: float = 15.0,
) -> np.ndarray:
    """
    Stack ELA maps computed at several JPEG qualities along the channel
    axis. Some forgeries only show up strongly at particular compression
    levels, so this gives ela_branch more signal than one fixed quality.

    Returns: float32 HxWx(3*len(qualities)) array in [0, 255].
    """
    maps = [compute_ela_image(image, quality=q, scale=scale) for q in qualities]
    return np.concatenate(maps, axis=-1)


def ela_to_grayscale_energy(ela_map: np.ndarray) -> np.ndarray:
    """Collapse an HxWxC ELA map to a single-channel HxW energy map (mean over channels)."""
    return ela_map.mean(axis=-1)
