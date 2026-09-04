"""
data/patches.py

Patch extraction utilities. The multi-stream model can reason at both
whole-image and patch level: PRNU/ELA signals are often more
discriminative in small, texture-poor regions, and pixel-wise fusion
benefits from local patches. This module turns a full image (or any
aligned per-stream map) into fixed-size patches.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np


def extract_patches(
    image: np.ndarray,
    patch_size: int = 64,
    stride: Optional[int] = None,
) -> np.ndarray:
    """
    Extract all patch_size x patch_size patches from `image` on a regular
    grid, dropping partial patches at the border.

    Args:
        image: HxW or HxWxC array.
        patch_size: side length of each square patch.
        stride: step between patches; defaults to patch_size (no overlap).

    Returns:
        Array of shape (N, patch_size, patch_size[, C]).
    """
    stride = stride or patch_size
    h, w = image.shape[:2]
    patches: List[np.ndarray] = []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patches.append(image[y:y + patch_size, x:x + patch_size, ...])

    if not patches:
        raise ValueError(
            f"Image of size {(h, w)} is smaller than patch_size={patch_size}; "
            "resize the image before patch extraction."
        )
    return np.stack(patches, axis=0)


def extract_random_patches(
    image: np.ndarray,
    patch_size: int = 64,
    num_patches: int = 16,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Sample `num_patches` random patch_size x patch_size patches from
    `image`. Used for training-time augmentation, where dense grid
    extraction (extract_patches) would be too slow/redundant per epoch.
    """
    rng = np.random.default_rng(seed)
    h, w = image.shape[:2]
    if h < patch_size or w < patch_size:
        raise ValueError(f"Image of size {(h, w)} is smaller than patch_size={patch_size}.")

    ys = rng.integers(0, h - patch_size + 1, size=num_patches)
    xs = rng.integers(0, w - patch_size + 1, size=num_patches)
    return np.stack(
        [image[y:y + patch_size, x:x + patch_size, ...] for y, x in zip(ys, xs)],
        axis=0,
    )


def patch_grid_size(image_size: int, patch_size: int = 64, stride: Optional[int] = None) -> int:
    """Number of patches per side that extract_patches() produces for a square image_size input."""
    stride = stride or patch_size
    return (image_size - patch_size) // stride + 1


def select_high_variance_patches(patches: np.ndarray, top_k: int) -> np.ndarray:
    """
    Keep the `top_k` patches with the highest pixel variance -- a cheap
    heuristic for skipping flat, uninformative regions (e.g. blank sky)
    so the model focuses on textured content.
    """
    variances = patches.reshape(patches.shape[0], -1).var(axis=1)
    top_idx = np.argsort(variances)[-top_k:][::-1]
    return patches[top_idx]
