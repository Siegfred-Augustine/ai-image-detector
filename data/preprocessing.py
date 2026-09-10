"""
data/preprocessing.py

Shared preprocessing utilities: image loading, dataset indexing (scanning
real/fake folders), train/val/test splitting, and the transform used for
the raw 'content' stream. ela.py, wavelet.py, and patches.py build on top
of what's loaded here.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass
class Sample:
    path: str
    label: int  # 0 = real, 1 = AI-generated


def load_image(path: str) -> Image.Image:
    """Load an image from disk as RGB."""
    return Image.open(path).convert("RGB")


def index_dataset(
    root_dir: str,
    real_dirname: str = "real",
    fake_dirname: str = "fake",
    max_per_class: Optional[int] = None,
    seed: int = 42,
) -> List[Sample]:
    """
    Scan `root_dir` for two class subfolders and return a flat list of
    Sample(path, label). Assumes:
        root_dir/
            real/*.jpg
            fake/*.jpg
    Adjust `real_dirname` / `fake_dirname` if your raw data layout differs.

    Args:
        max_per_class: if set, randomly subsample down to at most this
            many images per class (real/fake), BEFORE the train/val/test
            split -- so val_ratio/test_ratio still apply proportionally
            to the smaller pool. Useful for dataset-size experiments
            (e.g. "how does accuracy change with 500 vs 5000 images per
            class") without having to physically move files around.
            None (default) uses every image found.
        seed: controls which images get kept when subsampling, so the
            same `max_per_class` value always picks the same subset.
    """
    rng = random.Random(seed)
    samples: List[Sample] = []
    for dirname, label in ((real_dirname, 0), (fake_dirname, 1)):
        class_dir = os.path.join(root_dir, dirname)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"Expected class folder not found: {class_dir}")
        class_files = sorted(f for f in os.listdir(class_dir) if f.lower().endswith(VALID_EXTENSIONS))
        if max_per_class is not None and len(class_files) > max_per_class:
            class_files = rng.sample(class_files, max_per_class)
        for fname in class_files:
            samples.append(Sample(path=os.path.join(class_dir, fname), label=label))
    if not samples:
        raise RuntimeError(f"No images found under {root_dir} ({real_dirname}/, {fake_dirname}/)")
    return samples


def stratified_split(
    samples: List[Sample],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    """Split samples into train/val/test while preserving class balance."""
    rng = random.Random(seed)
    by_label = {0: [], 1: []}
    for s in samples:
        by_label[s.label].append(s)

    train, val, test = [], [], []
    for group in by_label.values():
        group = group[:]
        rng.shuffle(group)
        n = len(group)
        n_val = int(n * val_ratio)
        n_test = int(n * test_ratio)
        val.extend(group[:n_val])
        test.extend(group[n_val:n_val + n_test])
        train.extend(group[n_val + n_test:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def build_content_transform(image_size: int = 224) -> T.Compose:
    """
    Transform for the 'content' stream (models/content_branch.py).
    Uses ImageNet normalization since the content branch is a
    from-scratch CNN operating on natural RGB statistics.

    Deliberately does NOT include random flip/augmentation here — any
    geometric augmentation must be applied identically to the ela and
    prnu streams too (they're derived from the same underlying image),
    so flipping is handled once, upstream, in
    data/dataset.py:AIGeneratedImageDataset.__getitem__ and applied to
    the shared PIL image before any branch-specific transform runs.
    """
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_prnu_transform(image_size: int = 224) -> T.Compose:
    """
    Transform for the 'prnu' stream (models/prnu_branch.py).

    Resize + ToTensor ONLY — no ImageNet normalization. The Handoff doc
    (Section 2) is explicit that preprocessing "should remain minimal
    and non-destructive because forensic information, especially PRNU,
    must be preserved." Per-channel mean/std normalization would rescale
    the exact pixel amplitudes that the Hybrid Wavelet Layer
    (models/wavelet_layer.py) needs to compute the noise residual, so
    it is intentionally skipped here (unlike build_content_transform).
    """
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])


def to_unit_tensor(arr: np.ndarray) -> torch.Tensor:
    """
    Convert an HxW or HxWxC array (e.g. an ELA map or a wavelet/PRNU
    residual) into a [0, 1]-scaled CxHxW torch tensor. Forensic streams
    are intentionally NOT shifted with ImageNet stats -- their signal is
    the residual itself, not natural image content.
    """
    arr = arr.astype(np.float32)
    if arr.max() > 1.0 + 1e-6 or arr.min() < -1.0 - 1e-6:
        # Residuals can be negative; rescale to [0, 1] by min-max instead
        # of a flat /255 when values fall outside the standard range.
        lo, hi = arr.min(), arr.max()
        arr = (arr - lo) / (hi - lo + 1e-8) if hi > lo else np.zeros_like(arr)

    if arr.ndim == 2:
        arr = arr[None, :, :]
    else:
        arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(np.ascontiguousarray(arr))
