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
from typing import List, Tuple

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
) -> List[Sample]:
    """
    Scan `root_dir` for two class subfolders and return a flat list of
    Sample(path, label). Assumes:
        root_dir/
            real/*.jpg
            fake/*.jpg
    Adjust `real_dirname` / `fake_dirname` if your raw data layout differs.
    """
    samples: List[Sample] = []
    for dirname, label in ((real_dirname, 0), (fake_dirname, 1)):
        class_dir = os.path.join(root_dir, dirname)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"Expected class folder not found: {class_dir}")
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith(VALID_EXTENSIONS):
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


def build_content_transform(image_size: int = 224, train: bool = True) -> T.Compose:
    """
    Transform for the raw 'content' stream (models/content_branch.py).
    Uses ImageNet normalization since the content branch is typically a
    pretrained-backbone-style CNN.
    """
    ops = [T.Resize((image_size, image_size))]
    if train:
        ops.append(T.RandomHorizontalFlip(p=0.5))
    ops += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(ops)


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
