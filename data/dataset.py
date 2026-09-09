"""
data/dataset.py

PyTorch Dataset for the multi-stream AI-generated image detector. Reads
an ablation config (config/full.yaml, config/ela_only.yaml, etc.) to
decide which of the three streams -- content, ela, prnu -- are computed
and returned, so the same dataset class serves every experiment in
experiments/run_experiments.py without branching logic scattered
elsewhere.

Expected config shape -- adjust these keys to match your actual YAML
files if they differ:

    data:
      root: "./data/raw"          # contains real/ and fake/ subfolders
      image_size: 224
      val_ratio: 0.15
      test_ratio: 0.15
      seed: 42
    streams:
      content: true
      ela: true
      prnu: true
    ela:
      quality: 95                 # Handoff Section 3: 95% JPEG quality
      scale: 15.0
    augmentation:
      random_hflip: true
    training:
      batch_size: 32
      num_workers: 4

NOTE on the "prnu" stream: it is the RAW (Resize + ToTensor only, no
ImageNet normalization) image tensor, NOT a precomputed wavelet
residual. The D4 DWT -> learnable soft-threshold -> IDWT -> residual
pipeline (Handoff Section 4) now lives inside models/prnu_branch.py /
models/wavelet_layer.py as a differentiable nn.Module, because the
per-subband threshold is learnable and needs gradients from the
training loss (Handoff Section 13). See those files' docstrings.
data/wavelet.py's NumPy residual functions remain available for
offline visualization only -- they are not used by this dataset.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from data.ela import compute_ela_image
from data.patches import extract_patches
from data.preprocessing import (
    Sample,
    build_content_transform,
    build_prnu_transform,
    index_dataset,
    load_image,
    stratified_split,
    to_unit_tensor,
)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class AIGeneratedImageDataset(Dataset):
    """
    Returns a dict per item:
        {
          "content": FloatTensor [3, H, W]   (if streams.content)
          "ela":     FloatTensor [3, H, W]   (if streams.ela)
          "prnu":    FloatTensor [3, H, W]   (if streams.prnu; raw image --
                                               see module docstring)
          "label":   LongTensor  []          (0 = real, 1 = AI-generated)
        }
    Only the streams enabled in the config are included, so
    no_ela.yaml / ela_only.yaml / prnu_only.yaml / etc. each naturally
    produce the right batch shape for their ablation run.
    """

    def __init__(self, samples: List[Sample], config: dict, train: bool = True):
        self.samples = samples
        self.config = config
        self.train = train

        data_cfg = config.get("data", {})
        self.image_size = data_cfg.get("image_size", 224)

        stream_cfg = config.get("streams", {"content": True, "ela": True, "prnu": True})
        self.use_content = stream_cfg.get("content", True)
        self.use_ela = stream_cfg.get("ela", True)
        self.use_prnu = stream_cfg.get("prnu", True)

        ela_cfg = config.get("ela", {})
        # Handoff Section 3: "JPEG recompression at 95% quality" -- this
        # is a documented requirement, unlike most other hyperparameters.
        self.ela_quality = ela_cfg.get("quality", 95)
        self.ela_scale = ela_cfg.get("scale", 15.0)  # amplification -- NOT specified in doc

        aug_cfg = config.get("augmentation", {})
        # NOT specified in research doc (Handoff Section 19: DATA_AUGMENTATION).
        # A simple horizontal flip is used as a mild, label-preserving
        # default; disable via config if the team wants strictly
        # unaugmented training (safer for forensic signals like PRNU).
        self.random_hflip = aug_cfg.get("random_hflip", True) and train

        self.content_transform = build_content_transform(self.image_size)
        self.prnu_transform = build_prnu_transform(self.image_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        pil_image = load_image(sample.path).resize((self.image_size, self.image_size))

        # Apply any geometric augmentation ONCE, to the shared source
        # image, before branch-specific preprocessing -- so content,
        # ela, and prnu all see the *same* geometry for a given sample.
        # (Previously, only the content stream was flipped, which
        # decorrelates it from ela/prnu for augmented samples.)
        if self.random_hflip and torch.rand(1).item() < 0.5:
            pil_image = pil_image.transpose(Image.FLIP_LEFT_RIGHT)

        item: Dict[str, torch.Tensor] = {}

        if self.use_content:
            item["content"] = self.content_transform(pil_image)

        if self.use_ela:
            ela_map = compute_ela_image(pil_image, quality=self.ela_quality, scale=self.ela_scale)
            item["ela"] = to_unit_tensor(ela_map)

        if self.use_prnu:
            # Raw (minimally-preprocessed) image -- the wavelet residual
            # is computed inside models/prnu_branch.py, not here.
            item["prnu"] = self.prnu_transform(pil_image)

        item["label"] = torch.tensor(sample.label, dtype=torch.long)
        return item


def build_datasets(config: dict, data_root: Optional[str] = None):
    """Index the raw data folder, split it, and return (train_ds, val_ds, test_ds) built from the same config."""
    data_cfg = config.get("data", {})
    root = data_root or data_cfg.get("root", "./data/raw")
    val_ratio = data_cfg.get("val_ratio", 0.15)
    test_ratio = data_cfg.get("test_ratio", 0.15)
    seed = data_cfg.get("seed", 42)

    all_samples = index_dataset(root)
    train_samples, val_samples, test_samples = stratified_split(
        all_samples, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
    )

    train_ds = AIGeneratedImageDataset(train_samples, config, train=True)
    val_ds = AIGeneratedImageDataset(val_samples, config, train=False)
    test_ds = AIGeneratedImageDataset(test_samples, config, train=False)
    return train_ds, val_ds, test_ds


def build_dataloaders(config: dict, data_root: Optional[str] = None) -> Dict[str, DataLoader]:
    """Convenience wrapper used by training/train.py and experiments/run_experiments.py."""
    train_ds, val_ds, test_ds = build_datasets(config, data_root=data_root)
    train_cfg = config.get("training", {})
    batch_size = train_cfg.get("batch_size", 32)
    num_workers = train_cfg.get("num_workers", 4)

    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }


def get_patchwise_input(image: np.ndarray, patch_size: int = 64) -> np.ndarray:
    """
    Optional helper for patch-level (rather than whole-image) fusion
    experiments: turns one HxWxC image into an (N, patch_size,
    patch_size, C) array via data.patches.extract_patches.
    """
    return extract_patches(image, patch_size=patch_size)
