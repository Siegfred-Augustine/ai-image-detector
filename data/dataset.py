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
      quality: 90
      scale: 15.0
    wavelet:
      wavelet: "db8"
      level: 4
    training:
      batch_size: 32
      num_workers: 4
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from data.ela import compute_ela_image
from data.patches import extract_patches
from data.preprocessing import (
    Sample,
    build_content_transform,
    index_dataset,
    load_image,
    stratified_split,
    to_unit_tensor,
)
from data.wavelet import extract_noise_residual


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class AIGeneratedImageDataset(Dataset):
    """
    Returns a dict per item:
        {
          "content": FloatTensor [3, H, W]   (if streams.content)
          "ela":     FloatTensor [3, H, W]   (if streams.ela)
          "prnu":    FloatTensor [1, H, W]   (if streams.prnu)
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
        self.ela_quality = ela_cfg.get("quality", 90)
        self.ela_scale = ela_cfg.get("scale", 15.0)

        wavelet_cfg = config.get("wavelet", {})
        self.wavelet_name = wavelet_cfg.get("wavelet", "db8")
        self.wavelet_level = wavelet_cfg.get("level", 4)

        self.content_transform = build_content_transform(self.image_size, train=train)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        pil_image = load_image(sample.path).resize((self.image_size, self.image_size))
        image_arr = np.asarray(pil_image, dtype=np.float32)

        item: Dict[str, torch.Tensor] = {}

        if self.use_content:
            item["content"] = self.content_transform(pil_image)

        if self.use_ela:
            ela_map = compute_ela_image(pil_image, quality=self.ela_quality, scale=self.ela_scale)
            item["ela"] = to_unit_tensor(ela_map)

        if self.use_prnu:
            residual = extract_noise_residual(image_arr, wavelet=self.wavelet_name, level=self.wavelet_level)
            residual_gray = residual.mean(axis=-1) if residual.ndim == 3 else residual
            item["prnu"] = to_unit_tensor(residual_gray)

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
