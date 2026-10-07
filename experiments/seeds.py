"""Shared seed values used for paired model comparisons."""
from __future__ import annotations

import os
import random
from typing import List

import numpy as np
import torch

EXPERIMENT_SEEDS: List[int] = [42, 123, 2024]


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Seed every source of randomness the pipeline touches: Python's
    `random`, NumPy, and PyTorch (CPU + all CUDA devices).

    Args:
        seed: the seed value (one of EXPERIMENT_SEEDS during the
            default statistical protocol, or any int for ad-hoc runs).
        deterministic: if True, also request deterministic cuDNN
            kernels. This can slow training down (cuDNN falls back to
            slower deterministic algorithms) but makes runs bit-for-bit
            reproducible given the same seed -- worth it for the
            statistical-comparison protocol in Handoff Section 17,
            where run-to-run variance should come only from the
            documented "different random seeds," not from
            non-determinism in the CUDA kernels themselves.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """
    DataLoader `worker_init_fn`: reseeds NumPy/random inside each
    worker process from PyTorch's per-worker base seed, so multi-worker
    data loading doesn't silently reintroduce non-determinism that
    `set_seed` alone can't reach (each worker is a separate process).
    Usage: DataLoader(..., worker_init_fn=seed_worker, generator=g)
    where `g` is a torch.Generator seeded via g.manual_seed(seed).
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
