from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, random_split

from basil.config import BasilDataConfig
from basil.utils import setup_logging

logger = setup_logging(__name__)


class MemoryDataset(Dataset):
    """
    Standard dataset. Loads entire file into RAM.
    """

    def __init__(self, path: str):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"{path} not found")

        logger.info(f"Loading {self.path} into RAM...")
        # Loading to RAM immediately is necessary for random access (shuffling).
        # We cast to float32 now to avoid casting during the training hot loop.
        self.data = torch.from_numpy(np.load(str(self.path))).float()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]


class LinearMmapDataset(Dataset):
    """
    Streaming dataset using OS Page Cache.
    """

    def __init__(self, path: str):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"{path} not found")

        # We use mmap to allow the OS to swap pages in/out.
        # This allows training on datasets larger than physical RAM.
        self.data = np.load(str(self.path), mmap_mode="r")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # .copy() is critical: it triggers the physical disk read and
        # detaches the array from the mmap backing, preventing memory leaks.
        return torch.from_numpy(self.data[idx].copy())


def get_dataset(cfg: BasilDataConfig) -> tuple[Dataset, Dataset]:
    # The factory pattern abstracts away the I/O complexity (RAM vs Disk) from the trainer.

    ds_cls = LinearMmapDataset if cfg.stream else MemoryDataset
    ds = ds_cls(cfg.path)

    total_len = len(ds)
    val_len = int(total_len * cfg.val_set_size)
    train_len = total_len - val_len

    if cfg.stream:
        # For streaming/mmap, use contiguous blocks to minimize disk seeking
        # Train on the first part, validate on the last part
        train_ds = Subset(ds, range(0, train_len))
        val_ds = Subset(ds, range(train_len, total_len))
        return train_ds, val_ds
    else:
        # For memory, random split is fine (and preferred)
        # Uses global torch seed set in Trainer
        return random_split(ds, [train_len, val_len])
