from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

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


def get_dataset(cfg: BasilDataConfig) -> Dataset:
    # The factory pattern abstracts away the I/O complexity (RAM vs Disk) from the trainer.
    if cfg.stream:
        return LinearMmapDataset(cfg.path)
    return MemoryDataset(cfg.path)
