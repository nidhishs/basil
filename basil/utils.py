import logging
from typing import Optional, Union

import numpy as np
import torch


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Return a torch.device. If None, pick best available: cuda > mps > cpu.

    Args:
        device: Device string ('cuda', 'mps', 'cpu') or None for auto.

    Returns:
        torch.device: Selected device.
    """
    checks = {
        "cuda": torch.cuda.is_available,
        "mps": torch.backends.mps.is_available,
        "cpu": lambda: True,
    }
    order = ("cuda", "mps", "cpu")

    if device is None:
        device = next((d for d in order if checks[d]()), "cpu")

    key = device.lower()
    if key not in checks:
        raise ValueError(f"Unknown device: {key}")
    if not checks[key]():
        raise ValueError(f"{key.upper()} requested but not available")

    return torch.device(key)


def to_torch(
    data: Union[np.ndarray, torch.Tensor],
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Convert data to torch tensor.

    Args:
        data: Input data (numpy array or torch tensor).
        device: Target device (None to keep current).
        dtype: Target dtype.

    Returns:
        torch.Tensor: Converted tensor.
    """
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data.astype(np.float32))
    elif isinstance(data, torch.Tensor):
        tensor = data
    else:
        raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(data)}")

    return tensor.to(device=device, dtype=dtype) if device else tensor.to(dtype=dtype)


def to_numpy(data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Convert data to numpy array.

    Args:
        data: Input data (numpy array or torch tensor).

    Returns:
        np.ndarray: Converted array.
    """
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    if isinstance(data, np.ndarray):
        return data
    raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(data)}")


def set_seeds(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_entropy(counts: torch.Tensor) -> float:
    """
    Calculate normalized entropy for utilization measurement.

    Args:
        counts: Count tensor.

    Returns:
        float: Normalized entropy (0 to 1).
    """
    probs = counts.float() / counts.sum()
    probs = probs[probs > 0]
    entropy = -(probs * probs.log()).sum().item()
    max_entropy = np.log(len(counts))
    return entropy / max_entropy if max_entropy > 0 else 0.0


def validate_embeddings(embeddings: Union[np.ndarray, torch.Tensor]) -> None:
    """
    Validate embedding tensor shape and values.

    Args:
        embeddings: Input embeddings.

    Raises:
        ValueError: If embeddings are invalid.
    """

    if isinstance(embeddings, (np.ndarray, torch.Tensor)):
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")
        if embeddings.shape[0] == 0:
            raise ValueError("Empty embeddings")
        if embeddings.shape[1] == 0:
            raise ValueError("Zero-dimensional embeddings")
        if not np.isfinite(embeddings).all():
            raise ValueError("Embeddings contain NaN or Inf")
    else:
        raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(embeddings)}")


def chunk_iterator(n: int, chunk_size: int):
    """
    Generate chunks for batched processing.

    Args:
        n: Total number of items.
        chunk_size: Size of each chunk.

    Yields:
        tuple: (start_idx, end_idx) for each chunk.
    """
    for start_idx in range(0, n, chunk_size):
        end_idx = min(start_idx + chunk_size, n)
        yield start_idx, end_idx


def setup_logger(name: str = "basil", level: str = "INFO") -> logging.Logger:
    """
    Set up a logger with consistent formatting.

    Args:
        name: Logger name.
        level: Logging level.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
