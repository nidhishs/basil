from dataclasses import dataclass
from typing import Optional


@dataclass
class PreprocessConfig:
    """
    Configuration for the preprocessing pipeline.

    Attributes:
        variance_to_keep: Target proportion of variance to retain (0.0 to 1.0).
        dim_pca_max: Maximum PCA dimensions to keep after reduction.
        eps: Small epsilon for numerical stability.
        seed: Random seed for reproducibility.
    """

    variance_to_keep: float = 0.95
    dim_pca_max: int = 256
    eps: float = 1e-6
    seed: int = 42


@dataclass
class TrainerConfig:
    """
    Configuration for the BASIL trainer.

    Attributes:
        levels: Number of codebook levels.
        k_per_level: Number of codewords per level.
        max_iters: Maximum iterations per level.
        tol: Convergence tolerance for early stopping.
        size_penalty_lambda: Weight for utilization regularization.
        batch_size: Batch size for distance computations.
        device: Device to use ('cuda', 'mps', 'cpu', or None for auto).
        seed: Random seed for reproducibility.
        verbose: Whether to print training progress.
    """

    levels: int = 4
    k_per_level: int = 4096
    max_iters: int = 20
    tol: float = 1e-4
    size_penalty_lambda: float = 0.01
    batch_size: int = 8192
    device: Optional[str] = None
    seed: int = 42
    verbose: bool = True
    log_level: str = "INFO"  # Set to DEBUG for per-step logging
