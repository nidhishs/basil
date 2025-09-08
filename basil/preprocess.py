"""
Preprocessing pipeline for BASIL.

Provides normalization, PCA, and rescaling for embeddings.
"""

from typing import Any, Dict, Optional

import torch

from .config import PreprocessConfig
from .utils import set_seeds


class Preprocessor:
    """
    Fixed preprocessing pipeline for embeddings.

    Applies centering, PCA rotation, and per-axis scaling.
    Supports clean inverse transformation.
    """

    def __init__(self, cfg: PreprocessConfig):
        """
        Initialize preprocessor.

        Args:
            cfg: Preprocessing configuration.
        """

        self.cfg = cfg
        self.mean: Optional[torch.Tensor] = None
        self.components: Optional[torch.Tensor] = None
        self.scales: Optional[torch.Tensor] = None
        self.dim_in: Optional[int] = None
        self.dim_pca: Optional[int] = None
        self._fitted = False

    def fit(self, vectors: torch.Tensor) -> None:
        """
        Fit preprocessing parameters.

        Computes mean, PCA components, and scales from data.

        Args:
            vectors: Input vectors [N, D].

        Raises:
            ValueError: If vectors are invalid.
        """

        if vectors.ndim != 2:
            raise ValueError(f"Expected 2D tensor, got shape {vectors.shape}")

        set_seeds(self.cfg.seed)

        n, d = vectors.shape
        self.dim_in = d
        device = vectors.device
        dtype = vectors.dtype

        # Center data
        self.mean = vectors.mean(dim=0)
        centered = vectors - self.mean

        # Handle degenerate case
        if n == 1:
            # Single vector: use identity transform
            self.dim_pca = min(d, self.cfg.dim_pca_max)
            self.components = torch.eye(d, self.dim_pca, device=device, dtype=dtype)
            self.scales = torch.ones(self.dim_pca, device=device, dtype=dtype)
            self._fitted = True
            return

        # Compute covariance and eigendecomposition
        cov = (centered.T @ centered) / (n - 1)
        cov = cov + torch.eye(d, device=device, dtype=dtype) * self.cfg.eps

        # Eigendecomposition (sorted by eigenvalue)
        eigvals, eigvecs = torch.linalg.eigh(cov)

        # Sort in descending order
        idx = eigvals.argsort(descending=True)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Determine dimensionality based on variance
        eigvals = eigvals.clamp(min=self.cfg.eps)
        total_var = eigvals.sum()
        cumsum = eigvals.cumsum(dim=0)
        variance_ratio = cumsum / total_var

        # Find dimension that captures target variance
        n_components = (variance_ratio >= self.cfg.variance_to_keep).nonzero()[
            0
        ].item() + 1
        n_components = min(n_components, self.cfg.dim_pca_max, d)

        self.dim_pca = n_components
        self.components = eigvecs[:, :n_components].T  # [dim_pca, dim_in]

        # Project and compute scales
        projected = centered @ self.components.T
        stds = projected.std(dim=0)
        self.scales = 1.0 / (stds + self.cfg.eps)

        self._fitted = True

    def transform(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Apply preprocessing transformation.

        Args:
            vectors: Input vectors [N, D].

        Returns:
            Transformed vectors [N, D_pca].

        Raises:
            RuntimeError: If not fitted.
            ValueError: If dimension mismatch.
        """

        if not self._fitted:
            raise RuntimeError("Preprocessor not fitted")

        if vectors.shape[-1] != self.dim_in:
            raise ValueError(
                f"Dimension mismatch: expected {self.dim_in}, "
                f"got {vectors.shape[-1]}"
            )

        # Ensure same device/dtype
        device = self.mean.device
        dtype = self.mean.dtype
        vectors = vectors.to(device=device, dtype=dtype)

        # Center, rotate, scale
        centered = vectors - self.mean
        rotated = centered @ self.components.T
        scaled = rotated * self.scales

        return scaled

    def inverse_transform(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Apply inverse preprocessing transformation.

        Args:
            coords: Transformed coordinates [N, D_pca].

        Returns:
            Original space vectors [N, D].

        Raises:
            RuntimeError: If not fitted.
            ValueError: If dimension mismatch.
        """

        if not self._fitted:
            raise RuntimeError("Preprocessor not fitted")

        if coords.shape[-1] != self.dim_pca:
            raise ValueError(
                f"Dimension mismatch: expected {self.dim_pca}, "
                f"got {coords.shape[-1]}"
            )

        # Ensure same device/dtype
        device = self.mean.device
        dtype = self.mean.dtype
        coords = coords.to(device=device, dtype=dtype)

        # Unscale, unrotate, uncenter
        unscaled = coords / self.scales
        unrotated = unscaled @ self.components
        uncentered = unrotated + self.mean

        return uncentered

    def export_arrays(self) -> Dict[str, Any]:
        """
        Export preprocessing arrays for saving.

        Returns:
            Dictionary with mean, components, scales arrays.

        Raises:
            RuntimeError: If not fitted.
        """

        if not self._fitted:
            raise RuntimeError("Preprocessor not fitted")

        return {
            "mean": self.mean.cpu().numpy(),
            "components": self.components.cpu().numpy(),
            "scales": self.scales.cpu().numpy(),
        }

    @staticmethod
    def from_arrays(cfg: PreprocessConfig, arrays: Dict[str, Any]) -> "Preprocessor":
        """
        Create preprocessor from saved arrays.

        Args:
            cfg: Preprocessing configuration.
            arrays: Dictionary with mean, components, scales.

        Returns:
            Loaded preprocessor.

        Raises:
            KeyError: If required arrays are missing.
        """

        preprocessor = Preprocessor(cfg)

        # Load arrays as tensors
        preprocessor.mean = torch.from_numpy(arrays["mean"]).float()
        preprocessor.components = torch.from_numpy(arrays["components"]).float()
        preprocessor.scales = torch.from_numpy(arrays["scales"]).float()

        # Infer dimensions
        preprocessor.dim_in = preprocessor.components.shape[1]
        preprocessor.dim_pca = preprocessor.components.shape[0]
        preprocessor._fitted = True

        return preprocessor
