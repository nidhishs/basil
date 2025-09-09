from typing import Any, Dict, Optional

import torch

from basil.config import PreprocessConfig
from basil.utils import set_seeds


class Preprocessor:
    """
    Fixed preprocessing pipeline for embeddings.
    Center -> PCA rotate -> scale (and clean inverse).
    """

    def __init__(self, cfg: PreprocessConfig):
        """Initialize the Preprocessor.

        Args:
            cfg: Configuration object containing preprocessing parameters.
        """
        self.cfg = cfg
        self.mean: Optional[torch.Tensor] = None
        self.components: Optional[torch.Tensor] = None  # [dim_pca, dim_in]
        self.scales: Optional[torch.Tensor] = None
        self.dim_in: Optional[int] = None
        self.dim_pca: Optional[int] = None
        self._fitted = False

    def to(self, device: torch.device) -> "Preprocessor":
        """Move preprocessor parameters to specified device.

        Args:
            device: Target torch device.

        Returns:
            Self for method chaining.
        """
        if self._fitted:
            self.mean = self.mean.to(device)
            self.components = self.components.to(device)
            self.scales = self.scales.to(device)
        return self

    def fit(self, vectors: torch.Tensor) -> None:
        """Fit preprocessing parameters.

        Computes mean, PCA components, and scales from input vectors.

        Args:
            vectors: Input tensor of shape (n_samples, n_features).

        Raises:
            ValueError: If vectors is not a 2D tensor.
        """
        if vectors.ndim != 2:
            raise ValueError(f"Expected 2D tensor, got shape {vectors.shape}")

        set_seeds(self.cfg.seed)

        n, d = vectors.shape
        self.dim_in = d
        device, dtype = vectors.device, vectors.dtype

        self.mean = vectors.mean(dim=0)
        centered = vectors - self.mean

        if n == 1:
            self._fit_singleton(d, device, dtype)
            return

        comps, scales, dim_pca = self._compute_pca(centered, d, device, dtype, n)
        self.components = comps
        self.scales = scales
        self.dim_pca = dim_pca
        self._fitted = True

    def _fit_singleton(self, d: int, device, dtype) -> None:
        """Handle fitting for single vector case.

        Args:
            d: Input dimension.
            device: Torch device.
            dtype: Data type.
        """
        self.dim_pca = min(d, self.cfg.dim_pca_max)
        # components must be [dim_pca, dim_in] to match transform/inverse
        self.components = torch.eye(self.dim_pca, d, device=device, dtype=dtype)
        self.scales = torch.ones(self.dim_pca, device=device, dtype=dtype)
        self._fitted = True

    def _compute_pca(self, centered: torch.Tensor, d: int, device, dtype, n: int):
        """Compute PCA components from centered data.

        Args:
            centered: Mean-centered input vectors.
            d: Input dimension.
            device: Torch device.
            dtype: Data type.
            n: Number of samples.

        Returns:
            Tuple of (components, scales, n_components).
        """
        cov = (centered.T @ centered) / (n - 1)
        cov = cov + torch.eye(d, device=device, dtype=dtype) * self.cfg.eps

        eigvals, eigvecs = torch.linalg.eigh(cov)
        idx = eigvals.argsort(descending=True)
        eigvals = eigvals[idx].clamp(min=self.cfg.eps)
        eigvecs = eigvecs[:, idx]

        total_var = eigvals.sum()
        n_components = (
            eigvals.cumsum(0) / total_var >= self.cfg.variance_to_keep
        ).nonzero()[0].item() + 1
        n_components = min(n_components, self.cfg.dim_pca_max, d)

        components = eigvecs[:, :n_components].T  # [dim_pca, dim_in]
        projected = centered @ components.T
        stds = projected.std(dim=0)
        scales = 1.0 / (stds + self.cfg.eps)
        return components, scales, n_components

    def transform(self, vectors: torch.Tensor) -> torch.Tensor:
        """Transform vectors using fitted preprocessing parameters.

        Args:
            vectors: Input tensor to transform.

        Returns:
            Transformed tensor in PCA space.

        Raises:
            RuntimeError: If preprocessor not fitted.
            ValueError: If input dimension doesn't match fitted dimension.
        """
        if not self._fitted:
            raise RuntimeError("Preprocessor not fitted")
        if vectors.shape[-1] != self.dim_in:
            raise ValueError(
                f"Dimension mismatch: expected {self.dim_in}, got {vectors.shape[-1]}"
            )
        device, dtype = self.mean.device, self.mean.dtype
        vectors = vectors.to(device=device, dtype=dtype)
        return (vectors - self.mean) @ self.components.T * self.scales

    def inverse_transform(self, coords: torch.Tensor) -> torch.Tensor:
        """Inverse transform coordinates from PCA space to original space.

        Args:
            coords: Coordinates in PCA space.

        Returns:
            Reconstructed vectors in original space.

        Raises:
            RuntimeError: If preprocessor not fitted.
            ValueError: If PCA dimension doesn't match fitted dimension.
        """
        if not self._fitted:
            raise RuntimeError("Preprocessor not fitted")
        if coords.shape[-1] != self.dim_pca:
            raise ValueError(
                f"Dimension mismatch: expected {self.dim_pca}, got {coords.shape[-1]}"
            )
        device, dtype = self.mean.device, self.mean.dtype
        coords = coords.to(device=device, dtype=dtype)
        return (coords / self.scales) @ self.components + self.mean

    def export_arrays(self) -> Dict[str, Any]:
        """Export preprocessor parameters as numpy arrays.

        Returns:
            Dictionary containing mean, components, and scales as numpy arrays.

        Raises:
            RuntimeError: If preprocessor not fitted.
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
        """Create Preprocessor from exported arrays.

        Args:
            cfg: Configuration object.
            arrays: Dictionary containing mean, components, and scales arrays.

        Returns:
            Fitted Preprocessor instance.
        """
        preprocessor = Preprocessor(cfg)
        preprocessor.mean = torch.from_numpy(arrays["mean"]).float()
        preprocessor.components = torch.from_numpy(arrays["components"]).float()
        preprocessor.scales = torch.from_numpy(arrays["scales"]).float()
        preprocessor.dim_in = preprocessor.components.shape[1]
        preprocessor.dim_pca = preprocessor.components.shape[0]
        preprocessor._fitted = True
        return preprocessor
