"""
Training module for BASIL codebooks.

Implements sequential multi-level quantization with utilization balancing.
"""

import logging
from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch

from .config import TrainerConfig
from .io import save_artifacts
from .preprocess import Preprocessor
from .utils import (
    calculate_entropy,
    chunk_iterator,
    get_device,
    log_config,
    set_seeds,
    setup_logger,
    to_torch,
    validate_embeddings,
)


class BasilTrainer:
    """
    Trainer for multi-level additive codebooks.

    Learns codebooks sequentially to minimize reconstruction error
    while promoting balanced utilization.
    """

    def __init__(self, preprocess: Preprocessor, cfg: TrainerConfig):
        """
        Initialize trainer.

        Args:
            preprocess: Fitted preprocessor.
            cfg: Training configuration.
        """

        self.preprocess = preprocess
        self.cfg = cfg
        self.device = get_device(cfg.device)
        self.codebooks: Dict[str, torch.Tensor] = {}
        self._fitted = False
        self.logger = setup_logger("basil.trainer", level=cfg.log_level)

    def fit(self, embeddings: Union[np.ndarray, torch.Tensor]) -> None:
        """
        Train multi-level codebooks.

        Args:
            embeddings: Input embeddings [N, D].

        Raises:
            ValueError: If embeddings are invalid.
        """

        validate_embeddings(embeddings)
        set_seeds(self.cfg.seed)

        # Convert to torch and move to device
        embeddings = to_torch(embeddings, self.device)
        n, d = embeddings.shape

        if self.cfg.verbose:
            self.logger.info(f"Training BASIL on {n:,} embeddings (dim={d})")
            self.logger.info(f"Device: {self.device}")
            self.logger.info(
                f"Levels: {self.cfg.levels}, K per level: {self.cfg.k_per_level}"
            )
            log_config(self.logger, self.cfg, "TrainerConfig")

        # Fit preprocessor if not already fitted
        if not self.preprocess._fitted:
            if self.cfg.verbose:
                self.logger.info("Fitting preprocessor...")
                log_config(self.logger, self.preprocess.cfg, "PreprocessConfig")
            self.preprocess.fit(embeddings)

        # Move preprocessor to device
        self.preprocess.mean = self.preprocess.mean.to(self.device)
        self.preprocess.components = self.preprocess.components.to(self.device)
        self.preprocess.scales = self.preprocess.scales.to(self.device)

        # Transform to preprocessed space
        X = self.preprocess.transform(embeddings)
        residuals = X.clone()

        # Train each level sequentially
        for level in range(self.cfg.levels):
            if self.cfg.verbose:
                self.logger.info(f"Training level {level + 1}/{self.cfg.levels}")

            # Train single codebook on residuals
            codebook = self._train_single_level(
                residuals, level_seed=self.cfg.seed + level * 1000
            )

            # Store codebook
            self.codebooks[f"C{level + 1}"] = codebook

            # Compute assignments and update residuals
            assignments = self._assign_balanced(residuals, codebook)
            residuals = residuals - codebook[assignments]

            # Report utilization
            if self.cfg.verbose:
                unique_assigns = torch.unique(assignments).numel()
                counts = torch.bincount(assignments, minlength=self.cfg.k_per_level)
                entropy = calculate_entropy(counts)
                self.logger.info(
                    f"  Used clusters: {unique_assigns}/{self.cfg.k_per_level}"
                )
                self.logger.info(f"  Entropy: {entropy:.3f}")

        self._fitted = True

        if self.cfg.verbose:
            self.logger.info("Training complete!")

    def _train_single_level(self, X: torch.Tensor, level_seed: int) -> torch.Tensor:
        """
        Train a single codebook level using balanced k-means.

        Args:
            X: Input data [N, D].
            level_seed: Random seed for this level.

        Returns:
            Codebook [K, D].
        """

        set_seeds(level_seed)
        n, d = X.shape
        k = self.cfg.k_per_level

        # Initialize centroids with k-means++
        centroids = self._initialize_centroids(X, k)

        prev_loss = float("inf")
        for iter_num in range(self.cfg.max_iters):
            # Assign points with balancing
            assignments = self._assign_balanced(X, centroids)

            # Update centroids
            new_centroids = torch.zeros_like(centroids)
            counts = torch.zeros(k, device=self.device)

            for i in range(k):
                mask = assignments == i
                if mask.any():
                    new_centroids[i] = X[mask].mean(dim=0)
                    counts[i] = mask.sum()

            # Handle empty clusters
            empty_mask = counts == 0
            n_empty = empty_mask.sum().item()
            if n_empty > 0:
                # Reseed empty clusters from points far from centroids
                new_centroids = self._reseed_empty(
                    new_centroids, empty_mask, X, assignments
                )

            # Compute loss
            loss = 0.0
            for start, end in chunk_iterator(n, self.cfg.batch_size):
                batch = X[start:end]
                batch_assigns = assignments[start:end]
                batch_centroids = new_centroids[batch_assigns]
                loss += ((batch - batch_centroids) ** 2).sum().item()
            loss /= n

            # Check convergence
            improvement = abs(prev_loss - loss) / (prev_loss + 1e-10)

            # Per-step logging
            if self.cfg.verbose:
                used_clusters = len(torch.unique(assignments))
                self.logger.debug(
                    f"    Step {iter_num + 1}/{self.cfg.max_iters}: loss={loss:.6f}, improvement={improvement:.8f}, used_clusters={used_clusters}/{k}"
                )

            if improvement < self.cfg.tol:
                if self.cfg.verbose:
                    self.logger.info(f"  Converged at iteration {iter_num + 1}")
                break

            centroids = new_centroids
            prev_loss = loss

        return centroids

    def _initialize_centroids(self, X: torch.Tensor, k: int) -> torch.Tensor:
        """
        Initialize centroids using k-means++ algorithm.

        Args:
            X: Input data [N, D].
            k: Number of centroids.

        Returns:
            Initial centroids [K, D].
        """

        n, d = X.shape
        centroids = torch.zeros((k, d), device=self.device, dtype=X.dtype)

        # First centroid: random point
        idx = torch.randint(n, (1,)).item()
        centroids[0] = X[idx]

        # Remaining centroids: weighted by squared distance
        for i in range(1, k):
            # Compute min squared distances to existing centroids
            min_dists = torch.full((n,), float("inf"), device=self.device)

            for j in range(i):
                dists = ((X - centroids[j]) ** 2).sum(dim=1)
                min_dists = torch.minimum(min_dists, dists)

            # Sample proportional to squared distance
            probs = min_dists / min_dists.sum()
            idx = torch.multinomial(probs, 1).item()
            centroids[i] = X[idx]

        return centroids

    def _assign_balanced(
        self, X: torch.Tensor, centroids: torch.Tensor
    ) -> torch.Tensor:
        """
        Assign points to centroids with utilization balancing.

        Uses a simple size penalty to encourage balanced assignment.

        Args:
            X: Input data [N, D].
            centroids: Centroids [K, D].

        Returns:
            Assignments [N].
        """

        n = X.shape[0]
        k = centroids.shape[0]
        assignments = torch.zeros(n, dtype=torch.long, device=self.device)

        # Process in chunks for memory efficiency
        for start, end in chunk_iterator(n, self.cfg.batch_size):
            batch = X[start:end]
            batch_size = batch.shape[0]

            # Compute squared distances
            dists = torch.cdist(batch, centroids, p=2).pow(2)

            # Add size penalty based on current cluster sizes
            if self.cfg.size_penalty_lambda > 0:
                counts = torch.bincount(assignments[:start], minlength=k).float()
                size_penalty = self.cfg.size_penalty_lambda * counts / (n + 1e-10)
                dists = dists + size_penalty.unsqueeze(0)

            # Assign to nearest
            assignments[start:end] = dists.argmin(dim=1)

        return assignments

    def _reseed_empty(
        self,
        centroids: torch.Tensor,
        empty_mask: torch.Tensor,
        X: torch.Tensor,
        assignments: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reseed empty clusters from points far from their centroids.

        Args:
            centroids: Current centroids [K, D].
            empty_mask: Boolean mask of empty clusters [K].
            X: Input data [N, D].
            assignments: Current assignments [N].

        Returns:
            Updated centroids [K, D].
        """
        n_empty = empty_mask.sum().item()
        if n_empty == 0:
            return centroids

        # Find points with highest distance to their assigned centroid
        distances = ((X - centroids[assignments]) ** 2).sum(dim=1)

        # Reseed empty clusters with farthest points
        centroids[empty_mask] = X[distances.topk(n_empty).indices]

        return centroids

    def save(self, out_dir: Union[str, Path]) -> None:
        """
        Save trained model artifacts.

        Args:
            out_dir: Output directory.

        Raises:
            RuntimeError: If not fitted.
        """

        if not self._fitted:
            raise RuntimeError("Trainer not fitted")

        # Convert codebooks to numpy
        codebooks_np = {name: cb.cpu().numpy() for name, cb in self.codebooks.items()}

        # Get preprocessing arrays
        preprocess_arrays = self.preprocess.export_arrays()

        # Create metadata
        metadata = {
            "levels": self.cfg.levels,
            "k_per_level": self.cfg.k_per_level,
            "dim_in": self.preprocess.dim_in,
            "dim_pca": self.preprocess.dim_pca,
            "dim_pca_max": self.preprocess.cfg.dim_pca_max,
            "variance_to_keep": self.preprocess.cfg.variance_to_keep,
            "eps": self.preprocess.cfg.eps,
            "size_penalty_lambda": self.cfg.size_penalty_lambda,
            "seed": self.cfg.seed,
        }

        # Save artifacts
        save_artifacts(out_dir, codebooks_np, preprocess_arrays, metadata)

        if self.cfg.verbose:
            self.logger.info(f"Saved artifacts to {out_dir}")
