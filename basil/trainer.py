from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch

from basil.config import TrainerConfig
from basil.io import save_artifacts
from basil.preprocess import Preprocessor
from basil.utils import (
    calculate_entropy,
    chunk_iterator,
    get_device,
    set_seeds,
    setup_logger,
    to_torch,
    validate_embeddings,
)

# ------------------------------ trainer (orchestrator) ------------------------------


class BasilTrainer:
    """Trainer for multi-level additive codebooks.

    The trainer prepares data, orchestrates per-level balanced k-means, tracks
    codebooks per level, updates residuals, and saves artifacts.

    Attributes:
        preprocess (Preprocessor): Preprocessing pipeline.
        cfg (TrainerConfig): Trainer configuration.
        device (torch.device): Compute device.
        codebooks (Dict[str, torch.Tensor]): Learned codebooks per level.
        _fitted (bool): Flag indicating whether training has completed.
        logger: Logger instance.
    """

    def __init__(self, preprocess: Preprocessor, cfg: TrainerConfig):
        """Initialize the BASIL trainer.

        Args:
            preprocess (Preprocessor): Preprocessing pipeline.
            cfg (TrainerConfig): Trainer configuration.
        """
        self.preprocess = preprocess
        self.cfg = cfg
        self.device = get_device(cfg.device)
        self.codebooks: Dict[str, torch.Tensor] = {}
        self._fitted = False
        self.logger = setup_logger("basil.trainer", level=cfg.log_level)

    def fit(self, embeddings: Union[np.ndarray, torch.Tensor]) -> None:
        """Train multi-level codebooks.

        Args:
            embeddings (Union[np.ndarray, torch.Tensor]): Raw embeddings of shape (num_samples, dim_in).

        Raises:
            RuntimeError: If preprocessing or training encounters unrecoverable issues.
        """
        _, residuals = prepare_data(
            self.preprocess, self.cfg, embeddings, self.logger, self.device
        )
        self._train_all_levels(residuals)
        self._fitted = True
        if self.cfg.verbose:
            self.logger.info("Training complete!")

    def save(self, out_dir: Union[str, Path]) -> None:
        """Save trained model artifacts.

        Args:
            out_dir (Union[str, Path]): Output directory to save artifacts.

        Raises:
            RuntimeError: If called before the trainer has been fitted.
        """
        if not self._fitted:
            raise RuntimeError("Trainer not fitted")

        codebooks_np = {name: cb.cpu().numpy() for name, cb in self.codebooks.items()}
        preprocess_arrays = self.preprocess.export_arrays()
        actual_k = (
            next(iter(self.codebooks.values())).shape[0]
            if self.codebooks
            else self.cfg.k_per_level
        )

        metadata = {
            "levels": self.cfg.levels,
            "k_per_level": int(actual_k),
            "dim_in": self.preprocess.dim_in,
            "dim_pca": self.preprocess.dim_pca,
            "dim_pca_max": self.preprocess.cfg.dim_pca_max,
            "variance_to_keep": self.preprocess.cfg.variance_to_keep,
            "eps": self.preprocess.cfg.eps,
            "size_penalty_lambda": self.cfg.size_penalty_lambda,
            "seed": self.cfg.seed,
        }

        save_artifacts(out_dir, codebooks_np, preprocess_arrays, metadata)
        if self.cfg.verbose:
            self.logger.info(f"Saved artifacts to {out_dir}")

    # ------------------------------ orchestration ------------------------------

    def _train_all_levels(self, residuals: torch.Tensor) -> None:
        """Train each quantization level and update residuals sequentially.

        Args:
            residuals (torch.Tensor): Working residual tensor on device, shape (num_samples, dim_pca).
        """
        kmeans = BalancedKMeans(
            k=self.cfg.k_per_level,
            max_iters=self.cfg.max_iters,
            tol=self.cfg.tol,
            batch_size=self.cfg.batch_size,
            size_penalty_lambda=self.cfg.size_penalty_lambda,
            device=self.device,
            logger=self.logger,
            verbose=self.cfg.verbose,
        )

        with torch.inference_mode():
            for level_idx in range(self.cfg.levels):
                if self.cfg.verbose:
                    self.logger.info(
                        f"Training level {level_idx + 1}/{self.cfg.levels}"
                    )

                seed_for_level = self.cfg.seed + level_idx * 1000
                codebook, assignments = kmeans.fit_predict(residuals, seed_for_level)
                self.codebooks[f"C{level_idx + 1}"] = codebook

                residuals -= codebook[assignments]
                report_level_stats(
                    self.logger, self.cfg.verbose, assignments, codebook.shape[0]
                )


# ------------------------------ k-means ------------------------------


class BalancedKMeans:
    """Balanced k-means with k-means++ init and a simple size penalty.

    This class encapsulates clustering behavior so the trainer can remain an
    orchestration layer.

    Attributes:
        k (int): Target number of clusters per level.
        max_iters (int): Maximum k-means iterations.
        tol (float): Relative loss improvement threshold for early stop.
        batch_size (int): Mini-batch size for distance computations.
        size_penalty_lambda (float): Penalty strength to encourage balance.
        device (torch.device): Compute device.
        logger: Logger instance.
        verbose (bool): If True, emit progress logs.
    """

    def __init__(
        self,
        k,
        max_iters,
        tol,
        batch_size,
        size_penalty_lambda,
        device,
        logger,
        verbose,
    ):
        """Initialize a BalancedKMeans instance.

        Args:
            k (int): Target number of clusters per level.
            max_iters (int): Maximum k-means iterations.
            tol (float): Relative loss improvement threshold for early stop.
            batch_size (int): Mini-batch size for distance computations.
            size_penalty_lambda (float): Penalty strength to encourage balance.
            device (torch.device): Compute device.
            logger: Logger instance used for debug and info logs.
            verbose (bool): If True, emit progress logs.
        """
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.batch_size = batch_size
        self.size_penalty_lambda = size_penalty_lambda
        self.device = device
        self.logger = logger
        self.verbose = verbose

    @torch.inference_mode()
    def fit_predict(self, data_tensor, seed):
        """Cluster data and return centroids and assignments.

        Args:
            data_tensor (torch.Tensor): Input data of shape (num_samples, num_features).
            seed (int): Random seed for reproducibility.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple (centroids, assignments), where
                centroids has shape (num_clusters, num_features) and assignments has
                shape (num_samples,) with cluster indices.

        Notes:
            The number of clusters is clamped to the number of samples.
        """
        set_seeds(seed)
        num_samples, num_features = data_tensor.shape
        num_clusters = min(self.k, num_samples)

        centroids = self._kmeanspp_init(data_tensor, num_clusters)
        previous_loss = float("inf")

        for iteration_idx in range(self.max_iters):
            assignments = self._assign_balanced(data_tensor, centroids)
            new_centroids, counts_per_cluster = self._update_centroids(
                data_tensor, assignments, num_clusters, num_features, data_tensor.dtype
            )

            # Reseed any empty clusters using farthest points from current assignments
            empty_mask = counts_per_cluster == 0
            if empty_mask.any():
                new_centroids = self._reseed_empty(
                    new_centroids, empty_mask, data_tensor, assignments
                )

            loss_value = self._compute_loss(data_tensor, assignments, new_centroids)
            relative_improvement = abs(previous_loss - loss_value) / (
                previous_loss + 1e-10
            )
            self._log_step(
                iteration_idx,
                assignments,
                num_clusters,
                loss_value,
                relative_improvement,
            )

            centroids = new_centroids
            if relative_improvement < self.tol:
                if self.verbose:
                    self.logger.info(f"  Converged at iteration {iteration_idx + 1}")
                break
            previous_loss = loss_value

        # Final assignments after last centroid update
        assignments = self._assign_balanced(data_tensor, centroids)
        return centroids, assignments

    # ---- internals ----

    def _kmeanspp_init(self, data_tensor, num_clusters):
        """Initialize centroids using k-means++.

        Args:
            data_tensor (torch.Tensor): Data of shape (num_samples, num_features).
            num_clusters (int): Number of centroids to initialize.

        Returns:
            torch.Tensor: Initialized centroids of shape (num_clusters, num_features).
        """
        num_samples, num_features = data_tensor.shape
        centroids = torch.zeros(
            (num_clusters, num_features), device=self.device, dtype=data_tensor.dtype
        )
        first_index = torch.randint(num_samples, (1,)).item()
        centroids[0] = data_tensor[first_index]

        min_squared_distances = torch.full(
            (num_samples,), float("inf"), device=self.device
        )
        for centroid_idx in range(1, num_clusters):
            current_d2 = ((data_tensor - centroids[centroid_idx - 1]) ** 2).sum(1)
            min_squared_distances = torch.minimum(min_squared_distances, current_d2)
            probs = min_squared_distances / min_squared_distances.sum()
            next_index = torch.multinomial(probs, 1).item()
            centroids[centroid_idx] = data_tensor[next_index]
        return centroids

    def _assign_balanced(self, data_tensor, centroids):
        """Assign points to nearest centroids with a simple balance penalty.

        Args:
            data_tensor (torch.Tensor): Data of shape (num_samples, num_features).
            centroids (torch.Tensor): Centroids of shape (num_clusters, num_features).

        Returns:
            torch.Tensor: Assignments of shape (num_samples,) with cluster indices.
        """
        num_samples = data_tensor.shape[0]
        num_clusters = centroids.shape[0]
        assignments = torch.zeros(num_samples, dtype=torch.long, device=self.device)

        for start_idx, end_idx in chunk_iterator(num_samples, self.batch_size):
            batch = data_tensor[start_idx:end_idx]
            squared_distances = torch.cdist(batch, centroids, p=2).pow(2)
            if self.size_penalty_lambda > 0:
                counts_so_far = torch.bincount(
                    assignments[:start_idx], minlength=num_clusters
                ).float()
                penalty = (
                    self.size_penalty_lambda * counts_so_far / (num_samples + 1e-10)
                )
                squared_distances = squared_distances + penalty.unsqueeze(0)
            assignments[start_idx:end_idx] = squared_distances.argmin(dim=1)
        return assignments

    def _update_centroids(
        self, data_tensor, assignments, num_clusters, feature_dim, dtype
    ):
        """Compute new centroids as means of assigned points.

        Args:
            data_tensor (torch.Tensor): Data of shape (num_samples, num_features).
            assignments (torch.Tensor): Cluster ids of shape (num_samples,).
            num_clusters (int): Number of clusters.
            feature_dim (int): Feature dimensionality.
            dtype (torch.dtype): Desired dtype of centroids.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple (centroids, counts_per_cluster).
        """
        centroids = torch.zeros(
            (num_clusters, feature_dim), device=self.device, dtype=dtype
        )
        counts_per_cluster = torch.zeros(num_clusters, device=self.device)
        for cluster_idx in range(num_clusters):
            mask = assignments == cluster_idx
            if mask.any():
                centroids[cluster_idx] = data_tensor[mask].mean(0)
                counts_per_cluster[cluster_idx] = mask.sum()
        return centroids, counts_per_cluster

    def _reseed_empty(self, centroids, empty_mask, data_tensor, assignments):
        """Reseed empty centroids using the farthest currently assigned points.

        Args:
            centroids (torch.Tensor): Current centroids of shape (num_clusters, num_features).
            empty_mask (torch.Tensor): Boolean mask of empty clusters.
            data_tensor (torch.Tensor): Data of shape (num_samples, num_features).
            assignments (torch.Tensor): Assignments of shape (num_samples,).

        Returns:
            torch.Tensor: Updated centroids with reseeded entries for empty clusters.
        """
        num_empty = int(empty_mask.sum().item())
        if num_empty == 0:
            return centroids
        squared_distance_to_assigned = (
            (data_tensor - centroids[assignments]) ** 2
        ).sum(1)
        centroids[empty_mask] = data_tensor[
            squared_distance_to_assigned.topk(num_empty).indices
        ]
        return centroids

    def _compute_loss(self, data_tensor, assignments, centroids):
        """Compute average squared error for current assignments and centroids.

        Args:
            data_tensor (torch.Tensor): Data of shape (num_samples, num_features).
            assignments (torch.Tensor): Assignments of shape (num_samples,).
            centroids (torch.Tensor): Centroids of shape (num_clusters, num_features).

        Returns:
            float: Mean squared reconstruction error per sample.
        """
        num_samples = data_tensor.shape[0]
        total_squared_error = 0.0
        for start_idx, end_idx in chunk_iterator(num_samples, self.batch_size):
            batch = data_tensor[start_idx:end_idx]
            batch_assignments = assignments[start_idx:end_idx]
            total_squared_error += (
                ((batch - centroids[batch_assignments]) ** 2).sum().item()
            )
        return total_squared_error / num_samples

    def _log_step(
        self, iteration_idx, assignments, num_clusters, loss_value, relative_improvement
    ):
        """Log one k-means step if verbose.

        Args:
            iteration_idx (int): Current iteration index.
            assignments (torch.Tensor): Assignments of shape (num_samples,).
            num_clusters (int): Number of clusters.
            loss_value (float): Current loss value.
            relative_improvement (float): Relative improvement from previous step.
        """
        if not self.verbose:
            return
        used = len(torch.unique(assignments))
        self.logger.debug(
            f"    Step {iteration_idx + 1}/{self.max_iters}: loss={loss_value:.6f}, "
            f"improvement={relative_improvement:.8f}, used_clusters={used}/{num_clusters}"
        )


# ------------------------------ helpers ------------------------------


def prepare_data(preprocess, cfg, embeddings, logger, device):
    """Validate, seed, fit or reuse preprocessor, and return residual workspace.

    Args:
        preprocess (Preprocessor): Preprocessor instance, may be fitted or not.
        cfg (TrainerConfig): Trainer configuration.
        embeddings (Union[np.ndarray, torch.Tensor]): Raw input embeddings of shape (num_samples, dim_in).
        logger: Logger instance for status messages.
        device (torch.device): Compute device.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Standardized embeddings and a residual
        workspace tensor, both on the target device.
    """
    validate_embeddings(embeddings)
    set_seeds(cfg.seed)

    embeddings_t = to_torch(embeddings, device)
    num_samples, dim_in = embeddings_t.shape

    if cfg.verbose:
        logger.info(f"Training BASIL on {num_samples:,} embeddings (dim={dim_in})")
        logger.info(f"Device: {device}")
        logger.info(f"Levels: {cfg.levels}, K per level: {cfg.k_per_level}")

    if not preprocess._fitted:
        if cfg.verbose:
            logger.info("Fitting preprocessor...")
        preprocess.fit(embeddings_t)

    preprocess.to(device)
    standardized = preprocess.transform(embeddings_t)
    residuals = standardized.clone()
    return standardized, residuals


def report_level_stats(logger, verbose, assignments, num_clusters):
    """Report per-level cluster utilization statistics.

    Args:
        logger: Logger instance.
        verbose (bool): If True, logs are emitted.
        assignments (torch.Tensor): Cluster ids for each sample, shape (num_samples,).
        num_clusters (int): Number of clusters at this level.
    """
    if not verbose:
        return
    counts = torch.bincount(assignments, minlength=num_clusters)
    entropy = calculate_entropy(counts)
    used = (counts > 0).sum().item()
    logger.info(f"  Used clusters: {used}/{num_clusters}")
    logger.info(f"  Entropy: {entropy:.3f}")
