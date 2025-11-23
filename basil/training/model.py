from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    Single Vector Quantization layer with EMA updates and dead code restart.
    """

    def __init__(self, dim: int, codebook_size: int, ema_decay: float = 0.99):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.ema_decay = ema_decay
        self.epsilon = 1e-5

        # Initialize embedding uniformly in [-1/codebook_size, 1/codebook_size]
        scale = 1.0 / codebook_size
        embedding = torch.zeros(codebook_size, dim).uniform_(-scale, scale)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_cluster_size", torch.ones(codebook_size))
        self.register_buffer("ema_w", embedding.clone())

    @staticmethod
    def compute_distances(x: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        """
        Compute squared Euclidean distances in FP32 for numerical stability.
        Args:
            x: (batch_size, dim)
            codebook: (codebook_size, dim)
        Returns:
            distances: (batch_size, codebook_size)
        """
        with torch.autocast(device_type=x.device.type, enabled=False):
            x_32 = x.float()
            emb_32 = codebook.float()
            # ||x - e||^2 = ||x||^2 + ||e||^2 - 2*x.e
            distances = (
                torch.sum(x_32**2, dim=1, keepdim=True)
                + torch.sum(emb_32**2, dim=1)
                - 2 * torch.matmul(x_32, emb_32.t())
            )
        return distances

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch_size, dim)

        Returns:
            quantized: Quantized vectors (batch_size, dim)
            indices: Codebook indices (batch_size,)
            commitment_loss: Commitment loss scalar
        """
        input_shape = x.shape
        flat_x = x.reshape(-1, self.dim)

        # Find nearest codebook entries
        distances = self.compute_distances(flat_x, self.embedding)
        indices = torch.argmin(distances, dim=1)
        quantized = F.embedding(indices, self.embedding)

        # EMA Update (only during training)
        if self.training:
            self._ema_update(flat_x, indices)

        # Commitment loss: encourages encoder output to stay close to chosen code
        commitment_loss = F.mse_loss(quantized.detach(), flat_x)

        # Straight-through estimator: quantized for forward, x for backward
        quantized = flat_x + (quantized - flat_x).detach()
        quantized = quantized.reshape(input_shape)

        return quantized, indices, commitment_loss

    @torch.no_grad()
    def _ema_update(self, x: torch.Tensor, indices: torch.Tensor):
        """Update codebook using Exponential Moving Average with Laplace smoothing."""
        # Compute one-hot encodings
        encodings = F.one_hot(indices, self.codebook_size).float()

        # Update cluster sizes with EMA
        n_i = torch.sum(encodings, dim=0)
        self.ema_cluster_size.mul_(self.ema_decay).add_(n_i, alpha=1 - self.ema_decay)

        # Update embedding sum with EMA
        dw = torch.matmul(encodings.t(), x)
        self.ema_w.mul_(self.ema_decay).add_(dw, alpha=1 - self.ema_decay)

        # Laplace smoothing to prevent division by zero
        n = torch.sum(self.ema_cluster_size)
        cluster_size_smoothed = (
            (self.ema_cluster_size + self.epsilon)
            / (n + self.codebook_size * self.epsilon)
            * n
        )

        # Normalize to get updated embeddings
        self.embedding.copy_(self.ema_w / cluster_size_smoothed.unsqueeze(1))

        # Restart dead codes
        self._restart_dead_codes(x)

    @torch.no_grad()
    def _restart_dead_codes(self, x: torch.Tensor):
        """Reset codes with low usage by sampling from current batch."""
        dead_mask = self.ema_cluster_size < 1.0
        if not dead_mask.any():
            return

        n_dead = dead_mask.sum().item()
        random_indices = torch.randint(0, x.size(0), (n_dead,), device=x.device)
        replacements = x[random_indices]

        self.embedding[dead_mask] = replacements
        self.ema_w[dead_mask] = replacements
        self.ema_cluster_size[dead_mask] = 1.0


class ResidualVectorQuantizer(nn.Module):
    """
    Manager for multi-level residual quantization using VectorQuantizer layers.
    Supports hierarchical codebook sizes.
    """

    def __init__(
        self,
        num_levels: int,
        codebook_size: int,
        embedding_dim: int,
        beta: float = 0.25,
        use_hierarchical: bool = False,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.base_codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.use_hierarchical = use_hierarchical

        # Determine codebook sizes for each level
        if use_hierarchical:
            # Hierarchical: size, size//2, size//4, ...
            codebook_sizes = [
                max(codebook_size // (2**i), 8) for i in range(num_levels)
            ]
        else:
            # Uniform: all levels have the same size
            codebook_sizes = [codebook_size] * num_levels

        # Create VectorQuantizer layers
        self.layers = nn.ModuleList(
            [VectorQuantizer(embedding_dim, size, ema_decay) for size in codebook_sizes]
        )

    @staticmethod
    def _calculate_perplexity(
        indices: torch.Tensor, codebook_size: int
    ) -> torch.Tensor:
        """
        Metric: Measures codebook utilization entropy.
        """
        flattened = indices.reshape(-1)
        # Use the maximum codebook size for bincount to ensure all codes are counted
        counts = torch.bincount(flattened, minlength=codebook_size)
        probs = counts.float() / (flattened.numel() + 1e-10)

        # Filter zero probabilities to avoid log(0) -> NaN
        probs = probs[probs > 0]
        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
        return torch.exp(entropy)

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            z: Latent vectors (batch_size, embedding_dim)

        Returns:
            z_q: Quantized output (batch_size, embedding_dim)
            indices: Stacked codebook indices (batch_size, num_levels)
            metrics: Dictionary of loss components and metrics
        """
        residual = z
        quantized_sum = torch.zeros_like(z)
        indices_list = []

        total_commitment_loss = 0.0

        for layer in self.layers:
            # Quantize current residual
            quantized, indices, commitment_loss = layer(residual)

            # Accumulate
            indices_list.append(indices)
            total_commitment_loss += commitment_loss

            # Update residual for next level
            residual = residual - quantized
            quantized_sum = quantized_sum + quantized

        # Stack indices
        indices = torch.stack(indices_list, dim=-1)

        # Calculate metrics
        # Use the maximum codebook size to ensure all codes are counted
        max_codebook_size = max(layer.codebook_size for layer in self.layers)
        perplexity = self._calculate_perplexity(indices, max_codebook_size)
        vq_loss = self.beta * total_commitment_loss

        metrics = {
            "vq_loss": vq_loss,
            "commitment_loss": total_commitment_loss,
            "perplexity": perplexity,
        }

        # Final output with straight-through estimator
        z_q = z + (quantized_sum - z).detach()

        return z_q, indices, metrics


class RQVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.latent_dim),
            nn.LayerNorm(config.latent_dim),
        )

        self.quantizer = ResidualVectorQuantizer(
            num_levels=config.num_levels,
            codebook_size=config.codebook_size,
            embedding_dim=config.latent_dim,
            beta=config.commitment_beta,
            use_hierarchical=config.use_hierarchical,
            ema_decay=config.ema_decay,
        )

        self.decoder = nn.Sequential(
            nn.LayerNorm(config.latent_dim),
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        z_q, indices, metrics = self.quantizer(z)
        x_recon = self.decoder(z_q)
        return x_recon, indices, metrics

    # --- ONNX Export Helpers ---

    def export_encode(self, x):
        """Encode to indices without EMA updates."""
        z = self.encoder(x)
        residual = z
        indices_list = []

        for layer in self.quantizer.layers:
            distances = VectorQuantizer.compute_distances(residual, layer.embedding)
            idx = torch.argmin(distances, dim=1)
            indices_list.append(idx)
            residual = residual - F.embedding(idx, layer.embedding)

        return torch.stack(indices_list, dim=1).int()

    def export_decode(self, codes):
        """Decode from indices to embeddings."""
        quantized_sum = torch.zeros(
            codes.size(0), self.config.latent_dim, device=codes.device
        )
        for i, layer in enumerate(self.quantizer.layers):
            quantized_sum = quantized_sum + F.embedding(
                codes[:, i].long(), layer.embedding
            )
        return self.decoder(quantized_sum)
