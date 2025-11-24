from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    Single Vector Quantization layer with EMA updates and dead code restart.
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        ema_decay: float = 0.99,
        reset_code_interval: int = 1000,
        stochastic: bool = False,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.ema_decay = ema_decay
        self.stochastic_sampling = stochastic
        self.temperature = temperature
        self.epsilon = 1e-5
        self.usage_threshold = self.ema_decay**reset_code_interval
        self.scale = nn.Parameter(torch.ones(1))

        embedding = F.normalize(torch.randn(codebook_size, dim))
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch_size, dim)

        Returns:
            quantized: Quantized vectors (batch_size, dim)
            indices: Codebook indices (batch_size,)
            commit_loss: Commitment loss scalar
        """
        input_shape = x.shape
        flat_x = x.reshape(-1, self.dim)

        # Find nearest codebook entries
        distances = self.compute_distances(flat_x, self.embedding)

        if self.training and self.stochastic_sampling:
            logits = -distances / self.temperature
            indices = torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(-1)
        else:
            indices = torch.argmin(distances, dim=1)

        quantized = F.embedding(indices, self.embedding) * self.scale

        # EMA Update (only during training)
        if self.training:
            self._ema_update(flat_x, indices)

        # Commitment loss: encourages encoder output to stay close to chosen code
        commit_loss = F.mse_loss(quantized.detach(), flat_x)

        # Straight-through estimator: quantized for forward, x for backward
        quantized = flat_x + (quantized - flat_x).detach()
        quantized = quantized.reshape(input_shape)

        return quantized, indices, commit_loss

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
        dead_mask = self.ema_cluster_size < self.usage_threshold
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

    def __init__(self, config, embedding_dim: int):
        super().__init__()
        self.config = config
        self.num_levels = config.num_levels
        self.base_codebook_size = config.codebook_size
        self.embedding_dim = embedding_dim
        self.beta = config.commitment_beta
        self.use_hierarchical = config.use_hierarchical

        # Determine codebook sizes for each level
        if config.use_hierarchical:
            # Hierarchical: size, size//2, size//4, ...
            codebook_sizes = [
                max(config.codebook_size // (2**i), 8) for i in range(config.num_levels)
            ]
        else:
            # Uniform: all levels have the same size
            codebook_sizes = [config.codebook_size] * config.num_levels

        # Create VectorQuantizer layers
        self.layers = nn.ModuleList(
            [
                VectorQuantizer(
                    dim=embedding_dim,
                    codebook_size=size,
                    ema_decay=config.ema_decay,
                    reset_code_interval=config.reset_code_interval,
                    stochastic=config.stochastic_sampling,
                    temperature=config.stochastic_temperature,
                )
                for size in codebook_sizes
            ]
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
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
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

        total_commit_loss = 0.0

        for layer in self.layers:
            # Quantize current residual
            quantized, indices, commit_loss = layer(residual)

            # Accumulate
            indices_list.append(indices)
            total_commit_loss += commit_loss

            # Update residual for next level
            residual = residual - quantized
            quantized_sum = quantized_sum + quantized

        # Stack indices
        indices = torch.stack(indices_list, dim=-1)

        # Calculate metrics
        perplexities = []
        for i, layer in enumerate(self.layers):
            ppl = self._calculate_perplexity(indices[..., i], layer.codebook_size)
            perplexities.append(ppl)

        vq_loss = self.beta * total_commit_loss

        metrics = {
            "vq_loss": vq_loss,
            "commit_loss": total_commit_loss,
            "perplexity": torch.stack(perplexities),
        }

        # Final output with straight-through estimator
        z_q = z + (quantized_sum - z).detach()

        return z_q, indices, metrics


class MLP(nn.Module):
    """
    Multi-layer perceptron with configurable dimensions and optional normalization.
    """

    def __init__(
        self,
        dims: list[int],
        dropout: float = 0.1,
        initial_norm: bool = False,
        final_norm: bool = False,
    ):
        """
        Args:
            dims: List of layer dimensions [in_dim, hidden1, hidden2, ..., out_dim]
            dropout: Dropout probability for regularization
            initial_norm: Add LayerNorm before first layer
            final_norm: Add LayerNorm after last layer
        """
        super().__init__()
        layers = []

        if initial_norm:
            layers.append(nn.LayerNorm(dims[0]))

        # Build layers: all except last have activation and dropout
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(in_dim, out_dim))
            if not (i == len(dims) - 2):
                layers.extend([nn.SiLU(), nn.Dropout(dropout)])

        if final_norm:
            layers.append(nn.LayerNorm(dims[-1]))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class RQVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Build encoder: input_dim -> encoder_dims -> latent_dim (with final norm)
        self.encoder = MLP(
            dims=[config.input_dim] + config.encoder_dims,
            dropout=config.dropout,
            final_norm=True,
        )

        self.quantizer = ResidualVectorQuantizer(
            config=config,
            embedding_dim=config.latent_dim,
        )

        # Build decoder: inverse of encoder (with initial norm)
        self.decoder = MLP(
            dims=config.encoder_dims[::-1] + [config.input_dim],
            dropout=config.dropout,
            initial_norm=True,
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
