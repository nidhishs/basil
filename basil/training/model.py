from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualVectorQuantizer(nn.Module):
    def __init__(
        self,
        num_levels: int,
        codebook_size: int,
        embedding_dim: int,
        beta: float = 0.25,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.beta = beta

        self.codebooks = nn.ModuleList(
            [nn.Embedding(codebook_size, embedding_dim) for _ in range(num_levels)]
        )

        # Initialize standard normal to match latent space distribution
        # Decay initialization variance for deeper levels
        for i, cb in enumerate(self.codebooks):
            nn.init.normal_(cb.weight, mean=0.0, std=1.0 / (2**i))

    @staticmethod
    def _compute_distances(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Computes squared euclidean distance.
        Force FP32 because (x^2 + c^2 - 2xc) easily overflows/underflows in FP16/BF16.
        """
        with torch.autocast(device_type=x.device.type, enabled=False):
            x_32 = x.float()
            c_32 = c.float()

            dot_product = x_32 @ c_32.t()
            c_sq = (c_32**2).sum(dim=1)

            dists = c_sq - (2 * dot_product)
            return dists

    @staticmethod
    def _rotation_trick(x: torch.Tensor, quantized: torch.Tensor) -> torch.Tensor:
        """
        Householder Reflection gradient estimator.
        Provides more accurate gradients than standard Straight-Through Estimator (STE).
        """
        # Normalize inputs to unit vectors
        u = F.normalize(x, p=2, dim=-1, eps=1e-6)
        q = F.normalize(quantized, p=2, dim=-1, eps=1e-6)

        # Bisector
        w = F.normalize(u + q, p=2, dim=-1, eps=1e-6)

        xw_dot = (x * w.detach()).sum(dim=-1, keepdim=True)
        xu_dot = (x * u.detach()).sum(dim=-1, keepdim=True)

        # Reflection: x' = x - 2(x.w)w + 2(x.u)q
        rotated_x = x - (2 * xw_dot * w.detach()) + (2 * xu_dot * q.detach())
        return rotated_x

    def _calculate_perplexity(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Metric: Measures codebook utilization entropy.
        """
        flattened = indices.reshape(-1)
        counts = torch.bincount(flattened, minlength=self.codebook_size)
        probs = counts.float() / flattened.numel()

        # Filter zero probabilities to avoid log(0) -> NaN
        probs = probs[probs > 0]
        entropy = -torch.sum(probs * torch.log(probs))
        return torch.exp(entropy)

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        residual = z
        quantized_sum = 0.0
        indices_list = []

        total_codebook_loss = 0.0
        total_commitment_loss = 0.0

        for layer in self.codebooks:
            # 1. Distance & Selection
            dists = self._compute_distances(residual, layer.weight)
            min_idx = torch.argmin(dists, dim=-1)
            indices_list.append(min_idx)

            z_q_local = layer(min_idx)

            # 2. Loss Components
            total_codebook_loss += F.mse_loss(z_q_local, residual.detach())
            total_commitment_loss += F.mse_loss(z_q_local.detach(), residual)

            # 3. Gradient Estimation
            if self.training:
                z_q_st = self._rotation_trick(residual, z_q_local)
            else:
                z_q_st = z_q_local

            # 4. Update Residual
            residual = residual - z_q_st
            quantized_sum = quantized_sum + z_q_local

        indices = torch.stack(indices_list, dim=-1)
        loss = total_codebook_loss + (self.beta * total_commitment_loss)
        perplexity = self._calculate_perplexity(indices)

        metrics = {
            "vq_loss": loss,
            "codebook_loss": total_codebook_loss,
            "commitment_loss": total_commitment_loss,
            "perplexity": perplexity,
        }

        # Straight-through gradient for the sum
        z_q = z + (quantized_sum - z).detach()

        return z_q, indices, metrics

    @staticmethod
    def _wipe_optimizer_state(
        optimizer: torch.optim.Optimizer, param: torch.nn.Parameter, mask: torch.Tensor
    ):
        """
        Zeroes out momentum (exp_avg) and variance (exp_avg_sq) for specific indices in the optimizer state to prevent 'slingshotting'.
        """
        if optimizer is None or param not in optimizer.state:
            return

        state = optimizer.state[param]
        for key in ["exp_avg", "exp_avg_sq"]:
            if key in state:
                state[key][mask] = 0.0

    @torch.no_grad()
    def reset_dead_codes(
        self, batch_samples: torch.Tensor, optimizer: torch.optim.Optimizer = None
    ):
        """
        Replaces unused codes with random vectors from the current batch.
        """
        residual = batch_samples

        for layer in self.codebooks:
            # 1. Identify usage
            # Find which codebook entries are best matches for current residuals
            dists = self._compute_distances(residual, layer.weight)
            indices = torch.argmin(dists, dim=-1)

            counts = torch.bincount(indices, minlength=self.codebook_size)
            dead_mask = counts == 0

            # 2. Reset (if needed)
            if dead_mask.any():
                n_dead = dead_mask.sum().item()

                # Randomly sample vectors from the current batch to replace dead codes.
                # This ensures revived codes are immediately relevant to the data distribution.
                random_idxs = torch.randint(
                    0, residual.size(0), (n_dead,), device=residual.device
                )
                replacements = residual[random_idxs]

                # Update weights in-place
                layer.weight.data[dead_mask] = replacements

                # IMPORTANT: Reset optimizer state for these specific weights.
                # Keeping old momentum for new random weights causes massive gradient updates (slingshotting).
                self._wipe_optimizer_state(optimizer, layer.weight, dead_mask)

            # 3. Propagate Residual
            # Must update residual for the next layer using the ORIGINAL selections.
            # Even if we reset codes, they weren't selected in this pass, so they don't affect the residual yet.
            # This ensures the hierarchy remains consistent for subsequent layers.
            z_q_local = layer(indices)
            residual = residual - z_q_local


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
        z = self.encoder(x)
        residual = z
        indices_list = []
        for layer in self.quantizer.codebooks:
            # Re-implement distance calculation using standard ops for cleaner ONNX graph
            dot_prod = torch.matmul(residual, layer.weight.t())
            c_sq = torch.sum(layer.weight**2, dim=1)
            dists = c_sq - (2 * dot_prod)
            idx = torch.argmin(dists, dim=1)
            indices_list.append(idx)
            quantized = layer(idx)
            residual = residual - quantized
        return torch.stack(indices_list, dim=1).int()

    def export_decode(self, codes):
        quantized_sum = 0.0
        for i, layer in enumerate(self.quantizer.codebooks):
            idx = codes[:, i].long()
            quantized_sum = quantized_sum + layer(idx)
        return self.decoder(quantized_sum)
