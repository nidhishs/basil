"""
Codec module for BASIL encoding and decoding.

Provides the main API for converting embeddings to/from semantic IDs.
"""

from pathlib import Path
from typing import Iterable, List, Union

import numpy as np
import torch

from .config import PreprocessConfig
from .io import load_artifacts, validate_artifacts
from .preprocess import Preprocessor
from .utils import auto_select_device, chunk_iterator, to_numpy, to_torch


class BasilCodec:
    """
    BASIL codec for encoding/decoding embeddings.

    Converts between continuous embeddings and discrete semantic IDs
    using multi-level additive quantization.
    """

    def __init__(self, artifact_dir: Union[str, Path]):
        """
        Initialize codec from saved artifacts.

        Args:
            artifact_dir: Directory containing basil.npz and metadata.json.

        Raises:
            BasilError: If artifacts cannot be loaded.
        """

        # Load artifacts
        artifacts = load_artifacts(artifact_dir)
        validate_artifacts(artifacts)

        self.metadata = artifacts["metadata"]
        self.levels = self.metadata["levels"]
        self.k_per_level = self.metadata["k_per_level"]
        self.dim_in = self.metadata["dim_in"]
        self.dim_pca = self.metadata["dim_pca"]

        # Auto-select device
        self.device = auto_select_device()

        # Load codebooks
        self.codebooks = []
        for i in range(self.levels):
            cb_name = f"C{i + 1}"
            cb_array = artifacts["codebooks"][cb_name]
            cb_tensor = torch.from_numpy(cb_array).float().to(self.device)
            self.codebooks.append(cb_tensor)

        # Create preprocessor from arrays
        preprocess_cfg = PreprocessConfig(
            variance_to_keep=self.metadata.get("variance_to_keep", 0.95),
            dim_pca_max=self.metadata.get("dim_pca_max", 256),
            eps=self.metadata.get("eps", 1e-6),
            seed=self.metadata.get("seed", 42),
        )
        self.preprocess = Preprocessor.from_arrays(
            preprocess_cfg, artifacts["preprocess"]
        )

        # Move preprocessor to device
        self.preprocess.mean = self.preprocess.mean.to(self.device)
        self.preprocess.components = self.preprocess.components.to(self.device)
        self.preprocess.scales = self.preprocess.scales.to(self.device)

    def encode(self, embedding: Union[np.ndarray, torch.Tensor]) -> List[int]:
        """
        Encode a single embedding to semantic ID.

        Args:
            embedding: Input embedding vector [D].

        Returns:
            List of code indices, one per level.

        Raises:
            ValueError: If embedding dimension is invalid.
        """
        # Reshape 1D to 2D if needed
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        # Delegate to batch_encode
        return self.batch_encode(embedding)[0]

    def batch_encode(
        self, embeddings: Union[np.ndarray, torch.Tensor], batch_size: int = 8192
    ) -> List[List[int]]:
        """
        Encode multiple embeddings to semantic IDs.

        Args:
            embeddings: Input embeddings [N, D].
            batch_size: Batch size for processing.

        Returns:
            List of semantic IDs.

        Raises:
            ValueError: If embeddings dimension is invalid.
        """

        # Convert to torch tensor on device
        embeddings = to_torch(embeddings, self.device)

        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D tensor, got shape {embeddings.shape}")
        if embeddings.shape[1] != self.dim_in:
            raise ValueError(
                f"Dimension mismatch: expected {self.dim_in}, "
                f"got {embeddings.shape[1]}"
            )

        n = embeddings.shape[0]
        all_codes = []

        # Process in chunks
        for start, end in chunk_iterator(n, batch_size):
            batch = embeddings[start:end]

            # Transform batch to preprocessed space
            x_batch = self.preprocess.transform(batch)
            residuals = x_batch.clone()

            # Sequential encoding for batch
            batch_codes = []
            for level_idx, codebook in enumerate(self.codebooks):
                # Find nearest codewords for batch
                dists = torch.cdist(residuals, codebook, p=2)
                code_indices = dists.argmin(dim=1)

                # Store codes
                if level_idx == 0:
                    batch_codes = [[idx.item()] for idx in code_indices]
                else:
                    for i, idx in enumerate(code_indices):
                        batch_codes[i].append(idx.item())

                # Update residuals
                assigned_codewords = codebook[code_indices]
                residuals = residuals - assigned_codewords

            all_codes.extend(batch_codes)

        return all_codes

    def decode(self, sid: Iterable[int]) -> np.ndarray:
        """
        Decode a semantic ID to approximate embedding.

        Args:
            sid: Semantic ID (list of code indices).

        Returns:
            Decoded embedding [D].

        Raises:
            ValueError: If SID is invalid.
        """
        # Delegate to batch_decode
        return self.batch_decode([sid])[0]

    def batch_decode(self, sids: List[Iterable[int]]) -> np.ndarray:
        """
        Decode multiple semantic IDs to approximate embeddings.

        Args:
            sids: List of semantic IDs.

        Returns:
            Decoded embeddings [N, D].

        Raises:
            ValueError: If any SID is invalid.
        """

        if not sids:
            return np.empty((0, self.dim_in), dtype=np.float32)

        # Validate all SIDs
        for i, sid in enumerate(sids):
            sid = list(sid)
            if len(sid) != self.levels:
                raise ValueError(
                    f"Invalid SID length at index {i}: "
                    f"expected {self.levels}, got {len(sid)}"
                )
            for level_idx, code_idx in enumerate(sid):
                if not (0 <= code_idx < self.k_per_level):
                    raise ValueError(
                        f"Invalid code index at SID {i}, level {level_idx+1}: "
                        f"{code_idx} (must be 0-{self.k_per_level-1})"
                    )

        # Sum codewords for all SIDs
        n = len(sids)
        reconstructions = torch.zeros(
            (n, self.dim_pca), device=self.device, dtype=torch.float32
        )

        for i, sid in enumerate(sids):
            for level_idx, code_idx in enumerate(sid):
                reconstructions[i] += self.codebooks[level_idx][code_idx]

        # Inverse transform to original space
        decoded = self.preprocess.inverse_transform(reconstructions)

        # Convert to numpy
        return to_numpy(decoded)
