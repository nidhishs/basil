from pathlib import Path
from typing import Iterable, List, Union

import numpy as np
import torch

from basil.config import PreprocessConfig
from basil.io import load_artifacts, validate_artifacts
from basil.preprocess import Preprocessor
from basil.utils import chunk_iterator, get_device, to_numpy, to_torch


class BasilCodec:
    """BASIL codec for encoding/decoding embeddings using residual vector quantization.

    This codec loads pre-trained artifacts and provides methods to encode embeddings
    into semantic IDs and decode them back to approximate embeddings.

    Attributes:
        metadata: Dictionary containing codec configuration metadata.
        levels: Number of quantization levels.
        k_per_level: Number of centroids per quantization level.
        dim_in: Input embedding dimension.
        dim_pca: PCA-reduced dimension.
        device: PyTorch device for computations.
        codebooks: List of quantization codebooks as tensors.
        preprocess: Preprocessor instance for PCA transformation.
    """

    def __init__(self, artifact_dir: Union[str, Path]):
        """Initialize the BASIL codec with pre-trained artifacts.

        Args:
            artifact_dir: Path to directory containing trained codec artifacts
                including metadata, codebooks, and preprocessing components.

        Raises:
            FileNotFoundError: If artifact directory or required files don't exist.
            ValueError: If artifacts fail validation checks.
        """
        # Load and validate artifacts
        artifacts = load_artifacts(artifact_dir)
        validate_artifacts(artifacts)

        self.metadata = artifacts["metadata"]
        self.levels = self.metadata["levels"]
        self.k_per_level = self.metadata["k_per_level"]
        self.dim_in = self.metadata["dim_in"]
        self.dim_pca = self.metadata["dim_pca"]

        # Select device
        self.device = get_device(None)

        # Codebooks to device
        self.codebooks = []
        for i in range(self.levels):
            cb = artifacts["codebooks"][f"C{i + 1}"]
            self.codebooks.append(torch.from_numpy(cb).float().to(self.device))

        # Preprocessor from arrays, then move to device
        p_cfg = PreprocessConfig(
            variance_to_keep=self.metadata.get("variance_to_keep", 0.95),
            dim_pca_max=self.metadata.get("dim_pca_max", 256),
            eps=self.metadata.get("eps", 1e-6),
            seed=self.metadata.get("seed", 42),
        )
        self.preprocess = Preprocessor.from_arrays(p_cfg, artifacts["preprocess"]).to(
            self.device
        )

    # ------------------------------ public API ------------------------------

    def encode(self, embedding: Union[np.ndarray, torch.Tensor]) -> List[int]:
        """Encode a single embedding to a semantic ID.

        Args:
            embedding: Input embedding vector as numpy array or torch tensor.
                Can be 1D or 2D (single row).

        Returns:
            List of integer codes representing the quantized embedding,
            with one code per quantization level.

        Raises:
            ValueError: If embedding dimension doesn't match expected input dimension.
        """
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        return self.batch_encode(embedding)[0]

    def batch_encode(
        self, embeddings: Union[np.ndarray, torch.Tensor], batch_size: int = 8192
    ) -> List[List[int]]:
        """Encode multiple embeddings to semantic IDs.

        Args:
            embeddings: Input embeddings as 2D numpy array or torch tensor
                with shape (n_embeddings, embedding_dim).
            batch_size: Number of embeddings to process in each batch
                for memory efficiency.

        Returns:
            List of semantic IDs, where each semantic ID is a list of integer
            codes (one per quantization level).

        Raises:
            ValueError: If embeddings are not 2D or have wrong dimension.
        """
        embeddings = to_torch(embeddings, self.device)
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D tensor, got shape {embeddings.shape}")
        if embeddings.shape[1] != self.dim_in:
            raise ValueError(
                f"Dimension mismatch: expected {self.dim_in}, got {embeddings.shape[1]}"
            )

        n = embeddings.shape[0]
        all_codes: List[List[int]] = []
        with torch.inference_mode():
            for start, end in chunk_iterator(n, batch_size):
                x = self.preprocess.transform(embeddings[start:end])
                all_codes.extend(self._encode_residuals(x))
        return all_codes

    def decode(self, sid: Iterable[int]) -> np.ndarray:
        """Decode a single semantic ID to an approximate embedding.

        Args:
            sid: Semantic ID as an iterable of integer codes,
                with one code per quantization level.

        Returns:
            Reconstructed embedding as numpy array with shape (embedding_dim,).

        Raises:
            ValueError: If semantic ID has wrong length or invalid code indices.
        """
        return self.batch_decode([sid])[0]

    def batch_decode(self, sids: List[Iterable[int]]) -> np.ndarray:
        """Decode multiple semantic IDs to approximate embeddings.

        Args:
            sids: List of semantic IDs, where each semantic ID is an iterable
                of integer codes (one per quantization level).

        Returns:
            Reconstructed embeddings as numpy array with shape
            (n_embeddings, embedding_dim). Returns empty array if input is empty.

        Raises:
            ValueError: If any semantic ID has wrong length or invalid code indices.
        """
        sids_list = self._normalize_and_validate_sids(sids)
        if not sids_list:
            return np.empty((0, self.dim_in), dtype=np.float32)

        with torch.inference_mode():
            recon = self._reconstruct_from_sids(sids_list)
            decoded = self.preprocess.inverse_transform(recon)
            return to_numpy(decoded)

    # ------------------------------ helpers ------------------------------

    def _encode_residuals(self, x: torch.Tensor) -> List[List[int]]:
        """Encode a batch of embeddings using residual vector quantization.

        Args:
            x: Batch of embeddings already transformed to PCA space,
                with shape (batch_size, dim_pca).

        Returns:
            List of semantic IDs, where each semantic ID contains one code
            per quantization level.
        """
        residuals = x.clone()
        codes: List[List[int]] = [[] for _ in range(residuals.shape[0])]
        for codebook in self.codebooks:
            dists = torch.cdist(residuals, codebook, p=2)
            idxs = dists.argmin(dim=1)
            for i, idx in enumerate(idxs):
                codes[i].append(int(idx))
            residuals -= codebook[idxs]
        return codes

    def _normalize_and_validate_sids(
        self, sids: List[Iterable[int]]
    ) -> List[List[int]]:
        """Normalize and validate semantic IDs for batch operations.

        Args:
            sids: List of semantic IDs as iterables of integer codes.

        Returns:
            List of semantic IDs converted to lists, validated for correct
            length and code indices.

        Raises:
            ValueError: If any semantic ID has incorrect length or invalid codes.
        """
        if not sids:
            return []
        sids_list = [list(sid) for sid in sids]
        for i, sid in enumerate(sids_list):
            if len(sid) != self.levels:
                raise ValueError(
                    f"Invalid SID length at index {i}: expected {self.levels}, got {len(sid)}"
                )
            for level_idx, code_idx in enumerate(sid):
                if not (0 <= code_idx < self.k_per_level):
                    raise ValueError(
                        f"Invalid code index at SID {i}, level {level_idx+1}: {code_idx} (must be 0-{self.k_per_level-1})"
                    )
        return sids_list

    def _reconstruct_from_sids(self, sids_list: List[List[int]]) -> torch.Tensor:
        """Reconstruct embeddings from semantic IDs in PCA space.

        Args:
            sids_list: List of validated semantic IDs, where each semantic ID
                is a list of integer codes (one per quantization level).

        Returns:
            Reconstructed embeddings in PCA space as tensor with shape
            (n_embeddings, dim_pca).
        """
        n = len(sids_list)
        recon = torch.zeros((n, self.dim_pca), device=self.device, dtype=torch.float32)
        for i, sid in enumerate(sids_list):
            for level_idx, code_idx in enumerate(sid):
                recon[i] += self.codebooks[level_idx][code_idx]
        return recon
