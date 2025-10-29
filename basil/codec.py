import json
from pathlib import Path
from typing import List, Literal, Protocol, Union

import numpy as np

# Importing constants ensures we look for the same files the trainer saved
from basil.utils import CONFIG_FILENAME, MODEL_FILENAME


# -------------------------------------------------------------------------
# 1. Backend Interface (Strategy Pattern)
# -------------------------------------------------------------------------
class CodecBackend(Protocol):
    def batch_encode(self, vectors: np.ndarray) -> np.ndarray: ...
    def batch_decode(self, semantic_ids: np.ndarray) -> np.ndarray: ...


# -------------------------------------------------------------------------
# 2. Implementations
# -------------------------------------------------------------------------
class OnnxBackend:
    """
    Production backend. Zero Torch dependency.
    """

    def __init__(self, artifact_path: Path):
        import onnxruntime as ort

        # Suppress excessive ONNX runtime logs
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        providers = ["CPUExecutionProvider"]

        self.enc_sess = ort.InferenceSession(
            str(artifact_path / "encoder.onnx"), sess_options=opts, providers=providers
        )
        self.dec_sess = ort.InferenceSession(
            str(artifact_path / "decoder.onnx"), sess_options=opts, providers=providers
        )

    def batch_encode(self, vectors: np.ndarray) -> np.ndarray:
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        # Unpack list to get the first output (semantic_ids)
        return self.enc_sess.run(["semid"], {"emb": vectors})[0]

    def batch_decode(self, semantic_ids: np.ndarray) -> np.ndarray:
        if semantic_ids.dtype != np.int32:
            semantic_ids = semantic_ids.astype(np.int32)
        return self.dec_sess.run(["emb"], {"semid": semantic_ids})[0]


class TorchBackend:
    """
    Development backend. Lazy loads torch.
    """

    def __init__(self, artifact_path: Path, meta: dict):
        try:
            import torch
            from safetensors.torch import load_file

            from basil.config import BasilModelConfig
            from basil.training.model import RQVAE
        except ImportError as e:
            raise ImportError("backend='torch' requires 'basil[train]'.") from e

        self.torch = torch

        valid_keys = BasilModelConfig.__annotations__.keys()
        model_args = {k: v for k, v in meta["model"].items() if k in valid_keys}
        cfg = BasilModelConfig(**model_args)

        self.model = RQVAE(cfg)
        state_dict = load_file(artifact_path / MODEL_FILENAME)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.device = torch.device("cpu")
        self.model.to(self.device)

    def batch_encode(self, vectors: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            t_in = self.torch.from_numpy(vectors).float().to(self.device)
            return self.model.export_encode(t_in).cpu().numpy()

    def batch_decode(self, semantic_ids: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            t_in = self.torch.from_numpy(semantic_ids).long().to(self.device)
            return self.model.export_decode(t_in).cpu().numpy()


# -------------------------------------------------------------------------
# 3. Public API
# -------------------------------------------------------------------------
class BasilCodec:
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        backend: Literal["onnx", "torch"] = "onnx",
    ):
        self.path = Path(checkpoint_dir)
        if not self.path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {self.path}")

        with open(self.path / CONFIG_FILENAME, "r") as f:
            self.meta = json.load(f)

        if backend == "onnx":
            self._backend = OnnxBackend(self.path)
        elif backend == "torch":
            self._backend = TorchBackend(self.path, self.meta)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def encode(self, vector: np.ndarray) -> List[int]:
        if vector.ndim == 1:
            vector = vector[np.newaxis, :]
        return self._backend.batch_encode(vector)[0].tolist()

    def batch_encode(self, vectors: np.ndarray) -> np.ndarray:
        return self._backend.batch_encode(vectors)

    def decode(self, semantic_ids: List[int]) -> np.ndarray:
        arr = np.array(semantic_ids)[np.newaxis, :]
        return self._backend.batch_decode(arr)[0]

    def batch_decode(self, semantic_ids: np.ndarray) -> np.ndarray:
        return self._backend.batch_decode(semantic_ids)
