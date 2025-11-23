from __future__ import annotations

import copy
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

from basil.config import BasilDataConfig, BasilModelConfig, BasilTrainConfig

MODEL_FILENAME = "model.safetensors"
CONFIG_FILENAME = "config.json"

TORCH_IMPORT_ERROR = "This feature requires the optional 'torch' dependency. Please install it via 'pip install basil[train]'."

# --- Optional Imports (Safe for Inference) ---
# Guards ensure this module is safe for inference environments without Torch.
try:
    import torch
    from safetensors.torch import save_file
except ImportError:
    torch = None
    save_file = None


def setup_logging(name: str = "basil", level: str = "INFO") -> logging.Logger:
    """
    Configures a standard Python logger with a clean formatter.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


class MetricsTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = defaultdict(float)
        self.count = defaultdict(int)
        self.total_samples = 0

    def update(self, metrics: dict[str, float], n: int = 1):
        """
        metrics: Dictionary of metric names and their values.
        n: Batch size (weight).
        """
        self.total_samples += n
        for k, v in metrics.items():
            self.val[k] += v * n
            self.count[k] += n

    def average(self) -> dict[str, float]:
        return {k: self.val[k] / self.count[k] for k in self.val}


def setup_device(request: str) -> torch.device:
    """
    Auto-detects the best available hardware (CUDA > MPS > CPU).
    Raises ImportError if torch is not installed.
    """
    if torch is None:
        raise ImportError(TORCH_IMPORT_ERROR)

    if request != "auto":
        return torch.device(request)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(
    model: torch.nn.Module,
    model_cfg: BasilModelConfig,
    train_cfg: BasilTrainConfig,
    data_cfg: BasilDataConfig,
    out_dir: Path,
    export_onnx: bool = True,
):
    """
    Saves weights (safetensors), config (json), and optionally exports ONNX.
    Expects a model on any device (exports safely via copy).
    """
    from basil import __version__

    if torch is None:
        raise ImportError(TORCH_IMPORT_ERROR)

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save Weights (Safetensors)
    # Efficient zero-copy save. Safe to do from GPU.
    save_file(model.state_dict(), out_dir / MODEL_FILENAME)

    # 2. Save Config
    # fmt: off
    meta = {
        "model": (model_cfg.model_dump() if hasattr(model_cfg, "model_dump") else model_cfg),
        "train": (train_cfg.model_dump() if hasattr(train_cfg, "model_dump") else train_cfg),
        "data": (data_cfg.model_dump() if hasattr(data_cfg, "model_dump") else data_cfg),
        "format": f"basil-v{__version__}",
    }
    # fmt: on

    with open(out_dir / CONFIG_FILENAME, "w") as f:
        json.dump(meta, f, indent=2)

    # 3. Export ONNX (Optional, can be skipped for speed)
    if export_onnx:
        _export_onnx_set(model, model_cfg, out_dir)


# Wrappers expose specific methods (encode/decode) as the primary forward pass
class EncWrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m.export_encode(x)


class DecWrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, c):
        return self.m.export_decode(c)


def _export_onnx_set(model: torch.nn.Module, cfg: BasilModelConfig, out_dir: Path):
    """
    Internal helper to export Encoder/Decoder pairs. Uses deepcopy + CPU move.
    """
    if torch is None:
        raise ImportError(TORCH_IMPORT_ERROR)

    # Create a disposable CPU copy.
    export_model = copy.deepcopy(model).to("cpu")
    export_model.eval()

    dummy_emb = torch.randn(1, cfg.input_dim).to("cpu")
    dummy_codes = torch.zeros(1, cfg.num_levels).int().to("cpu")

    torch.onnx.export(
        EncWrapper(export_model),
        (dummy_emb,),
        out_dir / "encoder.onnx",
        input_names=["emb"],
        output_names=["semid"],
        dynamic_axes={"emb": {0: "batch_size"}, "semid": {0: "batch_size"}},
        opset_version=17,
    )

    torch.onnx.export(
        DecWrapper(export_model),
        (dummy_codes,),
        out_dir / "decoder.onnx",
        input_names=["semid"],
        output_names=["emb"],
        dynamic_axes={"semid": {0: "batch_size"}, "emb": {0: "batch_size"}},
        opset_version=17,
    )
