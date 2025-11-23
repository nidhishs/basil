from __future__ import annotations

from dataclasses import dataclass, field


# fmt: off
@dataclass
class BasilModelConfig:
    """
    Defines the RQ-VAE architecture parameters.
    """
    input_dim: int = field(metadata={"help": "Dimension of input embeddings (e.g. 768 for BERT)"})
    hidden_dim: int = field(default=512, metadata={"help": "Size of internal MLP projection layers"})
    latent_dim: int = field(default=32, metadata={"help": "Dimension of the quantized codebook vectors"})
    codebook_size: int = field(default=1024, metadata={"help": "Number of entries in each codebook level"})
    num_levels: int = field(default=3, metadata={"help": "Depth of residual quantization"})
    dropout: float = field(default=0.1, metadata={"help": "Dropout probability for regularization"})
    commitment_beta: float = field(default=0.25, metadata={"help": "Weight of the commitment loss"})
    use_hierarchical: bool = field(default=False, metadata={"help": "Use hierarchical codebook sizes where each level is half the size of the previous level"})
    ema_decay: float = field(default=0.99, metadata={"help": "EMA decay factor for codebook updates"})
    stochastic_sampling: bool = field(default=True, metadata={"help": "Use random sampling during training"})
    stochastic_temperature: float = field(default=0.6, metadata={"help": "Temperature for stochastic sampling (higher = more random)"})


@dataclass
class BasilDataConfig:
    """
    Separates I/O configuration. Allows switching between in-memory and streaming 
    strategies without changing model logic.
    """
    path: str = field(metadata={"help": "Path to the .npy dataset file"})
    val_set_size: float = field(default=0.05, metadata={"help": "Fraction of dataset to use for validation (0.0-1.0)"})
    batch_size: int = field(default=4096, metadata={"help": "Batch size. Keep > 2048 for healthy codebooks."})
    num_workers: int = field(default=8, metadata={"help": "DataLoader workers"})
    
    # Streaming allows training on datasets larger than RAM
    stream: bool = field(default=False, metadata={"help": "If True, use mmap+sequential read. If False, load to RAM."})
    # Safety flag prevents user error where streaming + unshuffled data = model collapse.
    is_pre_shuffled: bool = field(default=False, metadata={"help": "Must be True if streaming to confirm disk data is randomized."})
    prefetch_factor: int = field(default=2, metadata={"help": "Number of batches to prefetch per worker."})

    def __post_init__(self):
        if self.stream and not self.is_pre_shuffled:
            raise ValueError(
                "Configuration Error: `stream=True` requires `is_pre_shuffled=True`.\n"
                "Streaming reads sequentially. If data isn't randomized on disk, the model will collapse.\n"
            )


@dataclass
class BasilTrainConfig:
    """
    Hyperparameters and System config.
    """
    # Logging
    project_name: str | None = field(default=None, metadata={"help": "WandB project name. If None, WandB is disabled."})
    run_name: str | None = field(default=None, metadata={"help": "Name of this specific run"})
    log_interval: int = field(default=50, metadata={"help": "Log metrics every N steps"})
    
    # Optimization
    epochs: int = field(default=20, metadata={"help": "Total training epochs"})
    lr: float = field(default=1e-3, metadata={"help": "Peak learning rate"})
    min_lr: float = field(default=1e-5, metadata={"help": "Minimum learning rate"})
    warmup_ratio: float = field(default=0.1, metadata={"help": "Ratio of total steps for LR warmup"})
    weight_decay: float = field(default=1e-4, metadata={"help": "AdamW weight decay"})
    gradient_clip_norm: float = field(default=1.0, metadata={"help": "Max gradient norm"})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": "Number of steps to accumulate gradients before updating."})
    
    # System
    seed: int = field(default=42, metadata={"help": "Random seed"})
    device: str = field(default="auto", metadata={"help": "Accelerator: 'auto', 'cuda', 'mps', 'cpu'"})
    use_amp: bool = field(default=True, metadata={"help": "Enable Automatic Mixed Precision (AMP)"})

    def __post_init__(self):
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1")
# fmt: on
