from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


# fmt: off
class BasilModelConfig(BaseModel):
    """
    Defines the RQ-VAE architecture parameters.
    """
    model_config = ConfigDict(extra='forbid')
    
    input_dim: int = Field(..., description="Dimension of input embeddings (e.g. 768 for BERT)")
    hidden_dim: int = Field(default=512, description="Size of internal MLP projection layers")
    latent_dim: int = Field(default=32, description="Dimension of the quantized codebook vectors")
    codebook_size: int = Field(default=1024, description="Number of entries in each codebook level")
    num_levels: int = Field(default=3, description="Depth of residual quantization")
    dropout: float = Field(default=0.1, description="Dropout probability for regularization")
    commitment_beta: float = Field(default=0.25, description="Weight of the commitment loss")
    use_hierarchical: bool = Field(default=False, description="Use hierarchical codebook sizes where each level is half the size of the previous level")
    ema_decay: float = Field(default=0.99, description="EMA decay factor for codebook updates")
    reset_code_interval: int = Field(default=200, description="Number of steps a code can be unused before being reset.")
    stochastic_sampling: bool = Field(default=True, description="Use random sampling during training")
    stochastic_temperature: float = Field(default=0.6, description="Temperature for stochastic sampling (higher = more random)")


class BasilDataConfig(BaseModel):
    """
    Separates I/O configuration. Allows switching between in-memory and streaming 
    strategies without changing model logic.
    """
    model_config = ConfigDict(extra='forbid')
    
    path: str = Field(..., description="Path to the .npy dataset file")
    val_set_size: float = Field(default=0.05, description="Fraction of dataset to use for validation (0.0-1.0)")
    batch_size: int = Field(default=4096, description="Batch size. Keep > 2048 for healthy codebooks.")
    num_workers: int = Field(default=8, description="DataLoader workers")
    
    # Streaming allows training on datasets larger than RAM
    stream: bool = Field(default=False, description="If True, use mmap+sequential read. If False, load to RAM.")
    # Safety flag prevents user error where streaming + unshuffled data = model collapse.
    is_pre_shuffled: bool = Field(default=False, description="Must be True if streaming to confirm disk data is randomized.")
    prefetch_factor: int = Field(default=2, description="Number of batches to prefetch per worker.")

    @field_validator('is_pre_shuffled')
    def validate_streaming_config(cls, v, info):
        """Ensure streaming mode requires pre-shuffled data."""
        if info.data.get('stream') and not v:
            raise ValueError(
                "Configuration Error: `stream=True` requires `is_pre_shuffled=True`.\n"
                "Streaming reads sequentially. If data isn't randomized on disk, the model will collapse.\n"
            )
        return v


class BasilTrainConfig(BaseModel):
    """
    Hyperparameters and System config.
    """
    model_config = ConfigDict(extra='forbid')
    
    # Logging
    project_name: str | None = Field(default=None, description="WandB project name. If None, WandB is disabled.")
    run_name: str | None = Field(default=None, description="Name of this specific run")
    log_interval: int = Field(default=50, description="Log metrics every N steps")
    
    # Optimization
    epochs: int = Field(default=20, description="Total training epochs")
    lr: float = Field(default=1e-3, description="Peak learning rate")
    min_lr: float = Field(default=1e-5, description="Minimum learning rate")
    warmup_ratio: float = Field(default=0.1, description="Ratio of total steps for LR warmup")
    weight_decay: float = Field(default=1e-4, description="AdamW weight decay")
    gradient_clip_norm: float = Field(default=1.0, description="Max gradient norm")
    gradient_accumulation_steps: int = Field(default=1, ge=1, description="Number of steps to accumulate gradients before updating.")
    save_interval: int = Field(default=5, ge=0, description="Save checkpoint every N epochs. Set to 0 to disable intermediate checkpoints.")
    
    # System
    seed: int = Field(default=42, description="Random seed")
    device: str = Field(default="auto", description="Accelerator: 'auto', 'cuda', 'mps', 'cpu'")
    use_amp: bool = Field(default=True, description="Enable Automatic Mixed Precision (AMP)")
# fmt: on
