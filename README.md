# BASIL — Balanced Assignment Semantic ID Library

A production-focused Python library for converting high-dimensional embeddings into short, discrete semantic IDs and back to approximate vectors.

## Features

- **Multi-level additive quantization**: Sequential codebook learning with residual refinement
- **Balanced utilization**: Promotes healthy codeword usage to avoid hot buckets
- **Automatic device selection**: Seamlessly runs on CUDA, MPS (Apple Silicon), or CPU
- **Simple API**: One-line instantiation, minimal configuration
- **Production-ready**: Clean code, deterministic outputs, robust error handling
- **Efficient**: Handles 100k+ embeddings comfortably on developer hardware

## Design Philosophy

- **Production over research**: Simple, robust approaches that work reliably
- **Few knobs**: Sensible defaults with minimal configuration burden
- **Small surface area**: One codec, one trainer, one preprocessor
- **Readable code**: Brief functions, clear variable names, helpful comments

## Installation

```bash
pip install -e .
```

Dependencies:
- numpy >= 1.19.0
- torch >= 1.10.0

## Quick Start

### Training

1. Prepare your embeddings as a numpy array and save to `embeddings.npz`:
```python
import numpy as np

embeddings = # your embeddings [N, D]
np.savez_compressed("embeddings.npz", embeddings=embeddings)
```

2. Run training with fixed defaults:
```bash
./run.sh
```

This creates `artifacts/basil.npz` and `artifacts/metadata.json`.

### Encoding/Decoding

```python
from basil import BasilCodec

# Load trained codec
codec = BasilCodec("artifacts")

# Encode single embedding
embedding = np.random.randn(768)  # Your embedding
sid = codec.encode(embedding)
print(f"Semantic ID: {sid}")  # e.g., [42, 156, 789, 2341]

# Decode back to embedding
decoded = codec.decode(sid)
print(f"Decoded shape: {decoded.shape}")  # (768,)

# Batch operations
embeddings = np.random.randn(1000, 768)
sids = codec.batch_encode(embeddings)
decoded_batch = codec.batch_decode(sids)
```

### Custom Training

```python
from basil import BasilTrainer, Preprocessor
from basil.config import PreprocessConfig, TrainerConfig

# Configure preprocessing
preprocess_cfg = PreprocessConfig(
    variance_to_keep=0.95,  # Retain 95% of variance
    dim_pca_max=256,        # Max PCA dimensions
    seed=42
)

# Configure training
trainer_cfg = TrainerConfig(
    levels=4,               # Number of codebook levels
    k_per_level=4096,       # Codewords per level
    max_iters=20,           # Max iterations per level
    size_penalty_lambda=0.01,  # Utilization regularization
    device=None,            # Auto-select device
    verbose=True
)

# Train
preprocessor = Preprocessor(preprocess_cfg)
trainer = BasilTrainer(preprocessor, trainer_cfg)
trainer.fit(embeddings)
trainer.save("my_artifacts")

# Use trained model
codec = BasilCodec("my_artifacts")
```

## Architecture

BASIL uses multi-level additive quantization:

1. **Preprocessing**: 
   - Center data (subtract mean)
   - Apply PCA for decorrelation and optional dimensionality reduction
   - Scale axes for fair distance comparisons

2. **Sequential Codebook Learning**:
   - Train codebooks level by level on residuals
   - Each level refines the reconstruction from previous levels
   - Utilization balancing prevents codeword underuse

3. **Encoding**:
   - Transform embedding to preprocessed space
   - For each level, find nearest codeword and update residual
   - Return list of code indices (the semantic ID)

4. **Decoding**:
   - Sum selected codewords across levels
   - Apply inverse preprocessing to return to original space

## File Structure

```
basil/
├── __init__.py         # Package exports
├── config.py           # Configuration dataclasses
├── utils.py            # Device selection, helpers
├── io.py              # NPZ/JSON artifact I/O
├── preprocess.py       # PCA and normalization
├── trainer.py          # Codebook training
└── codec.py           # Main encode/decode API

scripts/
└── train_basil.py     # CLI training script

run.sh                 # Quick training with defaults
pyproject.toml         # Package configuration
README.md              # This file
```

## Performance Guidelines

- **Scale**: Tested up to 3k dimensions, millions of embeddings
- **Levels**: Typically 3-6 levels work well
- **Codewords**: 2k-8k per level balances quality and efficiency
- **Memory**: Batch processing prevents quadratic blowups
- **Speed**: 100k embeddings train in minutes on GPU/MPS

## Development

Format code with black and isort:
```bash
pip install -e .[dev]
black basil/ scripts/
isort basil/ scripts/
```

## License

MIT
