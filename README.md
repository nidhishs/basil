# 🌿 BASIL

**Balanced Assignment Semantic ID Library.** Convert high-dimensional embeddings into short, discrete semantic IDs and back.

## Installation

```bash
pip install -e .
```

Requires Python 3.11+ and `torch>=2.0.0`.

⚠️ **MPS Compatibility**: If you're using Apple Silicon with MPS backend, you must set the environment variable:
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```
This enables fallback to CPU for operations not yet supported on MPS.

## Quick Start

### 1. Train a Codec

Prepare embeddings and train:
```python
import numpy as np

# Save your embeddings
embeddings = np.random.randn(10000, 768)  # Your [N, D] embeddings
np.savez_compressed("embeddings.npz", embeddings=embeddings)
```

```bash
# Train with defaults
./scripts/run_train.sh
```

This creates `outputs/basil-01/` with trained artifacts.

### 2. Encode/Decode

```python
from basil import BasilCodec

# Load trained codec
codec = BasilCodec("outputs/basil-01")

# Encode embedding to semantic ID
embedding = np.random.randn(768)
sid = codec.encode(embedding)
print(f"Semantic ID: {sid}")  # e.g., [42, 156, 789]

# Decode back to embedding
decoded = codec.decode(sid)
print(f"Shape: {decoded.shape}")  # (768,)

# Batch operations
embeddings = np.random.randn(1000, 768)
sids = codec.batch_encode(embeddings)
decoded_batch = codec.batch_decode(sids)
```

### 3. Custom Training

```python
from basil import BasilTrainer, Preprocessor
from basil.config import PreprocessConfig, TrainerConfig

# Configure
preprocess_cfg = PreprocessConfig(variance_to_keep=0.95, dim_pca_max=256)
trainer_cfg = TrainerConfig(levels=3, k_per_level=1024, max_iters=20)

# Train
preprocessor = Preprocessor(preprocess_cfg)
trainer = BasilTrainer(preprocessor, trainer_cfg)
trainer.fit(embeddings)
trainer.save("my_artifacts")

# Use
codec = BasilCodec("my_artifacts")
```

## CLI Training

```bash
python scripts/train.py --input embeddings.npz --out_dir artifacts --levels 3 --k 1024
```

Key parameters:
- `--levels`: Number of codebook levels (default: 3)
- `--k`: Codewords per level (default: 1024)
- `--pca_max_dim`: Max PCA dimensions (default: 256)
- `--variance_to_keep`: Variance to retain in PCA (default: 0.95)

## How It Works

Multi-level additive quantization with balanced assignment:

1. **Preprocess**: Center, PCA, and normalize embeddings
2. **Train**: Learn codebooks level-by-level on residuals with utilization balancing
3. **Encode**: Find nearest codewords at each level, return indices as semantic ID
4. **Decode**: Sum selected codewords and apply inverse preprocessing
