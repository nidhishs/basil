# 🌿 BASIL

Compress embeddings into semantic IDs using residual quantization and Variational Autoencoders.

## Inference

Use ONNX-converted models to encode/decode embeddings. Only requires ONNX runtime.

### Install
```bash
pip install -e .
```

### Encode vectors to semantic IDs
```bash
basil encode -i embeddings.npy -c output/semid-model-1
```

### Decode semantic IDs to vectors
```bash
basil decode -i embeddings_encoded.npy -c output/semid-model-1
```

Run `basil encode --help` or `basil decode --help` for all options.

**Input format:** `.npy` file with 2D array of shape `[N, D]` where N = number of embeddings, D = embedding dimension

## Training

Train your own models. Requires PyTorch.

### Install with training dependencies
```bash
pip install -e .[train]
```

### Train a model
```bash
basil train configs/config.minimal.yaml
```

Run `basil train --help` for options.

**Config:** 
- [`configs/config.minimal.yaml`](configs/config.minimal.yaml) - Minimal template to get started
- [`configs/config.full.yaml`](configs/config.full.yaml) - Complete documentation of all parameters

**Output:** Checkpoints saved to `output_dir/epoch-{epoch}/` containing:
- `model.safetensors` - Model weights
- `config.json` - Model configuration

**Using checkpoints:** Training checkpoints require the `torch` backend:
```bash
basil encode -i embeddings.npy -c output/my_run/epoch-29 -b torch
```

### Training on Large Datasets

For datasets larger than available RAM, use streaming mode with memory-mapped files:

```yaml
data:
  stream: true              # Use mmap instead of loading to RAM
  is_pre_shuffled: true     # Confirm your data is pre-shuffled
```

**How it works:**
- Uses memory-mapped I/O to let the OS swap pages in/out as needed
- Reads data sequentially from disk to minimize seeking
- Requires pre-shuffled data (set `is_pre_shuffled: true` to confirm)
