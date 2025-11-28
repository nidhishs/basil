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

## Hyperparameter Optimization

Find optimal hyperparameters using Bayesian optimization (Optuna) with Ray Tune.

### Install with tuning dependencies
```bash
pip install -e .[tune]
```

### Run optimization
```bash
basil optimize configs/optimize.minimal.yaml
```

Run `basil optimize --help` for all options.

**Config:** 
- [`configs/optimize.minimal.yaml`](configs/optimize.minimal.yaml) - Minimal template to get started
- [`configs/optimize.full.yaml`](configs/optimize.full.yaml) - Complete documentation of all parameters

**Search space types:**
| Type | Description | Example |
|------|-------------|---------|
| `uniform` | Continuous [low, high] | `low: 0.0, high: 0.2` |
| `loguniform` | Log-uniform (for learning rates) | `low: 1e-4, high: 1e-2` |
| `randint` | Integer [low, high) | `low: 2, high: 5` |
| `choice` | Pick from list | `values: [true, false]` |
| `quniform` | Quantized uniform | `low: 32, high: 256, q: 32` |

**Example search space:**
```yaml
search_space:
  model:
    codebook_size:
      type: choice
      values: [256, 512, 1024]
  train:
    lr:
      type: loguniform
      low: 1.0e-4
      high: 5.0e-3
```

**Optimization settings:**
```yaml
optimization:
  num_samples: 20          # Number of trials to run
  metric: val_cos_sim      # Metric to optimize
  mode: max                # "max" or "min"
  max_concurrent: 1        # Concurrent trials (based on GPU count)
  scheduler: asha          # "asha" for early stopping or "none"
  grace_period: 3          # Min epochs before ASHA can stop a trial
  reduction_factor: 3      # ASHA stops bottom 1/N trials at each rung
```

**Algorithm details:**
- Uses **Optuna** (Tree-structured Parzen Estimator) for Bayesian search
- Optional **ASHA** (Async Successive Halving) scheduler aggressively stops underperforming trials early

**Output:**
- `best_config.yaml` - Best hyperparameters found, ready for `basil train`
- Trial results in `output_dir/basil_optimize/` (Ray Tune format)

**Train with best config:**
```bash
basil train output/tune_results/best_config.yaml
```
