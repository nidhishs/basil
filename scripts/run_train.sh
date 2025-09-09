#!/usr/bin/env bash
# Simple fixed-config runner for BASIL training.
# Usage: ./run_train.sh
# Expects: ./embeddings.npz (with array 'embeddings' [N, D] float32)
# Artifacts: ./outputs/basil-01/{basil.npz, metadata.json}
set -euo pipefail

# -----------------------------
# Fixed configuration (edit me)
# -----------------------------
INPUT="embeddings.npz"
OUT_DIR="outputs/basil-01"

LEVELS=3           # residual levels (L)
K=256              # centroids per level (K)
MAX_ITERS=20       # k-means iterations per level
VAR_KEEP=0.97      # target variance fraction for PCA
SIZE_LAMBDA=0.10   # soft-balance strength (0..)
LOG_LEVEL="DEBUG"   # logging level (DEBUG for per-step logs)

# Enable MPS fallback for compatibility with PyTorch operations on Apple Silicon
export PYTORCH_ENABLE_MPS_FALLBACK=1

# -----------------------------
# Checks
# -----------------------------
if [[ ! -f "$INPUT" ]]; then
  echo "ERROR: '$INPUT' not found."
  echo "Place an NPZ with array 'embeddings' of shape [N, D] float32 at: $INPUT"
  exit 1
fi

mkdir -p "$OUT_DIR"

# -----------------------------
# Run
# -----------------------------
python scripts/train.py \
  --input "$INPUT" \
  --out_dir "$OUT_DIR" \
  --levels "$LEVELS" \
  --k "$K" \
  --max_iters "$MAX_ITERS" \
  --variance_to_keep "$VAR_KEEP" \
  --size_penalty_lambda "$SIZE_LAMBDA" \
  --log_level "$LOG_LEVEL"