#!/usr/bin/env bash
# Simple fixed-config runner for BASIL training.
# Usage: ./run.sh
# Expects: ./embeddings.npz (with array 'embeddings' [N, D] float32)
# Artifacts: ./outputs/basil-v1/{basil.npz, metadata.json}
set -euo pipefail

# -----------------------------
# Fixed configuration (edit me)
# -----------------------------
INPUT="embeddings.npz"
OUT_DIR="outputs/basil-01"

LEVELS=3           # residual levels (L)
K=128              # centroids per level (K)
MAX_ITERS=20       # k-means iterations per level
PCA_MAX_DIM=1024   # cap for PCA dims (0 disables cap)
VAR_KEEP=0.95      # target variance fraction for PCA
SIZE_LAMBDA=0.10   # soft-balance strength (0..)
LOG_LEVEL="DEBUG"   # logging level (DEBUG for per-step logs)

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
  --pca_max_dim "$PCA_MAX_DIM" \
  --variance_to_keep "$VAR_KEEP" \
  --size_penalty_lambda "$SIZE_LAMBDA" \
  --log_level "$LOG_LEVEL"