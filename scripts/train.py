#!/usr/bin/env python3
"""
Training script for BASIL.

Trains codebooks from embeddings and saves artifacts.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from basil import BasilTrainer, Preprocessor
from basil.config import PreprocessConfig, TrainerConfig
from basil.utils import setup_logger


def parse_args():
    """Parse command-line arguments."""
    # fmt: off
    parser = argparse.ArgumentParser(description="Train BASIL codebooks from embeddings", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Input/Output
    parser.add_argument("--input", type=str, default="embeddings.npz", help="Path to input NPZ file with 'embeddings' array")
    parser.add_argument("--out_dir", type=str, default="artifacts", help="Output directory for artifacts")
    
    # Model architecture
    parser.add_argument("--levels", type=int, default=4, help="Number of residual levels (L)")
    parser.add_argument("--k", type=int, default=4096, help="Number of centroids per level (K)")
    
    # Training parameters
    parser.add_argument("--max_iters", type=int, default=20, help="Maximum k-means iterations per level")
    parser.add_argument("--size_penalty_lambda", type=float, default=0.01, help="Soft-balance strength (0 = no balancing)")
    parser.add_argument("--batch_size", type=int, default=8192, help="Batch size for processing")
    parser.add_argument("--tol", type=float, default=1e-4, help="Convergence tolerance for k-means")
    
    # PCA parameters
    parser.add_argument("--pca_max_dim", type=int, default=256, help="Maximum PCA dimensions (0 = no cap)")
    parser.add_argument("--variance_to_keep", type=float, default=0.95, help="Target variance fraction for PCA")
    
    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/mps/cpu, or None for auto)")
    parser.add_argument("--verbose", action="store_true", default=True, help="Enable verbose output")
    parser.add_argument("--quiet", action="store_true", help="Disable verbose output")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level (DEBUG shows per-step logs)")
    # fmt: on

    return parser.parse_args()


def main():
    """
    Main training function.
    """
    args = parse_args()

    # Set up logger
    logger = setup_logger("basil.train", level=args.log_level)

    # Handle verbose flag
    verbose = args.verbose and not args.quiet

    # Set up paths
    embeddings_path = Path(args.input)
    output_dir = Path(args.out_dir)

    # Check input exists
    if not embeddings_path.exists():
        logger.error(f"Error: {embeddings_path} not found")
        logger.error("Please provide an NPZ file with 'embeddings' array")
        sys.exit(1)

    # Load embeddings
    logger.info(f"Loading embeddings from {embeddings_path}")
    data = np.load(embeddings_path)

    if "embeddings" not in data:
        logger.error("Error: 'embeddings' array not found in NPZ file")
        logger.error(f"Available arrays: {list(data.keys())}")
        sys.exit(1)

    embeddings = data["embeddings"]
    logger.info(
        f"Loaded {embeddings.shape[0]:,} embeddings (dim={embeddings.shape[1]})"
    )

    # Create configurations from arguments
    preprocess_cfg = PreprocessConfig(
        variance_to_keep=args.variance_to_keep,
        dim_pca_max=args.pca_max_dim if args.pca_max_dim > 0 else None,
        eps=1e-6,
        seed=args.seed,
    )

    trainer_cfg = TrainerConfig(
        levels=args.levels,
        k_per_level=args.k,
        max_iters=args.max_iters,
        tol=args.tol,
        size_penalty_lambda=args.size_penalty_lambda,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        verbose=verbose,
        log_level=args.log_level,
    )

    # Initialize preprocessor and trainer
    preprocessor = Preprocessor(preprocess_cfg)
    trainer = BasilTrainer(preprocessor, trainer_cfg)

    trainer.fit(embeddings)

    # Save artifacts
    logger.info(f"Saving artifacts to {output_dir}/")
    trainer.save(output_dir)


if __name__ == "__main__":
    main()
