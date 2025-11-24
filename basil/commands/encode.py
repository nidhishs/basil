"""Encode command implementation."""

from pathlib import Path

import numpy as np

from basil.codec import BasilCodec
from basil.utils import process_batched_mmap, setup_logging

logger = setup_logging(__name__)


def add_parser(subparsers):
    """Add encode command parser to subparsers.

    Args:
        subparsers: The subparsers object from argparse
    """
    # fmt: off
    parser = subparsers.add_parser("encode", help="Encode vectors to semantic IDs")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to input .npy file containing vectors")
    parser.add_argument("-o", "--output", type=str, help="Path to output .npy file for semantic IDs (default: <input>_encoded.npy)")
    parser.add_argument("-c", "--checkpoint", type=str, required=True, help="Path to trained model checkpoint directory")
    parser.add_argument("-b", "--backend", type=str, default="onnx", choices=["onnx", "torch"], help="Backend to use (default: onnx)")
    parser.add_argument("-bs", "--batch-size", type=int, default=1024, help="Batch size for processing (default: 1024)")
    parser.set_defaults(func=encode_command)
    # fmt: on


def encode_command(args):
    """Execute the encode command.

    Args:
        args: Parsed command-line arguments
    """
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    # Validate input
    vectors = np.load(args.input, mmap_mode="r")
    if vectors.ndim != 2:
        raise ValueError(f"Expected 2D array of vectors, got shape {vectors.shape}")

    # Determine output path
    output_path = args.output or str(
        input_path.parent / f"{input_path.stem}_encoded.npy"
    )

    logger.info(f"Encoding {len(vectors):,} vectors from {args.input}")
    logger.info(f"Using {args.backend} backend from {args.checkpoint}")

    codec = BasilCodec(args.checkpoint, backend=args.backend)

    process_batched_mmap(
        input_path=args.input,
        output_path=output_path,
        process_fn=codec.batch_encode,
        batch_size=args.batch_size,
        log_fn=logger.info,
    )

    logger.info(f"Encoded vectors saved to {output_path}")
