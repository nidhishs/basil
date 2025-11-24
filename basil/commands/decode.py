"""Decode command implementation."""

from pathlib import Path

import numpy as np

from basil.codec import BasilCodec
from basil.utils import process_batched_mmap, setup_logging

logger = setup_logging(__name__)


def add_parser(subparsers):
    """Add decode command parser to subparsers.

    Args:
        subparsers: The subparsers object from argparse
    """
    # fmt: off
    parser = subparsers.add_parser("decode", help="Decode semantic IDs to vectors")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to input .npy file containing semantic IDs")
    parser.add_argument("-o", "--output", type=str, help="Path to output .npy file for vectors (default: <input>_decoded.npy)")
    parser.add_argument("-c", "--checkpoint", type=str, required=True, help="Path to trained model checkpoint directory")
    parser.add_argument("-b", "--backend", type=str, default="onnx", choices=["onnx", "torch"], help="Backend to use (default: onnx)")
    parser.add_argument("-bs", "--batch-size", type=int, default=1024, help="Batch size for processing (default: 1024)")
    parser.set_defaults(func=decode_command)
    # fmt: on


def decode_command(args):
    """Execute the decode command.

    Args:
        args: Parsed command-line arguments
    """
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    # Validate input
    semantic_ids = np.load(args.input, mmap_mode="r")
    if semantic_ids.ndim != 2:
        raise ValueError(
            f"Expected 2D array of semantic IDs, got shape {semantic_ids.shape}"
        )

    # Determine output path
    output_path = args.output or str(
        input_path.parent / f"{input_path.stem}_decoded.npy"
    )

    logger.info(f"Decoding {len(semantic_ids):,} semantic IDs from {args.input}")
    logger.info(f"Using {args.backend} backend from {args.checkpoint}")

    codec = BasilCodec(args.checkpoint, backend=args.backend)

    process_batched_mmap(
        input_path=args.input,
        output_path=output_path,
        process_fn=codec.batch_decode,
        batch_size=args.batch_size,
        log_fn=logger.info,
    )

    logger.info(f"Decoded vectors saved to {output_path}")
