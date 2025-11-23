"""Encode command implementation."""

from pathlib import Path

import numpy as np

from basil.codec import BasilCodec
from basil.utils import setup_logging

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
    parser.set_defaults(func=encode_command)
    # fmt: on


def encode_command(args):
    """Execute the encode command.

    Args:
        args: Parsed command-line arguments
    """
    # Determine output path
    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_encoded.npy")

    logger.info(f"Loading vectors from {args.input}")
    vectors = np.load(args.input)

    if vectors.ndim != 2:
        raise ValueError(f"Expected 2D array of vectors, got shape {vectors.shape}")

    logger.info(
        f"Initializing codec from {args.checkpoint} with {args.backend} backend"
    )
    codec = BasilCodec(args.checkpoint, backend=args.backend)

    logger.info(f"Encoding {len(vectors)} vectors...")
    semantic_ids = codec.batch_encode(vectors)

    logger.info(f"Saving semantic IDs to {args.output}")
    np.save(args.output, semantic_ids)

    logger.info(
        f"Encoded {len(vectors)} vectors to semantic IDs with shape {semantic_ids.shape}."
    )
