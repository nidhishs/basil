"""Decode command implementation."""

from pathlib import Path

import numpy as np

from basil.codec import BasilCodec
from basil.utils import setup_logging

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
    parser.set_defaults(func=decode_command)
    # fmt: on


def decode_command(args):
    """Execute the decode command.

    Args:
        args: Parsed command-line arguments
    """
    # Determine output path
    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_decoded.npy")

    logger.info(f"Loading semantic IDs from {args.input}")
    semantic_ids = np.load(args.input)

    if semantic_ids.ndim != 2:
        raise ValueError(
            f"Expected 2D array of semantic IDs, got shape {semantic_ids.shape}"
        )

    logger.info(
        f"Initializing codec from {args.checkpoint} with {args.backend} backend"
    )
    codec = BasilCodec(args.checkpoint, backend=args.backend)

    logger.info(f"Decoding {len(semantic_ids)} semantic IDs...")
    vectors = codec.batch_decode(semantic_ids)

    logger.info(f"Saving vectors to {args.output}")
    np.save(args.output, vectors)

    logger.info(
        f"Decoded {len(semantic_ids)} semantic IDs to vectors with shape {vectors.shape}."
    )
