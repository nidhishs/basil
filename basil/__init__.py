"""
BASIL - Balanced Assignment Semantic ID Library

A production-focused library for converting high-dimensional embeddings
to short semantic IDs and back.
"""

from basil.codec import BasilCodec
from basil.config import PreprocessConfig, TrainerConfig
from basil.preprocess import Preprocessor
from basil.trainer import BasilTrainer

__version__ = "1.0.0"
__all__ = [
    "BasilCodec",
    "BasilTrainer",
    "Preprocessor",
    "PreprocessConfig",
    "TrainerConfig",
]
