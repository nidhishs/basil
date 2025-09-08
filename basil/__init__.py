"""
BASIL - Balanced Assignment Semantic ID Library

A production-focused library for converting high-dimensional embeddings
to short semantic IDs and back.
"""

from .codec import BasilCodec
from .config import PreprocessConfig, TrainerConfig
from .preprocess import Preprocessor
from .trainer import BasilTrainer

__version__ = "1.0.0"
__all__ = [
    "BasilCodec",
    "BasilTrainer",
    "Preprocessor",
    "PreprocessConfig",
    "TrainerConfig",
]
