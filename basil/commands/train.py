"""Train command implementation."""

import yaml

from basil.config import BasilDataConfig, BasilModelConfig, BasilTrainConfig
from basil.training.trainer import BasilTrainer
from basil.utils import setup_logging

logger = setup_logging(__name__)


def add_parser(subparsers):
    """Add train command parser to subparsers.

    Args:
        subparsers: The subparsers object from argparse
    """
    # fmt: off
    parser = subparsers.add_parser("train", help="Train a Basil model")
    parser.add_argument("config", type=str, help="Path to training config YAML file")
    parser.set_defaults(func=train_command)
    # fmt: on


def train_command(args):
    """Execute the training command.

    Args:
        args: Parsed command-line arguments with 'config' attribute
    """
    logger.info(f"Loading config from {args.config}")
    with open(args.config, "r") as f:
        raw = yaml.safe_load(f)

    # Ensure the config file structure matches our expected schema
    if "data" not in raw:
        raise ValueError("Config must contain 'data' section")
    if "output_dir" not in raw:
        raise ValueError("Config must contain 'output_dir'")

    # Pydantic handles validation, type coercion, and extra field handling automatically
    model_cfg = BasilModelConfig.model_validate(raw.get("model", {}))
    train_cfg = BasilTrainConfig.model_validate(raw.get("train", {}))
    data_cfg = BasilDataConfig.model_validate(raw["data"])

    trainer = BasilTrainer(model_cfg, train_cfg, data_cfg, raw["output_dir"])
    trainer.train()
