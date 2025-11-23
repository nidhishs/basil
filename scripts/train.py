#!/usr/bin/env python3
import argparse

import yaml

from basil.config import BasilDataConfig, BasilModelConfig, BasilTrainConfig
from basil.training.trainer import BasilTrainer
from basil.utils import setup_logging

logger = setup_logging(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config.yaml")
    args = parser.parse_args()

    logger.info(f"Loading config from {args.config}")
    with open(args.config, "r") as f:
        raw = yaml.safe_load(f)

    # Ensure the config file structure matches our expected schema
    if "data" not in raw:
        raise ValueError("Config must contain 'data' section")
    if "output_dir" not in raw:
        raise ValueError("Config must contain 'output_dir'")

    # Introspect dataclasses to filter YAML keys, preventing crashes on
    # unexpected extra keys in the YAML file.
    model_keys = BasilModelConfig.__annotations__.keys()
    train_keys = BasilTrainConfig.__annotations__.keys()
    data_keys = BasilDataConfig.__annotations__.keys()

    model_args = {k: v for k, v in raw["model"].items() if k in model_keys}
    train_args = {k: v for k, v in raw["train"].items() if k in train_keys}
    data_args = {k: v for k, v in raw["data"].items() if k in data_keys}

    model_cfg = BasilModelConfig(**model_args)
    train_cfg = BasilTrainConfig(**train_args)
    data_cfg = BasilDataConfig(**data_args)

    trainer = BasilTrainer(model_cfg, train_cfg, data_cfg, raw["output_dir"])
    trainer.train()


if __name__ == "__main__":
    main()
