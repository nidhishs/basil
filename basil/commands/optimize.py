"""Hyperparameter optimization command using Ray Tune with Bayesian optimization."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from basil.config import BasilDataConfig, BasilModelConfig, BasilTrainConfig
from basil.training.trainer import BasilTrainer
from basil.utils import setup_logging

logger = setup_logging(__name__)

RAY_IMPORT_ERROR = (
    "Hyperparameter optimization requires Ray Tune and Optuna. "
    "Install with: pip install basil[tune]"
)

# Optional imports - set to None if not available
try:
    import ray
    from ray import tune
    from ray.tune.search.optuna import OptunaSearch
except ImportError:
    ray = None
    tune = None
    OptunaSearch = None


def add_parser(subparsers):
    """Add optimize command parser to subparsers."""
    parser = subparsers.add_parser(
        "optimize", help="Find optimal hyperparameters using Bayesian optimization"
    )
    parser.add_argument("config", type=str, help="Path to optimization config YAML")
    parser.set_defaults(func=optimize_command)


def _spec_to_tune(spec: dict):
    """Convert a single spec dict to a Ray Tune search space object."""
    t = spec["type"]
    if t == "uniform":
        return tune.uniform(spec["low"], spec["high"])
    elif t == "loguniform":
        return tune.loguniform(spec["low"], spec["high"])
    elif t == "randint":
        return tune.randint(spec["low"], spec["high"])
    elif t == "choice":
        return tune.choice(spec["values"])
    elif t == "quniform":
        return tune.quniform(spec["low"], spec["high"], spec["q"])
    else:
        raise ValueError(f"Unknown search space type: {t}")


def _build_tune_space(search_config: dict, base_config: dict) -> dict:
    """
    Build nested param_space for Ray Tune by merging base_config with search space.
    Returns a dict where search params are tune.* objects, others are fixed values.
    """
    param_space = copy.deepcopy(base_config)

    for section, params in search_config.items():
        if not isinstance(params, dict):
            continue
        if section not in param_space:
            raise ValueError(
                f"Search space references unknown section '{section}'. "
                f"Valid sections: {list(base_config.keys())}"
            )
        for param_name, spec in params.items():
            if not isinstance(spec, dict) or "type" not in spec:
                continue
            param_space[section][param_name] = _spec_to_tune(spec)

    return param_space


def _train_for_tune(config: dict):
    """Training function for Ray Tune - wraps BasilTrainer."""
    import tempfile

    from ray import tune

    model_cfg = BasilModelConfig.model_validate(config.get("model", {}))
    train_cfg = BasilTrainConfig.model_validate(config.get("train", {}))
    data_cfg = BasilDataConfig.model_validate(config["data"])

    trainer = BasilTrainer(
        model_cfg,
        train_cfg,
        data_cfg,
        output_dir=tempfile.mkdtemp(prefix="basil_tune_"),
        epoch_callback=lambda _, metrics: tune.report(metrics),
    )
    trainer.train()


def _build_tuner(param_space: dict, opt: dict, output_dir: str):
    """Build Ray Tune Tuner with search algorithm and scheduler."""
    num_samples = opt.get("num_samples", 20)
    metric = opt.get("metric", "val_cos_sim")
    mode = opt.get("mode", "max")
    epochs = param_space.get("train", {}).get("epochs", 20)

    search_alg = OptunaSearch(metric=metric, mode=mode)
    scheduler = (
        ray.tune.schedulers.ASHAScheduler(
            time_attr="epoch",
            metric=metric,
            mode=mode,
            max_t=epochs,
            grace_period=opt.get("grace_period", 3),
            reduction_factor=opt.get("reduction_factor", 3),
        )
        if opt.get("scheduler", "asha") == "asha"
        else None
    )

    return tune.Tuner(
        _train_for_tune,
        param_space=param_space,
        tune_config=tune.TuneConfig(
            num_samples=num_samples,
            search_alg=search_alg,
            scheduler=scheduler,
            max_concurrent_trials=opt.get("max_concurrent", 1),
        ),
        run_config=tune.RunConfig(
            name="basil_optimize", storage_path=Path(output_dir).resolve().as_posix()
        ),
    )


def _save_best_config(best_result, search_config: dict, output_dir: str):
    """Log best result and save config to YAML."""
    metrics = best_result.metrics
    config = best_result.config

    logger.info(f"\nBest val_cos_sim: {metrics.get('val_cos_sim')}")
    for section, params in search_config.items():
        if not isinstance(params, dict):
            continue
        for param_name in params:
            value = config.get(section, {}).get(param_name)
            if value is not None:
                logger.info(f"  {section}.{param_name}: {value}")

    best_path = Path(output_dir) / "best_config.yaml"
    best_cfg = copy.deepcopy(config)
    best_cfg["output_dir"] = str(Path(output_dir) / "best_model")

    with open(best_path, "w") as f:
        yaml.dump(best_cfg, f, default_flow_style=False)
    logger.info(f"Best config saved to: {best_path}")


def optimize_command(args):
    """Execute the hyperparameter optimization command."""
    if ray is None:
        raise ImportError(RAY_IMPORT_ERROR)

    with open(args.config, "r") as f:
        raw = yaml.safe_load(f)

    base_config = raw["base_config"]
    search_config = raw["search_space"]
    opt = raw.get("optimization", {})
    output_dir = raw.get("output_dir", "./output/tune_results")

    # Convert data path to absolute - Ray workers run in different directories
    if "data" in base_config and "path" in base_config["data"]:
        data_path = Path(base_config["data"]["path"])
        if not data_path.is_absolute():
            base_config["data"]["path"] = str(data_path.resolve())

    # Build nested param_space (base_config with tune.* objects for search params)
    param_space = _build_tune_space(search_config, base_config)

    logger.info(f"Starting optimization: {opt.get('num_samples', 20)} trials")

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    tuner = _build_tuner(param_space, opt, output_dir)
    results = tuner.fit()
    best = results.get_best_result(
        metric=opt.get("metric", "val_cos_sim"),
        mode=opt.get("mode", "max"),
    )
    _save_best_config(best, search_config, output_dir)
