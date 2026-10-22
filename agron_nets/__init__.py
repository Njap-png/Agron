"""Agron neural-network zoo.

This package bundles a growing family of neural network architectures,
optimisers and training algorithms behind a single, stable registry so the
Agron runtime can swap its brain at runtime.

Quick start
-----------
>>> from agron_nets import ARCHITECTURES, build_architecture, OPTIMIZERS, ALGORITHMS
>>> model, config = build_architecture("moe", num_hidden_layers=2, hidden_size=64)
>>> sorted(ARCHITECTURES)  # doctest: +SKIP
['conv', 'gru', 'hybrid', 'hyper', 'llama', 'lstm', 'mamba', 'mixer', 'moe', 'transformer']
"""

from __future__ import annotations

from .architectures import (
    ARCHITECTURES,
    DEFAULT_ARCHITECTURE,
    AgronConfig,
    AgronPreTrainedModel,
    architecture_description,
    architecture_label,
    architecture_names,
    architecture_report,
    build_architecture,
    build_config,
    get_architecture,
    register_auto_classes,
)
from .algorithms import ALGORITHMS, build_algorithm, algorithm_names, algorithm_report
from .optimizers import OPTIMIZERS, build_optimizer, optimizer_names, optimizer_report
from .schedulers import SCHEDULERS, build_scheduler, scheduler_names
from .training import (
    RECIPES,
    AgronTrainer,
    TrainerConfig,
    add_training_args,
    get_recipe,
    recipe_names,
    train_architecture,
    trainer_from_args,
)
from .utils import count_parameters, format_parameters, set_seed

__version__ = "1.2.0"

__all__ = [
    "ARCHITECTURES",
    "DEFAULT_ARCHITECTURE",
    "ALGORITHMS",
    "OPTIMIZERS",
    "RECIPES",
    "SCHEDULERS",
    "AgronConfig",
    "AgronPreTrainedModel",
    "AgronTrainer",
    "TrainerConfig",
    "add_training_args",
    "algorithm_names",
    "algorithm_report",
    "architecture_description",
    "architecture_label",
    "architecture_names",
    "architecture_report",
    "build_algorithm",
    "build_architecture",
    "build_config",
    "build_optimizer",
    "build_scheduler",
    "count_parameters",
    "format_parameters",
    "get_architecture",
    "get_recipe",
    "optimizer_names",
    "optimizer_report",
    "recipe_names",
    "register_auto_classes",
    "scheduler_names",
    "set_seed",
    "train_architecture",
    "trainer_from_args",
    "__version__",
]
