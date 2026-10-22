"""Unified training stack for the Agron zoo.

Public API::

    from agron_nets.training import (
        AgronTrainer, TrainerConfig, train_architecture,
        RECIPES, get_recipe, recipe_names,
        add_training_args, trainer_from_args,
    )
"""

from __future__ import annotations

from .cli import add_training_args, arch_overrides, build_model_from_args, trainer_from_args
from .engine import AgronTrainer, TrainerConfig, train_architecture
from .recipes import RECIPES, get_recipe, recipe_names, recipe_report

__all__ = [
    "AgronTrainer",
    "TrainerConfig",
    "train_architecture",
    "RECIPES",
    "get_recipe",
    "recipe_names",
    "recipe_report",
    "add_training_args",
    "arch_overrides",
    "build_model_from_args",
    "trainer_from_args",
]
