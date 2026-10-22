"""Shared command-line plumbing for the Agron training scripts.

Both ``train_from_scratch.py`` and ``train_agron.py`` expose the same
architecture / optimiser / algorithm / scheduler surface through
:func:`add_training_args` and :func:`trainer_from_args`.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..algorithms import algorithm_names
from ..architectures import ARCHITECTURES, build_architecture
from ..optimizers import optimizer_names
from ..schedulers import scheduler_names
from ..utils import count_parameters, format_parameters, read_json
from .engine import AgronTrainer, TrainerConfig
from .recipes import get_recipe, recipe_names


def add_training_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the shared model/training flags to ``parser``."""
    g = parser.add_argument_group("agron_nets")
    g.add_argument("--arch", choices=sorted(ARCHITECTURES), default=None,
                   help="architecture from the Agron zoo (default: llama)")
    g.add_argument("--optimizer", choices=optimizer_names(), default=None,
                   help="optimiser (default: from --recipe, else adamw)")
    g.add_argument("--scheduler", choices=scheduler_names() + ["none"], default=None,
                   help="learning-rate schedule (default: from --recipe, else cosine)")
    g.add_argument("--algorithm", action="append", default=None, metavar="NAME",
                   help=f"composable algorithm, repeatable (choices: {', '.join(algorithm_names())})")
    g.add_argument("--recipe", choices=recipe_names(), default=None,
                   help="named optimiser+algorithm bundle (see agron_nets.training.RECIPES)")
    g.add_argument("--lr", type=float, default=None, help="peak learning rate")
    g.add_argument("--warmup", type=int, default=0, help="warmup steps")
    g.add_argument("--grad-accum", type=int, default=1, help="gradient accumulation micro-steps")
    g.add_argument("--grad-clip", type=float, default=1.0, help="global grad-norm clip (0 disables)")
    g.add_argument("--weight-decay", type=float, default=None)
    g.add_argument("--batch-size", type=int, default=8)
    g.add_argument("--log-every", type=int, default=25)
    g.add_argument("--save-every", type=int, default=0)
    g.add_argument("--device", default=None, help="cpu / cuda / cuda:1 / mps")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--config-json", default=None,
                   help="JSON string or path with architecture overrides")
    g.add_argument("--sophia-hessian-every", type=int, default=0,
                   help="refresh Sophia's diagonal Hessian every N steps")
    return parser


def arch_overrides(config_json: Optional[str]) -> Dict[str, Any]:
    if not config_json:
        return {}
    if os.path.exists(config_json):
        data = read_json(config_json, default={}) or {}
    else:
        data = json.loads(config_json)
    return {k: v for k, v in data.items() if k not in ("model_type", "architectures", "transformers_version")}


def build_model_from_args(
    args: argparse.Namespace,
    preset: Optional[Dict[str, Any]] = None,
    model: Optional[torch.nn.Module] = None,
) -> Tuple[torch.nn.Module, Any]:
    """Build (or accept) the model described by ``args``."""
    if model is not None:
        return model, model.config
    cfg = dict(preset or {})
    cfg.update(arch_overrides(getattr(args, "config_json", None)))
    model, config = build_architecture(getattr(args, "arch", None) or "llama", **cfg)
    print(f"Architecture: {config.model_type} | {format_parameters(count_parameters(model))} params")
    return model, config


def trainer_from_args(
    args: argparse.Namespace,
    model: torch.nn.Module,
    output_dir: Optional[str] = None,
) -> AgronTrainer:
    """Resolve recipe + explicit flags into a configured :class:`AgronTrainer`."""
    recipe = get_recipe(args.recipe) if getattr(args, "recipe", None) else {}

    optimizer = args.optimizer or recipe.get("optimizer", "adamw")
    scheduler = args.scheduler or recipe.get("scheduler", "cosine")
    algorithms: List[Any] = list(recipe.get("algorithms", []))

    extra = []
    for item in getattr(args, "algorithm", None) or []:
        extra.extend(part.strip() for part in item.split(",") if part.strip())
    for name in extra:
        if name not in algorithms:
            algorithms.append(name)

    if getattr(args, "grad_accum", 1) > 1 and "gradient_accumulation" not in algorithms:
        algorithms.append("gradient_accumulation")

    if getattr(args, "sophia_hessian_every", 0) > 0 and "sophia" != optimizer:
        print("warning: --sophia-hessian-every only applies to --optimizer sophia")

    config = TrainerConfig(
        max_steps=args.steps,
        epochs=getattr(args, "epochs", 1),
        warmup_steps=args.warmup,
        log_every=args.log_every,
        save_every=args.save_every,
        output_dir=output_dir or args.output,
        grad_clip=(args.grad_clip or None),
        seed=args.seed,
        weight_decay=args.weight_decay,
        sophia_hessian_every=args.sophia_hessian_every,
    )

    optimizer_kwargs: Dict[str, Any] = {}
    if args.lr is not None:
        optimizer_kwargs["lr"] = args.lr

    trainer = AgronTrainer(
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        algorithms=algorithms,
        config=config,
        device=args.device,
        optimizer_kwargs=optimizer_kwargs,
    )

    # ``gradient_accumulation`` must know how many micro-steps to accumulate.
    if getattr(args, "grad_accum", 1) > 1:
        from ..algorithms.stability import GradientAccumulation

        for i, algo in enumerate(trainer.algorithms):
            if algo.name == "gradient_accumulation":
                trainer.algorithms[i] = GradientAccumulation(steps=args.grad_accum)

    print(
        f"Optimiser: {type(trainer.optimizer).__name__} | Scheduler: {scheduler} | "
        f"Algorithms: {', '.join(a.name for a in trainer.algorithms) or 'none'}"
    )
    return trainer


__all__ = [
    "add_training_args",
    "arch_overrides",
    "build_model_from_args",
    "trainer_from_args",
]
