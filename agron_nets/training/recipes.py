"""Named training recipes: ready-made optimiser + algorithm + schedule bundles.

A recipe is a plain dict so it can be passed straight to
:class:`agron_nets.training.AgronTrainer` or overridden field by field::

    >>> from agron_nets.training import RECIPES, AgronTrainer
    >>> trainer = AgronTrainer(model, **RECIPES["stable"])
"""

from __future__ import annotations

from typing import Any, Dict, List

RECIPES: Dict[str, Dict[str, Any]] = {
    "vanilla": {
        "optimizer": "adamw",
        "scheduler": "cosine",
        "algorithms": [],
        "description": "Plain AdamW + cosine decay. The reference baseline.",
    },
    "stable": {
        "optimizer": "adamw",
        "scheduler": "cosine",
        "algorithms": ["gradient_clip", "z_loss", "label_smoothing"],
        "description": "Clipping + z-loss + label smoothing for reliable convergence.",
    },
    "fast": {
        "optimizer": "lion",
        "scheduler": "linear",
        "algorithms": ["gradient_clip", "ema"],
        "description": "Lion sign-momentum with EMA weights: cheap and fast.",
    },
    "memory_efficient": {
        "optimizer": "adafactor",
        "scheduler": "cosine",
        "algorithms": ["gradient_accumulation", "activation_checkpointing"],
        "description": "Factored moments + accumulation + recomputation for small GPUs.",
    },
    "flat_minima": {
        "optimizer": "adamw",
        "scheduler": "wsd",
        "algorithms": ["sam", "gradient_clip"],
        "description": "Sharpness-Aware Minimisation for flatter, better-generalising minima.",
    },
    "long_horizon": {
        "optimizer": "kahan",
        "scheduler": "wsd",
        "algorithms": ["gradient_clip", "ema", "weight_decay_schedule"],
        "description": "Kahan-compensated AdamW + WSD + EMA for very long runs.",
    },
    "second_order": {
        "optimizer": "sophia",
        "scheduler": "cosine",
        "algorithms": ["gradient_clip"],
        "description": "Sophia with clipped second-order updates.",
    },
    "orthogonal": {
        "optimizer": "muon",
        "scheduler": "cosine",
        "algorithms": ["gradient_clip", "ema"],
        "description": "Muon's orthogonalised momentum updates.",
    },
    "curriculum": {
        "optimizer": "adamw",
        "scheduler": "cosine",
        "algorithms": ["curriculum", "gradient_clip"],
        "description": "Easy-to-hard sequence curriculum with AdamW.",
    },
    "mixup": {
        "optimizer": "adamw",
        "scheduler": "cosine",
        "algorithms": ["mixup", "gradient_clip"],
        "description": "Embedding mixup as a data-space regulariser.",
    },
}


def recipe_names() -> List[str]:
    return sorted(RECIPES)


def get_recipe(name: str) -> Dict[str, Any]:
    key = (name or "vanilla").strip().lower()
    if key not in RECIPES:
        raise KeyError(f"unknown recipe '{name}'. available: {', '.join(recipe_names())}")
    return dict(RECIPES[key])


def recipe_report() -> List[Dict[str, Any]]:
    return [
        {"name": name, **{k: v for k, v in recipe.items()}}
        for name, recipe in sorted(RECIPES.items())
    ]


__all__ = ["RECIPES", "get_recipe", "recipe_names", "recipe_report"]
