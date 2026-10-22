"""Learning-rate schedules and warmups.

Every function takes ``(step, total_steps, **kwargs)`` and returns a scalar
multiplier in ``(0, 1]``.  The factory :func:`build_scheduler` wires a
standard PyTorch scheduler when possible.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional

import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ExponentialLR,
    LambdaLR,
    LinearLR,
    OneCycleLR,
    PolynomialLR,
    ReduceLROnPlateau,
    SequentialLR,
)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
SCHEDULERS: Dict[str, Dict[str, Any]] = {}


def _register(name: str, fn_or_cls, description: str, family: str) -> None:
    SCHEDULERS[name] = {
        "name": name,
        "fn_or_cls": fn_or_cls,
        "description": description,
        "family": family,
    }


def scheduler_names() -> List[str]:
    return sorted(SCHEDULERS.keys())


def get_scheduler(name: str) -> Dict[str, Any]:
    key = (name or "cosine").strip().lower()
    if key not in SCHEDULERS:
        raise KeyError(f"unknown scheduler '{name}'. available: {', '.join(scheduler_names())}")
    return SCHEDULERS[key]


# --------------------------------------------------------------------------- #
# Pure Python schedule functions (for custom loops)
# --------------------------------------------------------------------------- #
def constant_lr(step: int, total: int = 0, warmup: int = 0, **kw) -> float:
    if warmup and step < warmup:
        return (step + 1) / max(1, warmup)
    return 1.0


def linear_warmup_linear_decay(step: int, total: int = 1, warmup: int = 0, min_ratio: float = 0.0, **kw) -> float:
    if warmup and step < warmup:
        return (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return max(min_ratio, 1.0 - t * (1.0 - min_ratio))


def linear_warmup_cosine_decay(step: int, total: int = 1, warmup: int = 0, min_ratio: float = 0.1, **kw) -> float:
    if warmup and step < warmup:
        return (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    t = min(max(t, 0.0), 1.0)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * t))


def cosine_restarts(step: int, total: int = 1, warmup: int = 0, cycles: int = 4, min_ratio: float = 0.0, **kw) -> float:
    if warmup and step < warmup:
        return (step + 1) / max(1, warmup)
    cycles = max(1, cycles)
    cycle_len = max(1, (total - warmup) // cycles)
    pos = (step - warmup) % cycle_len
    t = pos / cycle_len
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * t))


def inverse_sqrt(step: int, total: int = 0, warmup: int = 4000, **kw) -> float:
    step = max(step + 1, 1)
    if warmup and step < warmup:
        return step / max(1, warmup)
    return math.sqrt(warmup / step)


def polynomial(step: int, total: int = 1, warmup: int = 0, power: float = 0.5, min_ratio: float = 0.0, **kw) -> float:
    if warmup and step < warmup:
        return (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    t = min(max(t, 0.0), 1.0)
    return max(min_ratio, (1.0 - t) ** power)


def warmup_stable_decay(step: int, total: int = 1, warmup: int = 0, decay_start: float = 0.8, min_ratio: float = 0.0, **kw) -> float:
    if warmup and step < warmup:
        return (step + 1) / max(1, warmup)
    if total <= 0:
        return 1.0
    if step / total <= decay_start:
        return 1.0
    t = (step / total - decay_start) / max(1e-12, 1 - decay_start)
    t = min(max(t, 0.0), 1.0)
    return min_ratio + (1.0 - min_ratio) * (1.0 - t)


def one_cycle(step: int, total: int = 1, warmup_ratio: float = 0.3, max_ratio: float = 1.0, min_ratio: float = 0.01, **kw) -> float:
    if total <= 0:
        return max_ratio
    t = (step + 1) / total
    t = min(max(t, 0.0), 1.0)
    up = warmup_ratio
    if t <= up:
        pct = t / up
        return min_ratio + (max_ratio - min_ratio) * pct
    pct = (t - up) / max(1e-12, 1 - up)
    return max_ratio - (max_ratio - min_ratio) * pct


# --------------------------------------------------------------------------- #
# PyTorch scheduler factory
# --------------------------------------------------------------------------- #
def build_scheduler(optimizer, name: str = "cosine", total_steps: int = 1000,
                    warmup_steps: int = 0, **kwargs):
    """Return a ``LambdaLR`` driven by one of the pure functions, or a native scheduler."""
    name = (name or "cosine").strip().lower()

    if name in ("linear", "linear_decay", "linear_warmup_linear_decay"):
        def lr_lambda(step):
            return linear_warmup_linear_decay(step, total_steps, warmup_steps,
                                              kwargs.get("min_ratio", 0.0))
        return LambdaLR(optimizer, lr_lambda)

    if name in ("cosine", "cosine_warmup", "linear_warmup_cosine_decay"):
        def lr_lambda(step):
            return linear_warmup_cosine_decay(step, total_steps, warmup_steps,
                                               kwargs.get("min_ratio", 0.1))
        return LambdaLR(optimizer, lr_lambda)

    if name in ("cosine_restart", "cosine_restarts", "sgdr"):
        def lr_lambda(step):
            return cosine_restarts(step, total_steps, warmup_steps,
                                   kwargs.get("cycles", 4), kwargs.get("min_ratio", 0.0))
        return LambdaLR(optimizer, lr_lambda)

    if name == "inverse_sqrt":
        def lr_lambda(step):
            return inverse_sqrt(step, total_steps, kwargs.get("warmup", warmup_steps))
        return LambdaLR(optimizer, lr_lambda)

    if name == "polynomial":
        def lr_lambda(step):
            return polynomial(step, total_steps, warmup_steps,
                              kwargs.get("power", 0.5), kwargs.get("min_ratio", 0.0))
        return LambdaLR(optimizer, lr_lambda)

    if name in ("wsd", "warmup_stable_decay"):
        def lr_lambda(step):
            return warmup_stable_decay(step, total_steps, warmup_steps,
                                       kwargs.get("decay_start", 0.8), kwargs.get("min_ratio", 0.0))
        return LambdaLR(optimizer, lr_lambda)

    if name == "onecycle" or name == "one_cycle":
        return OneCycleLR(
            optimizer,
            max_lr=kwargs.get("max_lr", [g["lr"] for g in optimizer.param_groups]),
            total_steps=max(1, total_steps),
            pct_start=kwargs.get("warmup_ratio", 0.3),
            anneal_strategy="cos",
            div_factor=kwargs.get("div_factor", 25.0),
            final_div_factor=kwargs.get("final_div_factor", 10000.0),
        )

    if name == "cosine_anneal":
        return CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps),
                                 eta_min=kwargs.get("min_ratio", 0.1) * (optimizer.param_groups[0]["lr"]))

    if name == "exponential":
        return ExponentialLR(optimizer, gamma=kwargs.get("gamma", 0.995))

    if name == "polynomial_lr":
        return PolynomialLR(optimizer, total_iters=max(1, total_steps), power=kwargs.get("power", 1.0))

    # default: cosine warmup
    def lr_lambda(step):
        return linear_warmup_cosine_decay(step, total_steps, warmup_steps,
                                           kwargs.get("min_ratio", 0.1))
    return LambdaLR(optimizer, lr_lambda)


# --------------------------------------------------------------------------- #
# Register
# --------------------------------------------------------------------------- #
_register("constant", constant_lr, "Constant learning rate with optional warmup", "constant")
_register("linear", linear_warmup_linear_decay, "Linear warmup then linear decay", "linear")
_register("cosine", linear_warmup_cosine_decay, "Linear warmup then cosine decay", "cosine")
_register("cosine_restarts", cosine_restarts, "Cosine annealing with warm restarts (SGDR)", "cosine")
_register("inverse_sqrt", inverse_sqrt, "Inverse square root (Transformer/Noam style)", "exponential")
_register("polynomial", polynomial, "Polynomial decay", "polynomial")
_register("wsd", warmup_stable_decay, "Warmup-Stable-Decay", "modern")
_register("onecycle", one_cycle, "1cycle policy", "cyclic")


__all__ = [
    "SCHEDULERS",
    "constant_lr",
    "linear_warmup_linear_decay",
    "linear_warmup_cosine_decay",
    "cosine_restarts",
    "inverse_sqrt",
    "polynomial",
    "warmup_stable_decay",
    "one_cycle",
    "build_scheduler",
    "get_scheduler",
    "scheduler_names",
]
