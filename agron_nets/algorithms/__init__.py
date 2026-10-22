"""Registry of composable training algorithms.

Algorithms are grouped by category and can be combined freely::

    >>> from agron_nets import build_algorithm
    >>> algos = [build_algorithm("gradient_clip"), build_algorithm("ema")]
"""

from __future__ import annotations

from typing import Any, Dict, List

from .averaging import EMA, SWA
from .base import AlgorithmContext, BaseAlgorithm
from .data import Curriculum, DifficultyCurriculum
from .objectives import DPO, Distillation, distillation_loss, dpo_loss, sequence_log_probs
from .regularization import LabelSmoothing, Mixup, WeightDecaySchedule
from .stability import (
    ActivationCheckpointing,
    GradientAccumulation,
    GradientClip,
    GradientNoise,
    ZLoss,
)
from .wrappers import SAM, Lookahead


ALGORITHMS: Dict[str, Dict[str, Any]] = {}


def _register(cls) -> None:
    ALGORITHMS[cls.name] = {
        "name": cls.name,
        "class": cls,
        "category": getattr(cls, "category", "misc"),
        "description": getattr(cls, "description", ""),
    }


for _cls in (
    LabelSmoothing,
    Mixup,
    WeightDecaySchedule,
    GradientClip,
    GradientAccumulation,
    ZLoss,
    GradientNoise,
    ActivationCheckpointing,
    SAM,
    Lookahead,
    EMA,
    SWA,
    Distillation,
    DPO,
    Curriculum,
    DifficultyCurriculum,
):
    _register(_cls)


def algorithm_names(category: str | None = None) -> List[str]:
    if category:
        return sorted(k for k, v in ALGORITHMS.items() if v["category"] == category)
    return sorted(ALGORITHMS.keys())


def get_algorithm(name: str) -> Dict[str, Any]:
    key = (name or "").strip().lower()
    if key not in ALGORITHMS:
        raise KeyError(f"unknown algorithm '{name}'. available: {', '.join(algorithm_names())}")
    return ALGORITHMS[key]


def build_algorithm(name: str, **kwargs) -> BaseAlgorithm:
    """Instantiate one algorithm by registry name."""
    return get_algorithm(name)["class"](**kwargs)


def build_algorithms(names, **kwargs) -> List[BaseAlgorithm]:
    """Instantiate several algorithms (composable)."""
    if isinstance(names, str):
        names = [n.strip() for n in names.split(",") if n.strip()]
    return [build_algorithm(n, **kwargs) for n in names]


def algorithm_report() -> List[Dict[str, Any]]:
    return [
        {"name": k, "category": v["category"], "description": v["description"]}
        for k, v in sorted(ALGORITHMS.items(), key=lambda kv: (kv[1]["category"], kv[0]))
    ]


def categories() -> List[str]:
    return sorted({v["category"] for v in ALGORITHMS.values()})


__all__ = [
    "ALGORITHMS",
    "AlgorithmContext",
    "BaseAlgorithm",
    "EMA",
    "SWA",
    "Curriculum",
    "DifficultyCurriculum",
    "DPO",
    "Distillation",
    "LabelSmoothing",
    "Mixup",
    "WeightDecaySchedule",
    "GradientAccumulation",
    "GradientClip",
    "GradientNoise",
    "ZLoss",
    "ActivationCheckpointing",
    "SAM",
    "Lookahead",
    "algorithm_names",
    "algorithm_report",
    "build_algorithm",
    "build_algorithms",
    "categories",
    "distillation_loss",
    "dpo_loss",
    "get_algorithm",
    "sequence_log_probs",
]
