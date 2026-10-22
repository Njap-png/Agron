"""Composable training algorithms.

Each algorithm implements a small set of hooks that the Agron training engine
calls in order::

    algo.on_train_begin(ctx)
    for step, batch in loader:
        algo.on_step_begin(ctx, batch)
        loss = algo.train_step(ctx, batch)
        algo.on_step_end(ctx, loss)

Algorithms are deliberately composable: several can be active at once and they
will all see the same context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


@dataclass
class AlgorithmContext:
    """Mutable state shared between the engine and the active algorithms."""

    model: Any = None
    optimizer: Any = None
    scheduler: Any = None
    device: torch.device = torch.device("cpu")
    step: int = 0
    epoch: int = 0
    max_steps: int = 1000
    world_size: int = 1
    extra: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.extra.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.extra[key] = value


class BaseAlgorithm:
    """No-op baseline every algorithm inherits from."""

    name: str = "base"
    category: str = "misc"
    description: str = ""

    def __init__(self, **kwargs):
        self.config = kwargs

    # -- lifecycle -------------------------------------------------------- #
    def on_train_begin(self, ctx: AlgorithmContext) -> None:
        pass

    def on_train_end(self, ctx: AlgorithmContext) -> None:
        pass

    def on_epoch_begin(self, ctx: AlgorithmContext) -> None:
        pass

    def on_step_begin(self, ctx: AlgorithmContext, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return batch

    def compute_loss(self, ctx: AlgorithmContext, batch: Dict[str, torch.Tensor], outputs: Any, loss: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        return loss

    def train_step(self, ctx: AlgorithmContext, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Default single forward/backward pass."""
        batch = self.on_step_begin(ctx, batch)
        outputs = ctx.model(**batch)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        loss = self.compute_loss(ctx, batch, outputs, loss)
        loss = compose_loss(ctx, batch, outputs, loss, skip=self)
        if loss is not None:
            loss.backward()
        return loss

    def on_before_optimizer_step(self, ctx: AlgorithmContext) -> None:
        pass

    def should_step(self, ctx: AlgorithmContext) -> bool:
        """Whether the optimiser should run after this micro-step."""
        return True

    def on_step_end(self, ctx: AlgorithmContext, loss: Optional[torch.Tensor]) -> None:
        pass

    # -- introspection ---------------------------------------------------- #
    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        pass


def compose_loss(
    ctx: AlgorithmContext,
    batch: Dict[str, torch.Tensor],
    outputs: Any,
    loss: Optional[torch.Tensor],
    skip: Optional[BaseAlgorithm] = None,
) -> Optional[torch.Tensor]:
    """Apply every registered ``compute_loss`` hook in order.

    ``skip`` prevents the algorithm that already computed the loss (the step
    driver, e.g. SAM or Mixup) from being applied twice.
    """
    for algo in ctx.get("loss_hooks") or []:
        if algo is skip:
            continue
        loss = algo.compute_loss(ctx, batch, outputs, loss)
    return loss
