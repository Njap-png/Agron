"""Weight-averaging algorithms: EMA and Stochastic Weight Averaging."""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch

from .base import AlgorithmContext, BaseAlgorithm


class EMA(BaseAlgorithm):
    """Exponential moving average of model weights."""

    name = "ema"
    category = "averaging"
    description = "Tracks an exponential moving average of the weights."

    def __init__(self, decay: float = 0.999, warmup_steps: int = 0, use_buffers: bool = False):
        super().__init__(decay=decay, warmup_steps=warmup_steps, use_buffers=use_buffers)
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.use_buffers = use_buffers
        self.shadow: Dict[str, torch.Tensor] = {}
        self._backup: Dict[str, torch.Tensor] = {}

    def on_train_begin(self, ctx):
        self.shadow = {name: p.detach().clone() for name, p in ctx.model.named_parameters()}
        if self.use_buffers:
            for name, buf in ctx.model.named_buffers():
                self.shadow[name] = buf.detach().clone()

    @torch.no_grad()
    def on_step_end(self, ctx, loss):
        step = ctx.step + 1
        decay = self.decay
        if self.warmup_steps > 0:
            decay = min(decay, (1 + step) / (10 + step))
        for name, p in ctx.model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(decay).add_(p.detach(), alpha=1 - decay)
        if self.use_buffers:
            for name, buf in ctx.model.named_buffers():
                if name in self.shadow:
                    self.shadow[name].copy_(buf.detach())

    @torch.no_grad()
    def apply_shadow(self, model) -> None:
        """Swap in the averaged weights (backing up the live weights)."""
        self._backup = {}
        for name, p in model.named_parameters():
            if name in self.shadow:
                self._backup[name] = p.detach().clone()
                p.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model) -> None:
        """Restore the live weights saved by :meth:`apply_shadow`."""
        for name, p in model.named_parameters():
            if name in self._backup:
                p.copy_(self._backup[name])
        self._backup = {}

    def state_dict(self):
        return {"shadow": self.shadow}

    def load_state_dict(self, state):
        self.shadow = state.get("shadow", {})


class SWA(BaseAlgorithm):
    """Stochastic Weight Averaging.

    Averages checkpoints collected after ``start_step``.
    """

    name = "swa"
    category = "averaging"
    description = "Averages late-training checkpoints for better generalisation."

    def __init__(self, start_step: int = 100, interval: int = 10):
        super().__init__(start_step=start_step, interval=interval)
        self.start_step = start_step
        self.interval = max(1, interval)
        self._sum: Optional[Dict[str, torch.Tensor]] = None
        self._count = 0

    def on_train_begin(self, ctx):
        self._sum = None
        self._count = 0

    @torch.no_grad()
    def on_step_end(self, ctx, loss):
        if ctx.step < self.start_step or ctx.step % self.interval != 0:
            return
        if self._sum is None:
            self._sum = {name: p.detach().clone() for name, p in ctx.model.named_parameters()}
            self._count = 1
            return
        for name, p in ctx.model.named_parameters():
            if name in self._sum:
                self._sum[name].add_(p.detach())
        self._count += 1

    @torch.no_grad()
    def apply_average(self, model) -> bool:
        """Write the averaged weights into ``model``; returns success."""
        if self._sum is None or self._count == 0:
            return False
        for name, p in model.named_parameters():
            if name in self._sum:
                p.copy_(self._sum[name] / self._count)
        return True
