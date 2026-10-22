"""Stability algorithms: clipping, accumulation, z-loss, noise, checkpointing."""

from __future__ import annotations

import math
from typing import Optional

import torch

from .base import AlgorithmContext, BaseAlgorithm, compose_loss


class GradientClip(BaseAlgorithm):
    """Clip gradients by global norm (and optionally value)."""

    name = "gradient_clip"
    category = "stability"
    description = "Clips the global gradient norm before the optimiser step."

    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0, clip_values: float = 0.0):
        super().__init__(max_norm=max_norm, norm_type=norm_type, clip_values=clip_values)
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.clip_values = clip_values

    def on_before_optimizer_step(self, ctx):
        params = [p for p in ctx.model.parameters() if p.grad is not None]
        if not params:
            return
        torch.nn.utils.clip_grad_norm_(params, self.max_norm, norm_type=self.norm_type)
        if self.clip_values > 0:
            torch.nn.utils.clip_grad_value_(params, self.clip_values)


class GradientAccumulation(BaseAlgorithm):
    """Accumulate gradients over ``steps`` micro-batches before stepping.

    ``train_step`` performs the scaled backward pass; the engine consults
    :meth:`should_step` to decide whether to run ``optimizer.step()``.
    """

    name = "gradient_accumulation"
    category = "stability"
    description = "Simulate larger batch sizes via gradient accumulation."

    def __init__(self, steps: int = 4):
        super().__init__(steps=steps)
        self.steps = max(1, steps)

    def train_step(self, ctx, batch):
        batch = self.on_step_begin(ctx, batch)
        outputs = ctx.model(**batch)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        loss = self.compute_loss(ctx, batch, outputs, loss)
        loss = compose_loss(ctx, batch, outputs, loss, skip=self)
        if loss is not None:
            (loss / self.steps).backward()
        ctx.set("accum_step", (ctx.get("accum_step", 0) + 1) % self.steps)
        return loss

    def should_step(self, ctx) -> bool:
        return ctx.get("accum_step", 0) == 0


class ZLoss(BaseAlgorithm):
    """Auxiliary z-loss to keep the logits well-behaved (PaLM style)."""

    name = "z_loss"
    category = "stability"
    description = "Penalises large log-sum-exp values to stabilise logits."

    def __init__(self, weight: float = 1e-4):
        super().__init__(weight=weight)
        self.weight = weight

    def compute_loss(self, ctx, batch, outputs, loss):
        if loss is None:
            return None
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        z = logits.logsumexp(dim=-1).pow(2).mean()
        return loss + self.weight * z


class GradientNoise(BaseAlgorithm):
    """Add calibrated Gaussian noise to gradients.

    Useful both as a differential-privacy mechanism and as an exploration
    regulariser for small models that overfit quickly.
    """

    name = "gradient_noise"
    category = "stability"
    description = "Adds Gaussian noise to gradients (DP-style regularisation)."

    def __init__(self, sigma: float = 1e-3):
        super().__init__(sigma=sigma)
        self.sigma = sigma

    def on_before_optimizer_step(self, ctx):
        if self.sigma <= 0:
            return
        for p in ctx.model.parameters():
            if p.grad is not None:
                p.grad.add_(torch.randn_like(p.grad) * self.sigma)


class ActivationCheckpointing(BaseAlgorithm):
    """Enable gradient checkpointing on compatible models."""

    name = "activation_checkpointing"
    category = "memory"
    description = "Trades compute for memory by recomputing activations."

    def on_train_begin(self, ctx):
        model = ctx.model
        if hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable()
            except Exception:
                pass

    def on_train_end(self, ctx):
        model = ctx.model
        if hasattr(model, "gradient_checkpointing_disable"):
            try:
                model.gradient_checkpointing_disable()
            except Exception:
                pass
