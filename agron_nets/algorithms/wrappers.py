"""Optimiser wrappers: Sharpness-Aware Minimisation and Lookahead."""

from __future__ import annotations

from typing import Optional

import torch

from .base import AlgorithmContext, BaseAlgorithm, compose_loss


class SAM(BaseAlgorithm):
    """Sharpness-Aware Minimisation (Foret et al., 2021).

    Two-phase update: ascend to the worst point in a neighbourhood of radius
    ``rho``, then take the descent step from there.
    """

    name = "sam"
    category = "optimisation"
    description = "Seeks flat minima by perturbing weights to the local worst case."

    def __init__(self, rho: float = 0.05, adaptive: bool = False, **kwargs):
        super().__init__(rho=rho, adaptive=adaptive, **kwargs)
        self.rho = rho
        self.adaptive = adaptive

    def train_step(self, ctx, batch):
        model, optimizer = ctx.model, ctx.optimizer
        batch = self.on_step_begin(ctx, batch)

        # 1) forward / backward at the current point
        outputs = model(**batch)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        loss = self.compute_loss(ctx, batch, outputs, loss)
        loss = compose_loss(ctx, batch, outputs, loss, skip=self)
        loss.backward()

        # 2) compute the ascent perturbation
        with torch.no_grad():
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            grad_norm = torch.norm(torch.stack([g.norm(p=2) for g in grads]), p=2) if grads else torch.tensor(0.0)
            scale = self.rho / (grad_norm + 1e-12)
            self._perturbations = []
            for p in model.parameters():
                if p.grad is None:
                    continue
                if self.adaptive:
                    perturb = p.grad * scale * p.abs()
                else:
                    perturb = p.grad * scale
                p.add_(perturb)
                self._perturbations.append((p, perturb))
        optimizer.zero_grad(set_to_none=True)

        # 3) forward / backward at the perturbed point
        outputs = model(**batch)
        loss_perturbed = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        loss_perturbed = self.compute_loss(ctx, batch, outputs, loss_perturbed)
        loss_perturbed = compose_loss(ctx, batch, outputs, loss_perturbed, skip=self)
        loss_perturbed.backward()

        # 4) restore the original weights
        with torch.no_grad():
            for p, perturb in self._perturbations:
                p.sub_(perturb)
        self._perturbations = []
        return loss_perturbed


class Lookahead(BaseAlgorithm):
    """Lookahead optimiser (Zhang et al., 2019).

    Maintains slow weights that are periodically interpolated with the fast
    weights produced by the inner optimiser.
    """

    name = "lookahead"
    category = "optimisation"
    description = "Interpolates fast and slow weight trajectories every k steps."

    def __init__(self, k: int = 5, alpha: float = 0.5):
        super().__init__(k=k, alpha=alpha)
        self.k = max(1, k)
        self.alpha = alpha
        self._slow: Optional[list] = None

    def on_train_begin(self, ctx):
        self._slow = [p.detach().clone() for p in ctx.model.parameters()]

    @torch.no_grad()
    def on_step_end(self, ctx, loss):
        if self._slow is None:
            return
        if (ctx.step + 1) % self.k != 0:
            return
        for slow, p in zip(self._slow, ctx.model.parameters()):
            slow.lerp_(p.detach(), self.alpha)
            p.copy_(slow)
