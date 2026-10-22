"""Advanced optimisers for the Agron zoo.

Every optimiser is a standard ``torch.optim.Optimizer`` subclass, so it can be
used directly with PyTorch or routed through :func:`build_optimizer`.

Registered optimisers
---------------------
``adam``, ``adamw``, ``adamax``, ``adagrad``, ``rmsprop``, ``sgd``, ``nadam``,
``lion``, ``adafactor``, ``adan``, ``lamb``, ``muon``, ``shampoo``,
``schedulefree``, ``kahan``, ``sophia``
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch
from torch.optim import Optimizer


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
OPTIMIZERS: Dict[str, Dict[str, Any]] = {}


def _register(name: str, cls, description: str, family: str, defaults: Optional[Dict[str, Any]] = None) -> None:
    OPTIMIZERS[name] = {
        "name": name,
        "class": cls,
        "description": description,
        "family": family,
        "defaults": defaults or {},
    }


def optimizer_names() -> List[str]:
    return sorted(OPTIMIZERS.keys())


def get_optimizer(name: str) -> Dict[str, Any]:
    key = (name or "adamw").strip().lower()
    if key not in OPTIMIZERS:
        raise KeyError(f"unknown optimizer '{name}'. available: {', '.join(optimizer_names())}")
    return OPTIMIZERS[key]


def build_optimizer(name: str, params, **overrides):
    """Instantiate the named optimiser for ``params``."""
    entry = get_optimizer(name)
    defaults = dict(entry.get("defaults", {}))
    defaults.update(overrides)
    return entry["class"](params, **defaults)


def optimizer_report() -> List[Dict[str, Any]]:
    return [
        {"name": k, "family": v["family"], "description": v["description"], "defaults": v["defaults"]}
        for k, v in sorted(OPTIMIZERS.items())
    ]


def _check_lr(lr: float, eps: float = 0.0) -> None:
    if lr < 0.0:
        raise ValueError(f"Invalid learning rate: {lr}")
    if eps < 0.0:
        raise ValueError(f"Invalid epsilon value: {eps}")


def _check_beta(beta: float, index: int) -> None:
    if not 0.0 <= beta < 1.0:
        raise ValueError(f"Invalid beta parameter at index {index}: {beta}")


# --------------------------------------------------------------------------- #
# Adam family
# --------------------------------------------------------------------------- #
class Adam(Optimizer):
    """Vanilla Adam with optional AMSGrad."""

    def __init__(self, params, lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0, amsgrad: bool = False):
        _check_lr(lr, eps)
        _check_beta(betas[0], 0)
        _check_beta(betas[1], 1)
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients")
                if group["weight_decay"] != 0:
                    grad = grad.add(p, alpha=group["weight_decay"])
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                    if group["amsgrad"]:
                        state["max_v"] = torch.zeros_like(p)
                state["step"] += 1
                m, v = state["m"], state["v"]
                m.lerp_(grad, 1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bc1 = 1 - beta1 ** state["step"]
                bc2 = 1 - beta2 ** state["step"]
                if group["amsgrad"]:
                    max_v = state["max_v"]
                    max_v.copy_(torch.maximum(max_v, v))
                    denom = (max_v.sqrt() / math.sqrt(bc2)) + group["eps"]
                else:
                    denom = (v.sqrt() / math.sqrt(bc2)) + group["eps"]
                p.addcdiv_(m, denom, value=-(group["lr"] / bc1))
        return loss


class AdamW(Optimizer):
    """AdamW with decoupled weight decay and optional AMSGrad."""

    def __init__(self, params, lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.01, amsgrad: bool = False):
        _check_lr(lr, eps)
        _check_beta(betas[0], 0)
        _check_beta(betas[1], 1)
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                    if group["amsgrad"]:
                        state["max_v"] = torch.zeros_like(p)
                state["step"] += 1
                m, v = state["m"], state["v"]
                if group["weight_decay"] != 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                m.lerp_(grad, 1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bc1 = 1 - beta1 ** state["step"]
                bc2 = 1 - beta2 ** state["step"]
                if group["amsgrad"]:
                    max_v = state["max_v"]
                    max_v.copy_(torch.maximum(max_v, v))
                    denom = (max_v.sqrt() / math.sqrt(bc2)) + group["eps"]
                else:
                    denom = (v.sqrt() / math.sqrt(bc2)) + group["eps"]
                p.addcdiv_(m, denom, value=-(group["lr"] / bc1))
        return loss


class NAdam(Optimizer):
    """Nesterov-accelerated Adam."""

    def __init__(self, params, lr: float = 2e-3, betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0, momentum_decay: float = 4e-3):
        _check_lr(lr, eps)
        _check_beta(betas[0], 0)
        _check_beta(betas[1], 1)
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                                      momentum_decay=momentum_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            mu = group["momentum_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("NAdam does not support sparse gradients")
                if group["weight_decay"] != 0:
                    grad = grad.add(p, alpha=group["weight_decay"])
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                    state["mu_product"] = 1.0
                state["step"] += 1
                m, v = state["m"], state["v"]
                mu_product = state["mu_product"] * (1 - 0.5 * 0.96 ** (state["step"] * mu))
                mu_product_next = mu_product * (1 - 0.5 * 0.96 ** ((state["step"] + 1) * mu))
                state["mu_product"] = mu_product_next
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                m_hat = beta1 * m / (1 - mu_product_next) + (1 - beta1) * grad / (1 - mu_product)
                v_hat = v / (1 - beta2 ** state["step"])
                p.addcdiv_(m_hat, v_hat.sqrt().add_(group["eps"]), value=-group["lr"])
        return loss


class Adamax(Optimizer):
    """Adamax (infinity-norm variant of Adam)."""

    def __init__(self, params, lr: float = 2e-3, betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0):
        _check_lr(lr, eps)
        _check_beta(betas[0], 0)
        _check_beta(betas[1], 1)
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adamax does not support sparse gradients")
                if group["weight_decay"] != 0:
                    grad = grad.add(p, alpha=group["weight_decay"])
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["u"] = torch.zeros_like(p)
                state["step"] += 1
                m, u = state["m"], state["u"]
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                u.mul_(beta2).clamp_(min=0).add_(grad.abs().add_(group["eps"]), alpha=1 - beta2)
                bias_c = 1 - beta1 ** state["step"]
                p.addcdiv_(m, u, value=-(group["lr"] / bias_c))
        return loss


class Adagrad(Optimizer):
    """Adagrad with optional decoupled weight decay."""

    def __init__(self, params, lr: float = 1e-2, lr_decay: float = 0.0, weight_decay: float = 0.0,
                 initial_accumulator_value: float = 0.0, eps: float = 1e-10):
        _check_lr(lr, eps)
        if lr_decay < 0.0:
            raise ValueError(f"Invalid lr_decay value: {lr_decay}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if initial_accumulator_value < 0.0:
            raise ValueError("initial_accumulator_value must be non-negative")
        super().__init__(params, dict(lr=lr, lr_decay=lr_decay, weight_decay=weight_decay,
                                      initial_accumulator_value=initial_accumulator_value, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adagrad does not support sparse gradients")
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["sum"] = torch.full_like(p, group["initial_accumulator_value"])
                state["step"] += 1
                if group["weight_decay"] != 0:
                    grad = grad.add(p, alpha=group["weight_decay"])
                clr = group["lr"] / (1 + (state["step"] - 1) * group["lr_decay"])
                state["sum"].addcmul_(grad, grad, value=1)
                p.addcdiv_(grad, state["sum"].sqrt().add_(group["eps"]), value=-clr)
        return loss


class RMSprop(Optimizer):
    """RMSprop with momentum and centered variants."""

    def __init__(self, params, lr: float = 1e-2, alpha: float = 0.99, eps: float = 1e-8,
                 weight_decay: float = 0.0, momentum: float = 0.0, centered: bool = False):
        _check_lr(lr, eps)
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= alpha < 1.0:
            raise ValueError(f"Invalid alpha value: {alpha}")
        super().__init__(params, dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay,
                                      momentum=momentum, centered=centered))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("RMSprop does not support sparse gradients")
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["square_avg"] = torch.zeros_like(p)
                    if group["momentum"] > 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    if group["centered"]:
                        state["grad_avg"] = torch.zeros_like(p)
                state["step"] += 1
                alpha = group["alpha"]
                square_avg = state["square_avg"]
                square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)
                if group["centered"]:
                    grad_avg = state["grad_avg"]
                    grad_avg.mul_(alpha).add_(grad, alpha=1 - alpha)
                    avg = square_avg.addcmul_(grad_avg, grad_avg, value=-1).sqrt().add_(group["eps"])
                else:
                    avg = square_avg.sqrt().add_(group["eps"])
                if group["weight_decay"] != 0:
                    grad = grad.add(p, alpha=group["weight_decay"])
                if group["momentum"] > 0:
                    buf = state["momentum_buffer"]
                    buf.mul_(group["momentum"]).addcdiv_(grad, avg)
                    p.add_(buf, alpha=-group["lr"])
                else:
                    p.addcdiv_(grad, avg, value=-group["lr"])
        return loss


class SGD(Optimizer):
    """SGD with momentum, dampening, weight decay and Nesterov."""

    def __init__(self, params, lr: float = 1e-2, momentum: float = 0.0, dampening: float = 0.0,
                 weight_decay: float = 0.0, nesterov: bool = False):
        _check_lr(lr)
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires momentum > 0 and dampening == 0")
        super().__init__(params, dict(lr=lr, momentum=momentum, dampening=dampening,
                                      weight_decay=weight_decay, nesterov=nesterov))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad
                if group["weight_decay"] != 0:
                    d_p = d_p.add(p, alpha=group["weight_decay"])
                if group["momentum"] != 0:
                    buf = self.state[p].get("momentum_buffer")
                    if buf is None:
                        buf = self.state[p]["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf.mul_(group["momentum"]).add_(d_p, alpha=1 - group["dampening"])
                    d_p = d_p.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf
                p.add_(d_p, alpha=-group["lr"])
        return loss


# --------------------------------------------------------------------------- #
# Sign / momentum based
# --------------------------------------------------------------------------- #
class Lion(Optimizer):
    """Evolved Sign Momentum (Chen et al., 2023)."""

    def __init__(self, params, lr: float = 1e-4, betas: Tuple[float, float] = (0.9, 0.99),
                 weight_decay: float = 0.01):
        _check_lr(lr)
        _check_beta(betas[0], 0)
        _check_beta(betas[1], 1)
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Lion does not support sparse gradients")
                state = self.state[p]
                if not state:
                    state["m"] = torch.zeros_like(p)
                m = state["m"]
                if group["weight_decay"] != 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                update = m.mul(beta1).add(grad, alpha=1 - beta1)
                p.add_(torch.sign(update), alpha=-group["lr"])
                m.mul_(beta2).add_(grad, alpha=1 - beta2)
                state["m"] = m
        return loss


# --------------------------------------------------------------------------- #
# Factored / memory efficient
# --------------------------------------------------------------------------- #
class Adafactor(Optimizer):
    """Adafactor with factored second moments and update clipping."""

    def __init__(self, params, lr: float = 1e-3, beta2: float = 0.999,
                 eps: Tuple[float, float] = (1e-30, 1e-3), clip_threshold: float = 1.0,
                 weight_decay: float = 0.0, min_dim_size_to_factor: int = 128):
        _check_lr(lr)
        _check_beta(beta2, 1)
        super().__init__(params, dict(lr=lr, beta2=beta2, eps=eps, clip_threshold=clip_threshold,
                                      weight_decay=weight_decay,
                                      min_dim_size_to_factor=min_dim_size_to_factor))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta2 = group["beta2"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adafactor does not support sparse gradients")
                state = self.state[p]
                factored = (
                    grad.dim() >= 2
                    and min(grad.shape[-1], grad.shape[-2]) >= group["min_dim_size_to_factor"]
                )
                if not state:
                    state["step"] = 0
                    if factored:
                        state["row"] = torch.zeros(grad.shape[:-1], device=grad.device)
                        state["col"] = torch.zeros(grad.shape[:-2] + grad.shape[-1:], device=grad.device)
                    else:
                        state["v"] = torch.zeros_like(grad)
                state["step"] += 1
                step = state["step"]

                if factored:
                    row, col = state["row"], state["col"]
                    row.mul_(beta2).add_(grad.pow(2).mean(dim=-1), alpha=1 - beta2)
                    col.mul_(beta2).add_(grad.pow(2).mean(dim=-2), alpha=1 - beta2)
                    row_mean = row.mean(dim=-1, keepdim=True).clamp_min(1e-30)
                    col_mean = col.mean(dim=-2, keepdim=True).clamp_min(1e-30)
                    v = (row / row_mean).unsqueeze(-1) * (col / col_mean).unsqueeze(-2)
                else:
                    v = state["v"]
                    v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                update = grad / (v.sqrt() + group["eps"][1])
                update_norm = update.norm()
                if update_norm > 0:
                    update = update / max(1.0, (update_norm / group["clip_threshold"]).item())
                rel_step = group["lr"] * math.sqrt(max(1e-38, 1 - beta2 ** step))
                p.add_(update, alpha=-rel_step)
                if group["weight_decay"] != 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
        return loss


class Adan(Optimizer):
    """Adan - adaptive Nesterov momentum optimiser (Xie et al., 2022)."""

    def __init__(self, params, lr: float = 1e-3, betas: Tuple[float, float, float] = (0.98, 0.92, 0.99),
                 eps: float = 1e-8, weight_decay: float = 0.01):
        _check_lr(lr, eps)
        for i, b in enumerate(betas):
            _check_beta(b, i)
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2, beta3 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adan does not support sparse gradients")
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                    state["n"] = torch.zeros_like(p)
                    state["prev_grad"] = torch.zeros_like(p)
                state["step"] += 1
                m, v, n, prev_grad = state["m"], state["v"], state["n"], state["prev_grad"]
                if group["weight_decay"] != 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                grad_diff = grad - prev_grad
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).add_(grad_diff, alpha=1 - beta2)
                n.mul_(beta3).addcmul_(grad + (1 - beta2) * grad_diff,
                                       grad + (1 - beta2) * grad_diff, value=1 - beta3)
                prev_grad.copy_(grad)
                bc1 = 1 - beta1 ** state["step"]
                bc2 = 1 - beta2 ** state["step"]
                bc3 = 1 - beta3 ** state["step"]
                denom = (n.sqrt() / math.sqrt(bc3)) + group["eps"]
                update = (m + (1 - beta2) * v) / denom
                p.add_(update, alpha=-(group["lr"] / bc1))
        return loss


class LAMB(Optimizer):
    """LAMB - layer-wise adaptive moments for batched training."""

    def __init__(self, params, lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-6, weight_decay: float = 0.01):
        _check_lr(lr, eps)
        _check_beta(betas[0], 0)
        _check_beta(betas[1], 1)
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("LAMB does not support sparse gradients")
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                state["step"] += 1
                m, v = state["m"], state["v"]
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bc1 = 1 - beta1 ** state["step"]
                bc2 = 1 - beta2 ** state["step"]
                update = (m / bc1) / ((v / bc2).sqrt() + group["eps"])
                if group["weight_decay"] != 0:
                    update = update.add(p, alpha=group["weight_decay"])
                param_norm = p.norm()
                update_norm = update.norm()
                trust_ratio = 1.0
                if param_norm > 0 and update_norm > 0:
                    trust_ratio = (param_norm / update_norm).item()
                p.add_(update, alpha=-group["lr"] * trust_ratio)
        return loss


# --------------------------------------------------------------------------- #
# Second-order inspired
# --------------------------------------------------------------------------- #
def _zeropower_newton_schulz(matrix: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate the orthogonal polar factor via Newton-Schulz iteration."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = matrix.float()
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    if transposed:
        x = x.T
    return x.to(matrix.dtype)


class Muon(Optimizer):
    """Muon - momentum orthogonalised by Newton-Schulz iterations.

    Two-dimensional parameters are orthogonalised; one-dimensional parameters
    fall back to Adam-style updates.
    """

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, weight_decay: float = 0.0,
                 ns_steps: int = 5, eps: float = 1e-8,
                 adam_betas: Tuple[float, float] = (0.9, 0.999)):
        _check_lr(lr, eps)
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      weight_decay=weight_decay, ns_steps=ns_steps,
                                      eps=eps, adam_betas=adam_betas,
                                      ns_coefficients=(3.4445, -4.7750, 2.0315)))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            grads = [p.grad for p in params]
            if group["weight_decay"] != 0:
                for p in params:
                    p.mul_(1 - group["lr"] * group["weight_decay"])

            # 2D parameters: orthogonalised momentum
            ortho = [(p, g) for p, g in zip(params, grads) if p.dim() >= 2]
            flat = [(p, g) for p, g in zip(params, grads) if p.dim() < 2]

            for p, g in ortho:
                state = self.state[p]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(p)
                buf = state["momentum"]
                buf.mul_(group["momentum"]).add_(g)
                update = g.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf
                o = _zeropower_newton_schulz(update, group["ns_steps"])
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                p.add_(o, alpha=-group["lr"] * scale)

            beta1, beta2 = group["adam_betas"]
            for p, g in flat:
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                state["step"] += 1
                m, v = state["m"], state["v"]
                m.lerp_(g, 1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                bc1 = 1 - beta1 ** state["step"]
                bc2 = 1 - beta2 ** state["step"]
                denom = (v.sqrt() / math.sqrt(bc2)) + group["eps"]
                p.addcdiv_(m, denom, value=-(group["lr"] / bc1))
        return loss


def _matrix_inverse_root(mat: torch.Tensor, root: int = 2, eps: float = 1e-8) -> torch.Tensor:
    """``mat ** (-1/root)`` via a symmetric eigendecomposition."""
    mat = mat.float() + eps * torch.eye(mat.size(0), device=mat.device)
    eigvals, eigvecs = torch.linalg.eigh(mat)
    eigvals = eigvals.clamp_min(1e-12).pow(-1.0 / root)
    return (eigvecs * eigvals.unsqueeze(0)) @ eigvecs.T


class Shampoo(Optimizer):
    """Shampoo - full-matrix preconditioning of gradients.

    Preconditioners are refreshed every ``precondition_frequency`` steps and
    fall back to a diagonal preconditioner for very large dimensions.
    """

    def __init__(self, params, lr: float = 1e-2, betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0,
                 precondition_frequency: int = 25, max_dim: int = 2048):
        _check_lr(lr, eps)
        _check_beta(betas[0], 0)
        _check_beta(betas[1], 1)
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                                      precondition_frequency=precondition_frequency,
                                      max_dim=max_dim))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Shampoo does not support sparse gradients")
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["precond"] = None
                    state["factors"] = {}
                state["step"] += 1
                m = state["m"]
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                update = m
                if group["weight_decay"] != 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])

                if grad.dim() == 2:
                    factors = state["factors"]
                    if "L" not in factors:
                        factors["L"] = torch.zeros_like(grad @ grad.T)
                        factors["R"] = torch.zeros_like(grad.T @ grad)
                    factors["L"].mul_(beta2).add_(grad @ grad.T, alpha=1 - beta2)
                    factors["R"].mul_(beta2).add_(grad.T @ grad, alpha=1 - beta2)
                    if state["step"] % group["precondition_frequency"] == 0 or state["precond"] is None:
                        if max(grad.shape) > group["max_dim"]:
                            # diagonal fallback for very large layers
                            diagonal = factors["L"].diagonal().clamp_min(1e-12).pow(-0.25)
                            l_inv = torch.diag(diagonal)
                            r_diag = factors["R"].diagonal().clamp_min(1e-12).pow(-0.25)
                            r_inv = torch.diag(r_diag)
                        else:
                            l_inv = _matrix_inverse_root(factors["L"], 4)
                            r_inv = _matrix_inverse_root(factors["R"], 4)
                        state["precond"] = (l_inv, r_inv)
                    l_inv, r_inv = state["precond"]
                    try:
                        update = l_inv @ update @ r_inv
                    except Exception:
                        update = update
                elif grad.dim() > 2:
                    update = update / (grad.pow(2).mean(dim=tuple(range(1, grad.dim())), keepdim=True).sqrt() + group["eps"])

                p.add_(update, alpha=-group["lr"])
        return loss


class Sophia(Optimizer):
    """Sophia - second-order clipped stochastic optimisation.

    The diagonal Hessian estimate is updated by :meth:`update_hessian` using
    the Gauss-Newton-Bartlett (GNB) estimator with a Hutchinson probe.
    """

    def __init__(self, params, lr: float = 1e-3, betas: Tuple[float, float] = (0.965, 0.99),
                 eps: float = 1e-15, weight_decay: float = 0.01, rho: float = 0.04):
        _check_lr(lr, eps)
        _check_beta(betas[0], 0)
        _check_beta(betas[1], 1)
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, rho=rho))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["h"] = torch.zeros_like(p)
                state["step"] += 1
                m, h = state["m"], state["h"]
                if group["weight_decay"] != 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                ratio = (m / (group["rho"] * h + group["eps"])).clamp(-1.0, 1.0)
                p.add_(ratio, alpha=-group["lr"])
        return loss

    def update_hessian(self, loss: torch.Tensor, seed: Optional[int] = None) -> None:
        """Refresh the diagonal Hessian using a Hutchinson/GNB probe."""
        params = [p for group in self.param_groups for p in group["params"] if p.requires_grad]
        if not params:
            return
        generator = torch.Generator(device=params[0].device)
        if seed is not None:
            generator.manual_seed(seed)
        grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True, allow_unused=True)
        probes = [torch.randint_like(p, 0, 2, generator=generator).float() * 2 - 1 for p in params]
        dot = sum((g * v).sum() for g, v in zip(grads, probes) if g is not None and g.requires_grad)
        if not isinstance(dot, torch.Tensor) or not dot.requires_grad:
            return
        second = torch.autograd.grad(dot, params, retain_graph=False, allow_unused=True)
        for group in self.param_groups:
            beta2 = group["betas"][1]
            for p, hv in zip(params, second):
                if p not in self.state:
                    continue
                if hv is None:
                    continue
                state = self.state[p]
                h = state["h"]
                probe_sq = (hv * hv).abs()
                h.mul_(beta2).add_(probe_sq, alpha=1 - beta2)


# --------------------------------------------------------------------------- #
# Schedule-free and compensated
# --------------------------------------------------------------------------- #
class ScheduleFreeAdamW(Optimizer):
    """Schedule-Free AdamW (Defazio et al., 2024).

    Implements Algorithm 1 of the paper: ``z`` is the fast iterate, ``x`` the
    averaged eval point and ``y = (1-b1) z + b1 x`` the gradient point.

    Call :meth:`train` before the forward pass and :meth:`eval` before
    validation / inference.
    """

    def __init__(self, params, lr: float = 2.5e-3, betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0, warmup_steps: int = 0):
        _check_lr(lr, eps)
        _check_beta(betas[0], 0)
        _check_beta(betas[1], 1)
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                                      warmup_steps=warmup_steps))

    def _state(self, p):
        state = self.state[p]
        if not state:
            state["step"] = 0
            state["z"] = p.detach().clone()
            state["x"] = p.detach().clone()
            state["v"] = torch.zeros_like(p)
        return state

    @torch.no_grad()
    def train(self):
        """Set the parameters to the gradient point ``y``."""
        for group in self.param_groups:
            beta1 = group["betas"][0]
            for p in group["params"]:
                state = self._state(p)
                p.copy_((1 - beta1) * state["z"] + beta1 * state["x"])

    @torch.no_grad()
    def eval(self):
        """Set the parameters to the averaged eval point ``x``."""
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "x" in state:
                    p.copy_(state["x"])

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self._state(p)
                state["step"] += 1
                t = state["step"]
                z, x, v = state["z"], state["x"], state["v"]
                grad = p.grad
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if t == 1:
                    v.copy_(grad.pow(2))
                denom = v.sqrt().add_(group["eps"])
                if group["weight_decay"] != 0:
                    z.mul_(1 - lr * group["weight_decay"])
                z.addcdiv_(grad, denom, value=-lr)

                warmup = group["warmup_steps"]
                ckp1 = t / warmup if (warmup > 0 and t < warmup) else 1.0 / (t + 1)
                x.lerp_(z, ckp1)
                # prepare the next gradient point
                p.copy_((1 - beta1) * z + beta1 * x)
        return loss


class KahanAdamW(Optimizer):
    """AdamW with Kahan compensated summation on the parameter update.

    Compensated summation removes the low-precision drift that accumulates
    when training long runs in bf16/fp16.
    """

    def __init__(self, params, lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.01):
        _check_lr(lr, eps)
        _check_beta(betas[0], 0)
        _check_beta(betas[1], 1)
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if group["weight_decay"] != 0:
                    grad = grad.add(p, alpha=group["weight_decay"])
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                    state["comp"] = torch.zeros_like(p, dtype=torch.float32)
                state["step"] += 1
                m, v = state["m"], state["v"]
                m.lerp_(grad, 1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bc1 = 1 - beta1 ** state["step"]
                bc2 = 1 - beta2 ** state["step"]
                denom = (v.sqrt() / math.sqrt(bc2)) + group["eps"]
                delta = -(group["lr"] / bc1) * (m / denom)
                # Kahan compensated update
                comp = state["comp"]
                y = delta.float() - comp
                t = p.float() + y
                comp = (t - p.float()) - y
                state["comp"] = comp
                p.copy_(t.to(p.dtype))
        return loss


# --------------------------------------------------------------------------- #
# Register everything
# --------------------------------------------------------------------------- #
_register("adam", Adam, "Vanilla Adam with optional AMSGrad", "adaptive",
          {"lr": 1e-3, "betas": (0.9, 0.999)})
_register("adamw", AdamW, "AdamW with decoupled weight decay", "adaptive",
          {"lr": 1e-3, "betas": (0.9, 0.999), "weight_decay": 0.01})
_register("nadam", NAdam, "Nesterov-accelerated Adam", "adaptive",
          {"lr": 2e-3, "betas": (0.9, 0.999)})
_register("adamax", Adamax, "Adam with an infinity-norm second moment", "adaptive",
          {"lr": 2e-3, "betas": (0.9, 0.999)})
_register("adagrad", Adagrad, "Cumulative squared-gradient preconditioning", "adaptive",
          {"lr": 1e-2})
_register("rmsprop", RMSprop, "RMSprop with momentum / centering", "adaptive",
          {"lr": 1e-2, "alpha": 0.99})
_register("sgd", SGD, "SGD with momentum and Nesterov", "momentum",
          {"lr": 1e-2, "momentum": 0.9})
_register("lion", Lion, "Evolved Sign Momentum (Lion)", "sign-momentum",
          {"lr": 1e-4, "betas": (0.9, 0.99), "weight_decay": 0.01})
_register("adafactor", Adafactor, "Memory-efficient factored second moments", "factored",
          {"lr": 1e-3})
_register("adan", Adan, "Adaptive Nesterov momentum (Adan)", "adaptive",
          {"lr": 1e-3, "betas": (0.98, 0.92, 0.99), "weight_decay": 0.01})
_register("lamb", LAMB, "Layer-wise adaptive moments (LAMB)", "layer-adaptive",
          {"lr": 1e-3, "betas": (0.9, 0.999), "weight_decay": 0.01})
_register("muon", Muon, "Momentum orthogonalised with Newton-Schulz", "orthogonalised",
          {"lr": 0.02, "momentum": 0.95})
_register("shampoo", Shampoo, "Full-matrix gradient preconditioning", "second-order",
          {"lr": 1e-2, "precondition_frequency": 25})
_register("sophia", Sophia, "Second-order clipped optimisation with GNB Hessian", "second-order",
          {"lr": 1e-3, "betas": (0.965, 0.99)})
_register("schedulefree", ScheduleFreeAdamW, "Schedule-Free AdamW", "schedule-free",
          {"lr": 2.5e-3, "betas": (0.9, 0.999)})
_register("kahan", KahanAdamW, "AdamW with Kahan compensated updates", "compensated",
          {"lr": 1e-3, "betas": (0.9, 0.999), "weight_decay": 0.01})


__all__ = [
    "OPTIMIZERS",
    "Adam",
    "AdamW",
    "NAdam",
    "Adamax",
    "Adagrad",
    "RMSprop",
    "SGD",
    "Lion",
    "Adafactor",
    "Adan",
    "LAMB",
    "Muon",
    "Shampoo",
    "Sophia",
    "ScheduleFreeAdamW",
    "KahanAdamW",
    "build_optimizer",
    "get_optimizer",
    "optimizer_names",
    "optimizer_report",
]
