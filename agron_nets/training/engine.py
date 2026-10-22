"""A small, dependency-light training engine for the Agron architecture zoo.

The engine composes four independent axes:

* an **architecture** from :mod:`agron_nets.architectures`,
* an **optimiser** from :mod:`agron_nets.optimizers`,
* a **learning-rate schedule** from :mod:`agron_nets.schedulers`,
* any number of **algorithms** from :mod:`agron_nets.algorithms`.

Example
-------
>>> from agron_nets.training import AgronTrainer, TrainerConfig
>>> trainer = AgronTrainer(
...     model,
...     optimizer="muon",
...     scheduler="wsd",
...     algorithms=["gradient_clip", "ema", "z_loss"],
...     config=TrainerConfig(max_steps=1000, warmup_steps=50),
... )
>>> trainer.fit(train_loader)
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn as nn

from ..algorithms import AlgorithmContext, BaseAlgorithm, build_algorithm
from ..algorithms.base import compose_loss
from ..optimizers import OPTIMIZERS, build_optimizer
from ..schedulers import build_scheduler
from ..utils import model_device, set_seed, write_json


@dataclass
class TrainerConfig:
    """Everything the training loop needs that is not a model component."""

    max_steps: int = 1000
    epochs: int = 1
    warmup_steps: int = 0
    log_every: int = 10
    save_every: int = 0
    output_dir: Optional[str] = None
    grad_clip: Optional[float] = 1.0
    seed: int = 42
    weight_decay: Optional[float] = None
    sophia_hessian_every: int = 0
    log_tokens_per_second: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _truncate_batch(batch: Dict[str, torch.Tensor], max_len: int) -> Dict[str, torch.Tensor]:
    """Keep the last ``max_len`` positions so RoPE never runs out of cache."""
    if max_len <= 0:
        return batch
    out = dict(batch)
    for key in ("input_ids", "labels", "attention_mask", "position_ids"):
        value = out.get(key)
        if torch.is_tensor(value) and value.dim() >= 2 and value.shape[1] > max_len:
            out[key] = value[:, -max_len:]
    return out


def _extract_loss(outputs: Any) -> Optional[torch.Tensor]:
    if isinstance(outputs, dict):
        return outputs.get("loss")
    return getattr(outputs, "loss", None)


def _is_step_driver(algo: BaseAlgorithm) -> bool:
    """An algorithm that owns the forward/backward pass itself."""
    return type(algo).train_step is not BaseAlgorithm.train_step


class AgronTrainer:
    """Compose an architecture with an optimiser, schedule and algorithms."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Any = "adamw",
        scheduler: Any = "cosine",
        algorithms: Optional[Iterable[Any]] = None,
        config: Optional[TrainerConfig] = None,
        device: Optional[Any] = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        scheduler_kwargs: Optional[Dict[str, Any]] = None,
        algorithm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.config = config or TrainerConfig()
        self.device = torch.device(device or model_device(model))

        params = [p for p in model.parameters() if p.requires_grad]
        if isinstance(optimizer, str):
            overrides = dict(optimizer_kwargs or {})
            lr = overrides.pop("lr", None)
            if lr is None:
                lr = OPTIMIZERS.get(optimizer.lower(), {}).get("defaults", {}).get("lr", 1e-3)
            self.optimizer = build_optimizer(optimizer, params, lr=lr, **overrides)
        else:
            self.optimizer = optimizer

        if self.config.weight_decay is not None:
            for group in self.optimizer.param_groups:
                group["weight_decay"] = self.config.weight_decay

        if isinstance(scheduler, str) and scheduler.lower() not in ("none", "constant_none"):
            self.scheduler = build_scheduler(
                self.optimizer,
                scheduler,
                total_steps=max(1, self.config.max_steps),
                warmup_steps=self.config.warmup_steps,
                **(scheduler_kwargs or {}),
            )
        else:
            self.scheduler = scheduler if scheduler not in ("none", "constant_none") else None

        if algorithms is None:
            algorithms = []
        elif isinstance(algorithms, str):
            algorithms = [a.strip() for a in algorithms.split(",") if a.strip()]
        self.algorithms: List[BaseAlgorithm] = [
            a if isinstance(a, BaseAlgorithm) else build_algorithm(a, **(algorithm_kwargs or {}))
            for a in algorithms
        ]

        if self.config.grad_clip is not None and not any(
            a.name == "gradient_clip" for a in self.algorithms
        ):
            from ..algorithms.stability import GradientClip

            self.algorithms.append(GradientClip(max_norm=self.config.grad_clip))

        self.step = 0
        self.history: List[Dict[str, float]] = []
        self._ctx: Optional[AlgorithmContext] = None

    # ------------------------------------------------------------------ #
    # Setup helpers
    # ------------------------------------------------------------------ #
    def _make_context(self) -> AlgorithmContext:
        ctx = AlgorithmContext(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device,
            step=0,
            epoch=0,
            max_steps=self.config.max_steps,
        )
        ctx.set("loss_hooks", self.algorithms)
        ctx.set("algorithms", self.algorithms)
        return ctx

    def _call(self, name: str, *args) -> None:
        for algo in self.algorithms:
            fn = getattr(algo, name, None)
            if fn is not None:
                fn(self._ctx, *args)

    @property
    def current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def fit(
        self,
        train_dataloader,
        eval_dataloader=None,
        callbacks: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        set_seed(self.config.seed)
        self.model.to(self.device)
        self.model.train()
        self._ctx = self._make_context()
        self._ctx.step = self.step
        self._call("on_train_begin")

        max_len = int(getattr(self.model.config, "max_position_embeddings", 0) or 0)
        drivers = [a for a in self.algorithms if _is_step_driver(a)]
        if len(drivers) > 1:
            raise ValueError(
                "only one step-owning algorithm can be active at a time, got: "
                + ", ".join(a.name for a in drivers)
            )
        driver = drivers[0] if drivers else None
        schedule_free = hasattr(self.optimizer, "train") and self.optimizer.__class__.__name__ == "ScheduleFreeAdamW"

        started = time.time()
        tokens_seen = 0
        epoch = 0
        stopped = False

        while epoch < self.config.epochs and not stopped:
            self._ctx.epoch = epoch
            self._call("on_epoch_begin")
            for batch in train_dataloader:
                if self.step >= self.config.max_steps:
                    stopped = True
                    break

                batch = _truncate_batch(_batch_to_device(batch, self.device), max_len)
                self._ctx.step = self.step
                self._ctx.set("batch", batch)

                if schedule_free:
                    self.optimizer.train()

                for algo in self.algorithms:
                    batch = algo.on_step_begin(self._ctx, batch)

                if driver is not None:
                    loss = driver.train_step(self._ctx, batch)
                else:
                    outputs = self.model(**batch)
                    loss = _extract_loss(outputs)
                    loss = compose_loss(self._ctx, batch, outputs, loss)
                    if loss is not None:
                        loss.backward()

                tokens_seen += int(batch["input_ids"].numel()) if "input_ids" in batch else 0

                if not all(algo.should_step(self._ctx) for algo in self.algorithms):
                    continue

                self._sophia_hessian_step(batch)

                for algo in self.algorithms:
                    algo.on_before_optimizer_step(self._ctx)

                grad_norm = self._grad_norm()
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler is not None:
                    self.scheduler.step()

                for algo in self.algorithms:
                    algo.on_step_end(self._ctx, loss)

                self.step += 1

                if self.config.log_every and self.step % self.config.log_every == 0:
                    self._log(loss, grad_norm, tokens_seen, started)
                for cb in callbacks or []:
                    cb(self)
                if self.config.save_every and self.step % self.config.save_every == 0:
                    self.save_checkpoint(os.path.join(self.config.output_dir or "agron-checkpoints", f"step-{self.step}"))

            if not stopped:
                epoch += 1

        self._call("on_train_end")
        self.model.eval()
        if self.config.output_dir:
            self.save_checkpoint(os.path.join(self.config.output_dir, "final"))
            write_json(os.path.join(self.config.output_dir, "train_report.json"), {
                "steps": self.step,
                "tokens": tokens_seen,
                "seconds": round(time.time() - started, 2),
                "optimizer": self.optimizer.__class__.__name__,
                "scheduler": self.scheduler.__class__.__name__ if self.scheduler is not None else None,
                "algorithms": [a.name for a in self.algorithms],
                "config": self.config.to_dict(),
                "history": self.history,
            })

        final_eval = None
        if eval_dataloader is not None:
            final_eval = self.evaluate(eval_dataloader)

        return {
            "steps": self.step,
            "tokens": tokens_seen,
            "history": self.history,
            "eval_loss": final_eval,
            "seconds": round(time.time() - started, 2),
        }

    # ------------------------------------------------------------------ #
    # Evaluation / inference helpers
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def evaluate(self, dataloader, max_batches: Optional[int] = None) -> float:
        was_training = self.model.training
        self.model.eval()
        if hasattr(self.optimizer, "eval") and self.optimizer.__class__.__name__ == "ScheduleFreeAdamW":
            self.optimizer.eval()
        total_loss, total_tokens = 0.0, 0
        max_len = int(getattr(self.model.config, "max_position_embeddings", 0) or 0)
        for i, batch in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break
            batch = _truncate_batch(_batch_to_device(batch, self.device), max_len)
            outputs = self.model(**batch)
            loss = _extract_loss(outputs)
            if loss is None:
                continue
            n = int(batch["labels"].numel()) if "labels" in batch else 1
            total_loss += float(loss.detach()) * n
            total_tokens += n
        if was_training:
            self.model.train()
            if hasattr(self.optimizer, "train") and self.optimizer.__class__.__name__ == "ScheduleFreeAdamW":
                self.optimizer.train()
        return total_loss / max(1, total_tokens)

    def _grad_norm(self) -> float:
        grads = [p.grad.detach() for p in self.model.parameters() if p.grad is not None]
        if not grads:
            return 0.0
        return float(torch.norm(torch.stack([g.norm(2) for g in grads]), 2))

    def _log(self, loss, grad_norm: float, tokens_seen: int, started: float) -> None:
        entry = {
            "step": self.step,
            "loss": float(loss.detach()) if torch.is_tensor(loss) else float("nan"),
            "lr": self.current_lr,
            "grad_norm": grad_norm,
            "tokens": tokens_seen,
        }
        elapsed = max(1e-9, time.time() - started)
        if self.config.log_tokens_per_second:
            entry["tokens_per_s"] = round(tokens_seen / elapsed, 1)
        self.history.append(entry)
        print(
            f"step {self.step:>6} | loss {entry['loss']:.4f} | lr {entry['lr']:.3e} "
            f"| grad {grad_norm:.3f} | {entry.get('tokens_per_s', 0):.0f} tok/s",
            flush=True,
        )

    def _sophia_hessian_step(self, batch) -> None:
        every = self.config.sophia_hessian_every
        if every <= 0 or not hasattr(self.optimizer, "update_hessian"):
            return
        if self.step % every != 0:
            return
        self.optimizer.zero_grad(set_to_none=True)
        with torch.enable_grad():
            outputs = self.model(**batch)
            loss = _extract_loss(outputs)
            if loss is not None and loss.requires_grad:
                self.optimizer.update_hessian(loss, seed=self.config.seed + self.step)
        self.optimizer.zero_grad(set_to_none=True)

    # ------------------------------------------------------------------ #
    # Checkpointing
    # ------------------------------------------------------------------ #
    def save_checkpoint(self, directory: str) -> str:
        os.makedirs(directory, exist_ok=True)
        self.model.save_pretrained(directory)
        state = {
            "step": self.step,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "algorithms": [
                {"name": a.name, "state": a.state_dict()} for a in self.algorithms
            ],
            "config": self.config.to_dict(),
            "history": self.history,
        }
        torch.save(state, os.path.join(directory, "trainer_state.pt"))
        return directory

    def load_checkpoint(self, directory: str) -> None:
        path = os.path.join(directory, "trainer_state.pt")
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.step = int(state.get("step", 0))
        if state.get("optimizer") is not None:
            self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler is not None and state.get("scheduler") is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        algo_state = {item["name"]: item["state"] for item in state.get("algorithms", [])}
        for algo in self.algorithms:
            if algo.name in algo_state:
                algo.load_state_dict(algo_state[algo.name])
        self.history = state.get("history", [])


def train_architecture(
    architecture: str = "llama",
    config: Optional[Dict[str, Any]] = None,
    train_dataloader=None,
    eval_dataloader=None,
    optimizer: str = "adamw",
    scheduler: str = "cosine",
    algorithms: Optional[Iterable[str]] = None,
    trainer_config: Optional[TrainerConfig] = None,
    device: Optional[Any] = None,
) -> Dict[str, Any]:
    """One-liner: build an architecture, train it, return the report."""
    from ..architectures import build_architecture

    model, _ = build_architecture(architecture, **(config or {}))
    trainer = AgronTrainer(
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        algorithms=algorithms,
        config=trainer_config,
        device=device,
    )
    return trainer.fit(train_dataloader, eval_dataloader)


__all__ = [
    "AgronTrainer",
    "TrainerConfig",
    "train_architecture",
]
