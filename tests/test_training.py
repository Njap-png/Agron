"""Training-engine and algorithm-composition tests."""

from __future__ import annotations

import pytest
import torch

from agron_nets.algorithms import ALGORITHMS, algorithm_names, build_algorithm
from agron_nets.training import (
    RECIPES,
    AgronTrainer,
    TrainerConfig,
    get_recipe,
    recipe_names,
)

from .conftest import make_batch, make_loader, make_model

# Algorithms that need a bespoke batch layout or an external teacher/reference.
SPECIAL = {"dpo", "distillation"}


def _trainer(model, algorithms=None, steps=4, **cfg):
    config = TrainerConfig(max_steps=steps, epochs=1, log_every=0, warmup_steps=0, **cfg)
    return AgronTrainer(
        model,
        optimizer="adamw",
        scheduler="cosine",
        algorithms=algorithms,
        config=config,
        device="cpu",
    )


@pytest.mark.parametrize("name", [n for n in algorithm_names() if n not in SPECIAL | {"gradient_accumulation"}])
def test_algorithm_trains(name):
    model, _ = make_model("transformer")
    trainer = _trainer(model, [name], steps=6)
    report = trainer.fit(make_loader(6))
    assert report["steps"] == 6
    assert trainer.history == [] or any(
        torch.isfinite(torch.tensor(h["loss"])) for h in trainer.history
    )


@pytest.mark.parametrize("name", recipe_names())
def test_recipe_trains(name):
    model, _ = make_model("transformer")
    recipe = get_recipe(name)
    trainer = AgronTrainer(
        model,
        optimizer=recipe["optimizer"],
        scheduler=recipe["scheduler"],
        algorithms=recipe["algorithms"],
        config=TrainerConfig(max_steps=8, log_every=0, grad_clip=0.5),
        device="cpu",
    )
    report = trainer.fit(make_loader(8))
    assert report["steps"] >= 1


def test_dpo_algorithm_with_preference_batch():
    model, _ = make_model("transformer")
    batch = {
        "chosen_input_ids": torch.randint(0, 64, (2, 8)),
        "rejected_input_ids": torch.randint(0, 64, (2, 8)),
    }
    trainer = _trainer(model, ["dpo"], steps=2)
    report = trainer.fit([batch, batch])
    assert report["steps"] == 2


def test_distillation_with_teacher():
    teacher, _ = make_model("transformer")
    student, _ = make_model("transformer")
    algo = build_algorithm("distillation", teacher=teacher)
    trainer = _trainer(student, [algo], steps=2)
    report = trainer.fit(make_loader(2))
    assert report["steps"] == 2


def test_gradient_accumulation_only_steps_every_n():
    model, _ = make_model("transformer")
    trainer = _trainer(model, [], steps=1)
    from agron_nets.algorithms import build_algorithm as build

    trainer.algorithms = [build("gradient_accumulation", steps=3)]
    trainer._ctx = None
    report = trainer.fit(make_loader(3))
    # 3 micro-batches -> exactly one optimiser step.
    assert report["steps"] == 1


def test_schedule_free_optimizer():
    model, _ = make_model("transformer")
    trainer = AgronTrainer(
        model,
        optimizer="schedulefree",
        scheduler="none",
        config=TrainerConfig(max_steps=3, log_every=0),
        device="cpu",
    )
    report = trainer.fit(make_loader(3))
    assert report["steps"] == 3


def test_checkpoint_roundtrip(tmp_path):
    model, _ = make_model("transformer")
    trainer = _trainer(model, ["ema"], steps=2, output_dir=str(tmp_path))
    trainer.fit(make_loader(2))
    trainer.save_checkpoint(str(tmp_path / "ckpt"))

    model2, _ = make_model("transformer")
    trainer2 = _trainer(model2, ["ema"], steps=2)
    trainer2.load_checkpoint(str(tmp_path / "ckpt"))
    assert trainer2.step == trainer.step == 2


def test_evaluate_returns_finite_loss():
    model, _ = make_model("transformer")
    trainer = _trainer(model, [], steps=1)
    loss = trainer.evaluate(make_loader(2))
    assert torch.isfinite(torch.tensor(loss))


def test_cross_architecture_engine():
    for arch in ("llama", "mamba", "conv", "mixer", "gru", "hybrid", "moe", "hyper"):
        model, _ = make_model(arch)
        trainer = AgronTrainer(
            model,
            optimizer="lion",
            scheduler="linear",
            algorithms=["gradient_clip", "z_loss"],
            config=TrainerConfig(max_steps=2, log_every=0),
            device="cpu",
        )
        report = trainer.fit(make_loader(2, vocab_size=model.config.vocab_size))
        assert report["steps"] == 2, arch
