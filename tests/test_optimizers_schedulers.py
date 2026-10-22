"""Optimiser and scheduler registry tests."""

from __future__ import annotations

import copy

import pytest
import torch

from agron_nets.optimizers import OPTIMIZERS, build_optimizer, optimizer_names
from agron_nets.schedulers import SCHEDULERS, build_scheduler, scheduler_names

from .conftest import make_model


@pytest.mark.parametrize("name", optimizer_names())
def test_optimizer_takes_a_step(name):
    model, _ = make_model("transformer")
    model.train()
    params = list(model.parameters())
    before = [p.detach().clone() for p in params]

    opt = build_optimizer(name, params)
    input_ids = torch.randint(0, 64, (2, 8))
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        loss = model(input_ids=input_ids, labels=input_ids)["loss"]
        loss.backward()
        opt.step()

    assert torch.isfinite(loss)
    changed = [i for i, p in enumerate(params) if not torch.equal(before[i], p.detach())]
    assert changed, f"{name} did not update any parameter"


@pytest.mark.parametrize("name", optimizer_names())
def test_optimizer_state_dict_roundtrip(name):
    model, _ = make_model("transformer")
    params = list(model.parameters())
    opt = build_optimizer(name, params)
    ids = torch.randint(0, 64, (2, 8))
    loss = model(input_ids=ids, labels=ids)["loss"]
    loss.backward()
    opt.step()
    state = opt.state_dict()
    opt2 = build_optimizer(name, params)
    opt2.load_state_dict(copy.deepcopy(state))
    assert len(opt2.param_groups) == len(opt.param_groups)


@pytest.mark.parametrize("name", scheduler_names())
def test_scheduler_updates_lr(name):
    model, _ = make_model("transformer")
    opt = build_optimizer("adamw", list(model.parameters()), lr=1e-3)
    sched = build_scheduler(opt, name, total_steps=10, warmup_steps=2)
    seen = []
    for _ in range(10):
        seen.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert all(torch.isfinite(torch.tensor(v)) for v in seen)
    assert any(abs(v - seen[0]) > 1e-9 for v in seen), f"{name} never changed the LR"
