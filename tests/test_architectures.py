"""Architecture zoo tests: every network forwards, backwards and round-trips."""

from __future__ import annotations

import pytest
import torch

from agron_nets.architectures import ARCHITECTURES, build_architecture, build_config

from .conftest import make_batch, make_model


@pytest.mark.parametrize("arch", sorted(ARCHITECTURES))
def test_forward_backward(arch):
    model, config = make_model(arch)
    model.train()
    batch = make_batch(vocab_size=config.vocab_size)
    out = model(**batch)

    logits = out["logits"] if isinstance(out, dict) else out.logits
    loss = out["loss"] if isinstance(out, dict) else out.loss
    assert logits.shape == (2, 8, config.vocab_size)
    assert torch.isfinite(loss)

    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


@pytest.mark.parametrize("arch", sorted(ARCHITECTURES))
def test_generate(arch):
    model, config = make_model(arch)
    model.eval()
    prompt = torch.randint(0, config.vocab_size, (1, 5))
    with torch.no_grad():
        greedy = model.generate(prompt, max_new_tokens=4, do_sample=False,
                                eos_token_id=config.eos_token_id)
        sampled = model.generate(prompt, max_new_tokens=4, do_sample=True,
                                 temperature=0.9, top_k=10, top_p=0.95,
                                 eos_token_id=config.eos_token_id)
    assert greedy.shape[0] == 1
    assert 5 <= greedy.shape[1] <= 9
    assert sampled.shape[0] == 1
    assert 5 <= sampled.shape[1] <= 9


@pytest.mark.parametrize("arch", sorted(ARCHITECTURES))
def test_save_and_reload(arch, tmp_path):
    model, config = make_model(arch)
    model.eval()
    batch = make_batch(vocab_size=config.vocab_size)
    with torch.no_grad():
        before = model(**batch)["logits"]

    model.save_pretrained(tmp_path)
    reloaded = type(model).from_pretrained(tmp_path)
    if isinstance(reloaded, tuple):  # transformers < 5
        reloaded = reloaded[0]
    reloaded.eval()
    with torch.no_grad():
        after = reloaded(**batch)["logits"]
    assert torch.allclose(before, after, atol=1e-5)


@pytest.mark.parametrize("arch", sorted(ARCHITECTURES))
def test_config_is_hf_compatible(arch):
    config = build_config(arch)
    payload = config.to_dict()
    assert payload["model_type"] == config.model_type
    assert config.vocab_size == 16384
