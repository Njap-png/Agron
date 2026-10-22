"""End-to-end smoke test for every registered Agron architecture.

For each architecture this exercises: construction, forward, backward,
generation (greedy + sampled) and ``save_pretrained`` / ``from_pretrained``.

Usage::

    python -m agron_nets.smoke_test
    python -m agron_nets.smoke_test transformer mamba
"""

from __future__ import annotations

import sys
import tempfile
import traceback

import torch

from .architectures import ARCHITECTURES, build_architecture


def run(name: str, device: str = "cpu") -> None:
    cfg = dict(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    model, config = build_architecture(name, **cfg)
    model.to(device)
    model.train()

    input_ids = torch.randint(0, cfg["vocab_size"], (2, 16), device=device)
    labels = torch.randint(0, cfg["vocab_size"], (2, 16), device=device)

    out = model(input_ids=input_ids, labels=labels)
    logits, loss = out["logits"], out["loss"]
    assert logits.shape == (2, 16, cfg["vocab_size"]), logits.shape
    assert loss is not None and torch.isfinite(loss), loss
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients produced"

    model.eval()
    with torch.no_grad():
        greedy = model.generate(input_ids[:1, :4], max_new_tokens=5, do_sample=False)
        assert greedy.shape == (1, 9), greedy.shape
        sampled = model.generate(
            input_ids[:1, :4], max_new_tokens=5, do_sample=True, temperature=0.8, top_k=20, top_p=0.9
        )
        assert sampled.shape == (1, 9), sampled.shape

        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            reloaded = type(model).from_pretrained(tmp)
            if isinstance(reloaded, tuple):  # transformers < 5
                reloaded = reloaded[0]
            reloaded.to(device).eval()
            assert reloaded.config.model_type == config.model_type
            again = reloaded(input_ids=input_ids)
            assert torch.allclose(again["logits"], model(input_ids=input_ids)["logits"], atol=1e-5)


def main(argv):
    names = argv or list(ARCHITECTURES)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    passed, failed = [], {}
    for name in names:
        try:
            run(name, device)
            passed.append(name)
            print(f"PASS  {name}", flush=True)
        except Exception as exc:  # noqa: BLE001
            failed[name] = exc
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
    print(f"\n{len(passed)} passed, {len(failed)} failed")
    print("passed:", " ".join(passed))
    if failed:
        print("failed:", " ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
