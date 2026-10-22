"""Registry of every neural network architecture shipped with Agron.

The registry is the single source of truth used by the Gradio app, the
training scripts and the test-suite.  Every entry exposes a config class, a
model class and a short human readable description.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

from .base import AgronConfig, AgronPreTrainedModel
from .conv import AgronConvConfig, AgronConvForCausalLM
from .hybrid import AgronHybridConfig, AgronHybridForCausalLM
from .llama import AgronLlamaConfig, AgronLlamaForCausalLM, build_llama_config
from .mamba import AgronMambaConfig, AgronMambaForCausalLM
from .mixer import AgronMixerConfig, AgronMixerForCausalLM
from .rnn import AgronGRUConfig, AgronGRUForCausalLM, AgronLSTMConfig, AgronLSTMForCausalLM
from .transformer import (
    AgronHyperConfig,
    AgronHyperForCausalLM,
    AgronMoEConfig,
    AgronMoEForCausalLM,
    AgronTransformerConfig,
    AgronTransformerForCausalLM,
)
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
ARCHITECTURES: Dict[str, Dict[str, Any]] = {
    "llama": {
        "name": "llama",
        "label": "LLaMA (reference)",
        "family": "transformer",
        "description": "Stock LLaMA decoder. Compatible with the published Agron weights.",
        "config_cls": LlamaConfig,
        "model_cls": LlamaForCausalLM,
        "attention": True,
        "recurrent": False,
    },
    "transformer": {
        "name": "transformer",
        "label": "Transformer (native)",
        "family": "transformer",
        "description": "Decoder-only transformer with RoPE, RMSNorm and SwiGLU written from scratch.",
        "config_cls": AgronTransformerConfig,
        "model_cls": AgronTransformerForCausalLM,
        "attention": True,
        "recurrent": False,
    },
    "moe": {
        "name": "moe",
        "label": "Mixture-of-Experts",
        "family": "transformer",
        "description": "Sparse top-k routed experts with a shared expert and load-balancing loss.",
        "config_cls": AgronMoEConfig,
        "model_cls": AgronMoEForCausalLM,
        "attention": True,
        "recurrent": False,
    },
    "hybrid": {
        "name": "hybrid",
        "label": "Hybrid attention + SSM",
        "family": "hybrid",
        "description": "Interleaves causal attention layers with Mamba-lite state-space layers.",
        "config_cls": AgronHybridConfig,
        "model_cls": AgronHybridForCausalLM,
        "attention": True,
        "recurrent": True,
    },
    "mamba": {
        "name": "mamba",
        "label": "Mamba-lite (pure SSM)",
        "family": "state-space",
        "description": "Pure selective state-space decoder with a stable parallel scan.",
        "config_cls": AgronMambaConfig,
        "model_cls": AgronMambaForCausalLM,
        "attention": False,
        "recurrent": True,
    },
    "hyper": {
        "name": "hyper",
        "label": "Hypernetwork transformer",
        "family": "transformer",
        "description": "Transformer whose blocks are modulated by a hypernetwork conditioned on a task embedding.",
        "config_cls": AgronHyperConfig,
        "model_cls": AgronHyperForCausalLM,
        "attention": True,
        "recurrent": False,
    },
    "gru": {
        "name": "gru",
        "label": "GRU decoder",
        "family": "recurrent",
        "description": "Stacked GRU language model.",
        "config_cls": AgronGRUConfig,
        "model_cls": AgronGRUForCausalLM,
        "attention": False,
        "recurrent": True,
    },
    "lstm": {
        "name": "lstm",
        "label": "LSTM decoder",
        "family": "recurrent",
        "description": "Stacked LSTM language model.",
        "config_cls": AgronLSTMConfig,
        "model_cls": AgronLSTMForCausalLM,
        "attention": False,
        "recurrent": True,
    },
    "conv": {
        "name": "conv",
        "label": "Causal convolution decoder",
        "family": "convolutional",
        "description": "Depthwise separable causal convolutions with a gated activation.",
        "config_cls": AgronConvConfig,
        "model_cls": AgronConvForCausalLM,
        "attention": False,
        "recurrent": False,
    },
    "mixer": {
        "name": "mixer",
        "label": "Mixer decoder",
        "family": "mlp-mixer",
        "description": "Causal token-mixing convolution plus channel-mixing MLP (length-agnostic Mixer).",
        "config_cls": AgronMixerConfig,
        "model_cls": AgronMixerForCausalLM,
        "attention": False,
        "recurrent": False,
    },
}

DEFAULT_ARCHITECTURE = "llama"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def architecture_names() -> List[str]:
    """Sorted list of registered architecture keys."""
    return sorted(ARCHITECTURES.keys())


def get_architecture(name: str) -> Dict[str, Any]:
    """Look up a registry entry (raises ``KeyError`` when unknown)."""
    key = (name or "").strip().lower()
    if key not in ARCHITECTURES:
        raise KeyError(
            f"unknown architecture '{name}'. available: {', '.join(architecture_names())}"
        )
    return ARCHITECTURES[key]


def architecture_label(name: str) -> str:
    return get_architecture(name)["label"]


def architecture_description(name: str) -> str:
    return get_architecture(name)["description"]


def build_config(name: str, **overrides) -> Any:
    """Instantiate the config for ``name`` with Agron defaults + ``overrides``."""
    entry = get_architecture(name)
    config_cls = entry["config_cls"]
    if config_cls is LlamaConfig:
        cfg = build_llama_config(**overrides)
    else:
        cfg = config_cls(**overrides)
    return cfg


def build_architecture(name: str, **overrides):
    """Build a fresh (randomly initialised) model for ``name``."""
    entry = get_architecture(name)
    config = build_config(name, **overrides)
    model = entry["model_cls"](config)
    return model, config


def architecture_report(model_kwargs: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Return a description of every architecture (used by the UI + docs)."""
    model_kwargs = model_kwargs or {}
    report = []
    for key in architecture_names():
        entry = ARCHITECTURES[key]
        report.append(
            {
                "name": key,
                "label": entry["label"],
                "family": entry["family"],
                "description": entry["description"],
                "attention": entry["attention"],
                "recurrent": entry["recurrent"],
            }
        )
    return report


def register_auto_classes() -> None:
    """Make every Agron architecture loadable through the HF ``Auto*`` classes.

    Safe to call multiple times.  After this, ``AutoModelForCausalLM`` and
    ``AutoConfig`` can load Agron checkpoints directly.
    """
    for entry in ARCHITECTURES.values():
        config_cls = entry["config_cls"]
        model_cls = entry["model_cls"]
        model_type = getattr(config_cls, "model_type", None)

        if model_type:
            try:
                AutoConfig.register(model_type, config_cls, exist_ok=True)
            except Exception:
                pass

        for auto_cls in (AutoModelForCausalLM, AutoModel):
            try:
                mapping = getattr(auto_cls, "_model_mapping", None)
                if mapping is not None and config_cls in mapping:
                    continue
                auto_cls.register(config_cls, model_cls, exist_ok=True)
            except Exception:
                pass


register_auto_classes()


__all__ = [
    "ARCHITECTURES",
    "DEFAULT_ARCHITECTURE",
    "AgronConfig",
    "AgronPreTrainedModel",
    "AgronLlamaConfig",
    "AgronLlamaForCausalLM",
    "architecture_names",
    "architecture_label",
    "architecture_description",
    "architecture_report",
    "build_architecture",
    "build_config",
    "get_architecture",
    "register_auto_classes",
]
