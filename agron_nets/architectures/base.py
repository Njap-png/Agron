"""Base classes and shared building blocks for every Agron architecture.

All Agron networks are Hugging Face compatible: they subclass
``PreTrainedModel``, own a ``PretrainedConfig`` subclass and therefore support
``save_pretrained`` / ``from_pretrained`` / ``generate`` out of the box.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel

from ..utils import count_parameters, format_parameters, init_weights


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
class AgronConfig(PretrainedConfig):
    """Base configuration shared by the Agron architecture family."""

    model_type = "agron_base"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 16384,
        hidden_size: int = 512,
        intermediate_size: int = 2048,
        num_hidden_layers: int = 8,
        num_attention_heads: int = 4,
        num_key_value_heads: int = 4,
        max_position_embeddings: int = 512,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        hidden_act: str = "silu",
        tie_word_embeddings: bool = False,
        attention_bias: bool = False,
        mlp_bias: bool = False,
        initializer_range: float = 0.02,
        bos_token_id: int = 0,
        eos_token_id: int = 2,
        pad_token_id: int = 1,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.hidden_act = hidden_act
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        self.initializer_range = initializer_range
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


# --------------------------------------------------------------------------- #
# Shared layers
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation (used by every LLaMA-style block)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states.to(dtype))


class LayerNorm(nn.Module):
    """Standard layer norm with an optional bias (used by non-RMS variants)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size)) if bias else None
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.variance_epsilon)


def build_norm(hidden_size: int, eps: float, kind: str = "rms", bias: bool = False) -> nn.Module:
    """Factory for the normalisation layer used inside a block."""
    kind = (kind or "rms").lower()
    if kind in ("rms", "rmsnorm"):
        return RMSNorm(hidden_size, eps)
    if kind in ("layer", "layernorm", "ln"):
        return LayerNorm(hidden_size, eps, bias=bias)
    raise ValueError(f"unknown norm kind: {kind}")


class RotaryEmbedding(nn.Module):
    """Standard RoPE implementation (self-contained, version independent).

    ``inv_freq`` is deliberately *not* a buffer: Hugging Face v5 materialises
    fresh models on the meta device and copies parameters with ``to_empty()``,
    which leaves non-persistent buffers full of uninitialised memory. Deriving
    the frequencies from plain Python ints keeps the module correct no matter
    how it is instantiated, moved or loaded.
    """

    def __init__(self, dim: int, max_position_embeddings: int = 512, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.theta = float(theta)
        self._cache: Optional[torch.Tensor] = None
        self._cache_sin: Optional[torch.Tensor] = None
        self._cache_key: Tuple = ()

    def inv_freq(self, device=None, dtype=torch.float32) -> torch.Tensor:
        exponents = torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim
        return 1.0 / (self.theta ** exponents).to(dtype)

    def _cos_sin(self, seq_len: int, device, dtype):
        # ``forward`` rotates the two halves of the head dimension against each
        # other, so only ``dim / 2`` frequencies are needed here.
        key = (seq_len, str(device), dtype)
        if self._cache is None or self._cached_len < seq_len or self._cache_key[1:] != key[1:]:
            t = torch.arange(seq_len, device=device, dtype=torch.float32)
            freqs = torch.outer(t, self.inv_freq(device))
            self._cache = freqs.cos().to(dtype)
            self._cache_sin = freqs.sin().to(dtype)
            self._cached_len = seq_len
            self._cache_key = key
        return self._cache[:seq_len], self._cache_sin[:seq_len]

    def forward(self, x: torch.Tensor, position_ids: Optional[torch.Tensor] = None):
        """Rotate ``x`` (B, H, T, D) according to ``position_ids`` (B, T)."""
        seq_len = x.shape[-2]
        cos, sin = self._cos_sin(seq_len, x.device, x.dtype)
        if position_ids is not None:
            cos = cos[position_ids].unsqueeze(1)  # (B, 1, T, D)
            sin = sin[position_ids].unsqueeze(1)
        else:
            cos = cos[:seq_len].view(1, 1, seq_len, -1)
            sin = sin[:seq_len].view(1, 1, seq_len, -1)
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class SwiGLU(nn.Module):
    """Gated GELU/SiLU feed-forward block."""

    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE and grouped KV heads."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        max_position_embeddings: int = 512,
        rope_theta: float = 10000.0,
        bias: bool = False,
    ):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads

        self.q_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_attention_heads * self.head_dim, hidden_size, bias=bias)
        self.rotary = RotaryEmbedding(self.head_dim, max_position_embeddings, rope_theta)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        q = self.rotary(q, position_ids)
        k = self.rotary(k, position_ids)

        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        # additive mask: 0 -> keep, -inf -> masked
        mask = None
        if attention_mask is not None:
            mask = attention_mask[:, None, None, :].to(q.dtype)
            mask = (1.0 - mask) * torch.finfo(q.dtype).min
            causal = torch.full((seq_len, seq_len), torch.finfo(q.dtype).min, device=q.device, dtype=q.dtype)
            causal = torch.triu(causal, diagonal=1)[None, None, :, :]
            mask = mask + causal
        else:
            causal = torch.full((seq_len, seq_len), torch.finfo(q.dtype).min, device=q.device, dtype=q.dtype)
            mask = torch.triu(causal, diagonal=1)[None, None, :, :]

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(out)


# --------------------------------------------------------------------------- #
# Base model
# --------------------------------------------------------------------------- #
class AgronPreTrainedModel(PreTrainedModel, GenerationMixin):
    """Common base for all Agron causal language models."""

    config_class = AgronConfig
    base_model_prefix = "agron"
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _supports_flash_attn_2 = False
    _supports_attention_backend = False

    def _init_weights(self, module: nn.Module) -> None:
        init_weights(module, getattr(self.config, "initializer_range", 0.02))

    # -- embeddings ------------------------------------------------------- #
    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.embed_tokens = value

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, value: nn.Module) -> None:
        self.lm_head = value

    # -- helpers ---------------------------------------------------------- #
    def num_parameters(self, trainable_only: bool = False) -> int:
        return count_parameters(self, trainable_only)

    def parameter_summary(self) -> str:
        return f"{format_parameters(self.num_parameters())} parameters"

    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        shift_labels: bool = True,
        label_smoothing: float = 0.0,
    ) -> Optional[torch.Tensor]:
        """Standard next-token cross entropy with optional label smoothing."""
        if labels is None:
            return None
        if shift_labels:
            logits = logits[..., :-1, :]
            labels = labels[..., 1:]
        logits = logits.reshape(-1, logits.shape[-1]).float()
        labels = labels.reshape(-1)
        if label_smoothing > 0.0:
            return F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
        return F.cross_entropy(logits, labels)

    # -- generation ------------------------------------------------------- #
    @staticmethod
    def _apply_repetition_penalty(logits: torch.Tensor, sequences: torch.Tensor, penalty: float) -> torch.Tensor:
        if penalty == 1.0:
            return logits
        score = torch.gather(logits, 1, sequences)
        score = torch.where(score < 0, score * penalty, score / penalty)
        return logits.scatter(1, sequences, score)

    @staticmethod
    def _apply_no_repeat_ngram(logits: torch.Tensor, sequences: torch.Tensor, n: int) -> torch.Tensor:
        if n <= 0:
            return logits
        for b in range(sequences.shape[0]):
            seq = sequences[b].tolist()
            if len(seq) < n:
                continue
            banned = {}
            for i in range(len(seq) - n + 1):
                key = tuple(seq[i:i + n - 1])
                banned.setdefault(key, set()).add(seq[i + n - 1])
            prefix = tuple(seq[-(n - 1):]) if n > 1 else ()
            for tok in banned.get(prefix, ()):  # type: ignore[arg-type]
                logits[b, tok] = torch.finfo(logits.dtype).min
        return logits

    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        max_new_tokens: int = 64,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Uniform, architecture-agnostic sampling loop.

        Every Agron architecture (transformer, SSM, RNN, conv, mixer) supports
        the same generation contract, so the runtime never has to care which
        network is currently loaded.
        """
        if input_ids is None:
            raise ValueError("`input_ids` is required for generation")
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        device = input_ids.device
        pad_token_id = pad_token_id if pad_token_id is not None else getattr(self.config, "pad_token_id", 1)
        eos_token_id = eos_token_id if eos_token_id is not None else getattr(self.config, "eos_token_id", 2)

        sequences = input_ids
        finished = torch.zeros(sequences.shape[0], dtype=torch.bool, device=device)
        max_len = sequences.shape[1] + max(1, max_new_tokens)
        max_position = getattr(self.config, "max_position_embeddings", 512)

        for _ in range(max_new_tokens):
            window = sequences
            if window.shape[1] > max_position and max_position > 0:
                window = window[:, -max_position:]
            mask = None
            if attention_mask is not None:
                mask = attention_mask
                if mask.shape[1] < window.shape[1]:
                    mask = mask[:, -window.shape[1]:]
            logits = self(input_ids=window, attention_mask=mask)
            logits = logits["logits"] if isinstance(logits, dict) else logits.logits
            next_logits = logits[:, -1, :].float()

            if repetition_penalty != 1.0:
                next_logits = self._apply_repetition_penalty(next_logits, sequences, repetition_penalty)
            if no_repeat_ngram_size > 0:
                next_logits = self._apply_no_repeat_ngram(next_logits, sequences, no_repeat_ngram_size)

            if do_sample:
                temperature = max(temperature, 1e-5)
                next_logits = next_logits / temperature
                if top_k and top_k > 0:
                    k = min(top_k, next_logits.shape[-1])
                    threshold = torch.topk(next_logits, k, dim=-1).values[:, -1:]
                    next_logits = next_logits.masked_fill(next_logits < threshold, torch.finfo(next_logits.dtype).min)
                if top_p is not None and 0 < top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(next_logits, descending=True, dim=-1)
                    cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                    remove = cumulative - torch.softmax(sorted_logits, dim=-1) > top_p
                    sorted_logits = sorted_logits.masked_fill(remove, torch.finfo(next_logits.dtype).min)
                    next_logits = torch.full_like(next_logits, torch.finfo(next_logits.dtype).min).scatter(
                        -1, sorted_idx, sorted_logits
                    )
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

            next_token = torch.where(
                finished.unsqueeze(1), torch.full_like(next_token, pad_token_id), next_token
            )
            sequences = torch.cat([sequences, next_token], dim=1)
            if eos_token_id is not None:
                finished = finished | next_token.squeeze(1).eq(eos_token_id)
            if bool(finished.all()):
                break
        return sequences

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids, "attention_mask": kwargs.get("attention_mask")}


def causal_lm_loss(
    logits: torch.Tensor,
    labels: Optional[torch.Tensor],
    label_smoothing: float = 0.0,
) -> Optional[torch.Tensor]:
    """Helper mirroring :meth:`AgronPreTrainedModel._compute_loss`."""
    if labels is None:
        return None
    logits = logits[..., :-1, :]
    labels = labels[..., 1:]
    logits = logits.reshape(-1, logits.shape[-1]).float()
    labels = labels.reshape(-1)
    if label_smoothing > 0:
        return F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
    return F.cross_entropy(logits, labels)
