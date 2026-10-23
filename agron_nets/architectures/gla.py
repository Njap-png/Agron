"""Gated Linear Attention (GLA) decoder.

Each head keeps a matrix state that is decayed by a *data-dependent* per-channel
gate and updated with the outer product of the current key and value:

    S_t = diag(alpha_t) S_{t-1} + k_t^T v_t
    o_t = q_t S_t

with ``alpha_t = sigmoid(g_t)`` in (0, 1).  The forward pass is evaluated
chunkwise: inside a chunk the causal sum is a cumulative sum, and the state is
carried across chunks.  Gates are expressed relative to the first position of
the chunk so every exponential is bounded by 1, which keeps the recurrence
numerically stable for long sequences.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers.generation import GenerationMixin

from .base import AgronConfig, AgronPreTrainedModel, SwiGLU, build_norm, causal_lm_loss

CHUNK = 16


class AgronGLAConfig(AgronConfig):
    model_type = "agron_gla"

    def __init__(
        self,
        norm_type: str = "rms",
        chunk_size: int = CHUNK,
        gate_floor: float = 1e-4,
        gate_ceiling: float = 1.0 - 1e-4,
        **kwargs,
    ):
        self.norm_type = norm_type
        self.chunk_size = chunk_size
        self.gate_floor = gate_floor
        self.gate_ceiling = gate_ceiling
        super().__init__(**kwargs)


class GatedLinearAttention(nn.Module):
    def __init__(self, config: AgronGLAConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.chunk_size = config.chunk_size
        self.gate_floor = config.gate_floor
        self.gate_ceiling = config.gate_ceiling

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.g_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.out_gate = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.out_norm = build_norm(config.hidden_size, config.rms_norm_eps, "rms")

    def _split(self, x):
        b, t, _ = x.shape
        return x.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        heads, dim = self.num_heads, self.head_dim
        chunk = self.chunk_size

        q = self._split(self.q_proj(hidden_states))
        k = self._split(self.k_proj(hidden_states))
        v = self._split(self.v_proj(hidden_states))

        alpha = torch.sigmoid(self._split(self.g_proj(hidden_states)))
        alpha = alpha.clamp(self.gate_floor, self.gate_ceiling)
        log_alpha = alpha.log()

        state = torch.zeros(bsz, heads, dim, dim, device=hidden_states.device, dtype=hidden_states.dtype)
        out = torch.empty_like(v)

        for start in range(0, seq_len, chunk):
            end = min(start + chunk, seq_len)
            n = end - start
            qc, kc, vc = q[:, :, start:end], k[:, :, start:end], v[:, :, start:end]
            log_c = torch.cumsum(log_alpha[:, :, start:end], dim=2)  # (B,H,C,D), <= 0

            # Relative gates: exp(c_0 - c_j) <= 1 for every j.
            rel = torch.exp(log_c[:, :, :1] - log_c)                 # (B,H,C,D)
            kv = kc.unsqueeze(-1) * vc.unsqueeze(-2)                 # (B,H,C,D,D)
            cum_kv = torch.cumsum(rel.unsqueeze(-1) * kv, dim=2)     # (B,H,C,Dk,Dv)

            # intra-chunk read-out: q_t * exp(c_t - c_0) applied per key channel
            q_intra = qc * torch.exp(log_c - log_c[:, :, :1])
            intra = torch.einsum("bhck,bhckv->bhcv", q_intra, cum_kv)

            # inter-chunk contribution from the carried state, decayed by exp(c_t)
            q_inter = qc * torch.exp(log_c)
            inter = torch.einsum("bhck,bhkv->bhcv", q_inter, state)

            out[:, :, start:end] = intra + inter

            # roll the state to the end of the chunk
            c_last = log_c[:, :, -1]                                 # (B,H,D)
            state = state * torch.exp(c_last).unsqueeze(-1) + cum_kv[:, :, -1] * torch.exp(
                c_last - log_c[:, :, 0]
            ).unsqueeze(-1)

        out = out * torch.sigmoid(self.out_gate(hidden_states)).view(
            bsz, seq_len, heads, dim
        ).transpose(1, 2)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        return self.o_proj(self.out_norm(out))


class GLABlock(nn.Module):
    def __init__(self, config: AgronGLAConfig):
        super().__init__()
        self.linear_attention = GatedLinearAttention(config)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.input_layernorm = build_norm(config.hidden_size, config.rms_norm_eps, config.norm_type)
        self.post_attention_layernorm = build_norm(config.hidden_size, config.rms_norm_eps, config.norm_type)

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.linear_attention(self.input_layernorm(hidden_states))
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class AgronGLAForCausalLM(AgronPreTrainedModel, GenerationMixin):
    config_class = AgronGLAConfig
    base_model_prefix = "agron_gla"
    supports_gradient_checkpointing = True

    def __init__(self, config: AgronGLAConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([GLABlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = build_norm(config.hidden_size, config.rms_norm_eps, config.norm_type)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, inputs_embeds=None,
                labels=None, label_smoothing: float = 0.0, **kwargs):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        loss = causal_lm_loss(logits, labels, label_smoothing) if labels is not None else None
        return {"logits": logits, "loss": loss}
