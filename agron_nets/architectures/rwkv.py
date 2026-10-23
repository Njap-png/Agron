"""RWKV-4 style decoder: WKV time-mixing with chunked linear recurrence.

Each head keeps a weighted key-value state

    S_t = sum_{j<=t} exp(k_j + w * (j - t)) * v_j

and the "receptance" vector gates the read-out. ``w`` is a learned per-channel
decay (0 < w < 1) and ``u`` is the per-channel bonus applied to the current
token. The implementation below evaluates the recurrence in chunks with
cumulative sums, so it is exact and does not need a per-token Python loop.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers.generation import GenerationMixin

from .base import AgronConfig, AgronPreTrainedModel, SwiGLU, build_norm, causal_lm_loss

CHUNK = 16


class AgronRWKVConfig(AgronConfig):
    model_type = "agron_rwkv"

    def __init__(self, norm_type: str = "rms", chunk_size: int = CHUNK, **kwargs):
        self.norm_type = norm_type
        self.chunk_size = chunk_size
        super().__init__(**kwargs)


class WKVTimeMix(nn.Module):
    def __init__(self, config: AgronRWKVConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.chunk_size = config.chunk_size

        self.r_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.out_norm = build_norm(config.hidden_size, config.rms_norm_eps, "rms")

        # w in (0, 1): w = exp(-exp(w_param)); u is the current-token bonus.
        self.w_param = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
        self.u_param = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))

    def _split(self, x):
        b, t, _ = x.shape
        return x.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

    def _wkv(self, r, k, v):
        bsz, heads, seq_len, dim = k.shape
        device, dtype = k.device, k.dtype
        chunk = self.chunk_size

        w = torch.exp(-torch.exp(self.w_param.float())).to(dtype).view(1, heads, 1, dim)
        u = torch.exp(torch.clamp(self.u_param.float(), -10, 10)).to(dtype).view(1, heads, 1, dim)

        state_num = torch.zeros(bsz, heads, dim, dim, device=device, dtype=dtype)
        state_den = torch.zeros(bsz, heads, dim, device=device, dtype=dtype)
        out = torch.empty_like(v)

        k = torch.clamp(k, -10.0, 10.0)
        for start in range(0, seq_len, chunk):
            end = min(start + chunk, seq_len)
            n = end - start
            kc = k[:, :, start:end]          # (B,H,C,D)
            vc = v[:, :, start:end]
            pos = torch.arange(n, device=device, dtype=dtype).view(1, 1, n, 1)

            # within-chunk unnormalised weights exp(k_j + w*j)
            a = torch.exp(kc + w * pos)                     # (B,H,C,D)
            cum_num = torch.cumsum(a.unsqueeze(-1) * vc.unsqueeze(-2), dim=2)
            cum_den = torch.cumsum(a, dim=2)

            decay_to_t = torch.exp(-w * pos)                # (B,H,C,D)
            num = cum_num * decay_to_t.unsqueeze(-1) + state_num.unsqueeze(2) * torch.exp(
                -w * (pos + 1)
            ).unsqueeze(-1)
            den = cum_den * decay_to_t + state_den.unsqueeze(2) * torch.exp(-w * (pos + 1))

            # current-token bonus exp(u + k_t)
            bonus = torch.exp(u + kc)
            num = num + bonus.unsqueeze(-1) * vc.unsqueeze(-2)
            den = den + bonus

            out[:, :, start:end] = torch.einsum(
                "bhck,bhckv->bhcv", r[:, :, start:end], num / den.clamp(min=1e-6).unsqueeze(-1)
            )

            # roll the state to the end of this chunk (relative to its last token)
            roll = torch.exp(-w * (n - 1)).squeeze(2).unsqueeze(-1)
            state_num = cum_num[:, :, -1] * roll
            state_den = cum_den[:, :, -1] * torch.exp(-w * (n - 1)).squeeze(2)

        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        return self.o_proj(self.out_norm(out))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        r = self._split(self.r_proj(hidden_states))
        k = self._split(self.k_proj(hidden_states))
        v = self._split(self.v_proj(hidden_states))
        return self._wkv(r, k, v)


class RWKVBlock(nn.Module):
    def __init__(self, config: AgronRWKVConfig):
        super().__init__()
        self.time_mix = WKVTimeMix(config)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.input_layernorm = build_norm(config.hidden_size, config.rms_norm_eps, config.norm_type)
        self.post_attention_layernorm = build_norm(config.hidden_size, config.rms_norm_eps, config.norm_type)

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.time_mix(self.input_layernorm(hidden_states))
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class AgronRWKVForCausalLM(AgronPreTrainedModel, GenerationMixin):
    config_class = AgronRWKVConfig
    base_model_prefix = "agron_rwkv"
    supports_gradient_checkpointing = True

    def __init__(self, config: AgronRWKVConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([RWKVBlock(config) for _ in range(config.num_hidden_layers)])
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
