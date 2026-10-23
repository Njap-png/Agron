"""DeltaNet decoder: linear attention with the delta learning rule.

The state is a fast-weight matrix updated with the delta rule instead of a
plain additive outer product,

    S_t = S_{t-1} + beta_t * k_t (v_t - S_{t-1}^T k_t)^T
    o_t = S_{t-1}^T q_t

which lets the memory overwrite old associations instead of only accumulating
them.  Chunks are processed with the WY representation: the intra-chunk
pseudo-values are obtained from a unit lower-triangular solve, so the whole
chunk is evaluated with matrix products rather than one step per token.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers.generation import GenerationMixin

from .base import AgronConfig, AgronPreTrainedModel, SwiGLU, build_norm, causal_lm_loss

CHUNK = 16


class AgronDeltaNetConfig(AgronConfig):
    model_type = "agron_deltanet"

    def __init__(self, norm_type: str = "rms", chunk_size: int = CHUNK, **kwargs):
        self.norm_type = norm_type
        self.chunk_size = chunk_size
        super().__init__(**kwargs)


class DeltaRuleAttention(nn.Module):
    def __init__(self, config: AgronDeltaNetConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.chunk_size = config.chunk_size

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.beta_proj = nn.Linear(config.hidden_size, self.num_heads, bias=True)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.out_norm = build_norm(config.hidden_size, config.rms_norm_eps, "rms")

    def _split(self, x):
        b, t, _ = x.shape
        return x.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        heads, dim = self.num_heads, self.head_dim
        chunk = self.chunk_size
        device = hidden_states.device

        q = self._split(self.q_proj(hidden_states))
        k = self._split(self.k_proj(hidden_states))
        v = self._split(self.v_proj(hidden_states))
        beta = torch.sigmoid(self.beta_proj(hidden_states)).transpose(1, 2)  # (B,H,T)

        state = torch.zeros(bsz, heads, dim, dim, device=device, dtype=hidden_states.dtype)
        out = torch.empty_like(v)
        eye = torch.eye(chunk, device=device, dtype=hidden_states.dtype)

        for start in range(0, seq_len, chunk):
            end = min(start + chunk, seq_len)
            n = end - start
            qc, kc, vc = q[:, :, start:end], k[:, :, start:end], v[:, :, start:end]
            bc = beta[:, :, start:end]                                    # (B,H,C)

            # Strictly lower-triangular coupling between tokens of the chunk.
            kk = torch.matmul(kc, kc.transpose(-1, -2))                   # (B,H,C,C)
            coupling = torch.tril(kk, diagonal=-1) * bc.unsqueeze(-1)
            system = eye[:n, :n] + coupling

            # Residual targets w.r.t. the incoming state, then the triangular solve.
            residual = vc - torch.matmul(kc, state)                       # (B,H,C,Dv)
            rhs = residual * bc.unsqueeze(-1)
            pseudo = torch.linalg.solve_triangular(
                system, rhs, upper=False, unitriangular=True
            )

            # Read-out: inter-chunk part plus strictly causal intra-chunk part.
            scores = torch.tril(torch.matmul(qc, kc.transpose(-1, -2)), diagonal=-1)
            out[:, :, start:end] = torch.matmul(qc, state) + torch.matmul(scores, pseudo)

            # Delta-rule state update for the whole chunk.
            state = state + torch.matmul(kc.transpose(-1, -2), pseudo)

        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        return self.o_proj(self.out_norm(out))


class DeltaNetBlock(nn.Module):
    def __init__(self, config: AgronDeltaNetConfig):
        super().__init__()
        self.delta_attention = DeltaRuleAttention(config)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.input_layernorm = build_norm(config.hidden_size, config.rms_norm_eps, config.norm_type)
        self.post_attention_layernorm = build_norm(config.hidden_size, config.rms_norm_eps, config.norm_type)

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.delta_attention(self.input_layernorm(hidden_states))
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class AgronDeltaNetForCausalLM(AgronPreTrainedModel, GenerationMixin):
    config_class = AgronDeltaNetConfig
    base_model_prefix = "agron_deltanet"
    supports_gradient_checkpointing = True

    def __init__(self, config: AgronDeltaNetConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([DeltaNetBlock(config) for _ in range(config.num_hidden_layers)])
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
