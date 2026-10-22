"""Transformer-family Agron architectures.

Three variants are provided:

* ``transformer`` -- a clean, dependency-free decoder-only transformer
  (RoPE + RMSNorm + SwiGLU) written from scratch.
* ``moe``         -- the same backbone whose feed-forward blocks are replaced
  by a top-k Mixture-of-Experts router (plus a shared expert).
* ``hyper``       -- a hypernetwork that emits per-layer modulation vectors
  conditioning the transformer blocks on a learned task embedding.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.generation import GenerationMixin

from .base import (
    AgronConfig,
    AgronPreTrainedModel,
    CausalSelfAttention,
    RMSNorm,
    SwiGLU,
    build_norm,
    causal_lm_loss,
)


# --------------------------------------------------------------------------- #
# Plain transformer
# --------------------------------------------------------------------------- #
class AgronTransformerConfig(AgronConfig):
    model_type = "agron_transformer"

    def __init__(
        self,
        norm_type: str = "rms",
        norm_before: bool = True,
        **kwargs,
    ):
        self.norm_type = norm_type
        self.norm_before = norm_before
        super().__init__(**kwargs)


class TransformerBlock(nn.Module):
    """One pre-norm decoder block."""

    def __init__(self, config: AgronConfig):
        super().__init__()
        self.self_attn = CausalSelfAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            bias=config.attention_bias,
        )
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.input_layernorm = build_norm(config.hidden_size, config.rms_norm_eps, getattr(config, "norm_type", "rms"))
        self.post_attention_layernorm = build_norm(
            config.hidden_size, config.rms_norm_eps, getattr(config, "norm_type", "rms")
        )

    def forward(self, hidden_states, attention_mask=None, position_ids=None, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask, position_ids=position_ids)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class AgronTransformerForCausalLM(AgronPreTrainedModel, GenerationMixin):
    """Decoder-only transformer trained from scratch."""

    config_class = AgronTransformerConfig
    base_model_prefix = "agron"
    supports_gradient_checkpointing = True

    def __init__(self, config: AgronTransformerConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = build_norm(config.hidden_size, config.rms_norm_eps, getattr(config, "norm_type", "rms"))
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        label_smoothing: float = 0.0,
        **kwargs,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        if position_ids is None:
            position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = causal_lm_loss(logits, labels, label_smoothing) if labels is not None else None
        return {"logits": logits, "loss": loss}


# --------------------------------------------------------------------------- #
# Mixture of Experts
# --------------------------------------------------------------------------- #
class AgronMoEConfig(AgronConfig):
    model_type = "agron_moe"

    def __init__(
        self,
        num_experts: int = 4,
        num_experts_per_tok: int = 2,
        moe_intermediate_size: Optional[int] = None,
        shared_expert: bool = True,
        router_aux_loss_coef: float = 0.01,
        norm_type: str = "rms",
        intermediate_size: int = 2048,
        **kwargs,
    ):
        self.num_experts = num_experts
        self.num_experts_per_tok = min(num_experts_per_tok, num_experts)
        self.moe_intermediate_size = moe_intermediate_size or intermediate_size
        self.shared_expert = shared_expert
        self.router_aux_loss_coef = router_aux_loss_coef
        self.norm_type = norm_type
        super().__init__(intermediate_size=intermediate_size, **kwargs)


class MoEFeedForward(nn.Module):
    """Top-k routed experts with an optional always-on shared expert."""

    def __init__(self, config: AgronMoEConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                SwiGLU(config.hidden_size, config.moe_intermediate_size, bias=config.mlp_bias)
                for _ in range(config.num_experts)
            ]
        )
        self.shared = (
            SwiGLU(config.hidden_size, config.moe_intermediate_size, bias=config.mlp_bias)
            if config.shared_expert
            else None
        )

    def forward(self, x: torch.Tensor):
        bsz, seq, hidden = x.shape
        flat = x.view(-1, hidden)
        router_logits = self.router(flat)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        routing_weights = routing_weights.to(x.dtype)

        topk_weights, topk_idx = torch.topk(routing_weights, self.top_k, dim=-1)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-9)

        out = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            mask = (topk_idx == expert_id).any(dim=-1)
            if not bool(mask.any()):
                continue
            token_idx = mask.nonzero(as_tuple=True)[0]
            expert_out = expert(flat[token_idx])
            weight = (topk_weights * (topk_idx == expert_id)).sum(dim=-1)[token_idx]
            out.index_add_(0, token_idx, expert_out * weight.unsqueeze(-1).to(expert_out.dtype))

        if self.shared is not None:
            out = out + self.shared(flat)

        aux_loss = None
        if self.training:
            # Switch-Transformer style load-balancing loss.
            density = torch.zeros(self.num_experts, device=x.device, dtype=torch.float32)
            density.scatter_add_(0, topk_idx.reshape(-1), torch.ones_like(topk_weights).reshape(-1))
            density = density / max(1, topk_idx.numel())
            prob = routing_weights.mean(dim=0).float()
            aux_loss = self.num_experts * torch.sum(density * prob)

        return out.view(bsz, seq, hidden), aux_loss


class MoEBlock(nn.Module):
    def __init__(self, config: AgronMoEConfig):
        super().__init__()
        self.self_attn = CausalSelfAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            bias=config.attention_bias,
        )
        self.moe = MoEFeedForward(config)
        self.input_layernorm = build_norm(config.hidden_size, config.rms_norm_eps, config.norm_type)
        self.post_attention_layernorm = build_norm(config.hidden_size, config.rms_norm_eps, config.norm_type)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask, position_ids=position_ids)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, aux_loss = self.moe(hidden_states)
        return residual + hidden_states, aux_loss


class AgronMoEForCausalLM(AgronPreTrainedModel, GenerationMixin):
    """Sparse Mixture-of-Experts decoder."""

    config_class = AgronMoEConfig
    base_model_prefix = "agron"

    def __init__(self, config: AgronMoEConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([MoEBlock(config) for _ in range(config.num_hidden_layers)])
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

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        label_smoothing: float = 0.0,
        **kwargs,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        if position_ids is None:
            position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)

        aux_loss = None
        for layer in self.layers:
            hidden_states, layer_aux = layer(
                hidden_states, attention_mask=attention_mask, position_ids=position_ids
            )
            if layer_aux is not None:
                aux_loss = layer_aux if aux_loss is None else aux_loss + layer_aux

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        loss = causal_lm_loss(logits, labels, label_smoothing) if labels is not None else None
        if loss is not None and aux_loss is not None:
            loss = loss + self.config.router_aux_loss_coef * aux_loss
        return {"logits": logits, "loss": loss}


# --------------------------------------------------------------------------- #
# Hypernetwork-conditioned transformer
# --------------------------------------------------------------------------- #
class AgronHyperConfig(AgronConfig):
    model_type = "agron_hyper"

    def __init__(
        self,
        num_tasks: int = 8,
        hyper_hidden: int = 128,
        hyper_depth: int = 2,
        modulate_affine: bool = True,
        norm_type: str = "rms",
        **kwargs,
    ):
        self.num_tasks = num_tasks
        self.hyper_hidden = hyper_hidden
        self.hyper_depth = hyper_depth
        self.modulate_affine = modulate_affine
        self.norm_type = norm_type
        super().__init__(**kwargs)


class HyperBlock(nn.Module):
    """Transformer block whose norms are modulated by a hypernetwork."""

    def __init__(self, config: AgronHyperConfig):
        super().__init__()
        self.self_attn = CausalSelfAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            bias=config.attention_bias,
        )
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.input_layernorm = build_norm(config.hidden_size, config.rms_norm_eps, config.norm_type)
        self.post_attention_layernorm = build_norm(config.hidden_size, config.rms_norm_eps, config.norm_type)
        self.modulate_affine = config.modulate_affine

    def forward(self, hidden_states, scale=None, shift=None, attention_mask=None, position_ids=None, **kwargs):
        residual = hidden_states
        normed = self.input_layernorm(hidden_states)
        if scale is not None:
            normed = normed * scale
            if shift is not None and self.modulate_affine:
                normed = normed + shift
        hidden_states = residual + self.self_attn(
            normed, attention_mask=attention_mask, position_ids=position_ids
        )

        residual = hidden_states
        normed = self.post_attention_layernorm(hidden_states)
        if scale is not None:
            normed = normed * scale
            if shift is not None and self.modulate_affine:
                normed = normed + shift
        return residual + self.mlp(normed)


class AgronHyperForCausalLM(AgronPreTrainedModel, GenerationMixin):
    """Transformer conditioned on a task embedding produced by a hypernetwork."""

    config_class = AgronHyperConfig
    base_model_prefix = "agron"

    def __init__(self, config: AgronHyperConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.task_embeddings = nn.Embedding(config.num_tasks, config.hyper_hidden)
        self.hypernet = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(config.hyper_hidden, config.hyper_hidden), nn.SiLU())
                for _ in range(max(1, config.hyper_depth - 1))
            ]
        )
        # One (scale, shift) pair per block, emitted in a single projection.
        self.hyper_out = nn.Linear(
            config.hyper_hidden, config.num_hidden_layers * config.hidden_size * 2
        )
        self.layers = nn.ModuleList([HyperBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = build_norm(config.hidden_size, config.rms_norm_eps, config.norm_type)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        # Start as an identity modulation: scale = 1, shift = 0.
        with torch.no_grad():
            self.hyper_out.weight.zero_()
            self.hyper_out.bias.zero_()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        task_id: Optional[torch.LongTensor] = None,
        label_smoothing: float = 0.0,
        **kwargs,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        if position_ids is None:
            position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)

        if task_id is None:
            task_id = torch.zeros(hidden_states.shape[0], dtype=torch.long, device=hidden_states.device)
        task_id = task_id.clamp(0, self.config.num_tasks - 1)

        cond = self.task_embeddings(task_id)
        for layer in self.hypernet:
            cond = layer(cond)
        cond = self.hyper_out(cond)
        per_layer = cond.view(
            hidden_states.shape[0], self.config.num_hidden_layers, 2, self.config.hidden_size
        )

        for i, layer in enumerate(self.layers):
            scale = (1.0 + per_layer[:, i, 0]).unsqueeze(1)
            shift = per_layer[:, i, 1].unsqueeze(1)
            hidden_states = layer(
                hidden_states,
                scale=scale,
                shift=shift,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        loss = causal_lm_loss(logits, labels, label_smoothing) if labels is not None else None
        return {"logits": logits, "loss": loss}
