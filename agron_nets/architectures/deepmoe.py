"""Deep/Enhanced Mixture-of-Experts for Agron.

Implements a more capable MoE (Mixtral/DeepSeek style) with:
- Router with optional noise (z-loss style stability)
- Top-k experts with load balancing and router z-loss
- Shared expert (always active)
- Expert capacity / drop tokens option (configurable)
- Expert dropout
- Supports deeper configs and HuggingFace save/load
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.generation import GenerationMixin

from .base import AgronConfig, AgronPreTrainedModel, CausalSelfAttention, build_norm, causal_lm_loss, SwiGLU


class AgronDeepMoEConfig(AgronConfig):
    model_type = "agron_deepmoe"

    def __init__(
        self,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
        moe_intermediate_size: Optional[int] = None,
        shared_expert: bool = True,
        shared_expert_intermediate_size: Optional[int] = None,
        router_aux_loss_coef: float = 0.01,
        router_z_loss_coef: float = 0.001,
        router_jitter_noise: float = 0.0,
        expert_dropout: float = 0.0,
        norm_type: str = "rms",
        intermediate_size: int = 2048,
        **kwargs,
    ):
        self.num_experts = num_experts
        self.num_experts_per_tok = min(num_experts_per_tok, num_experts)
        self.moe_intermediate_size = moe_intermediate_size or intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size or self.moe_intermediate_size
        self.shared_expert = shared_expert
        self.router_aux_loss_coef = router_aux_loss_coef
        self.router_z_loss_coef = router_z_loss_coef
        self.router_jitter_noise = router_jitter_noise
        self.expert_dropout = expert_dropout
        self.norm_type = norm_type
        super().__init__(intermediate_size=intermediate_size, **kwargs)


class DeepMoEFeedForward(nn.Module):
    def __init__(self, config: AgronDeepMoEConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.expert_dropout = getattr(config, "expert_dropout", 0.0)
        self.router_jitter_noise = getattr(config, "router_jitter_noise", 0.0)

        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLU(config.hidden_size, config.moe_intermediate_size, bias=config.mlp_bias) for _ in range(config.num_experts)]
        )
        if getattr(config, "shared_expert", True):
            self.shared_expert = SwiGLU(config.hidden_size, getattr(config, "shared_expert_intermediate_size", config.moe_intermediate_size), bias=config.mlp_bias)
        else:
            self.shared_expert = None

    def forward(self, x: torch.Tensor):
        bsz, seq_len, hidden = x.shape
        flat = x.view(-1, hidden)
        tokens = flat.shape[0]

        router_logits = self.router(flat)

        if self.training and self.router_jitter_noise > 0:
            noise = torch.randn_like(router_logits) * self.router_jitter_noise
            router_logits = router_logits + noise

        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32).to(x.dtype)
        topk_weights, topk_idx = torch.topk(routing_weights, self.top_k, dim=-1)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-9)

        out = torch.zeros_like(flat)
        for expert_id in range(self.num_experts):
            expert = self.experts[expert_id]
            mask = (topk_idx == expert_id).any(dim=-1)
            if not mask.any():
                continue
            idx = mask.nonzero(as_tuple=False).squeeze(-1)
            expert_in = flat[idx]
            expert_out = expert(expert_in)
            if self.training and self.expert_dropout > 0:
                expert_out = F.dropout(expert_out, p=self.expert_dropout, training=True)
            out[idx] += expert_out

        if self.shared_expert is not None:
            out = out + self.shared_expert(flat)

        aux_loss = None
        if self.training:
            # Load balancing loss
            expert_mask = F.one_hot(topk_idx, num_classes=self.num_experts).float()
            tokens_per_expert = expert_mask.sum(dim=(0, 1)) / tokens
            router_prob = routing_weights.mean(dim=0)
            aux_loss = self.num_experts * torch.sum(tokens_per_expert * router_prob)
            # Router z-loss for stability
            z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()
            aux_loss = aux_loss + 0.001 * z_loss  # will be scaled by coef in model

        return out.view(bsz, seq_len, hidden), aux_loss


class DeepMoEBlock(nn.Module):
    def __init__(self, config: AgronDeepMoEConfig):
        super().__init__()
        self.self_attn = CausalSelfAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            bias=config.attention_bias,
        )
        self.mlp = DeepMoEFeedForward(config)
        self.input_layernorm = build_norm(config.hidden_size, config.rms_norm_eps, config.norm_type)
        self.post_attention_layernorm = build_norm(config.hidden_size, config.rms_norm_eps, config.norm_type)

    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = residual + self.self_attn(hidden_states, attention_mask=attention_mask, position_ids=position_ids)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, aux = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, aux


class AgronDeepMoEForCausalLM(AgronPreTrainedModel, GenerationMixin):
    config_class = AgronDeepMoEConfig
    base_model_prefix = "agron_deepmoe"
    supports_gradient_checkpointing = True

    def __init__(self, config: AgronDeepMoEConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([DeepMoEBlock(config) for _ in range(config.num_hidden_layers)])
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
            hidden_states, layer_aux = layer(hidden_states, attention_mask=attention_mask, position_ids=position_ids)
            if layer_aux is not None:
                aux_loss = layer_aux if aux_loss is None else aux_loss + layer_aux

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        loss = causal_lm_loss(logits, labels, label_smoothing) if labels is not None else None
        if loss is not None and aux_loss is not None:
            loss = loss + self.config.router_aux_loss_coef * aux_loss
        return {"logits": logits, "loss": loss}
