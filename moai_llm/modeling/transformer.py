"""
Transformer decoder layer for MOAI-LLM.

Implements Pre-LN architecture with RMSNorm, GQA, and SwiGLU.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from moai_llm.modeling.attention import MoaiAttention
from moai_llm.modeling.activations import MoaiMLP
from moai_llm.modeling.normalization import MoaiRMSNorm


class MoaiDecoderLayer(nn.Module):
    """
    Single transformer decoder layer with Pre-LN architecture.

    Architecture:
        x = x + Attention(LN(x))
        x = x + MLP(LN(x))

    Args:
        config: Model configuration
        layer_idx: Layer index
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        # Self-attention
        self.self_attn = MoaiAttention(config, layer_idx)

        # MLP (Feed-forward network with SwiGLU)
        self.mlp = MoaiMLP(config)

        # Layer normalization (Pre-LN style)
        self.input_layernorm = MoaiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MoaiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Forward pass for transformer decoder layer.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask
            position_ids: Position indices
            past_key_value: Cached key/value from previous steps
            output_attentions: Whether to return attention weights
            use_cache: Whether to return key/value cache

        Returns:
            Tuple of (hidden_states, present_key_value)
        """
        residual = hidden_states

        # Pre-LN: Normalize before attention
        hidden_states = self.input_layernorm(hidden_states)

        # Self-attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )

        # Residual connection
        hidden_states = residual + hidden_states

        # MLP block
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
