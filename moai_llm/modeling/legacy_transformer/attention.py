"""
Grouped Query Attention with Flash Attention 3 support for MOAI-LLM.

Grouped Query Attention (GQA) reduces KV cache size by sharing key/value heads
across multiple query heads, balancing between Multi-Head Attention (MHA) and
Multi-Query Attention (MQA).

References:
- GQA: https://arxiv.org/abs/2305.13245
- Flash Attention: https://arxiv.org/abs/2205.14135
- Flash Attention 2: https://arxiv.org/abs/2307.08691
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

from moai_llm.modeling.rope import MoaiRotaryEmbedding, apply_rotary_pos_emb
from moai_llm.modeling.normalization import MoaiRMSNorm

# Try to import flash attention
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False

# PyTorch 2.0+ SDPA (Scaled Dot Product Attention) - always available
SDPA_AVAILABLE = hasattr(F, 'scaled_dot_product_attention')


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key/value tensors to match the number of query heads.

    This is used for Grouped Query Attention to expand KV heads to match Q heads.
    Uses repeat_interleave for better memory access patterns.

    Args:
        hidden_states: Tensor of shape (batch, num_kv_heads, seq_len, head_dim)
        n_rep: Number of repetitions (num_query_heads // num_kv_heads)

    Returns:
        Tensor of shape (batch, num_query_heads, seq_len, head_dim)
    """
    if n_rep == 1:
        return hidden_states
    
    # repeat_interleave is more memory-efficient than expand+reshape
    # as it creates a contiguous tensor directly
    return hidden_states.repeat_interleave(n_rep, dim=1)


class MoaiAttention(nn.Module):
    """
    Grouped Query Attention with Flash Attention 3 support.

    Features:
    - GQA for efficient KV cache
    - QK-Norm for training stability
    - Flash Attention for memory-efficient computation
    - RoPE for position encoding

    Args:
        config: Model configuration
        layer_idx: Layer index (optional, for caching)
    """

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.attention_dropout = config.attention_dropout
        self.use_qk_norm = config.use_qk_norm

        # Validate head dimensions for Flash Attention compatibility
        if FLASH_ATTENTION_AVAILABLE and self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"For Flash Attention, num_heads ({self.num_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads})"
            )

        # Q, K, V projections
        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )

        # QK-Norm for training stability
        if self.use_qk_norm:
            self.q_norm = MoaiRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = MoaiRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

        # Rotary Position Embedding
        self.rotary_emb = MoaiRotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
            scaling_config=config.rope_scaling,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass for attention layer.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask (optional)
            position_ids: Position indices (optional)
            past_key_value: Cached key/value from previous steps (optional)
            output_attentions: Whether to return attention weights
            use_cache: Whether to return key/value cache

        Returns:
            Tuple of (output, attention_weights, past_key_value)
        """
        bsz, q_len, _ = hidden_states.size()

        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape to multi-head format
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply QK-Norm if enabled for stability
        if self.use_qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        # Get RoPE embeddings
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        # Apply rotary embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        # Handle KV cache
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # Repeat K/V for GQA
        key_states = repeat_kv(key_states, self.num_kv_groups)
        value_states = repeat_kv(value_states, self.num_kv_groups)

        # Compute attention (priority: Flash Attention > SDPA > Standard)
        # Flash Attention only supports fp16/bf16, fallback to SDPA for fp32
        use_flash = (
            FLASH_ATTENTION_AVAILABLE
            and not output_attentions
            and query_states.dtype in (torch.float16, torch.bfloat16)
        )
        
        if use_flash:
            # Use Flash Attention for efficient computation
            attn_output = self._flash_attention(
                query_states, key_states, value_states, attention_mask, q_len
            )
        elif SDPA_AVAILABLE and not output_attentions:
            # Use PyTorch 2.0+ SDPA (memory efficient, no flash-attn required)
            attn_output = self._sdpa_attention(
                query_states, key_states, value_states, attention_mask, q_len
            )
        else:
            # Fallback to standard attention (least efficient)
            attn_output = self._standard_attention(
                query_states, key_states, value_states, attention_mask, output_attentions
            )

        # Reshape and project output (single reshape for efficiency)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        attn_weights = None  # Flash attention doesn't return weights
        return attn_output, attn_weights, past_key_value

    def _flash_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        q_len: int,
    ) -> torch.Tensor:
        """
        Compute attention using Flash Attention.

        Args:
            query: Query tensor (batch, num_heads, seq_len, head_dim)
            key: Key tensor (batch, num_heads, seq_len, head_dim)
            value: Value tensor (batch, num_heads, seq_len, head_dim)
            attention_mask: Attention mask (optional)
            q_len: Query sequence length

        Returns:
            Attention output tensor
        """
        # Flash Attention expects (batch, seq_len, num_heads, head_dim)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Determine if causal masking is needed
        causal = q_len > 1 and (attention_mask is None or attention_mask.dim() == 2)

        # Apply Flash Attention
        attn_output = flash_attn_func(
            query,
            key,
            value,
            dropout_p=self.attention_dropout if self.training else 0.0,
            softmax_scale=1.0 / math.sqrt(self.head_dim),
            causal=causal,
        )

        return attn_output

    def _sdpa_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        q_len: int,
    ) -> torch.Tensor:
        """
        Compute attention using PyTorch 2.0+ Scaled Dot Product Attention.
        
        This is memory-efficient (O(n) instead of O(n²)) and doesn't require
        flash-attn installation. Works on all GPUs including consumer GPUs.

        Args:
            query: Query tensor (batch, num_heads, seq_len, head_dim)
            key: Key tensor (batch, num_heads, seq_len, head_dim)
            value: Value tensor (batch, num_heads, seq_len, head_dim)
            attention_mask: Attention mask (optional)
            q_len: Query sequence length

        Returns:
            Attention output tensor
        """
        # Determine if causal masking is needed
        is_causal = q_len > 1 and attention_mask is None
        
        # SDPA expects attention_mask as (batch, 1, seq, seq) or None
        # If we have a 4D mask, convert it to proper format for SDPA
        attn_mask = None
        if attention_mask is not None and not is_causal:
            # attention_mask is additive (0 for attend, -inf for mask)
            # SDPA expects the same format
            attn_mask = attention_mask
        
        # Use PyTorch SDPA - automatically selects best backend (Flash/Efficient/Math)
        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
            scale=1.0 / math.sqrt(self.head_dim),
        )
        
        return attn_output

    def _standard_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        output_attentions: bool,
    ) -> torch.Tensor:
        """
        Compute attention using standard scaled dot-product attention.

        Args:
            query: Query tensor (batch, num_heads, seq_len, head_dim)
            key: Key tensor (batch, num_heads, seq_len, head_dim)
            value: Value tensor (batch, num_heads, seq_len, head_dim)
            attention_mask: Attention mask (optional)
            output_attentions: Whether to return attention weights

        Returns:
            Attention output tensor
        """
        # Scaled dot-product attention
        attn_weights = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.head_dim)

        # Apply attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax (compute in float32 for numerical stability, then cast back)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        
        # Dropout only if needed (skip F.dropout call when dropout=0)
        if self.attention_dropout > 0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=True)

        # Compute output
        return torch.matmul(attn_weights, value)
