"""
Rotary Position Embedding (RoPE) with YaRN extension support for MOAI-LLM.

RoPE encodes position information by rotating query and key embeddings.
YaRN enables efficient context length extension with minimal training.

References:
- RoPE: https://arxiv.org/abs/2104.09864
- YaRN: https://arxiv.org/abs/2309.00071
- NTK-aware: https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class MoaiRotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding with support for scaling methods.

    Supports:
    - Standard RoPE
    - Linear scaling (simple interpolation)
    - Dynamic NTK scaling
    - YaRN (Yet another RoPE extensioN)

    Args:
        dim (int): Dimension of each attention head
        max_position_embeddings (int): Maximum sequence length
        base (float): Base frequency for RoPE (default: 10000.0)
        scaling_config (dict, optional): Configuration for RoPE scaling
            - type: "linear", "dynamic", "yarn", or "ntk"
            - factor: Scaling factor (required for all types)
            - original_max_position_embeddings: Original context length (for yarn/ntk)
        device: Device for computation
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 8192,
        base: float = 10000.0,
        scaling_config: Optional[dict] = None,
        device=None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_config = scaling_config
        self.scaling_type = scaling_config.get("type", None) if scaling_config else None

        # Compute inverse frequencies in FP32 for numerical precision
        inv_freq = self._compute_inv_freq(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build cosine and sine cache
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _compute_inv_freq(self, device) -> torch.Tensor:
        """Compute inverse frequencies based on scaling configuration."""
        # Standard inverse frequencies: 1 / (base^(2i/dim)) for i in [0, dim/2)
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim)
        )

        if self.scaling_config is None:
            return inv_freq

        scaling_factor = self.scaling_config.get("factor", 1.0)
        scaling_type = self.scaling_type

        if scaling_type == "linear":
            # Linear interpolation: simply divide frequencies by scaling factor
            inv_freq = inv_freq / scaling_factor

        elif scaling_type == "dynamic" or scaling_type == "ntk":
            # NTK-aware scaling: scale base frequency
            # More aggressive scaling for low frequencies, less for high frequencies
            original_max_pos = self.scaling_config.get(
                "original_max_position_embeddings", self.max_position_embeddings
            )
            # Adjust base using NTK scaling formula
            scaled_base = self.base * (scaling_factor ** (self.dim / (self.dim - 2)))
            inv_freq = 1.0 / (
                scaled_base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim)
            )

        elif scaling_type == "yarn":
            # YaRN scaling: more sophisticated interpolation
            # Uses attention scaling and NTK-by-parts
            original_max_pos = self.scaling_config.get(
                "original_max_position_embeddings", self.max_position_embeddings
            )
            alpha = self.scaling_config.get("alpha", 1.0)
            beta = self.scaling_config.get("beta", 32.0)

            # Compute wavelengths
            wavelengths = 2 * math.pi / inv_freq

            # Apply NTK-by-parts: different scaling for different frequency bands
            scale_factors = torch.ones_like(inv_freq)
            long_wavelengths = wavelengths > (original_max_pos * beta)
            medium_wavelengths = (wavelengths > original_max_pos) & ~long_wavelengths

            # Low frequencies: NTK scaling
            if long_wavelengths.any():
                scale_factors[long_wavelengths] = alpha * scaling_factor

            # Medium frequencies: interpolation
            if medium_wavelengths.any():
                # Linear interpolation between 1 and alpha
                ratio = (wavelengths[medium_wavelengths] - original_max_pos) / (
                    original_max_pos * (beta - 1)
                )
                scale_factors[medium_wavelengths] = 1 + (alpha - 1) * ratio

            # High frequencies: no scaling (scale_factor = 1)

            inv_freq = inv_freq / scale_factors

        return inv_freq

    def _set_cos_sin_cache(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ):
        """Precompute cos/sin cache for efficiency."""
        self.max_seq_len_cached = seq_len

        # Create position indices
        t = torch.arange(seq_len, device=device, dtype=torch.float32)

        # Apply scaling factor to positions if using linear scaling
        if self.scaling_type == "linear":
            scaling_factor = self.scaling_config.get("factor", 1.0)
            t = t / scaling_factor

        # Compute outer product: positions × inverse frequencies
        freqs = torch.outer(t, self.inv_freq)

        # Create embeddings by concatenating cos and sin
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(
        self, x: torch.Tensor, seq_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cosine and sine embeddings for given sequence length.

        Args:
            x: Input tensor (used for dtype/device, not actual values)
            seq_len: Sequence length (if None, use x.shape[1])

        Returns:
            Tuple of (cos, sin) embeddings
        """
        if seq_len is None:
            seq_len = x.shape[1] if x.dim() > 1 else 1

        # Extend cache if needed
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate half the hidden dims of the input.

    This is a helper function for applying rotary embeddings.
    Uses chunk() instead of slicing for better efficiency.

    Args:
        x: Input tensor of shape (..., head_dim)

    Returns:
        Rotated tensor with first and second half swapped and negated
    """
    x1, x2 = x.chunk(2, dim=-1)  # More efficient than slicing
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key tensors.

    Args:
        q: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
        k: Key tensor of shape (batch_size, num_kv_heads, seq_len, head_dim)
        cos: Cosine embeddings of shape (seq_len, head_dim)
        sin: Sine embeddings of shape (seq_len, head_dim)
        position_ids: Position indices (if None, assumes sequential positions)
        unsqueeze_dim: Dimension to unsqueeze for broadcasting

    Returns:
        Tuple of (rotated_q, rotated_k)
    """
    # cos/sin shape: (seq_len, head_dim) → (1, 1, seq_len, head_dim)
    # to broadcast with q/k: (batch, num_heads, seq_len, head_dim)
    # Use indexing instead of unsqueeze() for efficiency
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]

    # Apply rotation using the formula:
    # q_embed = q * cos + rotate_half(q) * sin
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin

    return q_embed, k_embed
