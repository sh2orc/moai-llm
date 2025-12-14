"""
Activation functions for MOAI-LLM.

SwiGLU (Swish-Gated Linear Unit) has been shown to provide the best performance
among GLU variants for LLM architectures.

Reference: https://arxiv.org/abs/2002.05202
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MoaiSwiGLU(nn.Module):
    """
    SwiGLU activation function with gating mechanism.

    SwiGLU(x, W, V, b, c) = Swish(xW + b) ⊗ (xV + c)
    where Swish(x) = x * sigmoid(x) and ⊗ is element-wise product.

    Note: This implementation uses the no-bias variant (Qwen3 style).

    Args:
        hidden_size (int): Input dimension
        intermediate_size (int): Intermediate dimension (typically ~2.67x hidden_size for SwiGLU)
        bias (bool): Whether to use bias (default: False)
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # Gate and up projections (combined for efficiency)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU activation.

        Args:
            x: Input tensor of shape (..., hidden_size)

        Returns:
            Output tensor of shape (..., hidden_size)
        """
        # Gate path: Swish(xW)
        gate = F.silu(self.gate_proj(x))  # SiLU = Swish

        # Up path: xV
        up = self.up_proj(x)

        # Element-wise product and down projection
        return self.down_proj(gate * up)

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, intermediate_size={self.intermediate_size}"


class MoaiMLP(nn.Module):
    """
    MLP block with SwiGLU activation for transformer.

    This is a convenience wrapper around SwiGLU for use in transformer blocks.

    Args:
        config: Model configuration
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.swiglu = MoaiSwiGLU(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            bias=config.mlp_bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply MLP transformation."""
        return self.swiglu(x)
