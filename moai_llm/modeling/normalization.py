"""
RMSNorm implementation for MOAI-LLM.

RMSNorm (Root Mean Square Layer Normalization) is more efficient than LayerNorm
as it only performs rescaling without recentering.

Reference: https://arxiv.org/abs/1910.07467
"""

import torch
import torch.nn as nn
from typing import Optional


class MoaiRMSNorm(nn.Module):
    """
    RMSNorm implementation.

    Args:
        hidden_size (int): Dimension of the input tensor
        eps (float): Epsilon for numerical stability (default: 1e-6)
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm to input tensor.

        Args:
            hidden_states: Input tensor of shape (..., hidden_size)

        Returns:
            Normalized tensor of same shape
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)

        # Compute RMS
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # Scale and cast back to original dtype
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self) -> str:
        return f"hidden_size={self.weight.shape[0]}, eps={self.variance_epsilon}"
