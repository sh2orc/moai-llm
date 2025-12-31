"""
RMSNorm implementation for MOAI-LLM.

RMSNorm (Root Mean Square Layer Normalization) is more efficient than LayerNorm
as it only performs rescaling without recentering.

Uses flash-attn's fused RMSNorm when available for ~30-50% speedup.

Reference: https://arxiv.org/abs/1910.07467
"""

import torch
import torch.nn as nn
from typing import Optional

# Try to import flash-attn's fused RMSNorm
try:
    from flash_attn.ops.rms_norm import rms_norm as flash_rms_norm
    FLASH_RMSNORM_AVAILABLE = True
except ImportError:
    FLASH_RMSNORM_AVAILABLE = False


class MoaiRMSNorm(nn.Module):
    """
    RMSNorm implementation with optional flash-attn fusion.

    When flash-attn is available, uses fused CUDA kernel for ~30-50% speedup.
    Falls back to PyTorch implementation otherwise.

    Args:
        hidden_size (int): Dimension of the input tensor
        eps (float): Epsilon for numerical stability (default: 1e-6)
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.hidden_size = hidden_size

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm to input tensor.

        Args:
            hidden_states: Input tensor of shape (..., hidden_size)

        Returns:
            Normalized tensor of same shape
        """
        # Use fused RMSNorm if available (flash-attn)
        if FLASH_RMSNORM_AVAILABLE and hidden_states.is_cuda:
            return flash_rms_norm(hidden_states, self.weight, self.variance_epsilon)
        
        # Fallback: PyTorch implementation
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states

    def extra_repr(self) -> str:
        fused = "fused" if FLASH_RMSNORM_AVAILABLE else "pytorch"
        return f"hidden_size={self.hidden_size}, eps={self.variance_epsilon}, backend={fused}"
