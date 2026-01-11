"""
MOAI-LLM: A 3B parameter language model.

This package provides state-of-the-art transformer implementations with:
- Grouped Query Attention (GQA) with Flash Attention 3
- RoPE with YaRN extension for long context
- SwiGLU activation and RMSNorm
- Multi-objective loss functions
- Hierarchical sequence packing

NEW: Moai-Mamba Hybrid Architecture
- Selective State Space Model for linear complexity
- Prefix-LM with bidirectional prefix
- 4-bit quantization for edge deployment
- Optimized for small LLMs (300M-1.5B parameters)
"""

__version__ = "0.2.0"

from moai_llm.config import MoaiConfig

# Moai-Mamba hybrid components
from moai_llm.modeling.ssm_config import (
    MoaiMambaConfig,
    load_mamba_config,
    get_mamba_config,
)
from moai_llm.modeling.moai_mamba import MoaiMambaForCausalLM
from moai_llm.modeling.quantization import quantize_model

__all__ = [
    "__version__",
    # Original Moai
    "MoaiConfig",
    # Moai-Mamba
    "MoaiMambaConfig",
    "load_mamba_config",
    "get_mamba_config",
    "MoaiMambaForCausalLM",
    "quantize_model",
]
