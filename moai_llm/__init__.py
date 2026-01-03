"""
MOAI-LLM: A 3B parameter language model.

This package provides a state-of-the-art transformer implementation with:
- Grouped Query Attention (GQA) with Flash Attention 3
- RoPE with YaRN extension for long context
- SwiGLU activation and RMSNorm
- Multi-objective loss functions
- Hierarchical sequence packing
"""

__version__ = "0.1.0"

from moai_llm.config import MoaiConfig

__all__ = ["MoaiConfig", "__version__"]
