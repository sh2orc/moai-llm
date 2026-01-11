"""
Legacy Transformer architecture for MOAI-LLM.

This module contains the original transformer-based implementation.
New models should use the pure Mamba SSM architecture instead.

Deprecated: Use moai_mamba.py for new models.
"""

from moai_llm.modeling.legacy_transformer.attention import MoaiAttention
from moai_llm.modeling.legacy_transformer.decoder import MoaiDecoderLayer
from moai_llm.modeling.legacy_transformer.model import (
    MoaiPreTrainedModel,
    MoaiModel,
    MoaiForCausalLM,
)

__all__ = [
    "MoaiAttention",
    "MoaiDecoderLayer",
    "MoaiPreTrainedModel",
    "MoaiModel",
    "MoaiForCausalLM",
]
