"""Modeling components for MOAI-LLM."""

# Legacy Transformer (deprecated - use Mamba for new models)
from moai_llm.modeling.normalization import MoaiRMSNorm
from moai_llm.modeling.activations import MoaiSwiGLU
from moai_llm.modeling.rope import MoaiRotaryEmbedding, apply_rotary_pos_emb
from moai_llm.modeling.legacy_transformer.attention import MoaiAttention
from moai_llm.modeling.legacy_transformer.decoder import MoaiDecoderLayer
from moai_llm.modeling.legacy_transformer.model import MoaiForCausalLM as MoaiTransformerForCausalLM
from moai_llm.modeling.legacy_transformer.model import MoaiModel as MoaiTransformerModel

# Moai-Mamba pure SSM architecture
from moai_llm.modeling.ssm_config import (
    MoaiMambaConfig,
    load_mamba_config,
    get_mamba_config,
)
from moai_llm.modeling.ssm import (
    MoaiSSM,
    MoaiSSMBlock,
)
from moai_llm.modeling.moai_mamba import (
    MoaiMambaPreTrainedModel,
    MoaiMambaModel,
    MoaiMambaForCausalLM,
)
from moai_llm.modeling.quantization import (
    QuantizationConfig,
    QuantizedLinear,
    quantize_model,
)
from moai_llm.modeling.advanced_quantization import (
    quantize_fp8,
    quantize_fp4,
    quantize_awq,
    quantize_gptq,
    quantize_bnb,
    get_quantization_info,
    print_quantization_comparison,
)

__all__ = [
    # Legacy Transformer components (deprecated - use Mamba for new models)
    "MoaiRMSNorm",
    "MoaiSwiGLU",
    "MoaiRotaryEmbedding",
    "apply_rotary_pos_emb",
    "MoaiAttention",
    "MoaiDecoderLayer",
    "MoaiTransformerModel",
    "MoaiTransformerForCausalLM",
    # Moai-Mamba pure SSM components (recommended for new models)
    "MoaiMambaConfig",
    "load_mamba_config",
    "get_mamba_config",
    "MoaiSSM",
    "MoaiSSMBlock",
    "MoaiMambaPreTrainedModel",
    "MoaiMambaModel",
    "MoaiMambaForCausalLM",
    # Quantization
    "QuantizationConfig",
    "QuantizedLinear",
    "quantize_model",
    # Advanced Quantization
    "quantize_fp8",
    "quantize_fp4",
    "quantize_awq",
    "quantize_gptq",
    "quantize_bnb",
    "get_quantization_info",
    "print_quantization_comparison",
]

# Backward compatibility: alias old names to legacy transformer
MoaiModel = MoaiTransformerModel
MoaiForCausalLM = MoaiTransformerForCausalLM
