"""
Configuration class for MOAI-LLM model.

Based on Qwen2.5 architecture with aggressive optimization:
- 28 layers, 3840 hidden size (~3B parameters)
- Grouped Query Attention (30 Q heads, 5 KV heads, 6:1 ratio)
- 128K vocabulary for multilingual support
- RoPE theta=1M for long context support
- QK-Norm for training stability
- YaRN extension capability
"""

from transformers import PretrainedConfig
from typing import Optional


class MoaiConfig(PretrainedConfig):
    """
    Configuration class for MOAI-LLM model.

    Args:
        vocab_size (int): Vocabulary size (default: 128000 for multilingual SentencePiece)
        hidden_size (int): Dimension of hidden layers (default: 3840)
        intermediate_size (int): Dimension of FFN intermediate layer (default: 10240, ~2.67x for SwiGLU)
        num_hidden_layers (int): Number of transformer layers (default: 28)
        num_attention_heads (int): Number of query attention heads (default: 30)
        num_key_value_heads (int): Number of key/value attention heads for GQA (default: 5)
        max_position_embeddings (int): Maximum sequence length (default: 32768, Qwen2.5 standard)
        rope_theta (float): Base frequency for RoPE (default: 1000000.0, Qwen2.5 for long context)
        rope_scaling (dict): Configuration for RoPE scaling (YaRN, NTK, etc.)
        rms_norm_eps (float): Epsilon for RMSNorm (default: 1e-6)
        use_qk_norm (bool): Whether to use QK normalization (default: True, Qwen2.5 feature)
        hidden_act (str): Activation function (default: "swiglu")
        tie_word_embeddings (bool): Tie input/output embeddings (default: False)
        attention_dropout (float): Dropout for attention (default: 0.0)
        initializer_range (float): Std for weight initialization (default: 0.02)
        use_cache (bool): Whether to use KV cache during generation (default: True)
        attention_bias (bool): Whether to use bias in attention projection (default: False, Qwen3 style)
        mlp_bias (bool): Whether to use bias in MLP (default: False)
    """

    model_type = "moai"

    def __init__(
        self,
        vocab_size: int = 128000,
        hidden_size: int = 3840,
        intermediate_size: int = 10240,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 30,
        num_key_value_heads: int = 5,
        max_position_embeddings: int = 32768,  # Qwen3: 40960, 우리는 32K로 시작
        rope_theta: float = 1000000.0,  # Qwen3 기본값
        rope_scaling: Optional[dict] = None,
        rms_norm_eps: float = 1e-6,  # Qwen3와 동일
        use_qk_norm: bool = True,  # Qwen3 안정화 기법
        hidden_act: str = "swiglu",  # SwiGLU (Qwen3은 silu)
        tie_word_embeddings: bool = False,  # Qwen3: False
        attention_dropout: float = 0.0,  # Qwen3: 0.0
        initializer_range: float = 0.02,  # Qwen3와 동일
        use_cache: bool = True,
        attention_bias: bool = False,  # Qwen3: False (no bias)
        mlp_bias: bool = False,  # Qwen3 스타일 (no bias)
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.rms_norm_eps = rms_norm_eps
        self.use_qk_norm = use_qk_norm
        self.hidden_act = hidden_act
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias

        # Validate configuration
        self._validate_config()

    def _validate_config(self):
        """Validate model configuration."""
        # Check if hidden_size is divisible by num_attention_heads
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )

        # Check if num_attention_heads is divisible by num_key_value_heads (for GQA)
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_key_value_heads}) for Grouped Query Attention"
            )

        # Validate RoPE scaling configuration
        if self.rope_scaling is not None:
            if not isinstance(self.rope_scaling, dict):
                raise ValueError("rope_scaling must be a dictionary")

            scaling_type = self.rope_scaling.get("type")
            if scaling_type not in [None, "linear", "dynamic", "yarn", "ntk", "longrope"]:
                raise ValueError(
                    f"rope_scaling type must be one of 'linear', 'dynamic', 'yarn', 'ntk', 'longrope', "
                    f"got {scaling_type}"
                )

            if scaling_type in ["yarn", "ntk", "longrope"]:
                if "factor" not in self.rope_scaling:
                    raise ValueError(f"rope_scaling with type '{scaling_type}' requires 'factor' parameter")

    @property
    def head_dim(self) -> int:
        """Dimension of each attention head."""
        return self.hidden_size // self.num_attention_heads

    @property
    def num_kv_groups(self) -> int:
        """Number of query groups per key/value head."""
        return self.num_attention_heads // self.num_key_value_heads


# Predefined configurations
class MoaiConfig3B(MoaiConfig):
    """
    3B parameter configuration optimized for Qwen3 architecture.

    - Qwen3-style embeddings (rope_theta=1M, max_seq=32K)
    - GQA with 6:1 ratio (30 Q heads, 5 KV heads)
    - QK-Norm for training stability
    - No bias in attention/MLP (modern LLM standard)
    """
    def __init__(self, **kwargs):
        super().__init__(
            vocab_size=128000,
            hidden_size=3840,
            intermediate_size=10240,
            num_hidden_layers=28,
            num_attention_heads=30,
            num_key_value_heads=5,
            max_position_embeddings=32768,  # Qwen3 long context
            rope_theta=1000000.0,  # Qwen3 default
            **kwargs
        )


class MoaiConfig3BConservative(MoaiConfig):
    """3B parameter configuration with conservative settings."""
    def __init__(self, **kwargs):
        super().__init__(
            vocab_size=64000,
            hidden_size=4096,
            intermediate_size=11008,
            num_hidden_layers=24,
            num_attention_heads=32,
            num_key_value_heads=8,
            max_position_embeddings=4096,
            **kwargs
        )
