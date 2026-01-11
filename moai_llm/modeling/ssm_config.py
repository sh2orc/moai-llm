"""
Moai-Mamba Configuration for Small LLM Optimization.

Pure Mamba Selective State Space Model (SSM) architecture:
- Selective State Space Model (SSM) for long-context efficiency
- Linear complexity O(L) for efficient processing
- 4-bit quantization support for edge deployment

Config files are stored in configs/ directory:
- mamba_config_2b.json: 2B parameters
- mamba_config_8b.json: 8B parameters
- mamba_config_16b.json: 16B parameters
"""

import json
import os
from transformers import PretrainedConfig
from typing import Optional, Dict, Any


class MoaiMambaConfig(PretrainedConfig):
    """
    Configuration for pure Moai-Mamba SSM model.

    Key Features:
    - Mamba SSM layers for efficient long-context processing
    - Linear complexity O(L) vs O(L²) for attention
    - 4-bit quantization ready

    Args:
        # Model Architecture
        vocab_size (int): Vocabulary size (default: 128000)
        hidden_size (int): Hidden dimension (default: 2048 for small LLM)
        num_hidden_layers (int): Total SSM layers (default: 24)
        intermediate_size (int): FFN intermediate size (default: 5632, ~2.75x)

        # Mamba Configuration
        state_size (int): SSM state dimension (default: 16)
        conv_kernel_size (int): Convolutional kernel size (default: 4)
        expand_factor (int): Expansion factor for Mamba (default: 2)
        time_step_rank (int): Rank for time step projection (default: None, auto)
        use_fast_scan (bool): Use fused selective scan (default: True)
        use_checkpoint (bool): Gradient checkpointing (default: True)

        # Normalization & Activation
        rms_norm_eps (float): RMSNorm epsilon (default: 1e-5)
        layer_norm_type (str): Norm type ("rms" or "layer", default: "rms")

        # Quantization (for edge deployment)
        quantization_bits (int): Quantization bits (default: 4)
        use_quantization (bool): Enable quantization (default: False)

        # Training
        initializer_range (float): Weight init std (default: 0.02)

        # Special Tokens
        pad_token_id (int): Padding token (default: 0)
        bos_token_id (int): BOS token (default: 1)
        eos_token_id (int): EOS token (default: 2)
    """

    model_type = "mamba_moai"

    def __init__(
        self,
        # Model Architecture
        vocab_size: int = 128000,
        hidden_size: int = 2048,
        num_hidden_layers: int = 24,
        intermediate_size: int = 5632,

        # Mamba Configuration
        state_size: int = 16,
        conv_kernel_size: int = 4,
        expand_factor: int = 2,
        time_step_rank: Optional[int] = None,
        use_fast_scan: bool = True,
        use_checkpoint: bool = True,

        # Normalization & Activation
        rms_norm_eps: float = 1e-5,
        layer_norm_type: str = "rms",

        # Quantization
        quantization_bits: int = 4,
        use_quantization: bool = False,

        # Training
        initializer_range: float = 0.02,

        # Special Tokens
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

        # Auto-calculate time_step_rank if not specified (Mamba convention)
        if time_step_rank is None:
            time_step_rank = max(1, hidden_size // 16)

        # Model Architecture
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size

        # Mamba Configuration
        self.state_size = state_size
        self.conv_kernel_size = conv_kernel_size
        self.expand_factor = expand_factor
        self.time_step_rank = time_step_rank
        self.use_fast_scan = use_fast_scan
        self.use_checkpoint = use_checkpoint

        # Normalization & Activation
        self.rms_norm_eps = rms_norm_eps
        self.layer_norm_type = layer_norm_type

        # Quantization
        self.quantization_bits = quantization_bits
        self.use_quantization = use_quantization

        # Training
        self.initializer_range = initializer_range

        # Validate configuration
        self._validate_config()

    def _validate_config(self):
        """Validate model configuration."""
        # Validate layer_norm_type
        if self.layer_norm_type not in ["rms", "layer"]:
            raise ValueError(
                f"layer_norm_type must be 'rms' or 'layer', got {self.layer_norm_type}"
            )

    @property
    def mamba_expand_dim(self) -> int:
        """Expanded dimension for Mamba projection."""
        return self.hidden_size * self.expand_factor


def load_mamba_config(config_path: str) -> MoaiMambaConfig:
    """
    Load Moai-Mamba config from JSON file.

    Args:
        config_path: Path to config JSON file (relative to project root or absolute)

    Returns:
        MoaiMambaConfig instance

    Example:
        >>> config = load_mamba_config("configs/mamba_config_2b.json")
        >>> model = MoaiMambaForCausalLM(config)
    """
    # Handle relative paths
    if not os.path.isabs(config_path):
        # Try relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_path = os.path.join(project_root, config_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    return MoaiMambaConfig(**config_dict)


def get_mamba_config(model_size: str = "2b") -> MoaiMambaConfig:
    """
    Get Moai-Mamba config for specified model size.

    Args:
        model_size: Model size ("2b", "8b", "16b")

    Returns:
        MoaiMambaConfig instance

    Example:
        >>> config = get_mamba_config("2b")
        >>> model = MoaiMambaForCausalLM(config)
    """
    config_map = {
        "2b": "configs/mamba_config_2b.json",
        "8b": "configs/mamba_config_8b.json",
        "16b": "configs/mamba_config_16b.json",
    }

    if model_size not in config_map:
        raise ValueError(
            f"Unknown model size: {model_size}. "
            f"Available sizes: {list(config_map.keys())}"
        )

    return load_mamba_config(config_map[model_size])
