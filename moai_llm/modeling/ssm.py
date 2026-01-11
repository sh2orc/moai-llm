"""
Mamba Selective State Space Model (SSM) Implementation.

Based on "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
https://arxiv.org/abs/2312.00752

Key Features:
- Selective mechanism: input-dependent parameters (A, B, C)
- Hardware-aware selective scan for efficiency
- Linear complexity O(L) vs quadratic O(L²) for attention
- 1D convolution for local context modeling
- Optimized for small LLMs with edge deployment

GLM 4.5 Integration:
- Compatible with GLM's prefix-LM structure
- RoPE position encoding support
- Efficient for bidirectional prefix processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from einops import rearrange, repeat


try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    MAMBA_SSM_AVAILABLE = True
except ImportError:
    MAMBA_SSM_AVAILABLE = False
    selective_scan_fn = None


class MoaiSSM(nn.Module):
    """
    Moai Selective State Space Model block.

    Architecture:
        1. Projection: (B, L, D) -> (B, L, ed) where ed = expand * D
        2. 1D Convolution: Local context modeling
        3. SiLU activation
        4. Selective SSM:
           - Input-dependent A, B, C parameters
           - Hardware-aware selective scan
           - Linear recurrence
        5. Gating: Output projection with gating mechanism

    Args:
        d_model: Hidden dimension
        d_state: SSM state dimension (default: 16)
        d_conv: Convolution kernel size (default: 4)
        expand: Expansion factor (default: 2)
        use_fast_scan: Use fused selective scan (default: True)
        layer_idx: Layer index for debugging
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        use_fast_scan: bool = True,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = expand * d_model
        self.use_fast_scan = use_fast_scan
        self.layer_idx = layer_idx

        # Input projection: d_model -> 2 * d_inner
        # Projects to delta and input paths
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # Convolution for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,  # Depthwise convolution
            bias=True,
        )

        # Projection for x, dt, B, C
        # x: input (d_inner)
        # dt: time step delta (d_inner)
        # B: input-dependent projection matrix (d_state)
        # C: input-dependent output matrix (d_state)
        self.x_proj = nn.Linear(self.d_inner, 2 * d_state + self.d_inner, bias=False)

        # Time step delta projection with bias
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        # SSM parameters
        # A: State transition matrix (learnable but NOT input-dependent)
        # Shape: (d_inner, d_state)
        self.A_log = nn.Parameter(torch.log(torch.rand(self.d_inner, d_state)))

        # D: Skip connection parameter
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection: d_inner -> d_model
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        # Initialize A_log properly (from Mamba paper)
        # A starts from random uniform in [0, 1], then log
        nn.init.uniform_(self.A_log, -1, 0)

        # Initialize conv bias
        nn.init.constant_(self.conv1d.bias, 0)

        # Initialize dt_proj bias
        nn.init.constant_(self.dt_proj.bias, 0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for Mamba SSM.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, d_model)
            attention_mask: Not used in Mamba (for compatibility)
            position_ids: Position indices (for compatibility, not used in Mamba)

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = hidden_states.shape

        # Input projection: (B, L, D) -> (B, L, 2*ed)
        in_proj_out = self.in_proj(hidden_states)
        # Split into delta and input paths
        # xz: (B, L, ed) - will be split into x and z
        xz = rearrange(in_proj_out, "b l (two ed) -> two b l ed", two=2)
        x, z = xz[0], xz[1]  # x: input, z: gate

        # 1D Convolution for local context
        # x: (B, L, ed) -> (B, ed, L) for conv
        x = rearrange(x, "b l ed -> b ed l")
        # Conv with padding
        x = self.conv1d(x)[:, :, :seq_len]  # Truncate to original length
        # x: (B, ed, L) -> (B, L, ed)
        x = rearrange(x, "b ed l -> b l ed")

        # SiLU activation
        x = F.silu(x)

        # Selective SSM
        y = self._selective_scan(x)

        # Gating: y = y * silu(z)
        y = y * F.silu(z)

        # Output projection
        output = self.out_proj(y)

        return output

    def _selective_scan(self, u: torch.Tensor) -> torch.Tensor:
        """
        Selective SSM scan with input-dependent parameters.

        Args:
            u: Input tensor of shape (B, L, d_inner)

        Returns:
            Output tensor of shape (B, L, d_inner)
        """
        batch, seq_len, d_inner = u.shape

        # Project to get B, C, and dt
        # x_proj_out: (B, L, 2*d_state + d_inner)
        x_proj_out = self.x_proj(u)

        # Split into B, C, and delta
        # B: (B, L, d_state) - input-dependent projection matrix
        # C: (B, L, d_state) - input-dependent output matrix
        # delta: (B, L, d_inner) - time step
        B, C, delta = x_proj_out.split([self.d_state, self.d_state, d_inner], dim=-1)

        # Apply softmax to B for normalization
        B = F.softmax(B, dim=-1)

        # Time step projection
        # Apply dt_proj and apply softplus for positivity
        # delta: (B, L, d_inner)
        delta = self.dt_proj(delta)
        delta = F.softplus(delta)

        # State transition matrix A
        # A: (d_inner, d_state) - log space, so exp
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # Use fast selective scan if available
        if self.use_fast_scan and MAMBA_SSM_AVAILABLE:
            return self._selective_scan_fast(u, delta, A, B, C)
        else:
            return self._selective_scan_torch(u, delta, A, B, C)

    def _selective_scan_fast(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """
        Hardware-aware selective scan using mamba_ssm.

        Args:
            u: Input (B, L, d_inner)
            delta: Time step (B, L, d_inner)
            A: State transition (d_inner, d_state)
            B: Projection matrix (B, L, d_state)
            C: Output matrix (B, L, d_state)

        Returns:
            Output (B, L, d_inner)
        """
        batch, seq_len, d_inner = u.shape
        d_state = A.shape[-1]

        # Prepare inputs for selective_scan_fn
        # u: (B, L, ed) -> (B, ed, L)
        u = rearrange(u, "b l ed -> b ed l")

        # delta: (B, L, ed) -> (B, ed, L)
        delta = rearrange(delta, "b l ed -> b ed l")

        # A: (ed, d_state) -> (ed, d_state)
        # B: (B, L, d_state) -> (B, d_state, L)
        B = rearrange(B, "b l d -> b d l")

        # C: (B, L, d_state) -> (B, d_state, L)
        C = rearrange(C, "b l d -> b d l")

        # D: skip connection (ed,)
        D = self.D

        # Call selective scan
        # y: (B, ed, L)
        y = selective_scan_fn(
            u=u,
            delta=delta,
            A=A,
            B=B,
            C=C,
            D=D,
            z=None,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        )

        # y: (B, ed, L) -> (B, L, ed)
        y = rearrange(y, "b ed l -> b l ed")

        return y

    def _selective_scan_torch(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pure PyTorch implementation of selective scan (slower but portable).

        Uses sequential scan: h[t] = A[t] * h[t-1] + B[t] * x[t]
        This is O(L * batch * d_inner * d_state) but avoids CUDA kernels.

        Args:
            u: Input (B, L, d_inner)
            delta: Time step (B, L, d_inner)
            A: State transition (d_inner, d_state)
            B: Projection matrix (B, L, d_state)
            C: Output matrix (B, L, d_state)

        Returns:
            Output (B, L, d_inner)
        """
        batch, seq_len, d_inner = u.shape
        d_state = A.shape[-1]

        # Discretize A and B
        # A_bar: (B, L, d_inner, d_state)
        A_bar = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))

        # B_bar: (B, L, d_inner, d_state)
        B_bar = delta.unsqueeze(-1) * B.unsqueeze(2)

        # Initialize hidden state
        h = torch.zeros(batch, d_inner, d_state, device=u.device, dtype=u.dtype)

        # Sequential scan
        outputs = []
        for t in range(seq_len):
            # h[t] = A_bar[t] * h[t-1] + B_bar[t] * u[t]
            h = A_bar[:, t] * h + B_bar[:, t] * u[:, t].unsqueeze(-1)

            # y[t] = C[t] @ h[t]
            # C: (B, L, d_state), h: (B, d_inner, d_state)
            y_t = torch.einsum("bsd,bds->bs", C[:, t], h)
            outputs.append(y_t)

        # Stack outputs
        y = torch.stack(outputs, dim=1)  # (B, L, d_inner)

        return y


class MoaiSSMBlock(nn.Module):
    """
    Complete Moai SSM block with normalization and residual connection.

    Architecture:
        x = x + SSM(LN(x))

    Compatible with prefix-LM architecture and can be used in
    hybrid SSM-Attention models.

    Args:
        config: MoaiMambaConfig
        layer_idx: Layer index
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # RMSNorm
        from moai_llm.modeling.normalization import MoaiRMSNorm
        self.norm = MoaiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Moai SSM
        self.ssm = MoaiSSM(
            d_model=config.hidden_size,
            d_state=config.state_size,
            d_conv=config.conv_kernel_size,
            expand=config.expand_factor,
            use_fast_scan=config.use_fast_scan,
            layer_idx=layer_idx,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for Moai SSM block.

        Args:
            hidden_states: Input tensor (B, L, D)
            attention_mask: Not used (for compatibility)
            position_ids: Position indices (for compatibility)

        Returns:
            Output tensor (B, L, D)
        """
        # Residual connection
        residual = hidden_states

        # Normalize
        hidden_states = self.norm(hidden_states)

        # Moai SSM
        hidden_states = self.ssm(hidden_states, attention_mask, position_ids)

        # Residual
        hidden_states = residual + hidden_states

        return hidden_states


class MoaiHybridBlock(nn.Module):
    """
    Hybrid Moai block combining SSM efficiency with attention for generation.

    This block implements:
    - Moai SSM for bidirectional prefix processing
    - Optional attention for autoregressive suffix
    - Prefix-LM structure
    - RoPE position encoding

    Args:
        config: MoaiMambaConfig
        layer_idx: Layer index
        layer_type: "ssm" or "attention"
    """

    def __init__(self, config, layer_idx: int, layer_type: str):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = layer_type

        # Input normalization
        from moai_llm.modeling.normalization import MoaiRMSNorm
        self.input_norm = MoaiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if layer_type == "ssm":
            # SSM branch
            self.ssm = MoaiSSM(
                d_model=config.hidden_size,
                d_state=config.state_size,
                d_conv=config.conv_kernel_size,
                expand=config.expand_factor,
                use_fast_scan=config.use_fast_scan,
                layer_idx=layer_idx,
            )
        else:
            # Attention branch (GLM-style)
            from moai_llm.modeling.attention import MoaiAttention
            self.attention = MoaiAttention(config, layer_idx)

        # Post-normalization
        self.post_norm = MoaiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # MLP (shared by both)
        from moai_llm.modeling.activations import MoaiMLP
        self.mlp = MoaiMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple:
        """
        Forward pass for hybrid Moai block.

        Args:
            hidden_states: Input tensor (B, L, D)
            attention_mask: Attention mask
            position_ids: Position indices
            past_key_value: Cached KV pairs (for attention layers)
            output_attentions: Whether to output attention weights
            use_cache: Whether to use KV cache

        Returns:
            Tuple of (hidden_states, present_key_value, attentions)
        """
        residual = hidden_states

        # Input normalization
        hidden_states = self.input_norm(hidden_states)

        # SSM or Attention
        if self.layer_type == "ssm":
            # Moai SSM (no cache, no attention weights)
            hidden_states = self.ssm(hidden_states, attention_mask, position_ids)
            present_key_value = None
            self_attn_weights = None
        else:
            # Attention (with cache and attention weights)
            hidden_states, self_attn_weights, present_key_value = self.attention(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        # Residual
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions and self.layer_type == "attention":
            outputs += (self_attn_weights,)

        if use_cache and self.layer_type == "attention":
            outputs += (present_key_value,)

        return outputs
