"""
Moai-Mamba Pure SSM Model Implementation.

Pure Mamba Selective State Space Model architecture:
- All layers use Mamba SSM (no attention)
- Linear complexity O(L) for efficient processing
- 4-bit quantization ready for edge deployment

Benefits for Small LLMs:
- Efficient: 16x faster than attention for long sequences
- Memory-effective: Linear memory vs quadratic for attention
- Long-context: Native support for 32K+ tokens without RAG
- Edge-friendly: Quantization ready for mobile/embedded deployment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast

from moai_llm.modeling.ssm_config import MoaiMambaConfig
from moai_llm.modeling.ssm import MoaiSSMBlock
from moai_llm.modeling.normalization import MoaiRMSNorm


class MoaiMambaPreTrainedModel(PreTrainedModel):
    """Base class for Moai-Mamba models."""

    config_class = MoaiMambaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MoaiSSMBlock"]

    def _init_weights(self, module):
        """Initialize model weights."""
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class MoaiMambaModel(MoaiMambaPreTrainedModel):
    """
    Pure Moai-Mamba SSM model.

    Architecture:
        - Embeddings -> [SSM layers] -> Norm
        - All layers use Mamba SSM for linear complexity
        - No attention layers

    Args:
        config: MoaiMambaConfig
    """

    def __init__(self, config: MoaiMambaConfig):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Token embeddings
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
        )

        # Build SSM layers
        self.layers = nn.ModuleList()
        for layer_idx in range(config.num_hidden_layers):
            self.layers.append(
                MoaiSSMBlock(config, layer_idx)
            )

        # Final layer normalization
        self.norm = MoaiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Gradient checkpointing
        self.gradient_checkpointing = False

        # Initialize weights
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        Forward pass for Moai-Mamba model.

        Args:
            input_ids: Token indices (batch_size, seq_len)
            inputs_embeds: Pre-computed embeddings
            output_hidden_states: Whether to output all hidden states
            return_dict: Whether to return ModelOutput

        Returns:
            Model outputs with hidden_states
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Retrieve input embeddings
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # Get embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Initialize hidden states
        hidden_states = inputs_embeds

        # SSM layers
        all_hidden_states = () if output_hidden_states else None

        for idx, ssm_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                # Use gradient checkpointing
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(ssm_layer),
                    hidden_states,
                    use_reentrant=False,
                )
            else:
                hidden_states = ssm_layer(hidden_states)

        # Final layer norm
        hidden_states = self.norm(hidden_states)

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=None,
        )


class MoaiMambaForCausalLM(MoaiMambaPreTrainedModel, GenerationMixin):
    """
    Moai-Mamba model with language modeling head.

    Pure SSM architecture with no attention layers.

    Args:
        config: MoaiMambaConfig
    """

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = MoaiMambaModel(config)
        self.vocab_size = config.vocab_size

        # Language modeling head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass for causal language modeling.

        Args:
            input_ids: Token indices
            inputs_embeds: Pre-computed embeddings
            labels: Labels for computing loss
            output_hidden_states: Whether to output all hidden states
            return_dict: Whether to return ModelOutput

        Returns:
            CausalLMOutputWithPast with loss, logits, hidden_states
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Decoder outputs
        outputs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        # Language modeling head
        logits = self.lm_head(hidden_states)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Use chunked cross-entropy for memory efficiency
            from moai_llm.losses import chunked_cross_entropy_loss
            loss = chunked_cross_entropy_loss(
                shift_logits,
                shift_labels,
                chunk_size=2048,
                ignore_index=-100,
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=outputs.hidden_states,
            attentions=None,
        )

    def prepare_inputs_for_generation(
        self, input_ids, **kwargs
    ):
        """Prepare inputs for generation."""
        return {
            "input_ids": input_ids,
            **kwargs
        }
