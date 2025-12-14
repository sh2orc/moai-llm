"""
Basic tests for MOAI-LLM model.

Run with: pytest tests/test_model.py
"""

import pytest
import torch
from moai_llm.config import MoaiConfig
from moai_llm.modeling.model import MoaiForCausalLM
from moai_llm.losses import create_loss_function


def test_model_initialization():
    """Test model can be initialized with default config."""
    config = MoaiConfig(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    model = MoaiForCausalLM(config)
    assert model is not None
    assert model.config.vocab_size == 1000


def test_forward_pass():
    """Test forward pass produces expected output shapes."""
    config = MoaiConfig(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    model = MoaiForCausalLM(config)
    model.eval()

    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    labels = input_ids.clone()

    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)

    assert outputs.loss is not None
    assert outputs.logits.shape == (batch_size, seq_len, 1000)


def test_gqa_heads():
    """Test GQA head configuration is correct."""
    config = MoaiConfig(
        num_attention_heads=28,
        num_key_value_heads=4,
    )

    assert config.num_kv_groups == 7  # 28 / 4 = 7


def test_loss_functions():
    """Test different loss functions work correctly."""
    vocab_size = 100
    logits = torch.randn(32, vocab_size)
    labels = torch.randint(0, vocab_size, (32,))

    # Test cross-entropy
    ce_loss = create_loss_function({"type": "cross_entropy"})
    loss_ce = ce_loss(logits, labels)
    assert loss_ce.item() > 0

    # Test focal loss
    focal_loss = create_loss_function({"type": "focal", "params": {"gamma": 2.0}})
    loss_focal = focal_loss(logits, labels)
    assert loss_focal.item() > 0

    # Test multi-objective
    multi_loss = create_loss_function({
        "type": "multi_objective",
        "params": {
            "ce_weight": 0.6,
            "focal_weight": 0.3,
            "smooth_weight": 0.1,
        }
    })
    loss_multi = multi_loss(logits, labels)
    assert loss_multi.item() > 0


def test_gradient_checkpointing():
    """Test gradient checkpointing can be enabled."""
    config = MoaiConfig(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    model = MoaiForCausalLM(config)
    model.gradient_checkpointing_enable()

    # Should work without errors
    input_ids = torch.randint(0, 1000, (2, 64))
    labels = input_ids.clone()

    model.train()
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    loss.backward()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
