#!/usr/bin/env python3
"""
Test script for Moai-Mamba model.

Verifies:
- Model creation from JSON config
- Forward pass
- Quantization
- Basic generation
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch


def test_moai_mamba_2b():
    """Test Moai-Mamba-2B model."""
    print("=" * 60)
    print("Testing Moai-Mamba-2B Model")
    print("=" * 60)

    try:
        from moai_llm import get_mamba_config, MoaiMambaForCausalLM

        # Create config from JSON
        print("\n1. Creating config from JSON...")
        config = get_mamba_config("2b")
        print("   Config created from configs/mamba_config_2b.json")
        print(f"   - Hidden size: {config.hidden_size}")
        print(f"   - Layers: {config.num_hidden_layers} ({config.mamba_layers} SSM + {config.attention_layers} Attention)")
        print(f"   - State size: {config.state_size}")
        print(f"   - Max position embeddings: {config.max_position_embeddings}")

        # Create model
        print("\n2. Creating model...")
        model = MoaiMambaForCausalLM(config)
        print("   Model created")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   - Total parameters: {total_params:,}")
        print(f"   - Trainable parameters: {trainable_params:,}")

        # Test forward pass
        print("\n3. Testing forward pass...")
        batch_size = 2
        seq_length = 128
        vocab_size = config.vocab_size

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        print("   Forward pass successful")
        print(f"   - Logits shape: {outputs.logits.shape}")
        print(f"   - Expected: ({batch_size}, {seq_length}, {vocab_size})")

        # Test generation
        print("\n4. Testing generation...")
        model.eval()
        with torch.no_grad():
            generated = model.generate(
                input_ids[:1],
                max_length=seq_length + 10,
                do_sample=False,
            )

        print("   Generation successful")
        print(f"   - Input shape: {input_ids[:1].shape}")
        print(f"   - Generated shape: {generated.shape}")

        print("\nAll tests passed for Moai-Mamba-2B!")
        return True

    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_moai_mamba_8b():
    """Test Moai-Mamba-8B model."""
    print("\n" + "=" * 60)
    print("Testing Moai-Mamba-8B Model")
    print("=" * 60)

    try:
        from moai_llm import get_mamba_config, MoaiMambaForCausalLM

        # Create config from JSON
        print("\n1. Creating config from JSON...")
        config = get_mamba_config("8b")
        print("   Config created from configs/mamba_config_8b.json")
        print(f"   - Hidden size: {config.hidden_size}")
        print(f"   - Layers: {config.num_hidden_layers} ({config.mamba_layers} SSM + {config.attention_layers} Attention)")
        print(f"   - State size: {config.state_size}")

        # Create model
        print("\n2. Creating model...")
        model = MoaiMambaForCausalLM(config)
        print("   Model created")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   - Total parameters: {total_params:,}")

        # Test forward pass
        print("\n3. Testing forward pass...")
        batch_size = 2
        seq_length = 64
        vocab_size = config.vocab_size

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        print("   Forward pass successful")
        print(f"   - Logits shape: {outputs.logits.shape}")

        print("\nAll tests passed for Moai-Mamba-8B!")
        return True

    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_quantization():
    """Test 4-bit quantization."""
    print("\n" + "=" * 60)
    print("Testing 4-bit Quantization")
    print("=" * 60)

    try:
        from moai_llm import get_mamba_config, MoaiMambaForCausalLM, quantize_model

        # Create model
        print("\n1. Creating model...")
        config = get_mamba_config("2b")
        model = MoaiMambaForCausalLM(config)
        print("   Model created")

        # Get original memory
        original_mem = sum(p.numel() * p.element_size() for p in model.parameters())
        print(f"\n2. Original model memory: {original_mem / 1024 / 1024:.2f} MB")

        # Quantize
        print("\n3. Quantizing to 4-bit...")
        model = quantize_model(model, bits=4, group_size=128)
        print("   Model quantized")

        # Get quantized memory (approximate)
        # 4-bit = 0.5 bytes per parameter vs 4 bytes for FP32
        quantized_mem = original_mem * 0.5 / 4
        print(f"\n4. Quantized model memory (approx): {quantized_mem / 1024 / 1024:.2f} MB")
        print(f"   - Memory reduction: {original_mem / quantized_mem:.2f}x")

        # Test forward pass
        print("\n5. Testing forward pass with quantized model...")
        vocab_size = config.vocab_size
        input_ids = torch.randint(0, vocab_size, (1, 32))
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        print("   Forward pass successful")
        print(f"   - Logits shape: {outputs.logits.shape}")

        print("\nAll tests passed for 4-bit quantization!")
        return True

    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Moai-Mamba Test Suite")
    print("=" * 60)

    results = []

    # Test 1: 2B model
    results.append(("Moai-Mamba-2B", test_moai_mamba_2b()))

    # Test 2: 8B model
    results.append(("Moai-Mamba-8B", test_moai_mamba_8b()))

    # Test 3: Quantization
    results.append(("4-bit Quantization", test_quantization()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for test_name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"{test_name}: {status}")

    all_passed = all(r[1] for r in results)

    if all_passed:
        print("\nAll tests passed!")
        return 0
    else:
        print("\nSome tests failed!")
        return 1


if __name__ == "__main__":
    exit(main())
