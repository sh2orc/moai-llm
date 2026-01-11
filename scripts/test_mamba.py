#!/usr/bin/env python3
"""
Test script for Moai-Mamba model (Pure SSM Architecture).

Verifies:
- Model creation from JSON config
- Forward pass
- Loss computation
- Basic generation
- Quantization
- All model sizes (2B, 8B, 16B)

Usage:
    # Test all models
    python scripts/test_mamba.py

    # Test specific model size
    python scripts/test_mamba.py --config 2b

    # Test with verbose output
    python scripts/test_mamba.py --verbose

    # Skip quantization test
    python scripts/test_mamba.py --skip-quantization
"""

import os
import sys
import argparse

import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moai_llm.modeling.ssm_config import get_mamba_config, load_mamba_config
from moai_llm.modeling.moai_mamba import MoaiMambaForCausalLM
from moai_llm.modeling.quantization import quantize_model, QuantizationConfig


def test_model_creation(config_size: str = "2b", verbose: bool = False):
    """Test model creation from JSON config."""
    print("=" * 60)
    print(f"Testing Moai-Mamba-{config_size.upper()} Model Creation")
    print("=" * 60)

    try:
        # Create config from JSON
        print(f"\n1. Creating config from JSON...")
        config = get_mamba_config(config_size)
        print(f"   Config loaded from configs/mamba_config_{config_size}.json")

        if verbose:
            print(f"\n   Configuration:")
            print(f"   - Hidden size: {config.hidden_size}")
            print(f"   - Num layers: {config.num_hidden_layers}")
            print(f"   - State size: {config.state_size}")
            print(f"   - Conv kernel size: {config.conv_kernel_size}")
            print(f"   - Expand factor: {config.expand_factor}")
            print(f"   - Intermediate size: {config.intermediate_size}")
            print(f"   - Max position embeddings: {config.max_position_embeddings}")

        # Create model
        print(f"\n2. Creating model...")
        model = MoaiMambaForCausalLM(config)
        print("   Model created successfully")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        if verbose:
            print(f"\n   Model statistics:")
            print(f"   - Total parameters: {total_params:,}")
            print(f"   - Trainable parameters: {trainable_params:,}")
            print(f"   - Model size (approx): {total_params * 4 / 1024 / 1024 / 1024:.2f} GB (FP32)")

        print(f"\n   Total parameters: {total_params:,}")

        return model, config, True

    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, False


def test_forward_pass(model, config, verbose: bool = False):
    """Test forward pass."""
    print("\n" + "=" * 60)
    print("Testing Forward Pass")
    print("=" * 60)

    try:
        model.eval()

        # Test inputs
        batch_size = 2
        seq_length = 128
        vocab_size = config.vocab_size

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))

        if verbose:
            print(f"\n   Input shape: {input_ids.shape}")
            print(f"   Batch size: {batch_size}")
            print(f"   Sequence length: {seq_length}")

        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids=input_ids)

        print("\n   Forward pass successful")

        if verbose:
            print(f"   Logits shape: {outputs.logits.shape}")
            print(f"   Expected: ({batch_size}, {seq_length}, {vocab_size})")
            print(f"   Logits range: [{outputs.logits.min():.2f}, {outputs.logits.max():.2f}]")

        print(f"   Logits shape: {outputs.logits.shape}")

        return True

    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_loss_computation(model, config, verbose: bool = False):
    """Test loss computation."""
    print("\n" + "=" * 60)
    print("Testing Loss Computation")
    print("=" * 60)

    try:
        model.eval()

        # Test inputs
        batch_size = 2
        seq_length = 64
        vocab_size = config.vocab_size

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
        labels = input_ids.clone()

        if verbose:
            print(f"\n   Input shape: {input_ids.shape}")
            print(f"   Labels shape: {labels.shape}")

        # Compute loss
        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=labels)

        print("\n   Loss computation successful")

        if verbose:
            print(f"   Loss value: {outputs.loss.item():.4f}")

        print(f"   Loss: {outputs.loss.item():.4f}")

        # Loss should be positive and finite
        assert outputs.loss.item() > 0, "Loss should be positive"
        assert not torch.isnan(outputs.loss), "Loss should not be NaN"
        assert not torch.isinf(outputs.loss), "Loss should not be infinite"

        print("   Loss validation passed")

        return True

    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_generation(model, config, verbose: bool = False):
    """Test text generation."""
    print("\n" + "=" * 60)
    print("Testing Text Generation")
    print("=" * 60)

    try:
        model.eval()

        # Test input
        batch_size = 1
        seq_length = 32
        vocab_size = config.vocab_size

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))

        if verbose:
            print(f"\n   Input shape: {input_ids.shape}")

        # Generate
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_new_tokens=10,
                do_sample=False,
            )

        print("\n   Generation successful")

        if verbose:
            print(f"   Input shape: {input_ids.shape}")
            print(f"   Generated shape: {generated.shape}")
            print(f"   Generated tokens: {generated[0].tolist()[-10:]}")

        print(f"   Input shape: {input_ids.shape}")
        print(f"   Generated shape: {generated.shape}")

        # Verify generation length
        expected_length = seq_length + 10
        assert generated.shape[1] == expected_length, f"Expected length {expected_length}, got {generated.shape[1]}"

        print("   Generation validation passed")

        return True

    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_quantization(verbose: bool = False):
    """Test 4-bit quantization."""
    print("\n" + "=" * 60)
    print("Testing 4-bit Quantization")
    print("=" * 60)

    try:
        # Create model
        print("\n1. Creating model...")
        config = get_mamba_config("2b")
        model = MoaiMambaForCausalLM(config)
        model.eval()

        # Get original memory
        original_params = sum(p.numel() for p in model.parameters())
        original_mem = original_params * 4  # FP32 = 4 bytes per param

        print(f"   Model created")
        print(f"   Original memory: {original_mem / 1024 / 1024:.2f} MB (FP32)")

        if verbose:
            print(f"   Original parameters: {original_params:,}")

        # Quantize
        print("\n2. Quantizing to 4-bit...")
        quantized_model = quantize_model(model, bits=4, group_size=128)
        print("   Model quantized")

        # Calculate memory reduction
        # 4-bit = 0.5 bytes per parameter vs 4 bytes for FP32
        quantized_mem = original_mem * 0.5 / 4

        print(f"\n3. Memory comparison:")
        print(f"   Original: {original_mem / 1024 / 1024:.2f} MB")
        print(f"   Quantized: {quantized_mem / 1024 / 1024:.2f} MB")
        print(f"   Reduction: {original_mem / quantized_mem:.2f}x")

        # Test forward pass with quantized model
        print("\n4. Testing forward pass with quantized model...")
        vocab_size = config.vocab_size
        input_ids = torch.randint(0, vocab_size, (1, 32))

        with torch.no_grad():
            outputs = quantized_model(input_ids=input_ids)

        print("   Forward pass successful")
        print(f"   Logits shape: {outputs.logits.shape}")

        print("\n   Quantization test passed")

        return True

    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_all_model_sizes(verbose: bool = False):
    """Test all available model sizes."""
    print("\n" + "=" * 60)
    print("Testing All Model Sizes")
    print("=" * 60)

    results = []

    for config_size in ["2b", "8b", "16b"]:
        print(f"\n--- Testing {config_size.upper()} ---")

        try:
            config = get_mamba_config(config_size)
            model = MoaiMambaForCausalLM(config)
            model.eval()

            total_params = sum(p.numel() for p in model.parameters())

            print(f"Parameters: {total_params:,}")

            # Quick forward pass test
            vocab_size = config.vocab_size
            input_ids = torch.randint(0, vocab_size, (1, 32))

            with torch.no_grad():
                outputs = model(input_ids=input_ids)

            print(f"Forward pass: OK")
            print(f"Logits shape: {outputs.logits.shape}")

            results.append((config_size.upper(), True, None))

        except Exception as e:
            print(f"FAILED: {e}")
            results.append((config_size.upper(), False, str(e)))

    # Summary
    print("\n" + "=" * 60)
    print("Model Size Test Summary")
    print("=" * 60)

    for size, passed, error in results:
        status = "PASSED" if passed else "FAILED"
        print(f"{size}: {status}")
        if error:
            print(f"  Error: {error}")

    return all(r[1] for r in results)


def main():
    parser = argparse.ArgumentParser(description="Test Moai-Mamba model")
    parser.add_argument("--config", type=str, default="2b", choices=["2b", "8b", "16b"],
                        help="Model size to test")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output")
    parser.add_argument("--skip-quantization", action="store_true",
                        help="Skip quantization test")
    parser.add_argument("--all-sizes", action="store_true",
                        help="Test all model sizes")

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("Moai-Mamba Test Suite")
    print("=" * 60)

    results = []

    # Test all model sizes if requested
    if args.all_sizes:
        results.append(("All Model Sizes", test_all_model_sizes(args.verbose)))
    else:
        # Test specified model size
        print(f"\nTesting model size: {args.config.upper()}\n")

        # Test 1: Model creation
        model, config, creation_ok = test_model_creation(args.config, args.verbose)
        results.append((f"Model Creation ({args.config.upper()})", creation_ok))

        if creation_ok:
            # Test 2: Forward pass
            results.append(("Forward Pass", test_forward_pass(model, config, args.verbose)))

            # Test 3: Loss computation
            results.append(("Loss Computation", test_loss_computation(model, config, args.verbose)))

            # Test 4: Generation
            results.append(("Text Generation", test_generation(model, config, args.verbose)))

        # Test 5: Quantization
        if not args.skip_quantization:
            results.append(("4-bit Quantization", test_quantization(args.verbose)))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for test_name, passed in results:
        status = "PASSED" if passed else "FAILED"
        symbol = "+" if passed else "-"
        print(f"{symbol} {test_name}: {status}")

    all_passed = all(r[1] for r in results)

    print("=" * 60)

    if all_passed:
        print("\nAll tests passed!")
        return 0
    else:
        print("\nSome tests failed!")
        return 1


if __name__ == "__main__":
    exit(main())
