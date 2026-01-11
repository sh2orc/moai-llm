#!/usr/bin/env python3
"""
Text generation script for Moai-Mamba model.

Supports:
- Interactive chat mode
- Batch text generation
- Temperature sampling
- Top-k and top-p sampling
- Quantized model inference

Usage:
    # Interactive chat
    python scripts/generate_mamba.py --model_path outputs/moai_mamba/final --chat

    # Single prompt generation
    python scripts/generate_mamba.py --model_path outputs/moai_mamba/final --prompt "Hello, world!"

    # Batch generation from file
    python scripts/generate_mamba.py --model_path outputs/moai_mamba/final --input_file prompts.txt

    # With custom parameters
    python scripts/generate_mamba.py --model_path outputs/moai_mamba/final --prompt "Hello" --temperature 0.8 --top_k 50
"""

import os
import sys
import argparse
from typing import List, Optional

import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moai_llm.modeling.ssm_config import load_mamba_config
from moai_llm.modeling.moai_mamba import MoaiMambaForCausalLM


def load_model(model_path: str, device: str = "cuda"):
    """Load model from checkpoint."""
    print(f"Loading model from {model_path}...")

    # Load config
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        config = load_mamba_config(config_path)
    else:
        # Use default config
        from moai_llm.modeling.ssm_config import get_mamba_config
        config = get_mamba_config("2b")

    # Load model
    model = MoaiMambaForCausalLM.from_pretrained(model_path)
    model = model.to(device)
    model.eval()

    print(f"Model loaded on {device}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    return model, config


def load_tokenizer(tokenizer_path: str):
    """Load tokenizer."""
    from transformers import AutoTokenizer

    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Tokenizer loaded (vocab_size: {tokenizer.vocab_size})")

    return tokenizer


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    do_sample: bool = True,
    device: str = "cuda",
):
    """Generate text from prompt."""
    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    input_length = inputs["input_ids"].shape[1]

    # Generate
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the generated part
    response = generated_text[len(prompt):].strip()

    return response


def interactive_chat(model, tokenizer, device: str = "cuda"):
    """Interactive chat mode."""
    print("\n" + "=" * 60)
    print("Moai-Mamba Interactive Chat")
    print("=" * 60)
    print("Type 'quit' or 'exit' to end the conversation")
    print("Type 'clear' to clear conversation history")
    print()

    conversation_history = []

    while True:
        # Get user input
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break

        if user_input.lower() == 'clear':
            conversation_history = []
            print("Conversation history cleared.")
            continue

        if not user_input:
            continue

        # Add to history
        conversation_history.append(f"User: {user_input}")

        # Format prompt
        prompt = "\n".join(conversation_history) + "\nAssistant:"

        # Generate response
        response = generate_text(
            model,
            tokenizer,
            prompt,
            max_new_tokens=256,
            temperature=0.7,
            device=device,
        )

        print(f"\nAssistant: {response}\n")

        # Add to history
        conversation_history.append(f"Assistant: {response}")


def batch_generation(
    model,
    tokenizer,
    prompts: List[str],
    output_file: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    device: str = "cuda",
):
    """Generate text for multiple prompts."""
    print(f"\nGenerating text for {len(prompts)} prompts...")

    results = []

    for i, prompt in enumerate(prompts):
        print(f"[{i+1}/{len(prompts)}] Generating...")

        response = generate_text(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=device,
        )

        results.append({
            "prompt": prompt,
            "response": response,
        })

        print(f"Response: {response[:100]}...")

    # Save results
    if output_file:
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate text with Moai-Mamba model")

    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Path to tokenizer (default: model_path)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")

    # Generation arguments
    parser.add_argument("--prompt", type=str, default=None,
                        help="Input prompt")
    parser.add_argument("--input_file", type=str, default=None,
                        help="File containing prompts (one per line)")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output file for batch generation")

    # Generation parameters
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling parameter")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling parameter")
    parser.add_argument("--no_sample", action="store_true",
                        help="Use greedy decoding (no sampling)")

    # Mode
    parser.add_argument("--chat", action="store_true",
                        help="Interactive chat mode")

    args = parser.parse_args()

    # Load model
    model, config = load_model(args.model_path, args.device)

    # Load tokenizer
    tokenizer_path = args.tokenizer_path if args.tokenizer_path else args.model_path
    tokenizer = load_tokenizer(tokenizer_path)

    # Run generation
    if args.chat:
        # Interactive chat mode
        interactive_chat(model, tokenizer, args.device)

    elif args.input_file:
        # Batch generation from file
        with open(args.input_file, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]

        batch_generation(
            model,
            tokenizer,
            prompts,
            args.output_file,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            device=args.device,
        )

    elif args.prompt:
        # Single prompt generation
        print(f"\nPrompt: {args.prompt}")
        print("-" * 60)

        response = generate_text(
            model,
            tokenizer,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            do_sample=not args.no_sample,
            device=args.device,
        )

        print(f"Response: {response}")

        # Save to file if specified
        if args.output_file:
            import json
            result = {
                "prompt": args.prompt,
                "response": response,
            }
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\nResponse saved to {args.output_file}")

    else:
        print("Error: Please provide --prompt, --input_file, or --chat")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
