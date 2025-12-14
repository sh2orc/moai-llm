"""
Evaluation script for MOAI-LLM.

Evaluates model on common benchmarks and computes metrics like perplexity.

Usage:
    python scripts/evaluate.py \
        --model_path outputs/moai-3b-run1/final_model \
        --tokenizer_path tokenizers/moai_tokenizer \
        --dataset wikipedia \
        --split test
"""

import argparse
import torch
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np

from transformers import AutoTokenizer
from datasets import load_dataset
import torch.nn.functional as F

from moai_llm.modeling.model import MoaiForCausalLM
from moai_llm.config import MoaiConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def compute_perplexity(model, tokenizer, dataset, max_length=2048, stride=512, device="cuda"):
    """
    Compute perplexity on a dataset.

    Args:
        model: Language model
        tokenizer: Tokenizer
        dataset: HuggingFace dataset
        max_length: Maximum sequence length
        stride: Stride for sliding window
        device: Device for computation

    Returns:
        Perplexity value
    """
    model.eval()
    model.to(device)

    nlls = []
    total_tokens = 0

    logger.info("Computing perplexity...")

    for example in tqdm(dataset, desc="Evaluating"):
        text = example["text"] if "text" in example else str(example)

        # Tokenize
        encodings = tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids.to(device)

        # Compute perplexity with sliding window
        seq_len = input_ids.size(1)

        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - begin_loc
            input_ids_chunk = input_ids[:, begin_loc:end_loc]

            target_ids = input_ids_chunk.clone()
            target_ids[:, :-trg_len] = -100  # Ignore prefix for loss

            with torch.no_grad():
                outputs = model(input_ids_chunk, labels=target_ids)
                neg_log_likelihood = outputs.loss * trg_len

            nlls.append(neg_log_likelihood)
            total_tokens += trg_len

            # Stop if we've covered the sequence
            if end_loc == seq_len:
                break

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / total_tokens)

    return ppl.item()


def generate_samples(model, tokenizer, prompts, max_new_tokens=128, temperature=0.7, top_p=0.9, device="cuda"):
    """
    Generate text samples from prompts.

    Args:
        model: Language model
        tokenizer: Tokenizer
        prompts: List of prompt strings
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        device: Device for computation

    Returns:
        List of generated texts
    """
    model.eval()
    model.to(device)

    generated_texts = []

    logger.info("Generating samples...")

    for prompt in tqdm(prompts, desc="Generating"):
        # Tokenize prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_texts.append(generated_text)

    return generated_texts


def main():
    parser = argparse.ArgumentParser(description="Evaluate MOAI-LLM")

    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
        help="Path to tokenizer",
    )

    # Data arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext",
        help="Dataset name for evaluation (default: wikitext)",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="wikitext-2-raw-v1",
        help="Dataset configuration (default: wikitext-2-raw-v1)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split (default: test)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (default: all)",
    )

    # Evaluation arguments
    parser.add_argument(
        "--compute_perplexity",
        action="store_true",
        help="Compute perplexity on dataset",
    )
    parser.add_argument(
        "--generate_samples",
        action="store_true",
        help="Generate text samples",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=["Once upon a time", "The future of AI is"],
        help="Prompts for text generation",
    )

    # Generation parameters
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate (default: 128)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling parameter (default: 0.9)",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for computation (default: cuda if available)",
    )

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("MOAI-LLM Evaluation")
    logger.info("="*80)

    # Load tokenizer
    logger.info(f"Loading tokenizer from: {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    logger.info(f"Loading model from: {args.model_path}")
    model = MoaiForCausalLM.from_pretrained(args.model_path)

    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,} ({total_params/1e9:.2f}B)")

    # Compute perplexity
    if args.compute_perplexity:
        logger.info("="*80)
        logger.info("Computing Perplexity")
        logger.info("="*80)

        # Load dataset
        logger.info(f"Loading dataset: {args.dataset} ({args.dataset_config})")
        dataset = load_dataset(args.dataset, args.dataset_config, split=args.split)

        if args.max_samples:
            dataset = dataset.select(range(min(args.max_samples, len(dataset))))

        logger.info(f"Dataset size: {len(dataset)}")

        # Compute perplexity
        ppl = compute_perplexity(model, tokenizer, dataset, device=args.device)

        logger.info("="*80)
        logger.info(f"Perplexity: {ppl:.2f}")
        logger.info("="*80)

    # Generate samples
    if args.generate_samples:
        logger.info("="*80)
        logger.info("Generating Text Samples")
        logger.info("="*80)

        generated_texts = generate_samples(
            model,
            tokenizer,
            args.prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=args.device,
        )

        logger.info("="*80)
        logger.info("Generated Samples")
        logger.info("="*80)
        for i, (prompt, generated) in enumerate(zip(args.prompts, generated_texts)):
            logger.info(f"\n[Sample {i+1}]")
            logger.info(f"Prompt: {prompt}")
            logger.info(f"Generated: {generated}")
            logger.info("-"*80)


if __name__ == "__main__":
    main()
