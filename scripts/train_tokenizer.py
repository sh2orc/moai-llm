"""
SentencePiece Tokenizer Training Script for MOAI-LLM.

This script trains a BPE tokenizer optimized for multilingual text
with special focus on Korean, Chinese, and Japanese.

Usage:
    python scripts/train_tokenizer.py \
        --input_files data/train.txt \
        --vocab_size 128000 \
        --model_prefix moai_tokenizer \
        --output_dir tokenizers/

References:
- SentencePiece: https://github.com/google/sentencepiece
- BPE-dropout: https://arxiv.org/abs/1910.13267
"""

import argparse
import sentencepiece as spm
from pathlib import Path
from typing import List, Optional
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_sentencepiece_tokenizer(
    input_files: List[str],
    vocab_size: int,
    model_prefix: str,
    output_dir: str,
    model_type: str = "bpe",
    character_coverage: float = 0.9995,
    num_threads: int = 16,
    max_sentence_length: int = 16384,
    shuffle_input_sentence: bool = True,
    train_extremely_large_corpus: bool = False,
    add_dummy_prefix: bool = True,
    remove_extra_whitespaces: bool = True,
    normalization_rule_name: str = "nmt_nfkc_cf",
    bpe_dropout: float = 0.1,
    user_defined_symbols: Optional[List[str]] = None,
):
    """
    Train SentencePiece BPE tokenizer.

    Args:
        input_files: List of input text files for training
        vocab_size: Target vocabulary size (e.g., 128000)
        model_prefix: Prefix for output model files
        output_dir: Directory to save tokenizer model
        model_type: Type of tokenizer ('bpe' or 'unigram')
        character_coverage: Character coverage for vocabulary (0.9995 for multilingual)
        num_threads: Number of threads for training
        max_sentence_length: Maximum sentence length to process
        shuffle_input_sentence: Whether to shuffle input sentences
        train_extremely_large_corpus: Enable for very large corpora (>10M sentences)
        add_dummy_prefix: Add dummy whitespace at beginning (recommended for BPE)
        remove_extra_whitespaces: Remove extra whitespaces
        normalization_rule_name: Text normalization rule ('nmt_nfkc_cf' recommended)
        bpe_dropout: BPE-dropout probability for regularization (0.0-0.1)
        user_defined_symbols: Additional symbols to preserve (e.g., special tokens)
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Full model prefix path
    full_model_prefix = str(output_path / model_prefix)

    # Prepare input files string
    input_str = ",".join(input_files)

    logger.info("="*80)
    logger.info("Training SentencePiece Tokenizer")
    logger.info("="*80)
    logger.info(f"Input files: {input_files}")
    logger.info(f"Vocabulary size: {vocab_size:,}")
    logger.info(f"Model type: {model_type}")
    logger.info(f"Character coverage: {character_coverage}")
    logger.info(f"BPE dropout: {bpe_dropout}")
    logger.info(f"Output: {full_model_prefix}")
    logger.info("="*80)

    # Special tokens
    # Following Qwen/Llama convention
    special_tokens = [
        "<|endoftext|>",  # End of text
        "<|im_start|>",   # Instruction message start
        "<|im_end|>",     # Instruction message end
    ]

    if user_defined_symbols:
        special_tokens.extend(user_defined_symbols)

    # Training arguments
    train_args = {
        "input": input_str,
        "model_prefix": full_model_prefix,
        "model_type": model_type,
        "vocab_size": vocab_size,
        "character_coverage": character_coverage,
        "num_threads": num_threads,
        "max_sentence_length": max_sentence_length,
        "shuffle_input_sentence": shuffle_input_sentence,
        "train_extremely_large_corpus": train_extremely_large_corpus,
        "add_dummy_prefix": add_dummy_prefix,
        "remove_extra_whitespaces": remove_extra_whitespaces,
        "normalization_rule_name": normalization_rule_name,
        # Control symbols
        "pad_id": 0,
        "bos_id": 1,
        "eos_id": 2,
        "unk_id": 3,
        "user_defined_symbols": ",".join(special_tokens),
        # BPE-specific parameters
        "split_digits": True,  # Treat each digit as separate token
        "split_by_unicode_script": True,  # Split by Unicode script (good for CJK)
        "split_by_whitespace": True,
        "split_by_number": True,
        "byte_fallback": True,  # Handle unknown bytes gracefully
        # Sampling parameters for BPE-dropout
        "enable_differential_privacy": False,
        "differential_privacy_noise_level": 0.0,
        # Optional: BPE-dropout (set to 0 during training, use during fine-tuning)
        # Note: BPE-dropout is applied during tokenization, not training
    }

    try:
        # Train tokenizer
        logger.info("Starting tokenizer training...")
        spm.SentencePieceTrainer.train(**train_args)
        logger.info("✓ Tokenizer training completed successfully!")

        # Load and test tokenizer
        sp = spm.SentencePieceProcessor()
        sp.load(f"{full_model_prefix}.model")

        # Print vocabulary statistics
        logger.info("="*80)
        logger.info("Tokenizer Statistics")
        logger.info("="*80)
        logger.info(f"Vocabulary size: {sp.vocab_size():,}")
        logger.info(f"BOS token: {sp.id_to_piece(sp.bos_id())} (id={sp.bos_id()})")
        logger.info(f"EOS token: {sp.id_to_piece(sp.eos_id())} (id={sp.eos_id()})")
        logger.info(f"UNK token: {sp.id_to_piece(sp.unk_id())} (id={sp.unk_id()})")
        logger.info(f"PAD token: {sp.id_to_piece(sp.pad_id())} (id={sp.pad_id()})")

        # Test tokenization with multilingual samples
        test_samples = [
            "Hello, world! This is a test.",
            "안녕하세요, 세상! 이것은 테스트입니다.",
            "你好，世界！这是一个测试。",
            "こんにちは、世界！これはテストです。",
        ]

        logger.info("="*80)
        logger.info("Tokenization Examples")
        logger.info("="*80)
        for sample in test_samples:
            tokens = sp.encode(sample, out_type=str)
            ids = sp.encode(sample, out_type=int)
            logger.info(f"\nText: {sample}")
            logger.info(f"Tokens ({len(tokens)}): {tokens[:20]}{'...' if len(tokens) > 20 else ''}")
            logger.info(f"IDs: {ids[:20]}{'...' if len(ids) > 20 else ''}")

        logger.info("="*80)
        logger.info(f"✓ Tokenizer saved to: {full_model_prefix}.model")
        logger.info(f"✓ Vocabulary saved to: {full_model_prefix}.vocab")
        logger.info("="*80)

    except Exception as e:
        logger.error(f"✗ Error during tokenizer training: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Train SentencePiece tokenizer for MOAI-LLM"
    )

    # Required arguments
    parser.add_argument(
        "--input_files",
        type=str,
        nargs="+",
        required=True,
        help="Input text files for training (space-separated)",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=128000,
        help="Target vocabulary size (default: 128000)",
    )

    # Optional arguments
    parser.add_argument(
        "--model_prefix",
        type=str,
        default="moai_tokenizer",
        help="Prefix for output model files (default: moai_tokenizer)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tokenizers",
        help="Output directory (default: tokenizers)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="bpe",
        choices=["bpe", "unigram"],
        help="Tokenizer type (default: bpe)",
    )
    parser.add_argument(
        "--character_coverage",
        type=float,
        default=0.9995,
        help="Character coverage (default: 0.9995 for multilingual)",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=16,
        help="Number of threads (default: 16)",
    )
    parser.add_argument(
        "--max_sentence_length",
        type=int,
        default=16384,
        help="Maximum sentence length (default: 16384)",
    )
    parser.add_argument(
        "--bpe_dropout",
        type=float,
        default=0.1,
        help="BPE-dropout probability (default: 0.1)",
    )
    parser.add_argument(
        "--user_defined_symbols",
        type=str,
        nargs="*",
        default=None,
        help="Additional user-defined symbols",
    )
    parser.add_argument(
        "--train_extremely_large_corpus",
        action="store_true",
        help="Enable for extremely large corpus (>10M sentences)",
    )

    args = parser.parse_args()

    # Train tokenizer
    train_sentencepiece_tokenizer(
        input_files=args.input_files,
        vocab_size=args.vocab_size,
        model_prefix=args.model_prefix,
        output_dir=args.output_dir,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
        num_threads=args.num_threads,
        max_sentence_length=args.max_sentence_length,
        bpe_dropout=args.bpe_dropout,
        user_defined_symbols=args.user_defined_symbols,
        train_extremely_large_corpus=args.train_extremely_large_corpus,
    )


if __name__ == "__main__":
    main()
