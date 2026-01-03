#!/usr/bin/env python3
"""
Rust Tokenizer Wrapper for MOAI-LLM

Converts HuggingFace datasets to Parquet, calls Rust tokenizer, loads results.
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

def dataset_to_parquet(dataset, output_path: str, text_column: str = "text"):
    """Convert HuggingFace dataset to Parquet"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    print(f"   Converting dataset to Parquet ({len(dataset):,} samples)...")

    # Extract text column as Arrow table
    texts = dataset[text_column]

    # Create Arrow table
    table = pa.table({text_column: texts})

    # Write to Parquet
    pq.write_table(table, output_path, compression='snappy')
    print(f"   ✅ Saved to {output_path}")


def parquet_to_dataset(parquet_path: str):
    """Load tokenized Parquet back to HuggingFace dataset"""
    from datasets import Dataset
    import pyarrow.parquet as pq

    print(f"   Loading tokenized Parquet...")
    table = pq.read_table(parquet_path)

    # Convert to HuggingFace Dataset
    dataset = Dataset(table)
    print(f"   ✅ Loaded {len(dataset):,} samples")

    return dataset


def tokenize_with_rust(
    dataset,
    tokenizer_path: str,
    text_column: str = "text",
    max_seq_length: int = 0,
    batch_size: int = 10000,
):
    """
    Tokenize dataset using Rust binary

    Args:
        dataset: HuggingFace Dataset
        tokenizer_path: Path to tokenizer directory (must contain tokenizer.json)
        text_column: Text column name
        max_seq_length: Max sequence length (0 = no truncation for packing)
        batch_size: Batch size for Rust processing

    Returns:
        Tokenized HuggingFace Dataset
    """
    # Find Rust binary
    rust_binary = Path(__file__).parent / "tokenize_rust" / "target" / "release" / "moai-tokenizer"

    if not rust_binary.exists():
        print("❌ Rust binary not found!")
        print(f"   Expected: {rust_binary}")
        print("")
        print("   Please build it first:")
        print("   cd tokenize_rust && cargo build --release")
        sys.exit(1)

    # Find tokenizer.json
    tokenizer_json = Path(tokenizer_path) / "tokenizer.json"
    if not tokenizer_json.exists():
        print(f"❌ tokenizer.json not found at {tokenizer_json}")
        sys.exit(1)

    # Create temporary files
    with tempfile.TemporaryDirectory(prefix="moai_rust_") as tmpdir:
        input_parquet = Path(tmpdir) / "input.parquet"
        output_parquet = Path(tmpdir) / "output.parquet"

        # Convert to Parquet
        print("📦 Step 1: Converting dataset to Parquet...")
        dataset_to_parquet(dataset, str(input_parquet), text_column)

        # Call Rust binary
        print("🚀 Step 2: Running Rust tokenizer...")
        cmd = [
            str(rust_binary),
            "--input", str(input_parquet),
            "--output", str(output_parquet),
            "--tokenizer", str(tokenizer_json),
            "--text-column", text_column,
            "--max-length", str(max_seq_length),
            "--batch-size", str(batch_size),
        ]

        result = subprocess.run(cmd, check=True)

        if result.returncode != 0:
            print(f"❌ Rust tokenizer failed with code {result.returncode}")
            sys.exit(1)

        # Load results
        print("📥 Step 3: Loading tokenized dataset...")
        tokenized_dataset = parquet_to_dataset(str(output_parquet))

        return tokenized_dataset


def main():
    parser = argparse.ArgumentParser(description="Test Rust tokenizer")
    parser.add_argument("--dataset", required=True, help="HuggingFace dataset name")
    parser.add_argument("--dataset_config", help="Dataset config")
    parser.add_argument("--tokenizer_path", required=True, help="Tokenizer directory")
    parser.add_argument("--text_column", default="text", help="Text column name")
    parser.add_argument("--max_seq_length", type=int, default=0, help="Max sequence length (0=no truncation)")
    parser.add_argument("--batch_size", type=int, default=10000, help="Batch size")
    parser.add_argument("--max_samples", type=int, help="Limit samples for testing")

    args = parser.parse_args()

    # Load dataset
    from datasets import load_dataset

    print(f"📚 Loading dataset: {args.dataset}")
    if args.dataset_config:
        dataset = load_dataset(args.dataset, args.dataset_config, split="train")
    else:
        dataset = load_dataset(args.dataset, split="train")

    if args.max_samples:
        dataset = dataset.select(range(args.max_samples))

    print(f"✅ Dataset loaded: {len(dataset):,} samples")

    # Tokenize
    tokenized = tokenize_with_rust(
        dataset,
        tokenizer_path=args.tokenizer_path,
        text_column=args.text_column,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
    )

    print("")
    print("=" * 80)
    print("✅ Test completed!")
    print(f"   Input samples:  {len(dataset):,}")
    print(f"   Output samples: {len(tokenized):,}")
    print(f"   First 3 token counts: {[len(ids) for ids in tokenized['input_ids'][:3]]}")
    print("=" * 80)


if __name__ == "__main__":
    main()
