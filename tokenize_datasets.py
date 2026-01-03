#!/usr/bin/env python3
"""
Tokenize Datasets Script (단일 프로세스)

모든 데이터셋을 단일 프로세스로 토크나이징하여 캐시에 저장합니다.
이후 train.py를 --skip_tokenization 플래그로 실행하면 캐시에서 바로 로드합니다.

Usage:
    python tokenize_datasets.py \\
        --dataset "dataset1" "dataset2" \\
        --tokenizer_path "tokenizers/moai" \\
        --max_seq_length 1024 \\
        --packing

이 스크립트는 torchrun 없이 단일 프로세스로 실행됩니다.
TOKENIZERS_PARALLELISM=true로 Rust Fast Tokenizer의 내부 병렬 처리를 최대한 활용합니다.
"""

# ⚠️ 중요: 모든 import 전에 TOKENIZERS_PARALLELISM 설정!
# tokenizers 라이브러리가 import 시점에 이 값을 캐싱하므로 가장 먼저 설정해야 함
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Rust Rayon 스레드 풀 최적화
cpu_count = os.cpu_count() or 8
os.environ["RAYON_NUM_THREADS"] = str(cpu_count)

import argparse
import hashlib
import logging
import sys
from pathlib import Path

# ============================================================================
# Logging Setup
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Import train.py functions
# ============================================================================
# train.py의 함수들을 재사용합니다
from train import (
    load_pretrain_dataset,
    tokenize_dataset,
    concatenate_sequences,
    get_cache_version_key,
    get_optimal_num_shards,
)

# ============================================================================
# Check for Rust tokenizer availability
# ============================================================================
def check_rust_tokenizer():
    """Check if Rust tokenizer binary is available"""
    rust_binary = Path(__file__).parent / "tokenize_rust" / "target" / "release" / "moai-tokenizer"
    return rust_binary.exists()

RUST_AVAILABLE = check_rust_tokenizer()


def tokenize_single_dataset(
    source: tuple,
    tokenizer,
    args,
    idx: int,
):
    """
    단일 데이터셋 토크나이징 (단일 프로세스)

    Args:
        source: ("hf", "dataset_name") 또는 ("file", "path")
        tokenizer: 토크나이저
        args: 인자
        idx: 소스 인덱스
    """
    import gc
    from datasets import Dataset as HFDataset

    src_type, src_name = source
    cache_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

    # 캐시 경로 계산
    cache_version = get_cache_version_key(
        tokenizer,
        additional_info=f"packing_{args.packing}_maxlen_{args.max_seq_length}_seq_{idx}"
    )
    dataset_hash = hashlib.md5(f"{src_name}_{cache_version}".encode()).hexdigest()[:16]
    tokenized_cache_path = Path(cache_home) / "datasets" / f"{dataset_hash}_tokenized"
    tokenized_marker = Path(cache_home) / "datasets" / f".{dataset_hash}_tokenized.marker"

    logger.info(f"📦 [{idx+1}] Dataset: {src_name}")

    # 캐시 확인
    if tokenized_cache_path.exists() and tokenized_marker.exists():
        logger.info(f"  ✅ Already tokenized (using cache)")
        tokenized_dataset = HFDataset.load_from_disk(str(tokenized_cache_path))
        num_samples = len(tokenized_dataset)
        logger.info(f"  ✅ Cached: {num_samples:,} samples")
        del tokenized_dataset
        gc.collect()
        return

    # 데이터셋 로드
    logger.info(f"  📚 Loading dataset...")
    if src_type == "hf":
        dataset, text_column = load_pretrain_dataset(
            dataset_names=[src_name],
            dataset_config=args.dataset_config if idx == 0 else None,
            train_files=None,
            text_column=args.text_column,
        )
    else:
        dataset, text_column = load_pretrain_dataset(
            dataset_names=None,
            dataset_config=None,
            train_files=[src_name],
            text_column=args.text_column,
        )

    # 토크나이징 - Rust 우선, Python fallback
    logger.info(f"  🔤 Tokenizing dataset...")

    if RUST_AVAILABLE:
        logger.info(f"  🚀 Using Rust tokenizer (ultra-fast mode)")
        from tokenize_rust_wrapper import tokenize_with_rust

        tokenized_ds = tokenize_with_rust(
            dataset=dataset["train"],
            tokenizer_path=args.tokenizer_path,
            text_column=text_column,
            max_seq_length=0 if args.packing else args.max_seq_length,
            batch_size=10000,
        )
    else:
        logger.info(f"  🐍 Using Python tokenizer (Rust not available)")
        logger.info(f"     TOKENIZERS_PARALLELISM={os.environ.get('TOKENIZERS_PARALLELISM', 'not set')}")
        logger.info(f"     Tip: Run ./setup_rust_tokenizer.sh for 25x faster tokenization!")
        tokenized_ds = tokenize_dataset(
            dataset=dataset["train"],
            tokenizer=tokenizer,
            text_column=text_column,
            max_seq_length=args.max_seq_length,
            packing=args.packing,
        )

    # Packing (선택적) - PyArrow 스트리밍 + 점진적 디스크 쓰기
    if args.packing:
        import tempfile
        import shutil
        from datasets import load_from_disk, concatenate_datasets

        logger.info(f"  📦 Packing sequences (incremental disk writing)...")

        total_samples = len(tokenized_ds)
        logger.info(f"     Total samples: {total_samples:,}")

        # 임시 디렉토리 생성
        temp_dir = Path(tempfile.mkdtemp(prefix="moai_packing_"))
        try:
            # 1. 토크나이징 결과를 임시 Arrow 파일로 저장
            logger.info(f"     Saving tokenized data to disk...")
            tokenized_ds.save_to_disk(str(temp_dir / "tokenized"))
            del tokenized_ds
            gc.collect()

            # 2. 배치별 packing 후 즉시 디스크에 저장
            STREAM_BATCH_SIZE = 500000  # 50만 샘플씩
            dataset_on_disk = load_from_disk(str(temp_dir / "tokenized"))
            num_batches = (total_samples + STREAM_BATCH_SIZE - 1) // STREAM_BATCH_SIZE

            logger.info(f"     Processing {num_batches} batches of {STREAM_BATCH_SIZE:,} samples")

            packed_shards_dir = temp_dir / "packed_shards"
            packed_shards_dir.mkdir()

            for batch_idx in range(num_batches):
                start_idx = batch_idx * STREAM_BATCH_SIZE
                end_idx = min(start_idx + STREAM_BATCH_SIZE, total_samples)

                logger.info(f"     Batch {batch_idx+1}/{num_batches}: Loading & packing {start_idx:,} - {end_idx:,}")

                # 배치 로드 및 packing
                batch_data = dataset_on_disk.select(range(start_idx, end_idx))
                tokenized_list = [{"input_ids": ids} for ids in batch_data["input_ids"]]
                del batch_data
                gc.collect()

                packed_batch = concatenate_sequences(
                    tokenized_sequences=tokenized_list,
                    max_seq_length=args.max_seq_length,
                    eos_token_id=tokenizer.eos_token_id,
                )
                del tokenized_list
                gc.collect()

                # 즉시 디스크에 저장 (메모리 누적 방지)
                packed_batch_ds = HFDataset.from_list(packed_batch)
                shard_path = packed_shards_dir / f"shard_{batch_idx:04d}"
                packed_batch_ds.save_to_disk(str(shard_path))

                logger.info(f"     ✓ Batch {batch_idx+1}/{num_batches} saved ({len(packed_batch):,} chunks)")

                del packed_batch, packed_batch_ds
                gc.collect()

            del dataset_on_disk
            gc.collect()

            # 3. 모든 샤드를 효율적으로 병합 (Arrow가 알아서 최적화)
            logger.info(f"     Merging {num_batches} shards...")
            shard_datasets = []
            for batch_idx in range(num_batches):
                shard_path = packed_shards_dir / f"shard_{batch_idx:04d}"
                shard_datasets.append(load_from_disk(str(shard_path)))

            tokenized_dataset = concatenate_datasets(shard_datasets)
            logger.info(f"  ✓ Total packed chunks: {len(tokenized_dataset):,}")

            del shard_datasets
            gc.collect()

        finally:
            # 임시 파일 삭제
            logger.info(f"     Cleaning up temporary files...")
            shutil.rmtree(temp_dir, ignore_errors=True)
    else:
        tokenized_dataset = tokenized_ds

    # 저장
    logger.info(f"  💾 Saving tokenized dataset...")
    num_shards = get_optimal_num_shards(len(tokenized_dataset), os.cpu_count() or 8)
    tokenized_dataset.save_to_disk(str(tokenized_cache_path), num_shards=num_shards)
    tokenized_marker.touch()
    num_samples = len(tokenized_dataset)
    logger.info(f"  ✅ Tokenized: {num_samples:,} samples (shards={num_shards})")

    # 메모리 해제
    del dataset
    del tokenized_dataset
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Tokenize datasets for MOAI-LLM")

    # 데이터셋 설정
    parser.add_argument("--dataset", nargs="+", help="HuggingFace dataset names")
    parser.add_argument("--dataset_config", type=str, help="Dataset config name")
    parser.add_argument("--train_file", nargs="+", help="Local training files (JSONL)")
    parser.add_argument("--text_column", type=str, default="text", help="Text column name")

    # 토크나이저 설정
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Tokenizer path")

    # 토크나이징 설정
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--packing", action="store_true", help="Enable sequence packing")

    args = parser.parse_args()

    # ========================================================================
    # STEP 1: 토크나이저 로드
    # ========================================================================
    logger.info("="*80)
    logger.info("🚀 MOAI-LLM Dataset Tokenization (Single Process)")
    logger.info("="*80)
    logger.info(f"Tokenizer: {args.tokenizer_path}")
    logger.info(f"Max seq length: {args.max_seq_length}")
    logger.info(f"Packing: {'Enabled' if args.packing else 'Disabled'}")
    logger.info("="*80)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    logger.info(f"✓ Tokenizer loaded: {tokenizer.__class__.__name__}")
    logger.info(f"  Vocab size: {tokenizer.vocab_size:,}")

    # ========================================================================
    # STEP 2: 데이터 소스 준비
    # ========================================================================
    all_sources = []
    if args.dataset:
        for ds in args.dataset:
            all_sources.append(("hf", ds))
    if args.train_file:
        for f in args.train_file:
            all_sources.append(("file", f))

    if not all_sources:
        logger.error("❌ No datasets specified! Use --dataset or --train_file")
        sys.exit(1)

    logger.info(f"📋 Total datasets: {len(all_sources)}")
    for i, (src_type, src_name) in enumerate(all_sources):
        logger.info(f"  {i+1}. [{src_type}] {src_name}")
    logger.info("="*80)

    # ========================================================================
    # STEP 3: 토크나이징
    # ========================================================================
    # TOKENIZERS_PARALLELISM 강제 설정 (Rust 내부 병렬 처리)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    logger.info("🔥 TOKENIZERS_PARALLELISM=true (Rust internal parallelism enabled)")
    logger.info("   Note: datasets library may still show 'Setting TOKENIZERS_PARALLELISM=false'")
    logger.info("   This is a library message, but Rust parallelism is active inside batch functions")
    logger.info("="*80)

    for idx, source in enumerate(all_sources):
        tokenize_single_dataset(source, tokenizer, args, idx)
        logger.info("")

    # ========================================================================
    # STEP 4: 완료
    # ========================================================================
    logger.info("="*80)
    logger.info("✅ All datasets tokenized successfully!")
    logger.info("="*80)
    logger.info("Next step: Run train.py with --skip_tokenization flag")
    logger.info("  Example:")
    logger.info(f"    torchrun --nproc_per_node=4 train.py \\")
    logger.info(f"      --skip_tokenization \\")
    logger.info(f"      --dataset {' '.join([src[1] for src in all_sources[:2]])} \\")
    logger.info(f"      --tokenizer_path {args.tokenizer_path} \\")
    logger.info(f"      ...")
    logger.info("="*80)


if __name__ == "__main__":
    main()
