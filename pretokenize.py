#!/usr/bin/env python3
"""
사전 토크나이징 스크립트
- 학습 전에 한 번만 실행
- 멀티프로세싱으로 빠른 처리
- 토크나이징된 데이터를 디스크에 저장
- 이후 모든 학습에서 재사용
"""

import os
import sys
import argparse
import hashlib
from pathlib import Path
import multiprocessing
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Pre-tokenize dataset for training")
    parser.add_argument("--dataset", type=str, required=True, help="HuggingFace dataset name")
    parser.add_argument("--tokenizer", type=str, required=True, help="Tokenizer path")
    parser.add_argument("--output_dir", type=str, default="./tokenized_data", help="Output directory")
    parser.add_argument("--num_proc", type=int, default=None, help="Number of processes (default: CPU count // 6)")
    parser.add_argument("--batch_size", type=int, default=10000, help="Batch size for tokenization")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--packing", action="store_true", help="Use sequence packing mode")
    parser.add_argument("--text_column", type=str, default="text", help="Text column name")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # CPU 코어 수 기반 최적 프로세스 수
    if args.num_proc is None:
        cpu_count = multiprocessing.cpu_count()
        args.num_proc = min(32, max(16, cpu_count // 6))
    
    print("="*80)
    print("🚀 Pre-tokenization Script")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Num processes: {args.num_proc}")
    print(f"Batch size: {args.batch_size}")
    print(f"Packing mode: {args.packing}")
    print(f"CPU cores: {multiprocessing.cpu_count()}")
    print()
    
    # TOKENIZERS_PARALLELISM 설정
    # multiprocessing 사용 시 datasets가 자동으로 false로 설정하지만 명시
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # datasets가 multiprocessing 시 강제 설정
    
    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 데이터셋 해시 생성
    dataset_hash = hashlib.md5(args.dataset.encode()).hexdigest()[:16]
    tokenized_path = output_dir / f"{dataset_hash}_tokenized"
    
    # 이미 토크나이징된 데이터가 있는지 확인
    if tokenized_path.exists():
        print(f"✅ Tokenized data already exists: {tokenized_path}")
        response = input("Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
        import shutil
        shutil.rmtree(tokenized_path)
    
    # 1. 토크나이저 로드
    print(f"📝 Loading tokenizer from {args.tokenizer}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    
    if not tokenizer.is_fast:
        print("⚠️  WARNING: Not using Fast Tokenizer! This will be slow.")
    else:
        print("✅ Using Fast Tokenizer (Rust-based)")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 2. 데이터셋 로드
    print(f"\n📚 Loading dataset: {args.dataset}...")
    dataset = load_dataset(args.dataset, split="train")
    print(f"✓ Loaded {len(dataset):,} samples")
    
    # 3. 토크나이징
    print(f"\n🔤 Tokenizing with {args.num_proc} processes...")
    print(f"   Batch size: {args.batch_size:,}")
    print(f"   This will take approximately: {len(dataset) / (args.num_proc * 7000) / 60:.1f} minutes")
    print()
    
    if args.packing:
        # Packing mode: 시퀀스 연결
        def batch_tokenize(examples):
            return tokenizer(
                examples[args.text_column],
                truncation=False,
                padding=False,
                add_special_tokens=True,
            )
        
        tokenized_ds = dataset.map(
            batch_tokenize,
            batched=True,
            batch_size=args.batch_size,
            num_proc=args.num_proc,
            remove_columns=dataset.column_names,
            desc="Tokenizing",
        )
        
        # 시퀀스 연결
        print("\n📦 Packing sequences...")
        def pack_sequences(examples):
            concatenated_ids = []
            current_length = 0
            current_batch = []
            
            for ids in examples["input_ids"]:
                if current_length + len(ids) <= args.max_seq_length:
                    current_batch.extend(ids)
                    current_length += len(ids)
                else:
                    if current_batch:
                        # 패딩
                        while len(current_batch) < args.max_seq_length:
                            current_batch.append(tokenizer.pad_token_id)
                        concatenated_ids.append(current_batch)
                    current_batch = ids[:args.max_seq_length]
                    current_length = len(current_batch)
            
            if current_batch:
                while len(current_batch) < args.max_seq_length:
                    current_batch.append(tokenizer.pad_token_id)
                concatenated_ids.append(current_batch)
            
            return {"input_ids": concatenated_ids}
        
        # 배치 처리
        all_packed = []
        batch_size = 50000
        for i in range(0, len(tokenized_ds), batch_size):
            batch = tokenized_ds[i:i+batch_size]
            packed = pack_sequences(batch)
            all_packed.extend(packed["input_ids"])
            if (i // batch_size + 1) % 10 == 0:
                print(f"  Packed {i+batch_size:,} / {len(tokenized_ds):,} samples...")
        
        final_dataset = Dataset.from_dict({"input_ids": all_packed})
        print(f"✓ Packed into {len(final_dataset):,} sequences")
        
    else:
        # 일반 mode: truncation
        def tokenize_function(examples):
            return tokenizer(
                examples[args.text_column],
                truncation=True,
                max_length=args.max_seq_length,
                padding=False,
            )
        
        final_dataset = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=args.batch_size,
            num_proc=args.num_proc,
            remove_columns=dataset.column_names,
            desc="Tokenizing",
        )
    
    # 4. 저장
    print(f"\n💾 Saving to {tokenized_path}...")
    final_dataset.save_to_disk(str(tokenized_path))
    
    # 메타데이터 저장
    metadata_path = tokenized_path / "metadata.txt"
    with open(metadata_path, 'w') as f:
        f.write(f"dataset={args.dataset}\n")
        f.write(f"tokenizer={args.tokenizer}\n")
        f.write(f"num_samples={len(final_dataset)}\n")
        f.write(f"max_seq_length={args.max_seq_length}\n")
        f.write(f"packing={args.packing}\n")
    
    print()
    print("="*80)
    print("✅ Pre-tokenization Complete!")
    print("="*80)
    print(f"Output: {tokenized_path}")
    print(f"Samples: {len(final_dataset):,}")
    print()
    print("📝 To use this in training, add to your train command:")
    print(f"   --pretokenized_data {tokenized_path}")
    print("="*80)

if __name__ == "__main__":
    main()

